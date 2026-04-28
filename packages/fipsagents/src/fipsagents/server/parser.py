"""File content parsing for /v1/files uploads.

Two-tier dispatch by MIME type:

- ``text/*`` (plus a small allowlist of structured text types like
  ``application/json`` and ``application/x-ndjson``) → ``PlaintextParser``
  decodes UTF-8 with replacement; no external dependency required.
- Everything else → ``DoclingParser`` if docling is installed, or
  ``parse_status="skipped"`` if not.

Parsing is invoked inline at upload time. Docling runs on a thread via
``asyncio.to_thread`` so it does not block the event loop. Background
parsing with a job queue is a future enhancement; today's simplifying
assumption is that uploads are bounded and infrequent enough that
inline latency is acceptable.

Install the binary-format support with::

    pip install fipsagents[files]
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# Plaintext-coverable types beyond the obvious ``text/*`` prefix. These
# are common formats that decode cleanly as UTF-8 and do not need
# Docling's structural extraction.
_PLAINTEXT_EXTRA_TYPES: frozenset[str] = frozenset({
    "application/json",
    "application/x-ndjson",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
})


def _is_plaintext_mime(mime_type: str) -> bool:
    return mime_type.startswith("text/") or mime_type in _PLAINTEXT_EXTRA_TYPES


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ParseOutcome:
    """Result of a parse attempt.

    Mirrors the ``parse_status`` lifecycle on :class:`FileRecord`:

    - ``completed`` — ``text`` is populated.
    - ``skipped``   — parser cannot handle this MIME type.
    - ``failed``    — parser tried and raised; ``error`` describes why.
    """

    __slots__ = ("status", "text", "error")

    def __init__(
        self,
        status: str,
        text: str | None = None,
        error: str | None = None,
    ) -> None:
        self.status = status
        self.text = text
        self.error = error

    @classmethod
    def completed(cls, text: str) -> "ParseOutcome":
        return cls("completed", text=text)

    @classmethod
    def skipped(cls, reason: str = "") -> "ParseOutcome":
        return cls("skipped", error=reason or None)

    @classmethod
    def failed(cls, error: str) -> "ParseOutcome":
        return cls("failed", error=error)

    def __repr__(self) -> str:  # pragma: no cover — debugging aid only
        return (
            f"ParseOutcome(status={self.status!r}, "
            f"text={'<{}b>'.format(len(self.text)) if self.text else None}, "
            f"error={self.error!r})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FileParser(ABC):
    """Pluggable file content extractor."""

    @abstractmethod
    async def parse(
        self, data: bytes, *, mime_type: str, filename: str,
    ) -> ParseOutcome:
        """Parse *data* into plaintext, dispatching by MIME type."""


# ---------------------------------------------------------------------------
# Null parser
# ---------------------------------------------------------------------------


class NullParser(FileParser):
    """Skip every file. Used when parsing is disabled in config."""

    async def parse(
        self, data: bytes, *, mime_type: str, filename: str,
    ) -> ParseOutcome:
        return ParseOutcome.skipped("parser disabled")


# ---------------------------------------------------------------------------
# Plaintext parser
# ---------------------------------------------------------------------------


class PlaintextParser(FileParser):
    """Decode UTF-8 with replacement for text-shaped MIME types.

    Falls through to ``skipped`` for binary formats so a higher-level
    parser (Docling) can take over.
    """

    async def parse(
        self, data: bytes, *, mime_type: str, filename: str,
    ) -> ParseOutcome:
        if not _is_plaintext_mime(mime_type):
            return ParseOutcome.skipped(
                f"plaintext parser: '{mime_type}' is not text-shaped",
            )
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover — replace never raises
            return ParseOutcome.failed(f"decode error: {exc}")
        return ParseOutcome.completed(text)


# ---------------------------------------------------------------------------
# Docling parser
# ---------------------------------------------------------------------------


class DoclingParser(FileParser):
    """Extract text from binary formats (PDF, DOCX, PPTX, etc.) via Docling.

    Falls through to plaintext if the input looks text-shaped, since
    Docling's pipeline is overkill (and slow) for ``.txt`` / ``.md``.
    """

    def __init__(self) -> None:
        self._converter: object | None = None
        self._plaintext = PlaintextParser()

    def _ensure_converter(self) -> object:
        if self._converter is not None:
            return self._converter
        # Lazy import: docling pulls in heavy ML deps (transformers,
        # torch). Only loaded when the [files] extra is installed.
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover — surfaced via factory
            raise ImportError(
                "DoclingParser requires the [files] extra. "
                "Install with: pip install 'fipsagents[files]'"
            ) from exc
        self._converter = DocumentConverter()
        return self._converter

    async def parse(
        self, data: bytes, *, mime_type: str, filename: str,
    ) -> ParseOutcome:
        if _is_plaintext_mime(mime_type):
            return await self._plaintext.parse(
                data, mime_type=mime_type, filename=filename,
            )
        try:
            text = await asyncio.to_thread(
                self._convert_sync, data, filename, mime_type,
            )
        except Exception as exc:
            logger.warning(
                "DoclingParser: failed to parse %s (%s): %s",
                filename, mime_type, exc,
            )
            return ParseOutcome.failed(
                f"docling: {type(exc).__name__}: {exc}",
            )
        if not text:
            return ParseOutcome.skipped("docling returned empty text")
        return ParseOutcome.completed(text)

    def _convert_sync(self, data: bytes, filename: str, mime_type: str) -> str:
        """Synchronous conversion — runs in a worker thread.

        Docling's API takes file paths or streams; the simplest
        cross-version approach is to write to a temp file and let it
        infer the format from the extension.
        """
        import os
        import tempfile

        converter = self._ensure_converter()
        # Preserve any extension on the filename so docling can dispatch
        # by suffix; if there's no extension, fall back to a guess from
        # the MIME type.
        suffix = ""
        if "." in filename:
            suffix = "." + filename.rsplit(".", 1)[-1]
        elif "/" in mime_type:
            suffix = "." + mime_type.split("/", 1)[1].split("+", 1)[0]

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False,
        ) as fh:
            tmp_path = fh.name
            fh.write(data)
        try:
            result = converter.convert(tmp_path)  # type: ignore[attr-defined]
            doc = getattr(result, "document", result)
            # Most stable text export across docling versions.
            return doc.export_to_markdown()  # type: ignore[no-any-return]
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _docling_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def create_parser(*, enabled: bool = True) -> FileParser:
    """Create a parser based on what's available in the environment.

    - ``enabled=False`` → :class:`NullParser` (every file marked skipped).
    - ``docling`` installed → :class:`DoclingParser` (handles plaintext
      internally too).
    - Otherwise → :class:`PlaintextParser` (binaries get marked skipped).
    """
    if not enabled:
        return NullParser()
    if _docling_available():
        return DoclingParser()
    logger.info(
        "create_parser: docling not installed; binary formats will be "
        "skipped. Install fipsagents[files] for PDF/DOCX/etc support.",
    )
    return PlaintextParser()
