"""Tests for fipsagents.server.parser."""


from __future__ import annotations

import sys

import pytest

from fipsagents.server.parser import (
    DoclingParser,
    FileParser,
    NullParser,
    ParseOutcome,
    PlaintextParser,
    _is_plaintext_mime,
    create_parser,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestIsPlaintextMime:
    @pytest.mark.parametrize("mime", [
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/html",
        "application/json",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    ])
    def test_recognises_text_shaped(self, mime):
        assert _is_plaintext_mime(mime) is True

    @pytest.mark.parametrize("mime", [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "image/png",
        "image/jpeg",
        "application/octet-stream",
    ])
    def test_rejects_binary(self, mime):
        assert _is_plaintext_mime(mime) is False


class TestParseOutcome:
    def test_completed_factory(self):
        out = ParseOutcome.completed("hello")
        assert out.status == "completed"
        assert out.text == "hello"
        assert out.error is None

    def test_skipped_factory(self):
        out = ParseOutcome.skipped("not my type")
        assert out.status == "skipped"
        assert out.text is None
        assert out.error == "not my type"

    def test_skipped_no_reason(self):
        out = ParseOutcome.skipped()
        assert out.status == "skipped"
        assert out.error is None

    def test_failed_factory(self):
        out = ParseOutcome.failed("boom")
        assert out.status == "failed"
        assert out.text is None
        assert out.error == "boom"


# ---------------------------------------------------------------------------
# NullParser
# ---------------------------------------------------------------------------


class TestNullParser:
    @pytest.mark.asyncio
    async def test_skips_everything(self):
        parser = NullParser()
        out = await parser.parse(
            b"hello", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "skipped"

    @pytest.mark.asyncio
    async def test_skips_binary(self):
        parser = NullParser()
        out = await parser.parse(
            b"\x00\x01", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "skipped"


# ---------------------------------------------------------------------------
# PlaintextParser
# ---------------------------------------------------------------------------


class TestPlaintextParser:
    @pytest.mark.asyncio
    async def test_decodes_utf8(self):
        parser = PlaintextParser()
        out = await parser.parse(
            b"hello world", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert out.text == "hello world"

    @pytest.mark.asyncio
    async def test_decodes_unicode(self):
        parser = PlaintextParser()
        body = "café — résumé".encode("utf-8")
        out = await parser.parse(
            body, mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert out.text == "café — résumé"

    @pytest.mark.asyncio
    async def test_replaces_invalid_bytes(self):
        parser = PlaintextParser()
        # 0xff is invalid UTF-8 mid-stream — should become U+FFFD.
        out = await parser.parse(
            b"hello\xffworld", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert "hello" in out.text
        assert "world" in out.text

    @pytest.mark.asyncio
    async def test_skips_pdf(self):
        parser = PlaintextParser()
        out = await parser.parse(
            b"%PDF-1.4", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "skipped"
        assert "not text-shaped" in out.error

    @pytest.mark.asyncio
    async def test_handles_json(self):
        parser = PlaintextParser()
        out = await parser.parse(
            b'{"hello": "world"}',
            mime_type="application/json",
            filename="x.json",
        )
        assert out.status == "completed"
        assert "hello" in out.text


# ---------------------------------------------------------------------------
# DoclingParser
# ---------------------------------------------------------------------------


class _FakeDoclingDoc:
    def __init__(self, text: str) -> None:
        self._text = text

    def export_to_markdown(self) -> str:
        return self._text


class _FakeDoclingResult:
    def __init__(self, text: str) -> None:
        self.document = _FakeDoclingDoc(text)


class _FakeConverter:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[str] = []

    def convert(self, path: str) -> _FakeDoclingResult:
        self.calls.append(path)
        return _FakeDoclingResult(self._text)


class TestDoclingParser:
    @pytest.mark.asyncio
    async def test_falls_through_to_plaintext(self):
        # Plaintext shouldn't even invoke docling.
        parser = DoclingParser()
        out = await parser.parse(
            b"hi there", mime_type="text/plain", filename="x.txt",
        )
        assert out.status == "completed"
        assert out.text == "hi there"
        # Converter never lazy-loaded.
        assert parser._converter is None

    @pytest.mark.asyncio
    async def test_invokes_converter_for_pdf(self):
        parser = DoclingParser()
        fake = _FakeConverter("# Heading\n\nbody text")
        parser._converter = fake  # bypass the lazy import

        out = await parser.parse(
            b"%PDF-1.4 stub", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "completed"
        assert out.text == "# Heading\n\nbody text"
        # Confirm the temp path docling saw kept the .pdf suffix.
        assert len(fake.calls) == 1
        assert fake.calls[0].endswith(".pdf")

    @pytest.mark.asyncio
    async def test_failed_converter_returns_failed_outcome(self):
        parser = DoclingParser()

        class _Boom:
            def convert(self, path):
                raise RuntimeError("docling exploded")

        parser._converter = _Boom()

        out = await parser.parse(
            b"%PDF-1.4 stub", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "failed"
        assert "RuntimeError" in out.error
        assert "docling exploded" in out.error

    @pytest.mark.asyncio
    async def test_empty_text_marks_skipped(self):
        parser = DoclingParser()
        parser._converter = _FakeConverter("")
        out = await parser.parse(
            b"%PDF-1.4 stub", mime_type="application/pdf", filename="x.pdf",
        )
        assert out.status == "skipped"
        assert "empty text" in out.error

    @pytest.mark.asyncio
    async def test_extension_inferred_from_mime_when_filename_lacks_one(self):
        parser = DoclingParser()
        fake = _FakeConverter("body")
        parser._converter = fake

        await parser.parse(
            b"data", mime_type="application/pdf", filename="noextension",
        )
        assert fake.calls[0].endswith(".pdf")

    @pytest.mark.asyncio
    async def test_lazy_import_raises_clear_error(self, monkeypatch):
        """Touching ._ensure_converter without docling installed should
        raise an ImportError that mentions the [files] extra."""
        parser = DoclingParser()
        # Simulate docling missing even if the host happens to have it.
        monkeypatch.setitem(sys.modules, "docling.document_converter", None)
        with pytest.raises(ImportError, match=r"\[files\] extra"):
            parser._ensure_converter()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateParser:
    def test_disabled_returns_null(self):
        parser = create_parser(enabled=False)
        assert isinstance(parser, NullParser)

    def test_enabled_with_docling_returns_docling(self, monkeypatch):
        # Pretend docling is importable.
        monkeypatch.setattr(
            "fipsagents.server.parser._docling_available", lambda: True,
        )
        parser = create_parser(enabled=True)
        assert isinstance(parser, DoclingParser)

    def test_enabled_without_docling_returns_plaintext(self, monkeypatch):
        monkeypatch.setattr(
            "fipsagents.server.parser._docling_available", lambda: False,
        )
        parser = create_parser(enabled=True)
        assert isinstance(parser, PlaintextParser)

    def test_real_environment_resolves_to_some_parser(self):
        """Smoke test against the actual environment — docling installed
        or not, factory must return a usable FileParser."""
        parser = create_parser(enabled=True)
        assert isinstance(parser, FileParser)
