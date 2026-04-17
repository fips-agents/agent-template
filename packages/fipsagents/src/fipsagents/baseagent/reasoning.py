"""Streaming parser for ``<think>…</think>`` reasoning blocks.

Some models (Granite, DeepSeek) embed chain-of-thought reasoning in the
content stream wrapped in ``<think>`` tags rather than using the
``reasoning_content`` delta field.  ``ThinkTagParser`` separates these
blocks so ``astep_stream`` can emit ``ReasoningDelta`` for thinking and
``ContentDelta`` for user-visible text.

When vLLM is started with ``--reasoning-parser granite``, it does this
extraction server-side and populates ``reasoning_content`` directly.
The parser is a fallback for deployments that don't set that flag.
"""

from __future__ import annotations

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _suffix_prefix_len(text: str, tag: str) -> int:
    """Length of the longest suffix of *text* that is a prefix of *tag*.

    Used to detect partial tag boundaries at chunk edges.  For example,
    ``_suffix_prefix_len("hello <thi", "<think>")`` returns 4 because
    ``"<thi"`` matches the first 4 characters of ``"<think>"``.
    """
    max_check = min(len(text), len(tag) - 1)
    for length in range(max_check, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0


class ThinkTagParser:
    """Streaming state machine that separates ``<think>`` blocks from content.

    Call :meth:`feed` with each content delta as it arrives.  Returns a
    list of ``("reasoning", text)`` and ``("content", text)`` tuples.
    Call :meth:`flush` after the stream ends to emit any buffered tail.

    Handles tags split across chunk boundaries and multiple think blocks
    in a single response.
    """

    __slots__ = ("_buf", "_in_think")

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def reset(self) -> None:
        """Reset parser state between model calls."""
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Process a content delta and return separated segments."""
        self._buf += text
        results: list[tuple[str, str]] = []

        while True:
            if self._in_think:
                idx = self._buf.find(_CLOSE_TAG)
                if idx == -1:
                    holdback = _suffix_prefix_len(self._buf, _CLOSE_TAG)
                    if holdback > 0:
                        emit = self._buf[:-holdback]
                        self._buf = self._buf[-holdback:]
                    else:
                        emit = self._buf
                        self._buf = ""
                    if emit:
                        results.append(("reasoning", emit))
                    break
                else:
                    if idx > 0:
                        results.append(("reasoning", self._buf[:idx]))
                    self._buf = self._buf[idx + len(_CLOSE_TAG) :]
                    self._in_think = False
            else:
                idx = self._buf.find(_OPEN_TAG)
                if idx == -1:
                    holdback = _suffix_prefix_len(self._buf, _OPEN_TAG)
                    if holdback > 0:
                        emit = self._buf[:-holdback]
                        self._buf = self._buf[-holdback:]
                    else:
                        emit = self._buf
                        self._buf = ""
                    if emit:
                        results.append(("content", emit))
                    break
                else:
                    if idx > 0:
                        results.append(("content", self._buf[:idx]))
                    self._buf = self._buf[idx + len(_OPEN_TAG) :]
                    self._in_think = True

        return results

    def flush(self) -> list[tuple[str, str]]:
        """Emit any remaining buffered text."""
        if not self._buf:
            return []
        kind = "reasoning" if self._in_think else "content"
        result = [(kind, self._buf)]
        self._buf = ""
        self._in_think = False
        return result


def create_reasoning_parser(model_name: str) -> ThinkTagParser | None:
    """Return a ``ThinkTagParser`` if the model is known to use think tags.

    Returns ``None`` for models that emit ``reasoning_content`` natively
    (or don't support reasoning at all).
    """
    name = model_name.lower()
    if "granite" in name or "deepseek" in name:
        return ThinkTagParser()
    return None
