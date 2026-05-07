"""Tests for ThinkTagParser and its integration with BaseAgent.astep_stream (#34).

Covers the streaming state machine that separates <think>…</think> reasoning
blocks from user-visible content, the create_reasoning_parser factory, and the
end-to-end flow through astep_stream.
"""

from __future__ import annotations

import pytest

from fipsagents.baseagent.reasoning import ThinkTagParser, create_reasoning_parser


# ---------------------------------------------------------------------------
# ThinkTagParser unit tests
# ---------------------------------------------------------------------------


def test_no_think_tags_passes_through_as_content():
    p = ThinkTagParser()
    assert p.feed("Hello world") == [("content", "Hello world")]


def test_complete_think_block_separated():
    p = ThinkTagParser()
    result = p.feed("<think>reasoning here</think>visible answer")
    assert result == [("reasoning", "reasoning here"), ("content", "visible answer")]


def test_think_block_at_start():
    p = ThinkTagParser()
    result = p.feed("<think>thinking</think>answer")
    assert result[0] == ("reasoning", "thinking")
    assert result[1] == ("content", "answer")


def test_content_before_and_after_think_block():
    p = ThinkTagParser()
    result = p.feed("before<think>middle</think>after")
    assert result == [
        ("content", "before"),
        ("reasoning", "middle"),
        ("content", "after"),
    ]


def test_tag_split_across_chunks():
    p = ThinkTagParser()
    # "<thi" is a prefix of "<think>" — the parser holds it back.
    result1 = p.feed("Hello <thi")
    assert result1 == [("content", "Hello ")]
    # Completing the open tag, then the close tag and trailing content.
    result2 = p.feed("nk>reasoning</think>done")
    assert result2 == [("reasoning", "reasoning"), ("content", "done")]


def test_close_tag_split_across_chunks():
    p = ThinkTagParser()
    # "</thi" is a prefix of "</think>" — held back while inside think block.
    result1 = p.feed("<think>thinking</thi")
    assert result1 == [("reasoning", "thinking")]
    result2 = p.feed("nk>answer")
    assert result2 == [("content", "answer")]


def test_multiple_think_blocks():
    p = ThinkTagParser()
    result = p.feed("<think>first</think>middle<think>second</think>end")
    assert result == [
        ("reasoning", "first"),
        ("content", "middle"),
        ("reasoning", "second"),
        ("content", "end"),
    ]


def test_unclosed_think_block_emits_as_reasoning():
    p = ThinkTagParser()
    # No partial close-tag boundary, so the text is emitted immediately.
    result = p.feed("<think>thinking forever")
    assert result == [("reasoning", "thinking forever")]
    # Nothing buffered — flush has nothing to add.
    assert p.flush() == []


def test_flush_emits_buffered_partial_tag():
    p = ThinkTagParser()
    # "</thi" is a prefix of "</think>", held back inside think block.
    result = p.feed("<think>thinking</thi")
    assert result == [("reasoning", "thinking")]
    flushed = p.flush()
    assert flushed == [("reasoning", "</thi")]


def test_reset_clears_state():
    p = ThinkTagParser()
    p.feed("<think>start")
    p.reset()
    result = p.feed("normal content")
    assert result == [("content", "normal content")]


def test_empty_think_block():
    p = ThinkTagParser()
    result = p.feed("<think></think>content")
    assert result == [("content", "content")]


def test_single_char_feeds():
    p = ThinkTagParser()
    source = "<think>hi</think>ok"
    all_results: list[tuple[str, str]] = []
    for ch in source:
        all_results.extend(p.feed(ch))
    all_results.extend(p.flush())

    reasoning = "".join(t for kind, t in all_results if kind == "reasoning")
    content = "".join(t for kind, t in all_results if kind == "content")
    assert reasoning == "hi", f"reasoning={reasoning!r}"
    assert content == "ok", f"content={content!r}"


# ---------------------------------------------------------------------------
# create_reasoning_parser tests
# ---------------------------------------------------------------------------


def test_creates_parser_for_granite():
    parser = create_reasoning_parser("RedHatAI/granite-3.3-8b-instruct")
    assert isinstance(parser, ThinkTagParser)


def test_creates_parser_for_deepseek():
    parser = create_reasoning_parser("deepseek-r1")
    assert isinstance(parser, ThinkTagParser)


def test_returns_none_for_gpt_oss():
    assert create_reasoning_parser("RedHatAI/gpt-oss-20b") is None


def test_returns_none_for_openai():
    assert create_reasoning_parser("gpt-4o") is None


def test_creates_implicit_open_parser_for_nemotron():
    """Nemotron's chat template opens <think> implicitly; parser must start
    already inside a think block."""
    parser = create_reasoning_parser("nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8")
    assert isinstance(parser, ThinkTagParser)
    # Empirically: a Nemotron stream starts with reasoning content and the
    # first explicit tag the model emits is the closing </think>.
    result = parser.feed("Okay, let me think about this...</think>final answer")
    assert result == [
        ("reasoning", "Okay, let me think about this..."),
        ("content", "final answer"),
    ]


# ---------------------------------------------------------------------------
# implicit_open mode
# ---------------------------------------------------------------------------


def test_implicit_open_starts_in_think_state():
    """Parser constructed with implicit_open=True treats opening content as
    reasoning until the first </think>."""
    p = ThinkTagParser(implicit_open=True)
    result = p.feed("thinking out loud</think>visible answer")
    assert result == [
        ("reasoning", "thinking out loud"),
        ("content", "visible answer"),
    ]


def test_implicit_open_handles_chunked_thinking():
    """Reasoning split across chunks accumulates correctly before the close."""
    p = ThinkTagParser(implicit_open=True)
    assert p.feed("first chunk ") == [("reasoning", "first chunk ")]
    assert p.feed("second chunk") == [("reasoning", "second chunk")]
    assert p.feed("</think>answer") == [("content", "answer")]


def test_implicit_open_close_tag_split_across_chunks():
    """Close-tag boundary handling still works with implicit_open=True."""
    p = ThinkTagParser(implicit_open=True)
    # "</thi" is held back as a possible close-tag prefix.
    assert p.feed("thinking</thi") == [("reasoning", "thinking")]
    assert p.feed("nk>answer") == [("content", "answer")]


def test_implicit_open_subsequent_think_block_recognised():
    """After the implicit block closes, a subsequent explicit <think>...</think>
    is parsed normally — the parser is a regular state machine again once
    the implicit block ends."""
    p = ThinkTagParser(implicit_open=True)
    result = p.feed("first thoughts</think>answer<think>more thoughts</think>more answer")
    assert result == [
        ("reasoning", "first thoughts"),
        ("content", "answer"),
        ("reasoning", "more thoughts"),
        ("content", "more answer"),
    ]


def test_implicit_open_reset_returns_to_in_think():
    """reset() between turns must restore the implicit-open initial state,
    not the default not-in-think state."""
    p = ThinkTagParser(implicit_open=True)
    p.feed("first turn thoughts</think>first turn answer")
    p.reset()
    # Next turn: again starts inside think.
    result = p.feed("second turn thoughts</think>second turn answer")
    assert result == [
        ("reasoning", "second turn thoughts"),
        ("content", "second turn answer"),
    ]


def test_implicit_open_unclosed_still_emits_as_reasoning():
    """Stream that ends without </think> emits everything as reasoning."""
    p = ThinkTagParser(implicit_open=True)
    result = p.feed("thinking forever and ever")
    assert result == [("reasoning", "thinking forever and ever")]
    assert p.flush() == []


def test_default_constructor_unchanged():
    """Backward compatibility: ThinkTagParser() with no args behaves
    identically to the pre-implicit_open implementation."""
    p = ThinkTagParser()
    # Plain content with no tags is content, not reasoning.
    assert p.feed("just some content") == [("content", "just some content")]


# ---------------------------------------------------------------------------
# Integration: astep_stream separates think tags end-to-end
# ---------------------------------------------------------------------------


def _chunk(*, content=None, finish_reason=None):
    """Build a minimal mock OpenAI streaming chunk with no tool calls."""
    from unittest.mock import MagicMock

    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = None
    delta.tool_calls = None
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


@pytest.mark.asyncio
async def test_astep_stream_separates_think_tags():
    """astep_stream emits ReasoningDelta/ContentDelta and strips tags from message."""
    from unittest.mock import MagicMock

    from fipsagents.baseagent import BaseAgent
    from fipsagents.baseagent.events import ContentDelta, ReasoningDelta
    from fipsagents.baseagent.tools import ToolRegistry

    agent = BaseAgent.__new__(BaseAgent)
    agent.messages = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "think about this"},
    ]
    agent.tools = ToolRegistry()
    agent._tool_inspector = None
    agent._reasoning_parser = ThinkTagParser()

    stream_chunks = [
        _chunk(content="<think>reasoning", finish_reason=None),
        _chunk(content=" text</think>", finish_reason=None),
        _chunk(content="visible answer", finish_reason=None),
        _chunk(finish_reason="stop"),
    ]

    async def mock_stream_raw(*args, **kwargs):
        for c in stream_chunks:
            yield c

    agent.llm = MagicMock()
    agent.llm.call_model_stream_raw = mock_stream_raw

    events = []
    async for event in agent.astep_stream():
        events.append(event)

    reasoning_events = [e for e in events if isinstance(e, ReasoningDelta)]
    content_events = [e for e in events if isinstance(e, ContentDelta)]

    full_reasoning = "".join(e.content for e in reasoning_events)
    full_content = "".join(e.content for e in content_events)

    assert full_reasoning == "reasoning text", (
        f"Expected 'reasoning text', got {full_reasoning!r}"
    )
    assert full_content == "visible answer", (
        f"Expected 'visible answer', got {full_content!r}"
    )

    # The assistant message stored in agent.messages must contain only the
    # visible answer — think tags and reasoning must be stripped out.
    assistant_msgs = [
        m for m in agent.messages
        if m.get("role") == "assistant" and not m.get("tool_calls")
    ]
    assert assistant_msgs, "Expected an assistant message in agent.messages"
    stored_content = assistant_msgs[-1].get("content", "")
    assert "think" not in stored_content.lower(), (
        f"Think tag leaked into stored message: {stored_content!r}"
    )
    assert "visible answer" in stored_content, (
        f"Visible answer missing from stored message: {stored_content!r}"
    )
