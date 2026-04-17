"""Tests for fipsagents.serialization.anthropic_messages."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.serialization.anthropic_messages import (
    stream_events_as_anthropic_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MSG_ID = "msg_test_123"
MODEL = "test-model"


async def _iter(items):
    """Build an async iterator from a plain list."""
    for item in items:
        yield item


async def _collect(
    gen: AsyncIterator[str],
) -> list[dict]:
    """Consume Anthropic named-event SSE output, returning parsed payloads.

    Each yielded string has the shape ``event: <type>\\ndata: <json>\\n\\n``.
    We parse the JSON and return it. The ``type`` field is already present
    inside each JSON payload, so no information is lost.
    """
    results: list[dict] = []
    async for frame in gen:
        assert frame.startswith("event: "), f"Unexpected frame: {frame!r}"
        lines = frame.strip().split("\n")
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {frame!r}"
        assert lines[1].startswith("data: "), f"Missing data line: {frame!r}"
        payload = json.loads(lines[1][len("data: "):])
        results.append(payload)
    return results


def _events_of_type(frames: list[dict], event_type: str) -> list[dict]:
    """Filter collected frames by their ``type`` field."""
    return [f for f in frames if f.get("type") == event_type]


def _stream(events: list | None = None) -> AsyncIterator[str]:
    """Shorthand to create the generator with default id/model."""
    return stream_events_as_anthropic_messages(
        _iter(events or []),
        message_id=MSG_ID,
        model_name=MODEL,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_empty_stream_emits_message_start_and_stop():
    """No events -> message_start, ping, message_delta, message_stop."""
    frames = await _collect(_stream([]))
    types = [f["type"] for f in frames]
    assert types[0] == "message_start"
    assert "ping" in types
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"


async def test_content_delta_becomes_text_block():
    frames = await _collect(_stream([
        ContentDelta(content="Hello"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 1
    assert starts[0]["content_block"]["type"] == "text"
    assert starts[0]["content_block"]["text"] == ""
    assert starts[0]["index"] == 0

    deltas = _events_of_type(frames, "content_block_delta")
    text_deltas = [d for d in deltas if d["delta"]["type"] == "text_delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["delta"]["text"] == "Hello"
    assert text_deltas[0]["index"] == 0


async def test_reasoning_delta_becomes_thinking_block():
    frames = await _collect(_stream([
        ReasoningDelta(content="Let me think..."),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 1
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[0]["content_block"]["thinking"] == ""

    deltas = _events_of_type(frames, "content_block_delta")
    thinking_deltas = [
        d for d in deltas if d["delta"]["type"] == "thinking_delta"
    ]
    assert len(thinking_deltas) == 1
    assert thinking_deltas[0]["delta"]["thinking"] == "Let me think..."


async def test_first_tool_call_delta_opens_tool_use_block():
    frames = await _collect(_stream([
        ToolCallDelta(
            index=0, call_id="toolu_abc", name="get_weather",
            arguments_delta='{"location":',
        ),
        StreamComplete(finish_reason="tool_calls", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 1
    block = starts[0]["content_block"]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_abc"
    assert block["name"] == "get_weather"
    assert block["input"] == {}


async def test_subsequent_tool_call_delta_emits_input_json_delta():
    frames = await _collect(_stream([
        ToolCallDelta(
            index=0, call_id="toolu_abc", name="get_weather",
            arguments_delta='{"loc',
        ),
        ToolCallDelta(index=0, arguments_delta='ation": "NYC"}'),
        StreamComplete(finish_reason="tool_calls", metrics=StreamMetrics()),
    ]))
    deltas = _events_of_type(frames, "content_block_delta")
    json_deltas = [
        d for d in deltas if d["delta"]["type"] == "input_json_delta"
    ]
    assert len(json_deltas) == 2
    assert json_deltas[0]["delta"]["partial_json"] == '{"loc'
    assert json_deltas[1]["delta"]["partial_json"] == 'ation": "NYC"}'


async def test_two_tool_calls_get_separate_blocks():
    frames = await _collect(_stream([
        ToolCallDelta(
            index=0, call_id="toolu_1", name="search",
            arguments_delta="{}",
        ),
        ToolCallDelta(
            index=1, call_id="toolu_2", name="lookup",
            arguments_delta="{}",
        ),
        StreamComplete(finish_reason="tool_calls", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 2
    assert starts[0]["content_block"]["id"] == "toolu_1"
    assert starts[0]["content_block"]["name"] == "search"
    assert starts[1]["content_block"]["id"] == "toolu_2"
    assert starts[1]["content_block"]["name"] == "lookup"
    # Different block indexes.
    assert starts[0]["index"] != starts[1]["index"]


async def test_tool_result_event_is_skipped():
    frames = await _collect(_stream([
        ToolResultEvent(call_id="call_1", name="search", content="result"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    # No content_block_start for the tool result.
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 0


async def test_stream_complete_closes_block_and_emits_message_delta_stop():
    frames = await _collect(_stream([
        ContentDelta(content="hi"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    types = [f["type"] for f in frames]
    # Should have: message_start, ping, content_block_start,
    # content_block_delta, content_block_stop, message_delta, message_stop
    assert "content_block_stop" in types
    stop_idx = types.index("content_block_stop")
    assert types[stop_idx + 1] == "message_delta"
    assert types[stop_idx + 2] == "message_stop"


async def test_usage_in_message_delta():
    metrics = StreamMetrics(
        prompt_tokens=42,
        completion_tokens=17,
        total_tokens=59,
    )
    frames = await _collect(_stream([
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]))
    msg_deltas = _events_of_type(frames, "message_delta")
    assert len(msg_deltas) == 1
    assert msg_deltas[0]["usage"]["output_tokens"] == 17


@pytest.mark.parametrize(
    "internal_reason, expected_reason",
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("length", "max_tokens"),
        ("custom_reason", "custom_reason"),
    ],
)
async def test_stop_reason_mapping(internal_reason, expected_reason):
    frames = await _collect(_stream([
        StreamComplete(finish_reason=internal_reason, metrics=StreamMetrics()),
    ]))
    msg_deltas = _events_of_type(frames, "message_delta")
    assert len(msg_deltas) == 1
    assert msg_deltas[0]["delta"]["stop_reason"] == expected_reason


async def test_exception_yields_error_then_stop():
    async def _failing():
        yield ContentDelta(content="partial")
        raise ValueError("boom")

    frames = await _collect(
        stream_events_as_anthropic_messages(
            _failing(), message_id=MSG_ID, model_name=MODEL,
        )
    )
    types = [f["type"] for f in frames]
    # Should end with: ..., content_block_stop (original), text block start
    # (error), text delta (error msg), content_block_stop (error),
    # message_delta, message_stop
    assert types[-1] == "message_stop"
    assert types[-2] == "message_delta"

    # Find the error text delta.
    text_deltas = [
        f for f in frames
        if f.get("type") == "content_block_delta"
        and f.get("delta", {}).get("type") == "text_delta"
    ]
    error_texts = [
        d for d in text_deltas
        if "Error" in d["delta"]["text"]
    ]
    assert len(error_texts) == 1
    assert "ValueError" in error_texts[0]["delta"]["text"]
    assert "boom" in error_texts[0]["delta"]["text"]


async def test_message_id_and_model_in_message_start():
    frames = await _collect(
        stream_events_as_anthropic_messages(
            _iter([]),
            message_id="msg_custom_42",
            model_name="granite-8b",
        )
    )
    msg_starts = _events_of_type(frames, "message_start")
    assert len(msg_starts) == 1
    msg = msg_starts[0]["message"]
    assert msg["id"] == "msg_custom_42"
    assert msg["model"] == "granite-8b"
    assert msg["role"] == "assistant"
    assert msg["type"] == "message"


async def test_block_transition_reasoning_to_content():
    frames = await _collect(_stream([
        ReasoningDelta(content="thinking..."),
        ContentDelta(content="answer"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 2
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[1]["content_block"]["type"] == "text"
    # Indexes are sequential.
    assert starts[0]["index"] == 0
    assert starts[1]["index"] == 1

    # A content_block_stop should appear between the thinking and text blocks.
    stops = _events_of_type(frames, "content_block_stop")
    thinking_stop = [s for s in stops if s["index"] == 0]
    assert len(thinking_stop) == 1


async def test_multiple_content_deltas_share_one_block():
    frames = await _collect(_stream([
        ContentDelta(content="Hello"),
        ContentDelta(content=" world"),
        ContentDelta(content="!"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    # Only one text block opened, not three.
    assert len(starts) == 1
    assert starts[0]["content_block"]["type"] == "text"

    deltas = _events_of_type(frames, "content_block_delta")
    text_deltas = [d for d in deltas if d["delta"]["type"] == "text_delta"]
    assert len(text_deltas) == 3
    assert text_deltas[0]["delta"]["text"] == "Hello"
    assert text_deltas[1]["delta"]["text"] == " world"
    assert text_deltas[2]["delta"]["text"] == "!"
    # All share the same block index.
    assert all(d["index"] == starts[0]["index"] for d in text_deltas)


async def test_stream_metrics_extension():
    metrics = StreamMetrics(
        time_to_first_reasoning=0.05,
        time_to_first_content=0.12,
        total_time=1.5,
        inter_token_latencies=[0.01, 0.02],
        prompt_tokens=42,
        completion_tokens=17,
        total_tokens=59,
        model_calls=2,
        tool_calls=1,
    )
    frames = await _collect(_stream([
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]))
    msg_deltas = _events_of_type(frames, "message_delta")
    assert len(msg_deltas) == 1
    sm = msg_deltas[0]["stream_metrics"]
    assert sm["time_to_first_reasoning"] == 0.05
    assert sm["time_to_first_content"] == 0.12
    assert sm["total_time"] == 1.5
    assert sm["inter_token_latencies"] == [0.01, 0.02]
    assert sm["prompt_tokens"] == 42
    assert sm["total_tokens"] == 59
    assert sm["model_calls"] == 2
    assert sm["tool_calls"] == 1


async def test_tool_call_delta_without_call_id_is_skipped():
    """First ToolCallDelta for an index with no call_id is skipped."""
    frames = await _collect(_stream([
        ToolCallDelta(index=0, arguments_delta='{"x": 1}'),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 0


async def test_interleaved_tool_call_deltas():
    """Deltas for two tool indexes interleaved: index 0, 1, 0, 1."""
    frames = await _collect(_stream([
        ToolCallDelta(index=0, call_id="toolu_a", name="search", arguments_delta='{"q":'),
        ToolCallDelta(index=1, call_id="toolu_b", name="lookup", arguments_delta='{"k":'),
        ToolCallDelta(index=0, arguments_delta='"foo"}'),
        ToolCallDelta(index=1, arguments_delta='"bar"}'),
        StreamComplete(finish_reason="tool_calls", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 2
    assert starts[0]["content_block"]["id"] == "toolu_a"
    assert starts[1]["content_block"]["id"] == "toolu_b"

    deltas = _events_of_type(frames, "content_block_delta")
    json_deltas = [d for d in deltas if d["delta"]["type"] == "input_json_delta"]
    assert len(json_deltas) == 4
    # Deltas reference the correct block indexes.
    block_a = starts[0]["index"]
    block_b = starts[1]["index"]
    assert json_deltas[0]["index"] == block_a
    assert json_deltas[1]["index"] == block_b
    assert json_deltas[2]["index"] == block_a
    assert json_deltas[3]["index"] == block_b


async def test_content_after_tool_calls():
    """Text block opens correctly after tool_use blocks."""
    frames = await _collect(_stream([
        ToolCallDelta(index=0, call_id="toolu_1", name="search", arguments_delta="{}"),
        ContentDelta(content="Based on the results..."),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 2
    assert starts[0]["content_block"]["type"] == "tool_use"
    assert starts[1]["content_block"]["type"] == "text"
    # Tool block index < text block index.
    assert starts[0]["index"] < starts[1]["index"]


async def test_tool_call_first_delta_empty_args_no_delta_emitted():
    """First ToolCallDelta with empty arguments_delta opens block but
    does not emit an input_json_delta frame."""
    frames = await _collect(_stream([
        ToolCallDelta(index=0, call_id="toolu_x", name="ping", arguments_delta=""),
        StreamComplete(finish_reason="tool_calls", metrics=StreamMetrics()),
    ]))
    starts = _events_of_type(frames, "content_block_start")
    assert len(starts) == 1
    assert starts[0]["content_block"]["name"] == "ping"

    deltas = _events_of_type(frames, "content_block_delta")
    json_deltas = [d for d in deltas if d["delta"]["type"] == "input_json_delta"]
    assert len(json_deltas) == 0
