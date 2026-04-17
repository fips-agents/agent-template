"""Tests for fipsagents.serialization.openai_sse.stream_events_as_sse."""

from __future__ import annotations

import json
import re
from typing import AsyncIterator

from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.serialization.openai_sse import stream_events_as_sse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _iter(items):
    """Build an async iterator from a plain list."""
    for item in items:
        yield item


async def _collect(gen: AsyncIterator[str]) -> list[dict | str]:
    """Consume SSE output, parsing JSON frames; return raw string for [DONE]."""
    results = []
    async for frame in gen:
        assert frame.startswith("data: "), f"Unexpected frame: {frame!r}"
        payload = frame[len("data: "):].rstrip("\n")
        if payload == "[DONE]":
            results.append("[DONE]")
        else:
            results.append(json.loads(payload))
    return results


def _delta(parsed: dict) -> dict:
    """Extract the delta dict from a parsed chunk. Empty-choices chunks
    (e.g. the trailing usage chunk) return ``{}`` so filter expressions
    can safely call this on any chunk."""
    choices = parsed.get("choices") or []
    if not choices:
        return {}
    return choices[0]["delta"]


def _finish_reason(parsed: dict) -> str | None:
    choices = parsed.get("choices") or []
    if not choices:
        return None
    return choices[0]["finish_reason"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_empty_stream_still_emits_role_and_done():
    frames = await _collect(stream_events_as_sse(_iter([]), model_name="test-model"))
    # Role chunk + [DONE]
    assert len(frames) == 2
    assert _delta(frames[0]) == {"role": "assistant"}
    assert frames[1] == "[DONE]"


async def test_content_delta_becomes_content_chunk():
    events = [ContentDelta(content="hi")]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    # role chunk, content chunk, [DONE]
    assert len(frames) == 3
    assert _delta(frames[1]) == {"content": "hi"}
    assert frames[2] == "[DONE]"


async def test_reasoning_delta_becomes_reasoning_content_chunk():
    events = [ReasoningDelta(content="thinking")]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    assert len(frames) == 3
    assert _delta(frames[1]) == {"reasoning_content": "thinking"}


async def test_first_tool_call_delta_carries_id_name_args():
    events = [
        ToolCallDelta(index=0, call_id="call_1", name="search", arguments_delta='{"q":')
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    # role, tool_call opening, [DONE]
    assert len(frames) == 3
    tc = _delta(frames[1])["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    assert tc["function"]["arguments"] == '{"q":'
    assert tc["index"] == 0


async def test_subsequent_tool_call_delta_carries_only_arguments():
    events = [
        ToolCallDelta(index=0, call_id="call_1", name="search", arguments_delta='{"q":'),
        ToolCallDelta(index=0, arguments_delta='"foo"}'),
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    # role, opening, continuation, [DONE]
    assert len(frames) == 4
    continuation = _delta(frames[2])["tool_calls"][0]
    assert "id" not in continuation
    assert "type" not in continuation
    assert continuation["function"] == {"arguments": '"foo"}'}


async def test_two_tool_calls_on_different_indexes():
    events = [
        ToolCallDelta(index=0, call_id="call_0", name="search", arguments_delta="{}"),
        ToolCallDelta(index=1, call_id="call_1", name="lookup", arguments_delta="{}"),
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    # role, tc0 opening, tc1 opening, [DONE]
    assert len(frames) == 4
    tc0 = _delta(frames[1])["tool_calls"][0]
    tc1 = _delta(frames[2])["tool_calls"][0]
    assert tc0["id"] == "call_0"
    assert tc0["function"]["name"] == "search"
    assert tc1["id"] == "call_1"
    assert tc1["function"]["name"] == "lookup"
    assert tc1["index"] == 1


async def test_tool_result_event_becomes_tool_role_chunk():
    events = [ToolResultEvent(call_id="call_1", name="search", content="result text")]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    assert len(frames) == 3
    d = _delta(frames[1])
    assert d["role"] == "tool"
    assert d["tool_call_id"] == "call_1"
    assert d["content"] == "result text"


async def test_stream_complete_emits_finish_reason_then_usage_then_done():
    events = [StreamComplete(finish_reason="stop", metrics=StreamMetrics())]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    # role, finish_reason chunk, usage chunk, [DONE]
    assert len(frames) == 4
    assert _delta(frames[1]) == {}
    assert _finish_reason(frames[1]) == "stop"
    # Usage chunk: empty choices, usage + stream_metrics populated.
    assert frames[2]["choices"] == []
    assert "usage" in frames[2]
    assert "stream_metrics" in frames[2]
    assert frames[3] == "[DONE]"


async def test_usage_chunk_carries_token_counts_and_stream_metrics():
    metrics = StreamMetrics(
        time_to_first_reasoning=0.05,
        time_to_first_content=0.12,
        total_time=1.5,
        inter_token_latencies=[0.01, 0.02, 0.015],
        prompt_tokens=42,
        completion_tokens=17,
        total_tokens=59,
        model_calls=2,
        tool_calls=1,
    )
    events = [StreamComplete(finish_reason="stop", metrics=metrics)]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    usage_chunk = frames[2]
    assert usage_chunk["usage"] == {
        "prompt_tokens": 42,
        "completion_tokens": 17,
        "total_tokens": 59,
    }
    sm = usage_chunk["stream_metrics"]
    assert sm["time_to_first_reasoning"] == 0.05
    assert sm["time_to_first_content"] == 0.12
    assert sm["total_time"] == 1.5
    assert sm["inter_token_latencies"] == [0.01, 0.02, 0.015]
    assert sm["model_calls"] == 2
    assert sm["tool_calls"] == 1


async def test_usage_chunk_shares_completion_id_and_model_name():
    events = [StreamComplete(finish_reason="stop", metrics=StreamMetrics())]
    frames = await _collect(
        stream_events_as_sse(_iter(events), model_name="granite-8b", completion_id="cid-1")
    )
    json_frames = [f for f in frames if isinstance(f, dict)]
    assert all(f["id"] == "cid-1" for f in json_frames)
    assert all(f["model"] == "granite-8b" for f in json_frames)
    assert all(f["object"] == "chat.completion.chunk" for f in json_frames)


async def test_exception_in_source_yields_error_chunk_then_done():
    async def _failing():
        yield ContentDelta(content="partial")
        raise ValueError("boom")

    frames = await _collect(stream_events_as_sse(_failing(), model_name="m"))
    # role, content, error, [DONE]
    assert len(frames) == 4
    error_frame = frames[2]
    assert "error" in error_frame, f"Expected error frame, got: {error_frame}"
    assert error_frame["error"]["type"] == "ValueError"
    assert error_frame["error"]["message"] == "boom"
    assert frames[3] == "[DONE]"


async def test_completion_id_auto_generated_when_not_provided():
    events = [ContentDelta(content="x")]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))
    chunks = [f for f in frames if f != "[DONE]"]
    ids = {f["id"] for f in chunks}
    assert len(ids) == 1, "All chunks must share the same completion_id"
    (cid,) = ids
    assert re.fullmatch(r"chatcmpl-[0-9a-f]{24}", cid), f"Bad id format: {cid!r}"


async def test_completion_id_used_when_provided():
    events = [ContentDelta(content="x"), ContentDelta(content="y")]
    frames = await _collect(
        stream_events_as_sse(_iter(events), model_name="m", completion_id="custom-id")
    )
    chunks = [f for f in frames if f != "[DONE]"]
    assert all(f["id"] == "custom-id" for f in chunks)


async def test_model_name_included_in_every_chunk():
    events = [ContentDelta(content="x")]
    frames = await _collect(
        stream_events_as_sse(_iter(events), model_name="granite-8b")
    )
    chunks = [f for f in frames if f != "[DONE]"]
    assert all(f["model"] == "granite-8b" for f in chunks)
