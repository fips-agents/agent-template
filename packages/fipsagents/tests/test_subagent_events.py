"""Tests for subagent-related StreamEvent variants and SSE serialization."""

from __future__ import annotations

import json
from typing import AsyncIterator

from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    StreamEvent,
    StreamMetrics,
    SubagentCompleted,
    SubagentDelta,
    SubagentFailed,
    SubagentInvoked,
)
from fipsagents.serialization.openai_sse import stream_events_as_sse


# ---------------------------------------------------------------------------
# Helpers (reusable from test_serialization_openai_sse)
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


# ---------------------------------------------------------------------------
# Tests: Event Construction
# ---------------------------------------------------------------------------


class TestSubagentInvokedEvent:
    def test_construction(self):
        ev = SubagentInvoked(
            agent_name="research_helper",
            task="Search for policy on remote work",
            span_id="span-123",
            transport="remote",
            depth=1,
        )
        assert ev.agent_name == "research_helper"
        assert ev.task == "Search for policy on remote work"
        assert ev.span_id == "span-123"
        assert ev.transport == "remote"
        assert ev.depth == 1

    def test_is_member_of_stream_event_union(self):
        ev: StreamEvent = SubagentInvoked(
            agent_name="test", task="task", span_id="s1", transport="remote", depth=0
        )
        assert isinstance(ev, SubagentInvoked)


class TestSubagentCompletedEvent:
    def test_construction(self):
        ev = SubagentCompleted(
            agent_name="research_helper",
            span_id="span-123",
            content="Found 3 relevant policies",
            tokens_used={"prompt": 500, "completion": 200},
            tool_calls_made=2,
            cost_usd=0.05,
        )
        assert ev.agent_name == "research_helper"
        assert ev.span_id == "span-123"
        assert ev.content == "Found 3 relevant policies"
        assert ev.tokens_used == {"prompt": 500, "completion": 200}
        assert ev.tool_calls_made == 2
        assert ev.cost_usd == 0.05

    def test_is_member_of_stream_event_union(self):
        ev: StreamEvent = SubagentCompleted(
            agent_name="test",
            span_id="s1",
            content="done",
            tokens_used={},
            tool_calls_made=0,
            cost_usd=0.0,
        )
        assert isinstance(ev, SubagentCompleted)


class TestSubagentFailedEvent:
    def test_construction(self):
        ev = SubagentFailed(
            agent_name="research_helper",
            span_id="span-123",
            error_type="Timeout",
            error_message="Request timed out after 60 seconds",
        )
        assert ev.agent_name == "research_helper"
        assert ev.span_id == "span-123"
        assert ev.error_type == "Timeout"
        assert ev.error_message == "Request timed out after 60 seconds"

    def test_is_member_of_stream_event_union(self):
        ev: StreamEvent = SubagentFailed(
            agent_name="test",
            span_id="s1",
            error_type="RemoteError",
            error_message="500 Internal Server Error",
        )
        assert isinstance(ev, SubagentFailed)


class TestSubagentDeltaEvent:
    def test_construction_with_content_delta(self):
        nested = ContentDelta(content="hello from subagent")
        ev = SubagentDelta(
            agent_name="worker", span_id="span-456", delta=nested
        )
        assert ev.agent_name == "worker"
        assert ev.span_id == "span-456"
        assert isinstance(ev.delta, ContentDelta)
        assert ev.delta.content == "hello from subagent"

    def test_is_member_of_stream_event_union(self):
        ev: StreamEvent = SubagentDelta(
            agent_name="test",
            span_id="s1",
            delta=ContentDelta(content="nested"),
        )
        assert isinstance(ev, SubagentDelta)


class TestStreamEventUnionWithSubagents:
    def test_all_subagent_variants_are_assignable(self):
        """Smoke check that all subagent variants round-trip through
        the union without type errors."""
        events: list[StreamEvent] = [
            SubagentInvoked(
                agent_name="a", task="t", span_id="s", transport="remote", depth=0
            ),
            SubagentCompleted(
                agent_name="a",
                span_id="s",
                content="done",
                tokens_used={},
                tool_calls_made=0,
                cost_usd=0.0,
            ),
            SubagentFailed(
                agent_name="a",
                span_id="s",
                error_type="Error",
                error_message="msg",
            ),
            SubagentDelta(
                agent_name="a",
                span_id="s",
                delta=ContentDelta(content="x"),
            ),
        ]
        assert len(events) == 4


# ---------------------------------------------------------------------------
# Tests: SSE Serialization
# ---------------------------------------------------------------------------


async def test_subagent_invoked_serializes_to_sse():
    events = [
        SubagentInvoked(
            agent_name="research_helper",
            task="Search policies",
            span_id="span-123",
            transport="remote",
            depth=1,
        ),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))

    # role + subagent_invoked + complete + usage + [DONE]
    assert len(frames) >= 3

    # Find the subagent frame (skip role chunk)
    subagent_frame = None
    for frame in frames:
        if isinstance(frame, dict) and "delta" in frame["choices"][0]:
            delta = frame["choices"][0]["delta"]
            if "subagent" in delta:
                subagent_frame = delta
                break

    assert subagent_frame is not None
    assert subagent_frame["subagent"]["type"] == "invoked"
    assert subagent_frame["subagent"]["agent_name"] == "research_helper"
    assert subagent_frame["subagent"]["task"] == "Search policies"
    assert subagent_frame["subagent"]["span_id"] == "span-123"
    assert subagent_frame["subagent"]["transport"] == "remote"
    assert subagent_frame["subagent"]["depth"] == 1


async def test_subagent_completed_serializes_to_sse():
    events = [
        SubagentCompleted(
            agent_name="research_helper",
            span_id="span-123",
            content="Found important policies",
            tokens_used={"prompt": 500, "completion": 200},
            tool_calls_made=2,
            cost_usd=0.05,
        ),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))

    subagent_frame = None
    for frame in frames:
        if isinstance(frame, dict) and "delta" in frame["choices"][0]:
            delta = frame["choices"][0]["delta"]
            if "subagent" in delta and delta["subagent"].get("type") == "completed":
                subagent_frame = delta
                break

    assert subagent_frame is not None
    assert subagent_frame["subagent"]["agent_name"] == "research_helper"
    assert subagent_frame["subagent"]["span_id"] == "span-123"
    assert subagent_frame["subagent"]["content"] == "Found important policies"
    assert subagent_frame["subagent"]["tokens_used"] == {"prompt": 500, "completion": 200}
    assert subagent_frame["subagent"]["tool_calls_made"] == 2
    assert subagent_frame["subagent"]["cost_usd"] == 0.05


async def test_subagent_failed_serializes_to_sse():
    events = [
        SubagentFailed(
            agent_name="research_helper",
            span_id="span-123",
            error_type="Timeout",
            error_message="Request timed out after 60 seconds",
        ),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))

    subagent_frame = None
    for frame in frames:
        if isinstance(frame, dict) and "delta" in frame["choices"][0]:
            delta = frame["choices"][0]["delta"]
            if "subagent" in delta and delta["subagent"].get("type") == "failed":
                subagent_frame = delta
                break

    assert subagent_frame is not None
    assert subagent_frame["subagent"]["agent_name"] == "research_helper"
    assert subagent_frame["subagent"]["span_id"] == "span-123"
    assert subagent_frame["subagent"]["error_type"] == "Timeout"
    assert subagent_frame["subagent"]["error_message"] == "Request timed out after 60 seconds"


async def test_subagent_delta_serializes_to_sse():
    # v1: SubagentDelta with a nested event serializes without raising
    nested_delta = ContentDelta(content="nested response")
    events = [
        SubagentDelta(
            agent_name="worker",
            span_id="span-456",
            delta=nested_delta,
        ),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))

    subagent_frame = None
    for frame in frames:
        if isinstance(frame, dict) and "delta" in frame["choices"][0]:
            delta = frame["choices"][0]["delta"]
            if "subagent" in delta and delta["subagent"].get("type") == "delta":
                subagent_frame = delta
                break

    assert subagent_frame is not None
    assert subagent_frame["subagent"]["agent_name"] == "worker"
    assert subagent_frame["subagent"]["span_id"] == "span-456"
    # v1 uses repr() placeholder; v2 will recursively serialize the delta
    assert "delta" in subagent_frame["subagent"]


async def test_mixed_subagent_and_content_events():
    """Test a realistic stream with both subagent events and regular content."""
    events = [
        SubagentInvoked(
            agent_name="helper", task="help", span_id="s1", transport="remote", depth=1
        ),
        SubagentCompleted(
            agent_name="helper",
            span_id="s1",
            content="result",
            tokens_used={},
            tool_calls_made=0,
            cost_usd=0.0,
        ),
        ContentDelta(content="Processed result:"),
        ContentDelta(content=" done"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    frames = await _collect(stream_events_as_sse(_iter(events), model_name="m"))

    # Should have: role, invoked, completed, content x2, complete, usage, [DONE]
    assert len(frames) >= 5
    assert frames[-1] == "[DONE]"
