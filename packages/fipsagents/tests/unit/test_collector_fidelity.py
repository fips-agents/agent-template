"""Tests for TraceCollector fidelity levels."""

from __future__ import annotations

import pytest

from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.server.collector import TraceCollector
from fipsagents.server.tracing import NullTraceStore


async def _aiter(items):
    for item in items:
        yield item


class TestMinimalFidelity:
    @pytest.mark.asyncio
    async def test_no_span_events_recorded(self):
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="minimal")
        collector.begin_request()
        events = [
            ContentDelta(content="Hello"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        for span in collector._spans:
            assert span.events == [], f"Span {span.name} has events at minimal fidelity"


class TestStandardFidelity:
    @pytest.mark.asyncio
    async def test_messages_snapshot_recorded(self):
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="standard")
        messages = [{"role": "user", "content": "Hello"}]
        collector.begin_request(messages=messages)
        events = [
            ContentDelta(content="Hi"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        request_span = next(s for s in collector._spans if s.name == "request")
        assert len(request_span.events) == 1
        assert request_span.events[0]["name"] == "messages_snapshot"

    @pytest.mark.asyncio
    async def test_tool_result_recorded(self):
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="standard")
        collector.begin_request()
        events = [
            ToolCallDelta(index=0, call_id="call_1", name="search"),
            ToolResultEvent(call_id="call_1", name="search", content="found it"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        tool_span = next(
            (s for s in collector._spans if s.name == "tool:search"), None,
        )
        assert tool_span is not None
        assert len(tool_span.events) == 1
        assert tool_span.events[0]["name"] == "tool_result"

    @pytest.mark.asyncio
    async def test_no_content_deltas_at_standard(self):
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="standard")
        collector.begin_request()
        events = [
            ContentDelta(content="chunk1"),
            ContentDelta(content="chunk2"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        model_span = next(
            (s for s in collector._spans if s.name == "model_call"), None,
        )
        assert model_span is not None
        assert model_span.events == []


class TestFullFidelity:
    @pytest.mark.asyncio
    async def test_content_deltas_recorded(self):
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="full")
        collector.begin_request()
        events = [
            ContentDelta(content="chunk1"),
            ContentDelta(content="chunk2"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        model_span = next(
            (s for s in collector._spans if s.name == "model_call"), None,
        )
        assert model_span is not None
        content_events = [e for e in model_span.events if e["name"] == "content_delta"]
        assert len(content_events) == 2

    @pytest.mark.asyncio
    async def test_reasoning_deltas_recorded(self):
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="full")
        collector.begin_request()
        events = [
            ReasoningDelta(content="thinking..."),
            ContentDelta(content="answer"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        model_span = next(
            (s for s in collector._spans if s.name == "model_call"), None,
        )
        assert model_span is not None
        reasoning_events = [e for e in model_span.events if e["name"] == "reasoning_delta"]
        assert len(reasoning_events) == 1

    @pytest.mark.asyncio
    async def test_tool_call_deltas_recorded(self):
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="full")
        collector.begin_request()
        events = [
            ToolCallDelta(index=0, call_id="call_1", name="search"),
            ToolCallDelta(index=0, arguments_delta='{"q":'),
            ToolCallDelta(index=0, arguments_delta='"hello"}'),
            ToolResultEvent(call_id="call_1", name="search", content="result"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        model_span = next(
            (s for s in collector._spans if s.name == "model_call"), None,
        )
        assert model_span is not None
        tc_events = [e for e in model_span.events if e["name"] == "tool_call_delta"]
        assert len(tc_events) == 3

    @pytest.mark.asyncio
    async def test_full_includes_messages_and_tool_results(self):
        """Full fidelity is a superset of standard."""
        store = NullTraceStore()
        collector = TraceCollector(store, fidelity="full")
        messages = [{"role": "user", "content": "test"}]
        collector.begin_request(messages=messages)
        events = [
            ToolCallDelta(index=0, call_id="c1", name="calc"),
            ToolResultEvent(call_id="c1", name="calc", content="42"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        observed = collector.observe(_aiter(events))
        async for _ in observed:
            pass
        await collector.end_request()
        request_span = next(s for s in collector._spans if s.name == "request")
        assert any(e["name"] == "messages_snapshot" for e in request_span.events)
        tool_span = next(s for s in collector._spans if s.name == "tool:calc")
        assert any(e["name"] == "tool_result" for e in tool_span.events)
