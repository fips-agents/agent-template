"""Reproduction test for #185: subagent SSE events dropped by observers.

Wires the full observer chain (MetricsCollector, TraceCollector) around a
mock event stream containing SubagentInvoked/SubagentCompleted/SubagentFailed
events, then serialises to SSE. If the observers swallow events, the SSE
output will be missing the ``subagent`` delta chunks.

Result: determines whether #185 is reproducible in the observer chain or is
environment/deployment-specific.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    StreamEvent,
    StreamMetrics,
    SubagentCompleted,
    SubagentFailed,
    SubagentInvoked,
)
from fipsagents.serialization.openai_sse import stream_events_as_sse
from fipsagents.server.collector import TraceCollector
from fipsagents.server.metrics import NullMetricsCollector
from fipsagents.server.tracing import NullTraceStore

try:
    from prometheus_client import CollectorRegistry

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

if _HAS_PROMETHEUS:
    from fipsagents.server.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Mock event streams
# ---------------------------------------------------------------------------

MODEL = "test-model"


async def _mock_stream() -> AsyncIterator[StreamEvent]:
    """Simulate an agent stream with a successful subagent delegation."""
    yield SubagentInvoked(
        agent_name="specialist",
        task="analyze data",
        span_id="span-001",
        transport="remote",
        depth=1,
    )
    yield SubagentCompleted(
        agent_name="specialist",
        span_id="span-001",
        content="Analysis complete",
        tokens_used={"prompt": 60, "completion": 40},
        tool_calls_made=2,
        cost_usd=0.01,
    )
    yield ContentDelta(content="Based on the analysis...")
    yield StreamComplete(
        finish_reason="stop",
        metrics=StreamMetrics(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            time_to_first_content=0.1,
            total_time=0.5,
            model_calls=1,
            tool_calls=0,
        ),
    )


async def _mock_stream_with_failure() -> AsyncIterator[StreamEvent]:
    """Simulate an agent stream with a failed subagent delegation."""
    yield SubagentInvoked(
        agent_name="specialist",
        task="analyze data",
        span_id="span-002",
        transport="remote",
        depth=1,
    )
    yield SubagentFailed(
        agent_name="specialist",
        span_id="span-002",
        error_type="Timeout",
        error_message="Subagent did not respond within 30s",
    )
    yield ContentDelta(content="The analysis failed, falling back...")
    yield StreamComplete(
        finish_reason="stop",
        metrics=StreamMetrics(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            time_to_first_content=0.1,
            total_time=0.5,
            model_calls=1,
            tool_calls=0,
        ),
    )


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------


async def _collect_sse(stream: AsyncIterator[str]) -> list[dict]:
    """Parse SSE output into a list of JSON chunks."""
    chunks = []
    async for line in stream:
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def _find_subagent_chunks(chunks: list[dict]) -> list[dict]:
    """Filter SSE chunks that carry a ``subagent`` delta field."""
    return [
        c
        for c in chunks
        if c.get("choices")
        and c["choices"][0].get("delta", {}).get("subagent")
    ]


# ---------------------------------------------------------------------------
# Observer wrappers
# ---------------------------------------------------------------------------


def _make_metrics_observer():
    """Return a MetricsCollector (real or null) depending on availability."""
    if _HAS_PROMETHEUS:
        return MetricsCollector(registry=CollectorRegistry())
    return NullMetricsCollector()


def _make_trace_observer() -> TraceCollector:
    collector = TraceCollector(NullTraceStore())
    collector.begin_request({"model": MODEL, "stream": True})
    return collector


# ---------------------------------------------------------------------------
# Baseline: no observers
# ---------------------------------------------------------------------------


class TestSubagentEventsWithoutObservers:
    """Verify SSE serialisation emits subagent chunks when no observers
    are in the pipeline. This is the control group."""

    @pytest.mark.asyncio
    async def test_subagent_events_appear_in_sse(self):
        sse = stream_events_as_sse(_mock_stream(), MODEL)
        chunks = await _collect_sse(sse)

        subagent = _find_subagent_chunks(chunks)
        assert len(subagent) == 2, (
            f"Expected 2 subagent chunks (invoked + completed), got {len(subagent)}: "
            f"{json.dumps(subagent, indent=2)}"
        )

        types = [c["choices"][0]["delta"]["subagent"]["type"] for c in subagent]
        assert types == ["invoked", "completed"]

    @pytest.mark.asyncio
    async def test_subagent_invoked_payload(self):
        sse = stream_events_as_sse(_mock_stream(), MODEL)
        chunks = await _collect_sse(sse)
        subagent = _find_subagent_chunks(chunks)

        invoked = subagent[0]["choices"][0]["delta"]["subagent"]
        assert invoked["agent_name"] == "specialist"
        assert invoked["task"] == "analyze data"
        assert invoked["span_id"] == "span-001"
        assert invoked["transport"] == "remote"
        assert invoked["depth"] == 1

    @pytest.mark.asyncio
    async def test_subagent_completed_payload(self):
        sse = stream_events_as_sse(_mock_stream(), MODEL)
        chunks = await _collect_sse(sse)
        subagent = _find_subagent_chunks(chunks)

        completed = subagent[1]["choices"][0]["delta"]["subagent"]
        assert completed["agent_name"] == "specialist"
        assert completed["span_id"] == "span-001"
        assert completed["content"] == "Analysis complete"
        assert completed["tokens_used"] == {"prompt": 60, "completion": 40}
        assert completed["tool_calls_made"] == 2
        assert completed["cost_usd"] == 0.01


# ---------------------------------------------------------------------------
# MetricsCollector observer
# ---------------------------------------------------------------------------


class TestSubagentEventsWithMetricsObserver:
    """Verify subagent SSE chunks survive the MetricsCollector observer."""

    @pytest.mark.asyncio
    async def test_subagent_events_survive_metrics_observer(self):
        metrics = _make_metrics_observer()
        observed = metrics.observe(_mock_stream(), model=MODEL)
        sse = stream_events_as_sse(observed, MODEL)
        chunks = await _collect_sse(sse)

        subagent = _find_subagent_chunks(chunks)
        assert len(subagent) == 2, (
            f"MetricsCollector dropped subagent events. "
            f"Got {len(subagent)} subagent chunks, expected 2. "
            f"All chunks: {json.dumps([c.get('choices', [{}])[0].get('delta', {}) for c in chunks], indent=2)}"
        )
        types = [c["choices"][0]["delta"]["subagent"]["type"] for c in subagent]
        assert types == ["invoked", "completed"]


# ---------------------------------------------------------------------------
# TraceCollector observer
# ---------------------------------------------------------------------------


class TestSubagentEventsWithTraceObserver:
    """Verify subagent SSE chunks survive the TraceCollector observer."""

    @pytest.mark.asyncio
    async def test_subagent_events_survive_trace_observer(self):
        collector = _make_trace_observer()
        observed = collector.observe(_mock_stream())
        sse = stream_events_as_sse(observed, MODEL)
        chunks = await _collect_sse(sse)

        subagent = _find_subagent_chunks(chunks)
        assert len(subagent) == 2, (
            f"TraceCollector dropped subagent events. "
            f"Got {len(subagent)} subagent chunks, expected 2. "
            f"All chunks: {json.dumps([c.get('choices', [{}])[0].get('delta', {}) for c in chunks], indent=2)}"
        )
        types = [c["choices"][0]["delta"]["subagent"]["type"] for c in subagent]
        assert types == ["invoked", "completed"]


# ---------------------------------------------------------------------------
# Full observer chain (metrics + trace + capture)
# ---------------------------------------------------------------------------


class TestSubagentEventsWithAllObservers:
    """Verify subagent SSE chunks survive the full observer chain as wired
    in ``OpenAIChatServer._stream()``."""

    @pytest.mark.asyncio
    async def test_subagent_events_survive_full_chain(self):
        # Wire chain in the same order as app.py:
        #   agent stream -> metrics.observe -> trace.observe -> SSE
        metrics = _make_metrics_observer()
        trace = _make_trace_observer()

        events = _mock_stream()
        events = metrics.observe(events, model=MODEL)
        events = trace.observe(events)
        sse = stream_events_as_sse(events, MODEL)
        chunks = await _collect_sse(sse)

        subagent = _find_subagent_chunks(chunks)
        assert len(subagent) == 2, (
            f"Full observer chain dropped subagent events. "
            f"Got {len(subagent)} subagent chunks, expected 2. "
            f"All deltas: {json.dumps([c.get('choices', [{}])[0].get('delta', {}) for c in chunks], indent=2)}"
        )
        types = [c["choices"][0]["delta"]["subagent"]["type"] for c in subagent]
        assert types == ["invoked", "completed"]

    @pytest.mark.asyncio
    async def test_full_chain_preserves_all_event_types(self):
        """All event types (role, subagent, content, finish, usage) must
        appear in the SSE output."""
        metrics = _make_metrics_observer()
        trace = _make_trace_observer()

        events = _mock_stream()
        events = metrics.observe(events, model=MODEL)
        events = trace.observe(events)
        sse = stream_events_as_sse(events, MODEL)
        chunks = await _collect_sse(sse)

        # Role chunk (leading), subagent invoked, subagent completed,
        # content delta, finish_reason chunk, usage chunk = 6 chunks.
        assert len(chunks) >= 6, (
            f"Expected at least 6 SSE chunks, got {len(chunks)}"
        )

        # Check finish_reason is present on exactly one chunk.
        finish_chunks = [
            c for c in chunks
            if c.get("choices")
            and c["choices"][0].get("finish_reason") is not None
        ]
        assert len(finish_chunks) == 1
        assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_full_chain_with_capture_metrics_passthrough(self):
        """Mimic the _capture_metrics wrapper from app.py to ensure the
        additional pass-through observer does not eat events."""
        from fipsagents.baseagent.events import StreamComplete as _SC

        metrics = _make_metrics_observer()
        trace = _make_trace_observer()
        captured_metrics = None

        events = _mock_stream()
        events = metrics.observe(events, model=MODEL)
        events = trace.observe(events)

        # Inline pass-through (mirrors app.py::_capture_metrics)
        async def _capture(stream):
            nonlocal captured_metrics
            async for ev in stream:
                if isinstance(ev, _SC):
                    captured_metrics = ev.metrics
                yield ev

        events = _capture(events)
        sse = stream_events_as_sse(events, MODEL)
        chunks = await _collect_sse(sse)

        subagent = _find_subagent_chunks(chunks)
        assert len(subagent) == 2
        assert captured_metrics is not None
        assert captured_metrics.total_tokens == 80


# ---------------------------------------------------------------------------
# SubagentFailed event
# ---------------------------------------------------------------------------


class TestSubagentFailedEvent:
    """Verify SubagentFailed events survive the full observer chain."""

    @pytest.mark.asyncio
    async def test_failed_event_survives_observers(self):
        metrics = _make_metrics_observer()
        trace = _make_trace_observer()

        events = _mock_stream_with_failure()
        events = metrics.observe(events, model=MODEL)
        events = trace.observe(events)
        sse = stream_events_as_sse(events, MODEL)
        chunks = await _collect_sse(sse)

        subagent = _find_subagent_chunks(chunks)
        assert len(subagent) == 2, (
            f"Expected 2 subagent chunks (invoked + failed), got {len(subagent)}"
        )

        types = [c["choices"][0]["delta"]["subagent"]["type"] for c in subagent]
        assert types == ["invoked", "failed"]

    @pytest.mark.asyncio
    async def test_failed_event_payload(self):
        metrics = _make_metrics_observer()
        trace = _make_trace_observer()

        events = _mock_stream_with_failure()
        events = metrics.observe(events, model=MODEL)
        events = trace.observe(events)
        sse = stream_events_as_sse(events, MODEL)
        chunks = await _collect_sse(sse)

        subagent = _find_subagent_chunks(chunks)
        failed = subagent[1]["choices"][0]["delta"]["subagent"]
        assert failed["type"] == "failed"
        assert failed["agent_name"] == "specialist"
        assert failed["span_id"] == "span-002"
        assert failed["error_type"] == "Timeout"
        assert "30s" in failed["error_message"]
