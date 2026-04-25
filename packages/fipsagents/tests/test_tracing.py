"""Tests for tracing data model, collector, and stores."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.server.collector import TraceCollector
from fipsagents.server.tracing import (
    NullTraceStore,
    Span,
    SqliteTraceStore,
    Trace,
    create_trace_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_trace_store(tmp_path):
    store = SqliteTraceStore(str(tmp_path / "test.db"))
    yield store
    await store.close()


@pytest_asyncio.fixture
async def null_store():
    return NullTraceStore()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TestSpan:
    def test_duration_ms(self):
        span = Span(trace_id="t1", span_id="s1", start_time=1.0, end_time=1.5)
        assert span.duration_ms == 500.0

    def test_duration_ms_none_when_not_ended(self):
        span = Span(trace_id="t1", span_id="s1", start_time=1.0, end_time=None)
        assert span.duration_ms is None


class TestTrace:
    def test_to_summary_counts_tools(self):
        spans = [
            Span(trace_id="t1", span_id="s1", name="request"),
            Span(trace_id="t1", span_id="s2", name="tool:search"),
            Span(trace_id="t1", span_id="s3", name="tool:calc"),
        ]
        trace = Trace(trace_id="t1", started_at="2026-01-01T00:00:00Z", spans=spans)
        summary = trace.to_summary()
        assert summary.tool_calls == 2

    def test_to_summary_aggregates_tokens(self):
        spans = [
            Span(
                trace_id="t1",
                span_id="s1",
                name="model_call",
                attributes={"prompt_tokens": 100, "completion_tokens": 50},
            ),
            Span(
                trace_id="t1",
                span_id="s2",
                name="model_call",
                attributes={"prompt_tokens": 200, "completion_tokens": 75},
            ),
        ]
        trace = Trace(trace_id="t1", started_at="2026-01-01T00:00:00Z", spans=spans)
        summary = trace.to_summary()
        assert summary.prompt_tokens == 300
        assert summary.completion_tokens == 125


# ---------------------------------------------------------------------------
# Collector helpers
# ---------------------------------------------------------------------------


async def _emit_events(*events):
    """Async generator yielding events."""
    for e in events:
        yield e


def _default_metrics(**overrides) -> StreamMetrics:
    """Build a StreamMetrics with sensible defaults."""
    kwargs = {"total_time": 0.5, "prompt_tokens": None, "completion_tokens": None}
    kwargs.update(overrides)
    return StreamMetrics(**kwargs)


# ---------------------------------------------------------------------------
# TraceCollector
# ---------------------------------------------------------------------------


class TestTraceCollector:
    @pytest.mark.asyncio
    async def test_simple_response(self, null_store):
        """ContentDelta + StreamComplete produces request, step, and model_call spans."""
        collector = TraceCollector(null_store, trace_id="t-simple")
        collector.begin_request({"model": "test-model"})

        events = _emit_events(
            ContentDelta("Hello"),
            StreamComplete(finish_reason="stop", metrics=_default_metrics()),
        )
        collected = [e async for e in collector.observe(events)]
        await collector.end_request()

        span_names = [s.name for s in collector._spans]
        assert "request" in span_names
        assert "step:1" in span_names
        assert "model_call" in span_names
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_tool_call_flow(self, null_store):
        """ToolCallDelta + ToolResultEvent + ContentDelta + StreamComplete
        creates tool span and two steps."""
        collector = TraceCollector(null_store, trace_id="t-tool")
        collector.begin_request()

        events = _emit_events(
            ToolCallDelta(index=0, call_id="call_1", name="search"),
            ToolResultEvent(call_id="call_1", name="search", content="result text"),
            ContentDelta("Based on the search..."),
            StreamComplete(finish_reason="stop", metrics=_default_metrics()),
        )
        collected = [e async for e in collector.observe(events)]
        await collector.end_request()

        span_names = [s.name for s in collector._spans]
        assert "tool:search" in span_names
        assert "step:1" in span_names
        assert "step:2" in span_names
        assert len(collected) == 4

    @pytest.mark.asyncio
    async def test_multiple_tools(self, null_store):
        """Two tool calls in the same step produce two tool spans."""
        collector = TraceCollector(null_store, trace_id="t-multi")
        collector.begin_request()

        events = _emit_events(
            ToolCallDelta(index=0, call_id="call_a", name="search"),
            ToolCallDelta(index=1, call_id="call_b", name="calc"),
            ToolResultEvent(call_id="call_a", name="search", content="r1"),
            ToolResultEvent(call_id="call_b", name="calc", content="r2"),
            ContentDelta("Done"),
            StreamComplete(finish_reason="stop", metrics=_default_metrics()),
        )
        [e async for e in collector.observe(events)]
        await collector.end_request()

        tool_spans = [s for s in collector._spans if s.name.startswith("tool:")]
        assert len(tool_spans) == 2
        tool_names = {s.name for s in tool_spans}
        assert tool_names == {"tool:search", "tool:calc"}

    @pytest.mark.asyncio
    async def test_error_resilience(self, null_store):
        """Collector handles unexpected event types without crashing."""
        collector = TraceCollector(null_store, trace_id="t-err")
        collector.begin_request()

        # Simulate an unexpected object type mixed in with valid events
        class UnknownEvent:
            pass

        events = _emit_events(
            ContentDelta("Hello"),
            UnknownEvent(),
            StreamComplete(finish_reason="stop", metrics=_default_metrics()),
        )
        collected = [e async for e in collector.observe(events)]
        await collector.end_request()

        # All events pass through despite the unknown one
        assert len(collected) == 3

    @pytest.mark.asyncio
    async def test_metrics_attached(self, null_store):
        """StreamMetrics from StreamComplete are attached to the model_call span."""
        metrics = _default_metrics(
            prompt_tokens=150, completion_tokens=42, total_time=1.2
        )
        collector = TraceCollector(null_store, trace_id="t-metrics")
        collector.begin_request()

        events = _emit_events(
            ContentDelta("answer"),
            StreamComplete(finish_reason="stop", metrics=metrics),
        )
        [e async for e in collector.observe(events)]
        await collector.end_request()

        model_spans = [s for s in collector._spans if s.name == "model_call"]
        assert len(model_spans) == 1
        attrs = model_spans[0].attributes
        assert attrs["prompt_tokens"] == 150
        assert attrs["completion_tokens"] == 42
        assert attrs["total_time"] == 1.2

    @pytest.mark.asyncio
    async def test_events_pass_through(self, null_store):
        """All original events are yielded unchanged."""
        original = [
            ContentDelta("Hello"),
            ContentDelta(" world"),
            StreamComplete(finish_reason="stop", metrics=_default_metrics()),
        ]
        collector = TraceCollector(null_store, trace_id="t-pass")
        collector.begin_request()

        collected = [e async for e in collector.observe(_emit_events(*original))]
        await collector.end_request()

        assert collected == original


# ---------------------------------------------------------------------------
# NullTraceStore
# ---------------------------------------------------------------------------


class TestNullTraceStore:
    @pytest.mark.asyncio
    async def test_get_returns_none(self, null_store):
        assert await null_store.get_trace("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_returns_empty(self, null_store):
        assert await null_store.list_traces() == []


# ---------------------------------------------------------------------------
# SqliteTraceStore
# ---------------------------------------------------------------------------


class TestSqliteTraceStore:
    @pytest.mark.asyncio
    async def test_save_and_get(self, sqlite_trace_store):
        spans = [
            Span(
                trace_id="t1",
                span_id="s1",
                name="request",
                start_time=1.0,
                end_time=2.0,
            ),
            Span(
                trace_id="t1",
                span_id="s2",
                parent_span_id="s1",
                name="model_call",
                start_time=1.1,
                end_time=1.9,
                attributes={"prompt_tokens": 100},
            ),
        ]
        trace = Trace(
            trace_id="t1",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:02Z",
            model="test-model",
            session_id="sess_abc",
            status="ok",
            spans=spans,
        )
        await sqlite_trace_store.save_trace(trace)

        loaded = await sqlite_trace_store.get_trace("t1")
        assert loaded is not None
        assert loaded.trace_id == "t1"
        assert loaded.model == "test-model"
        assert loaded.session_id == "sess_abc"
        assert len(loaded.spans) == 2
        assert loaded.spans[1].attributes["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_list_traces(self, sqlite_trace_store):
        t1 = Trace(
            trace_id="t-old",
            started_at="2026-01-01T00:00:00Z",
            spans=[Span(trace_id="t-old", span_id="s1", name="request")],
        )
        t2 = Trace(
            trace_id="t-new",
            started_at="2026-01-02T00:00:00Z",
            spans=[
                Span(trace_id="t-new", span_id="s1", name="request"),
                Span(trace_id="t-new", span_id="s2", name="tool:search"),
            ],
        )
        await sqlite_trace_store.save_trace(t1)
        await sqlite_trace_store.save_trace(t2)

        summaries = await sqlite_trace_store.list_traces()
        assert len(summaries) == 2
        # Ordered by started_at DESC: t-new first
        assert summaries[0].trace_id == "t-new"
        assert summaries[1].trace_id == "t-old"

    @pytest.mark.asyncio
    async def test_delete_before(self, sqlite_trace_store):
        old_trace = Trace(
            trace_id="t-old",
            started_at="2025-01-01T00:00:00Z",
            spans=[Span(trace_id="t-old", span_id="s1", name="request")],
        )
        new_trace = Trace(
            trace_id="t-new",
            started_at="2026-04-01T00:00:00Z",
            spans=[Span(trace_id="t-new", span_id="s1", name="request")],
        )
        await sqlite_trace_store.save_trace(old_trace)
        await sqlite_trace_store.save_trace(new_trace)

        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        deleted = await sqlite_trace_store.delete_before(cutoff)

        assert deleted == 1
        assert await sqlite_trace_store.get_trace("t-old") is None
        assert await sqlite_trace_store.get_trace("t-new") is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateTraceStore:
    def test_null(self):
        store = create_trace_store(None)
        assert isinstance(store, NullTraceStore)

    def test_sqlite(self):
        store = create_trace_store("sqlite")
        assert isinstance(store, SqliteTraceStore)

    def test_postgres_requires_url(self):
        with pytest.raises(ValueError, match="requires database_url"):
            create_trace_store("postgres")
