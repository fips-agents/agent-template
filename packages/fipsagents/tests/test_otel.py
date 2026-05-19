"""Tests for OpenTelemetry trace export."""

import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

from fipsagents.server.otel import OTELTraceStore  # noqa: E402
from fipsagents.server.propagation import _string_to_span_id, _string_to_trace_id  # noqa: E402
from fipsagents.server.tracing import (  # noqa: E402
    NullTraceStore,
    Span,
    Trace,
    create_trace_store,
)


class TestSpanIdConversion:
    def test_trace_id_is_128_bit(self):
        tid = _string_to_trace_id("trace_abc123")
        assert 0 < tid < (1 << 128)

    def test_span_id_is_64_bit(self):
        sid = _string_to_span_id("span_abc123")
        assert 0 < sid < (1 << 64)

    def test_deterministic(self):
        assert _string_to_trace_id("foo") == _string_to_trace_id("foo")
        assert _string_to_span_id("bar") == _string_to_span_id("bar")

    def test_different_inputs_different_outputs(self):
        assert _string_to_trace_id("a") != _string_to_trace_id("b")
        assert _string_to_span_id("a") != _string_to_span_id("b")


class TestOTELTraceStore:
    def _make_trace(self) -> Trace:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return Trace(
            trace_id="trace_test123",
            started_at=now.isoformat(),
            ended_at=now.isoformat(),
            model="test-model",
            session_id="sess_test",
            status="ok",
            spans=[
                Span(
                    trace_id="trace_test123",
                    span_id="span_root",
                    name="request",
                    start_time=0.0,
                    end_time=1.0,
                    status="ok",
                    attributes={"model": "test-model"},
                ),
                Span(
                    trace_id="trace_test123",
                    span_id="span_step",
                    parent_span_id="span_root",
                    name="step:1",
                    start_time=0.1,
                    end_time=0.9,
                    status="ok",
                ),
                Span(
                    trace_id="trace_test123",
                    span_id="span_tool",
                    parent_span_id="span_step",
                    name="tool:search",
                    start_time=0.2,
                    end_time=0.5,
                    status="ok",
                    attributes={"tool_name": "search", "is_error": False},
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_save_trace_writes_to_inner(self):
        inner = NullTraceStore()
        # NullTraceStore.save_trace just logs, so we verify no exception
        store = OTELTraceStore(inner=inner, service_name="test")
        trace = self._make_trace()
        await store.save_trace(trace)  # Should not raise
        await store.close()

    @pytest.mark.asyncio
    async def test_get_trace_delegates_to_inner(self):
        inner = NullTraceStore()
        store = OTELTraceStore(inner=inner, service_name="test")
        result = await store.get_trace("nonexistent")
        assert result is None  # NullTraceStore returns None
        await store.close()

    @pytest.mark.asyncio
    async def test_list_traces_delegates_to_inner(self):
        inner = NullTraceStore()
        store = OTELTraceStore(inner=inner, service_name="test")
        result = await store.list_traces()
        assert result == []  # NullTraceStore returns empty
        await store.close()

    @pytest.mark.asyncio
    async def test_delete_before_delegates_to_inner(self):
        from datetime import datetime, timezone
        inner = NullTraceStore()
        store = OTELTraceStore(inner=inner, service_name="test")
        result = await store.delete_before(datetime.now(timezone.utc))
        assert result == 0  # NullTraceStore returns 0
        await store.close()

    @pytest.mark.asyncio
    async def test_export_error_does_not_propagate(self):
        """OTEL export errors are logged but don't crash save_trace."""
        inner = NullTraceStore()
        store = OTELTraceStore(
            endpoint="http://nonexistent:4317",
            inner=inner,
            service_name="test",
        )
        trace = self._make_trace()
        # save_trace should succeed even if OTLP export fails
        await store.save_trace(trace)
        await store.close()

    @pytest.mark.asyncio
    async def test_span_events_exported(self):
        """Span events should be exported as OTEL span events."""
        inner = NullTraceStore()
        store = OTELTraceStore(inner=inner, service_name="test")
        trace = self._make_trace()
        trace.spans[1].events = [
            {"name": "messages_snapshot", "timestamp": 0.15, "body": '[{"role":"user"}]'},
        ]
        await store.save_trace(trace)
        await store.close()

    @pytest.mark.asyncio
    async def test_empty_trace_no_export(self):
        """A trace with no spans should not crash."""
        inner = NullTraceStore()
        store = OTELTraceStore(inner=inner, service_name="test")
        trace = Trace(
            trace_id="empty",
            started_at="2024-01-01T00:00:00+00:00",
        )
        await store.save_trace(trace)
        await store.close()


class TestCreateTraceStoreWithOTEL:
    def test_otel_exporter_creates_otel_store(self):
        store = create_trace_store(
            None, exporter="otel", service_name="test",
        )
        assert isinstance(store, OTELTraceStore)
        # Cleanup
        store._provider.shutdown()

    def test_otel_wraps_inner_store(self):
        store = create_trace_store(
            None, exporter="otel", service_name="test",
        )
        assert isinstance(store._inner, NullTraceStore)
        store._provider.shutdown()

    def test_no_exporter_returns_plain_store(self):
        store = create_trace_store(None)
        assert isinstance(store, NullTraceStore)
