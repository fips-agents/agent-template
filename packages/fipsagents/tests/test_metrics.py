"""Tests for Prometheus metrics collector."""

from __future__ import annotations

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from fipsagents.baseagent.events import (  # noqa: E402
    ContentDelta,
    StreamComplete,
    StreamMetrics,
    ToolResultEvent,
)
from fipsagents.server.metrics import (  # noqa: E402
    MetricsCollector,
    NullMetricsCollector,
    create_metrics_collector,
)


async def _emit_events(*events):
    for e in events:
        yield e


class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    @pytest.mark.asyncio
    async def test_observe_passes_events_through(self, collector):
        events = [ContentDelta(content="hello"), ContentDelta(content=" world")]
        result = []
        async for e in collector.observe(_emit_events(*events), model="test"):
            result.append(e)
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == " world"

    @pytest.mark.asyncio
    async def test_tool_call_counter(self, collector):
        events = [
            ToolResultEvent(call_id="c1", name="search", content="result"),
            ToolResultEvent(
                call_id="c2", name="search", content="", is_error=True,
            ),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        async for _ in collector.observe(_emit_events(*events), model="test"):
            pass
        ok_val = collector.tool_calls_total.labels(
            tool_name="search", status="ok",
        )._value.get()
        err_val = collector.tool_calls_total.labels(
            tool_name="search", status="error",
        )._value.get()
        assert ok_val == 1
        assert err_val == 1

    @pytest.mark.asyncio
    async def test_token_counters(self, collector):
        m = StreamMetrics(prompt_tokens=100, completion_tokens=50)
        events = [StreamComplete(finish_reason="stop", metrics=m)]
        async for _ in collector.observe(_emit_events(*events), model="gpt"):
            pass
        prompt_val = collector.tokens_total.labels(
            model="gpt", direction="prompt",
        )._value.get()
        completion_val = collector.tokens_total.labels(
            model="gpt", direction="completion",
        )._value.get()
        assert prompt_val == 100
        assert completion_val == 50

    @pytest.mark.asyncio
    async def test_model_call_duration(self, collector):
        m = StreamMetrics(total_time=1.5)
        events = [StreamComplete(finish_reason="stop", metrics=m)]
        async for _ in collector.observe(_emit_events(*events), model="gpt"):
            pass
        data = collector.generate_metrics().decode()
        assert "agent_model_call_duration_seconds" in data

    def test_request_duration(self, collector):
        start = collector.record_request_start()
        collector.record_request_end("gpt", False, "ok", start)
        data = collector.generate_metrics().decode()
        assert "agent_request_duration_seconds" in data
        assert "agent_requests_total" in data

    def test_generate_metrics_returns_text(self, collector):
        data = collector.generate_metrics()
        assert isinstance(data, bytes)
        text = data.decode()
        assert "agent_requests_total" in text


class TestNullMetricsCollector:
    @pytest.mark.asyncio
    async def test_observe_passes_through(self):
        null = NullMetricsCollector()
        events = [ContentDelta(content="hi")]
        result = []
        async for e in null.observe(_emit_events(*events), model="test"):
            result.append(e)
        assert len(result) == 1

    def test_generate_metrics_returns_empty(self):
        null = NullMetricsCollector()
        assert null.generate_metrics() == b""


class TestCreateMetricsCollector:
    def test_disabled_returns_null(self):
        c = create_metrics_collector(enabled=False)
        assert isinstance(c, NullMetricsCollector)

    def test_enabled_returns_real(self):
        c = create_metrics_collector(enabled=True)
        assert isinstance(c, MetricsCollector)
