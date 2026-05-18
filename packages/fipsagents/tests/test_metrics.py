"""Tests for Prometheus metrics collector."""

from __future__ import annotations

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from fipsagents.baseagent.events import (  # noqa: E402
    ContentDelta,
    StreamComplete,
    StreamMetrics,
    SubagentInvoked,
    SubagentCompleted,
    SubagentFailed,
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

    def test_token_label_mode_threaded_through(self):
        c = create_metrics_collector(enabled=True, token_label_mode="tenant")
        assert isinstance(c, MetricsCollector)
        assert c._token_labelnames == ["model", "direction", "tenant_id"]


class TestTokenLabelModes:
    """Verify ``token_label_mode`` controls which labels land on tokens_total."""

    @pytest.mark.asyncio
    async def test_default_mode_only_model_and_direction(self):
        c = MetricsCollector()
        m = StreamMetrics(prompt_tokens=10, completion_tokens=5)
        async for _ in c.observe(
            _emit_events(StreamComplete(finish_reason="stop", metrics=m)),
            model="m1",
            tenant_id="acme",
            session_id="sess1",
        ):
            pass
        # tenant_id / session_id were passed but not recorded as labels.
        val = c.tokens_total.labels(model="m1", direction="prompt")._value.get()
        assert val == 10

    @pytest.mark.asyncio
    async def test_tenant_mode_records_tenant_id(self):
        c = MetricsCollector(token_label_mode="tenant")
        m = StreamMetrics(prompt_tokens=10, completion_tokens=5)
        async for _ in c.observe(
            _emit_events(StreamComplete(finish_reason="stop", metrics=m)),
            model="m1",
            tenant_id="acme",
            session_id="sess1",
        ):
            pass
        val = c.tokens_total.labels(
            model="m1", direction="prompt", tenant_id="acme",
        )._value.get()
        assert val == 10

    @pytest.mark.asyncio
    async def test_tenant_mode_missing_tenant_uses_default(self):
        c = MetricsCollector(token_label_mode="tenant")
        m = StreamMetrics(prompt_tokens=7)
        async for _ in c.observe(
            _emit_events(StreamComplete(finish_reason="stop", metrics=m)),
            model="m1",
            tenant_id=None,
            session_id=None,
        ):
            pass
        val = c.tokens_total.labels(
            model="m1", direction="prompt", tenant_id="default",
        )._value.get()
        assert val == 7

    @pytest.mark.asyncio
    async def test_session_mode_records_both(self):
        c = MetricsCollector(token_label_mode="session")
        m = StreamMetrics(prompt_tokens=3, completion_tokens=2)
        async for _ in c.observe(
            _emit_events(StreamComplete(finish_reason="stop", metrics=m)),
            model="m1",
            tenant_id="acme",
            session_id="sess1",
        ):
            pass
        prompt_val = c.tokens_total.labels(
            model="m1", direction="prompt",
            tenant_id="acme", session_id="sess1",
        )._value.get()
        completion_val = c.tokens_total.labels(
            model="m1", direction="completion",
            tenant_id="acme", session_id="sess1",
        )._value.get()
        assert prompt_val == 3
        assert completion_val == 2

    @pytest.mark.asyncio
    async def test_session_mode_missing_session_uses_none(self):
        c = MetricsCollector(token_label_mode="session")
        m = StreamMetrics(prompt_tokens=5)
        async for _ in c.observe(
            _emit_events(StreamComplete(finish_reason="stop", metrics=m)),
            model="m1",
            tenant_id="acme",
            session_id=None,
        ):
            pass
        val = c.tokens_total.labels(
            model="m1", direction="prompt",
            tenant_id="acme", session_id="none",
        )._value.get()
        assert val == 5

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            MetricsCollector(token_label_mode="bogus")


class TestSubagentEventPassthrough:
    """Verify MetricsCollector passes through subagent events unchanged."""

    @pytest.mark.asyncio
    async def test_subagent_invoked_passes_through(self):
        collector = MetricsCollector()
        events = [
            SubagentInvoked(
                agent_name="helper",
                task="help",
                span_id="s1",
                transport="remote",
                depth=1,
            ),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        result = []
        async for e in collector.observe(_emit_events(*events), model="test"):
            result.append(e)
        assert len(result) == 2
        assert isinstance(result[0], SubagentInvoked)
        assert result[0].agent_name == "helper"

    @pytest.mark.asyncio
    async def test_subagent_completed_passes_through(self):
        collector = MetricsCollector()
        events = [
            SubagentCompleted(
                agent_name="helper",
                span_id="s1",
                content="done",
                tokens_used={},
                tool_calls_made=0,
                cost_usd=0.0,
            ),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        result = []
        async for e in collector.observe(_emit_events(*events), model="test"):
            result.append(e)
        assert len(result) == 2
        assert isinstance(result[0], SubagentCompleted)
        assert result[0].content == "done"

    @pytest.mark.asyncio
    async def test_subagent_failed_passes_through(self):
        collector = MetricsCollector()
        events = [
            SubagentFailed(
                agent_name="helper",
                span_id="s1",
                error_type="Timeout",
                error_message="timeout",
            ),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        result = []
        async for e in collector.observe(_emit_events(*events), model="test"):
            result.append(e)
        assert len(result) == 2
        assert isinstance(result[0], SubagentFailed)
        assert result[0].error_type == "Timeout"


class TestFoundationEventPassthrough:
    """Verify MetricsCollector passes through foundation events unchanged."""

    @pytest.mark.asyncio
    async def test_compaction_events_pass_through(self):
        from fipsagents.baseagent.events import CompactionStarted, CompactionSkipped
        collector = MetricsCollector()
        events = [
            CompactionStarted(session_id="s1", message_count=50),
            CompactionSkipped(reason="pending_state"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        result = []
        async for e in collector.observe(_emit_events(*events), model="test"):
            result.append(e)
        assert len(result) == 3
        assert isinstance(result[0], CompactionStarted)
        assert isinstance(result[1], CompactionSkipped)

    @pytest.mark.asyncio
    async def test_permission_events_pass_through(self):
        from fipsagents.baseagent.events import PermissionDecisionMade
        collector = MetricsCollector()
        events = [
            PermissionDecisionMade(tool="search", action="allow"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        result = []
        async for e in collector.observe(_emit_events(*events), model="test"):
            result.append(e)
        assert len(result) == 2
        assert isinstance(result[0], PermissionDecisionMade)
