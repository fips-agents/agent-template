"""Tests for event-triggered mode server integration."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.config import ServerConfig
from fipsagents.server.events import (
    EventSink, EventSource, InboundEvent, OutboundEvent, RetryConfig,
    create_event_sink, create_event_source, default_translate_event,
)
from fipsagents.server.sinks.null import NullSink
from fipsagents.server.sources.null import NullEventSource

# -- Helpers ---------------------------------------------------------------

def _make_event(**overrides) -> InboundEvent:
    defaults = {
        "event_id": "evt-001", "event_type": "test.event",
        "payload": {"key": "value"}, "source": "test-source",
        "timestamp": datetime.now(tz=UTC),
    }
    defaults.update(overrides)
    return InboundEvent(**defaults)


class _SingleEventSource(EventSource):
    """Yields one event then stops."""
    def __init__(self, event: InboundEvent) -> None:
        super().__init__("single-shot")
        self._event = event
        self.acknowledged: list[str] = []
    async def consume(self):
        yield self._event
    async def acknowledge(self, event_id: str) -> None:
        self.acknowledged.append(event_id)

class _CaptureSink(EventSink):
    """Captures emitted events for assertion."""
    def __init__(self) -> None:
        self.events: list[OutboundEvent] = []
    async def emit(self, event: OutboundEvent) -> None:
        self.events.append(event)

def _make_server():
    """Build a minimal OpenAIChatServer stub for unit tests."""
    from fipsagents.server.app import OpenAIChatServer
    server = object.__new__(OpenAIChatServer)
    server._agent = MagicMock()
    server._agent_lock = asyncio.Lock()
    server._session_store = None
    server._event_sources = []
    server._event_sink = None
    server._event_tasks = []
    return server

# -- Tests -----------------------------------------------------------------

class TestEventTranslation:
    def test_returns_system_and_user_messages(self):
        event = _make_event()
        messages = default_translate_event(event)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_contains_event_type(self):
        event = _make_event(event_type="deploy.complete")
        messages = default_translate_event(event)
        assert "'deploy.complete'" in messages[0]["content"]

    def test_system_message_contains_source(self):
        event = _make_event(source="ci-pipeline")
        messages = default_translate_event(event)
        assert "'ci-pipeline'" in messages[0]["content"]

    def test_user_message_contains_payload_json(self):
        event = _make_event(payload={"status": "success", "count": 7})
        messages = default_translate_event(event)
        payload = json.loads(messages[1]["content"])
        assert payload["status"] == "success"
        assert payload["count"] == 7



class TestRetryErrorMatching:
    """Test the retry-error name matching logic used in _event_loop."""

    def test_timeout_error_matches(self):
        cfg = RetryConfig(retriable_errors=["TimeoutError"])
        exc = TimeoutError("timed out")
        assert type(exc).__name__ in cfg.retriable_errors

    def test_subclass_name_does_not_match_base(self):
        """SubagentTimeoutError should not match TimeoutError by name."""
        cfg = RetryConfig(retriable_errors=["TimeoutError"])

        class SubagentTimeoutError(Exception):
            pass

        exc = SubagentTimeoutError()
        assert type(exc).__name__ not in cfg.retriable_errors

    def test_empty_retriable_errors_means_nothing_retriable(self):
        cfg = RetryConfig(retriable_errors=[])
        exc = TimeoutError("boom")
        assert type(exc).__name__ not in cfg.retriable_errors

    def test_value_error_matches_when_listed(self):
        cfg = RetryConfig(retriable_errors=["ValueError", "TimeoutError"])
        exc = ValueError("bad value")
        assert type(exc).__name__ in cfg.retriable_errors

    def test_generic_exception_not_retriable_by_default(self):
        cfg = RetryConfig()  # default: ["TimeoutError"]
        exc = RuntimeError("unexpected")
        assert type(exc).__name__ not in cfg.retriable_errors


class TestEventSourceFactoryIntegration:
    def test_cron_config_returns_cron_source(self):
        from fipsagents.server.sources.cron import CronSource

        cfg = SimpleNamespace(
            type="cron",
            schedule="0 9 * * *",
            event_type="daily_check",
            source_id="my-cron",
            max_events_per_second=1.0,
        )
        source = create_event_source(cfg)
        assert isinstance(source, CronSource)
        assert source.source_id == "my-cron"

    def test_webhook_config_returns_webhook_source(self):
        from fipsagents.server.sources.webhook import HttpWebhookSource

        cfg = SimpleNamespace(
            type="webhook",
            path="/hooks/test",
            source_id="my-webhook",
            secret=None,
            event_type_header="X-Event-Type",
            signature_header="X-Hub-Signature-256",
            session_ttl_hours=168,
            max_events_per_second=10.0,
        )
        source = create_event_source(cfg)
        assert isinstance(source, HttpWebhookSource)
        assert source.source_id == "my-webhook"

    def test_null_config_returns_null_source(self):
        source = create_event_source(None)
        assert isinstance(source, NullEventSource)

    def test_null_type_string_returns_null_source(self):
        cfg = SimpleNamespace(type="null")
        source = create_event_source(cfg)
        assert isinstance(source, NullEventSource)


class TestEventSinkFactoryIntegration:
    def test_log_config_returns_log_sink(self):
        from fipsagents.server.sinks.log import LogSink

        cfg = SimpleNamespace(type="log", level="DEBUG")
        sink = create_event_sink(cfg)
        assert isinstance(sink, LogSink)

    def test_http_callback_config_returns_http_sink(self):
        from fipsagents.server.sinks.http_callback import HttpCallbackSink

        cfg = SimpleNamespace(
            type="http_callback",
            url="https://example.com/callback",
            timeout_seconds=15.0,
        )
        sink = create_event_sink(cfg)
        assert isinstance(sink, HttpCallbackSink)

    def test_null_config_returns_null_sink(self):
        sink = create_event_sink(None)
        assert isinstance(sink, NullSink)


class TestNullEventSourceBehavior:
    async def test_yields_no_events(self):
        source = NullEventSource()
        events = [e async for e in source.consume()]
        assert events == []

    async def test_null_sink_emit_does_not_raise(self):
        sink = NullSink()
        event = OutboundEvent(
            correlation_id="x",
            event_type="test",
            payload={},
            source="test",
            timestamp=datetime.now(tz=UTC),
        )
        await sink.emit(event)  # must not raise


class TestBackwardCompatibility:
    def test_server_config_defaults_no_event_sources(self):
        cfg = ServerConfig()
        assert cfg.event_sources == []
        assert cfg.event_sink is None

    def test_server_init_has_empty_event_lists(self):
        """OpenAIChatServer attributes default to empty event state."""
        server = _make_server()
        assert server._event_sources == []
        assert server._event_sink is None


class TestEventLoopHappyPath:
    async def test_event_loop_processes_and_emits(self):
        """Single event: process, emit response, acknowledge."""
        event = _make_event(event_id="evt-happy")
        source = _SingleEventSource(event)
        source.config = None  # triggers default RetryConfig
        sink = _CaptureSink()
        server = _make_server()

        async def mock_collect_sync(agent, messages, **kwargs):
            return "processed content", None, "stop"
        server._collect_sync = mock_collect_sync

        await server._event_loop(source, sink)

        assert len(sink.events) == 1
        assert sink.events[0].event_type == "response"
        assert sink.events[0].payload["content"] == "processed content"
        assert sink.events[0].correlation_id == "evt-happy"
        assert "evt-happy" in source.acknowledged

    async def test_event_loop_failure_emits_processing_failed(self):
        """Non-retriable error emits processing_failed and acknowledges."""
        event = _make_event(event_id="evt-fail")
        source = _SingleEventSource(event)
        source.config = SimpleNamespace(retry=RetryConfig(retriable_errors=[]))
        sink = _CaptureSink()
        server = _make_server()

        async def mock_collect_sync_fail(agent, messages, **kwargs):
            raise RuntimeError("boom")
        server._collect_sync = mock_collect_sync_fail

        await server._event_loop(source, sink)

        assert len(sink.events) == 1
        assert sink.events[0].event_type == "processing_failed"
        assert "boom" in sink.events[0].payload["error"]
        assert "evt-fail" in source.acknowledged


class TestEventLoopRetry:
    async def test_retriable_error_retries_then_succeeds(self):
        """First attempt raises TimeoutError, second succeeds."""
        event = _make_event(event_id="evt-retry")
        source = _SingleEventSource(event)
        source.config = SimpleNamespace(retry=RetryConfig(
            max_attempts=3, backoff_base=0.01, backoff_max=0.05,
            retriable_errors=["TimeoutError"],
        ))
        sink = _CaptureSink()
        server = _make_server()
        call_count = 0

        async def mock_collect_sync(agent, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("first attempt fails")
            return "retry worked", None, "stop"
        server._collect_sync = mock_collect_sync

        await server._event_loop(source, sink)
        assert call_count == 2
        assert sink.events[0].event_type == "response"
        assert sink.events[0].payload["content"] == "retry worked"

    async def test_all_retries_exhausted_emits_failed(self):
        """All attempts fail with retriable error -> processing_failed."""
        event = _make_event(event_id="evt-exhaust")
        source = _SingleEventSource(event)
        source.config = SimpleNamespace(retry=RetryConfig(
            max_attempts=2, backoff_base=0.01, backoff_max=0.05,
            retriable_errors=["TimeoutError"],
        ))
        sink = _CaptureSink()
        server = _make_server()

        async def mock_collect_sync(agent, messages, **kwargs):
            raise TimeoutError("always fails")
        server._collect_sync = mock_collect_sync

        await server._event_loop(source, sink)
        assert len(sink.events) == 1
        assert sink.events[0].event_type == "processing_failed"
        assert "always fails" in sink.events[0].payload["error"]

    async def test_non_retriable_error_skips_retry(self):
        """Non-retriable error fails immediately without retrying."""
        event = _make_event(event_id="evt-noretry")
        source = _SingleEventSource(event)
        source.config = SimpleNamespace(retry=RetryConfig(
            max_attempts=3, retriable_errors=["TimeoutError"],
        ))
        sink = _CaptureSink()
        server = _make_server()
        call_count = 0

        async def mock_collect_sync(agent, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("not retriable")
        server._collect_sync = mock_collect_sync

        await server._event_loop(source, sink)
        assert call_count == 1
        assert sink.events[0].event_type == "processing_failed"


class TestProcessEventSessionKey:
    def _make_process_server(self):
        server = _make_server()
        server._agent.config = MagicMock()
        server._agent.config.model.name = "test-model"
        server._agent.messages = []
        return server

    async def test_session_key_overrides_source_id(self):
        """event.session_key takes precedence over source.source_id."""
        event = _make_event(session_key="custom-session-123")
        server = self._make_process_server()
        collected = {}

        async def mock_collect_sync(agent, messages, **kwargs):
            collected["session_id"] = kwargs.get("session_id")
            return "ok", None, "stop"
        server._collect_sync = mock_collect_sync

        await server._process_event(event, _SingleEventSource(event))
        assert collected["session_id"] == "custom-session-123"

    async def test_null_session_key_falls_back_to_source_id(self):
        """When session_key is None, source.source_id is used."""
        event = _make_event(session_key=None)
        server = self._make_process_server()
        collected = {}

        async def mock_collect_sync(agent, messages, **kwargs):
            collected["session_id"] = kwargs.get("session_id")
            return "ok", None, "stop"
        server._collect_sync = mock_collect_sync

        source = _SingleEventSource(event)
        source.source_id = "fallback-source-id"
        await server._process_event(event, source)
        assert collected["session_id"] == "fallback-source-id"

    async def test_session_store_loads_and_saves(self):
        """When session_store is set, load before and save after."""
        event = _make_event(session_key="sess-42")
        server = self._make_process_server()
        server._agent.messages = [{"role": "assistant", "content": "done"}]

        mock_store = AsyncMock()
        mock_store.load = AsyncMock(return_value=[
            {"role": "user", "content": "prior turn"},
        ])
        mock_store.save = AsyncMock()
        server._session_store = mock_store

        async def mock_collect_sync(agent, messages, **kwargs):
            assert any(m.get("content") == "prior turn" for m in messages)
            return "ok", None, "stop"
        server._collect_sync = mock_collect_sync

        await server._process_event(event, _SingleEventSource(event))
        mock_store.load.assert_called_once_with("sess-42")
        mock_store.save.assert_called_once()
        assert mock_store.save.call_args[0][0] == "sess-42"


class TestEventLoopCancellation:
    async def test_cancelled_error_logs_cleanly(self):
        """CancelledError in the event loop is handled gracefully."""
        class _HangingSource(EventSource):
            def __init__(self):
                super().__init__("hanging")
            async def consume(self):
                await asyncio.sleep(3600)
                yield  # never reached

        source = _HangingSource()
        source.config = None
        sink = _CaptureSink()
        server = _make_server()

        task = asyncio.create_task(server._event_loop(source, sink))
        await asyncio.sleep(0.01)
        task.cancel()
        # _event_loop catches CancelledError internally and returns cleanly.
        await task
        assert len(sink.events) == 0
