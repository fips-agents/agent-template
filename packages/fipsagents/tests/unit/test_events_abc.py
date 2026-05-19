"""Tests for event-triggered mode ABCs, models, factories, and rate limiter."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import get_args

import pytest

from fipsagents.baseagent.events import (
    EventFailed,
    EventProcessed,
    EventReceived,
    StreamEvent,
)
from fipsagents.server.events import (
    EventSink,
    EventSource,
    InboundEvent,
    OutboundEvent,
    RetryConfig,
    TokenBucketRateLimiter,
    create_event_sink,
    create_event_source,
    default_translate_event,
)
from fipsagents.server.sinks.null import NullSink
from fipsagents.server.sources.null import NullEventSource


# -- Helpers ---------------------------------------------------------------

def _make_inbound(**overrides) -> InboundEvent:
    defaults = {
        "event_id": "evt-1",
        "event_type": "push",
        "payload": {"ref": "refs/heads/main"},
        "source": "github",
        "timestamp": datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return InboundEvent(**defaults)


def _make_outbound(**overrides) -> OutboundEvent:
    defaults = {
        "correlation_id": "evt-1",
        "event_type": "push.processed",
        "payload": {"status": "ok"},
        "source": "agent",
        "timestamp": datetime(2026, 5, 18, 12, 0, 1, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return OutboundEvent(**defaults)


# -- Model tests -----------------------------------------------------------

class TestInboundEvent:
    def test_construct_minimal(self):
        event = _make_inbound()
        assert event.event_id == "evt-1"
        assert event.event_type == "push"
        assert event.source == "github"
        assert event.metadata == {}
        assert event.session_key is None

    def test_construct_with_metadata(self):
        event = _make_inbound(
            metadata={"delivery": "abc123"},
            session_key="repo:my-org/my-repo",
        )
        assert event.metadata["delivery"] == "abc123"
        assert event.session_key == "repo:my-org/my-repo"


class TestOutboundEvent:
    def test_construct_minimal(self):
        event = _make_outbound()
        assert event.correlation_id == "evt-1"
        assert event.event_type == "push.processed"
        assert event.metadata == {}

    def test_construct_with_metadata(self):
        event = _make_outbound(metadata={"duration_ms": 42.0})
        assert event.metadata["duration_ms"] == 42.0


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.backoff_base == 2.0
        assert cfg.backoff_max == 60.0
        assert "TimeoutError" in cfg.retriable_errors

    def test_custom_values(self):
        cfg = RetryConfig(max_attempts=5, backoff_base=1.5, backoff_max=120.0)
        assert cfg.max_attempts == 5
        assert cfg.backoff_base == 1.5

    def test_max_attempts_minimum(self):
        with pytest.raises(Exception):
            RetryConfig(max_attempts=0)


# -- Null source/sink tests ------------------------------------------------

class TestNullEventSource:
    @pytest.mark.asyncio
    async def test_consume_yields_nothing(self):
        source = NullEventSource()
        events = [e async for e in source.consume()]
        assert events == []

    @pytest.mark.asyncio
    async def test_source_id(self):
        source = NullEventSource()
        assert source.source_id == "null"

    @pytest.mark.asyncio
    async def test_acknowledge_noop(self):
        source = NullEventSource()
        await source.acknowledge("evt-1")  # should not raise

    @pytest.mark.asyncio
    async def test_close_noop(self):
        source = NullEventSource()
        await source.close()  # should not raise

    @pytest.mark.asyncio
    async def test_setup_noop(self):
        source = NullEventSource()
        await source.setup()  # should not raise


class TestNullSink:
    @pytest.mark.asyncio
    async def test_emit_noop(self):
        sink = NullSink()
        event = _make_outbound()
        await sink.emit(event)  # should not raise

    @pytest.mark.asyncio
    async def test_setup_noop(self):
        sink = NullSink()
        await sink.setup()  # should not raise

    @pytest.mark.asyncio
    async def test_close_noop(self):
        sink = NullSink()
        await sink.close()  # should not raise


# -- Translation tests -----------------------------------------------------

class TestDefaultTranslateEvent:
    def test_returns_system_and_user_messages(self):
        event = _make_inbound()
        messages = default_translate_event(event)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_contains_event_type(self):
        event = _make_inbound(event_type="pull_request")
        messages = default_translate_event(event)
        assert "'pull_request'" in messages[0]["content"]

    def test_system_message_contains_source(self):
        event = _make_inbound(source="gitlab")
        messages = default_translate_event(event)
        assert "'gitlab'" in messages[0]["content"]

    def test_user_message_contains_payload_json(self):
        event = _make_inbound(payload={"action": "opened", "number": 42})
        messages = default_translate_event(event)
        import json
        payload = json.loads(messages[1]["content"])
        assert payload["action"] == "opened"
        assert payload["number"] == 42


# -- Factory tests ---------------------------------------------------------

class TestCreateEventSource:
    def test_none_returns_null(self):
        source = create_event_source(None)
        assert isinstance(source, NullEventSource)

    def test_null_type_returns_null(self):
        cfg = SimpleNamespace(type="null")
        source = create_event_source(cfg)
        assert isinstance(source, NullEventSource)

    def test_no_type_attr_returns_null(self):
        cfg = SimpleNamespace()
        source = create_event_source(cfg)
        assert isinstance(source, NullEventSource)

    def test_unknown_type_raises(self):
        cfg = SimpleNamespace(type="kafka")
        with pytest.raises(ValueError, match="Unknown event source type"):
            create_event_source(cfg)


class TestCreateEventSink:
    def test_none_returns_null(self):
        sink = create_event_sink(None)
        assert isinstance(sink, NullSink)

    def test_null_type_returns_null(self):
        cfg = SimpleNamespace(type="null")
        sink = create_event_sink(cfg)
        assert isinstance(sink, NullSink)

    def test_no_type_attr_returns_null(self):
        cfg = SimpleNamespace()
        sink = create_event_sink(cfg)
        assert isinstance(sink, NullSink)

    def test_unknown_type_raises(self):
        cfg = SimpleNamespace(type="kafka")
        with pytest.raises(ValueError, match="Unknown event sink type"):
            create_event_sink(cfg)


# -- Rate limiter tests ----------------------------------------------------

class TestTokenBucketRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_when_tokens_available(self):
        limiter = TokenBucketRateLimiter(rate=100.0)
        await limiter.acquire()  # should not block

    @pytest.mark.asyncio
    async def test_rate_zero_passthrough(self):
        limiter = TokenBucketRateLimiter(rate=0)
        await limiter.acquire()  # disabled, should return immediately

    @pytest.mark.asyncio
    async def test_negative_rate_passthrough(self):
        limiter = TokenBucketRateLimiter(rate=-1.0)
        await limiter.acquire()  # disabled, should return immediately

    def test_custom_capacity(self):
        limiter = TokenBucketRateLimiter(rate=5.0, capacity=10.0)
        assert limiter.capacity == 10.0

    def test_default_capacity(self):
        limiter = TokenBucketRateLimiter(rate=5.0)
        assert limiter.capacity == 5.0

    def test_default_capacity_minimum_one(self):
        limiter = TokenBucketRateLimiter(rate=0.5)
        assert limiter.capacity == 1.0

    @pytest.mark.asyncio
    async def test_multiple_acquires_drain_tokens(self):
        limiter = TokenBucketRateLimiter(rate=1000.0, capacity=3.0)
        # Start with 3 tokens, drain all three immediately
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        # 4th would need to wait; just verify we got here without error


# -- StreamEvent union tests -----------------------------------------------

class TestStreamEvents:
    def test_event_received_is_stream_event(self):
        """EventReceived must be in the StreamEvent union."""
        assert EventReceived in get_args(StreamEvent)

    def test_event_processed_is_stream_event(self):
        """EventProcessed must be in the StreamEvent union."""
        assert EventProcessed in get_args(StreamEvent)

    def test_event_failed_is_stream_event(self):
        """EventFailed must be in the StreamEvent union."""
        assert EventFailed in get_args(StreamEvent)

    def test_event_received_fields(self):
        ev = EventReceived(event_id="e1", event_type="push", source="gh")
        assert ev.event_id == "e1"
        assert ev.event_type == "push"
        assert ev.source == "gh"

    def test_event_processed_fields(self):
        ev = EventProcessed(event_id="e1", source="gh", duration_ms=42.5)
        assert ev.event_id == "e1"
        assert ev.duration_ms == 42.5

    def test_event_failed_fields(self):
        ev = EventFailed(
            event_id="e1", source="gh", error="boom", retriable=True,
        )
        assert ev.error == "boom"
        assert ev.retriable is True
