"""Tests for RedisStreamSink with mocked redis.asyncio."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fipsagents.server.events import OutboundEvent


@pytest.fixture
def redis_sink_config():
    return SimpleNamespace(
        type="redis",
        url="redis://localhost:6379",
        stream="results-stream",
        maxlen=10000,
    )


def _make_event(**overrides) -> OutboundEvent:
    defaults = {
        "correlation_id": "evt-1",
        "event_type": "response",
        "payload": {"content": "hello"},
        "source": "test",
        "timestamp": datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return OutboundEvent(**defaults)


class TestRedisStreamSinkEmit:
    @pytest.mark.asyncio
    async def test_emit_xadds_to_stream(self, redis_sink_config):
        from fipsagents.server.sinks.redis import RedisStreamSink

        sink = RedisStreamSink(config=redis_sink_config)
        mock_client = AsyncMock()
        sink._client = mock_client

        event = _make_event()
        await sink.emit(event)

        mock_client.xadd.assert_awaited_once()
        call_args = mock_client.xadd.call_args
        assert call_args[0][0] == "results-stream"
        fields = call_args[0][1]
        assert fields["correlation_id"] == "evt-1"
        assert fields["event_type"] == "response"
        assert call_args[1]["maxlen"] == 10000
        assert call_args[1]["approximate"] is True

    @pytest.mark.asyncio
    async def test_emit_no_maxlen(self):
        from fipsagents.server.sinks.redis import RedisStreamSink

        config = SimpleNamespace(
            type="redis", url="redis://localhost",
            stream="s", maxlen=None,
        )
        sink = RedisStreamSink(config=config)
        mock_client = AsyncMock()
        sink._client = mock_client

        event = _make_event()
        await sink.emit(event)

        call_kwargs = mock_client.xadd.call_args[1]
        assert "maxlen" not in call_kwargs

    @pytest.mark.asyncio
    async def test_emit_error_does_not_raise(self, redis_sink_config):
        from fipsagents.server.sinks.redis import RedisStreamSink

        sink = RedisStreamSink(config=redis_sink_config)
        mock_client = AsyncMock()
        mock_client.xadd.side_effect = RuntimeError("connection lost")
        sink._client = mock_client

        event = _make_event()
        await sink.emit(event)  # should not raise


class TestRedisStreamSinkClose:
    @pytest.mark.asyncio
    async def test_close_disconnects(self, redis_sink_config):
        from fipsagents.server.sinks.redis import RedisStreamSink

        sink = RedisStreamSink(config=redis_sink_config)
        mock_client = AsyncMock()
        sink._client = mock_client

        await sink.close()

        mock_client.aclose.assert_awaited_once()
        assert sink._client is None

    @pytest.mark.asyncio
    async def test_close_no_client_noop(self, redis_sink_config):
        from fipsagents.server.sinks.redis import RedisStreamSink

        sink = RedisStreamSink(config=redis_sink_config)
        await sink.close()  # should not raise
