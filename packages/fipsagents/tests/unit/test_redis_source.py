"""Tests for RedisStreamSource with mocked redis.asyncio."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def redis_config():
    return SimpleNamespace(
        type="redis",
        source_id="test-redis",
        url="redis://localhost:6379",
        stream="test-stream",
        consumer_group="test-group",
        consumer_name="worker-0",
        block_ms=100,
        max_events_per_second=0,
        retry=SimpleNamespace(
            max_attempts=3, backoff_base=2.0,
            backoff_max=60.0, retriable_errors=["TimeoutError"],
        ),
    )


class TestRedisStreamSourceConstruction:
    def test_source_id(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        assert source.source_id == "test-redis"

    def test_config_stored(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        assert source.config is redis_config


class TestRedisStreamSourceConsume:
    @pytest.mark.asyncio
    async def test_consume_yields_events(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        source._client = AsyncMock()
        source._client.xreadgroup = AsyncMock(side_effect=[
            [("test-stream", [
                ("1-0", {
                    "payload": '{"key": "val"}',
                    "event_type": "test.event",
                }),
            ])],
            [],  # empty to break the loop
        ])

        events = []
        async for event in source.consume():
            events.append(event)
            break  # consume one then stop

        assert len(events) == 1
        assert events[0].event_id == "1-0"
        assert events[0].payload == {"key": "val"}
        assert events[0].event_type == "test.event"
        assert events[0].session_key == "event:redis:test-stream:test-group"

    @pytest.mark.asyncio
    async def test_consume_non_json_payload(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        source._client = AsyncMock()
        source._client.xreadgroup = AsyncMock(side_effect=[
            [("test-stream", [("2-0", {"data": "plain-value"})])],
            [],
        ])

        events = []
        async for event in source.consume():
            events.append(event)
            break

        assert events[0].payload == {"data": "plain-value"}

    @pytest.mark.asyncio
    async def test_consume_default_event_type(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        source._client = AsyncMock()
        source._client.xreadgroup = AsyncMock(side_effect=[
            [("test-stream", [("3-0", {"payload": '{"x": 1}'})])],
            [],
        ])

        events = []
        async for event in source.consume():
            events.append(event)
            break

        assert events[0].event_type == "redis.test-stream"


class TestRedisStreamSourceAcknowledge:
    @pytest.mark.asyncio
    async def test_acknowledge_xacks(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        source._client = AsyncMock()

        await source.acknowledge("1-0")

        source._client.xack.assert_awaited_once_with(
            "test-stream", "test-group", "1-0",
        )

    @pytest.mark.asyncio
    async def test_acknowledge_no_client_noop(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        await source.acknowledge("1-0")  # should not raise


class TestRedisStreamSourceClose:
    @pytest.mark.asyncio
    async def test_close_disconnects(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        mock_client = AsyncMock()
        source._client = mock_client

        await source.close()

        mock_client.aclose.assert_awaited_once()
        assert source._client is None

    @pytest.mark.asyncio
    async def test_close_no_client_noop(self, redis_config):
        from fipsagents.server.sources.redis import RedisStreamSource

        source = RedisStreamSource("test-redis", config=redis_config)
        await source.close()  # should not raise
