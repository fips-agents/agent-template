"""Tests for KafkaSource with mocked aiokafka."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def kafka_config():
    return SimpleNamespace(
        type="kafka",
        source_id="test-kafka",
        bootstrap_servers="localhost:9092",
        topic="test-topic",
        consumer_group="test-group",
        auto_offset_reset="earliest",
        security_protocol=None,
        sasl_mechanism=None,
        sasl_username=None,
        sasl_password=None,
        max_events_per_second=0,
        retry=SimpleNamespace(
            max_attempts=3, backoff_base=2.0,
            backoff_max=60.0, retriable_errors=["TimeoutError"],
        ),
    )


class TestKafkaSourceConstruction:
    def test_source_id(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)
        assert source.source_id == "test-kafka"

    def test_config_stored(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)
        assert source.config is kafka_config


class TestKafkaSourceConsume:
    @pytest.mark.asyncio
    async def test_consume_yields_events(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)

        mock_msg = MagicMock()
        mock_msg.value = json.dumps({"action": "test"}).encode()
        mock_msg.partition = 0
        mock_msg.offset = 42
        mock_msg.key = b"key1"
        mock_msg.topic = "test-topic"

        async def _fake_consumer():
            yield mock_msg

        source._consumer = _fake_consumer()
        events = []
        async for event in source.consume():
            events.append(event)
            break  # only consume one

        assert len(events) == 1
        assert events[0].payload == {"action": "test"}
        assert events[0].event_type == "kafka.test-topic"
        assert events[0].metadata["partition"] == 0
        assert events[0].metadata["offset"] == 42
        assert events[0].session_key == "event:kafka:test-topic:test-group"

    @pytest.mark.asyncio
    async def test_consume_non_json_payload(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)

        mock_msg = MagicMock()
        mock_msg.value = b"not-json"
        mock_msg.partition = 0
        mock_msg.offset = 0
        mock_msg.key = None
        mock_msg.topic = "test-topic"

        async def _fake_consumer():
            yield mock_msg

        source._consumer = _fake_consumer()
        events = []
        async for event in source.consume():
            events.append(event)
            break

        assert events[0].payload == {"raw": "not-json"}

    @pytest.mark.asyncio
    async def test_consume_null_key(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)

        mock_msg = MagicMock()
        mock_msg.value = json.dumps({"x": 1}).encode()
        mock_msg.partition = 1
        mock_msg.offset = 0
        mock_msg.key = None
        mock_msg.topic = "test-topic"

        async def _fake_consumer():
            yield mock_msg

        source._consumer = _fake_consumer()
        events = []
        async for event in source.consume():
            events.append(event)
            break

        assert events[0].metadata["key"] is None


class TestKafkaSourceAcknowledge:
    @pytest.mark.asyncio
    async def test_acknowledge_commits(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)
        source._consumer = AsyncMock()
        source._pending_offsets["evt-1"] = ("test-topic", 42)

        await source.acknowledge("evt-1")

        source._consumer.commit.assert_awaited_once()
        assert "evt-1" not in source._pending_offsets

    @pytest.mark.asyncio
    async def test_acknowledge_unknown_event_id(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)
        source._consumer = AsyncMock()

        await source.acknowledge("nonexistent")
        source._consumer.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_acknowledge_no_consumer_noop(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)
        await source.acknowledge("evt-1")  # should not raise


class TestKafkaSourceClose:
    @pytest.mark.asyncio
    async def test_close_stops_consumer(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)
        mock_consumer = AsyncMock()
        source._consumer = mock_consumer

        await source.close()

        mock_consumer.stop.assert_awaited_once()
        assert source._consumer is None

    @pytest.mark.asyncio
    async def test_close_no_consumer_noop(self, kafka_config):
        from fipsagents.server.sources.kafka import KafkaSource

        source = KafkaSource("test-kafka", config=kafka_config)
        await source.close()  # should not raise
