"""Tests for KafkaSink with mocked aiokafka."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fipsagents.server.events import OutboundEvent


@pytest.fixture
def kafka_sink_config():
    return SimpleNamespace(
        type="kafka",
        bootstrap_servers="localhost:9092",
        topic="results-topic",
        security_protocol=None,
        sasl_mechanism=None,
        sasl_username=None,
        sasl_password=None,
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


class TestKafkaSinkEmit:
    @pytest.mark.asyncio
    async def test_emit_sends_to_topic(self, kafka_sink_config):
        from fipsagents.server.sinks.kafka import KafkaSink

        sink = KafkaSink(config=kafka_sink_config)
        mock_producer = AsyncMock()
        sink._producer = mock_producer

        event = _make_event()
        await sink.emit(event)

        mock_producer.send_and_wait.assert_awaited_once()
        call_args = mock_producer.send_and_wait.call_args
        assert call_args[0][0] == "results-topic"

    @pytest.mark.asyncio
    async def test_emit_error_does_not_raise(self, kafka_sink_config):
        from fipsagents.server.sinks.kafka import KafkaSink

        sink = KafkaSink(config=kafka_sink_config)
        mock_producer = AsyncMock()
        mock_producer.send_and_wait.side_effect = RuntimeError("network")
        sink._producer = mock_producer

        event = _make_event()
        await sink.emit(event)  # should not raise


class TestKafkaSinkClose:
    @pytest.mark.asyncio
    async def test_close_stops_producer(self, kafka_sink_config):
        from fipsagents.server.sinks.kafka import KafkaSink

        sink = KafkaSink(config=kafka_sink_config)
        mock_producer = AsyncMock()
        sink._producer = mock_producer

        await sink.close()

        mock_producer.stop.assert_awaited_once()
        assert sink._producer is None

    @pytest.mark.asyncio
    async def test_close_no_producer_noop(self, kafka_sink_config):
        from fipsagents.server.sinks.kafka import KafkaSink

        sink = KafkaSink(config=kafka_sink_config)
        await sink.close()  # should not raise
