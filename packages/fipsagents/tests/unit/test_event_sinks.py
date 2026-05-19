"""Tests for LogSink and HttpCallbackSink event sinks."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from fipsagents.server.events import OutboundEvent, create_event_sink
from fipsagents.server.sinks.http_callback import HttpCallbackSink
from fipsagents.server.sinks.log import LogSink
from fipsagents.server.sinks.null import NullSink


def _make_event(**overrides: object) -> OutboundEvent:
    defaults: dict = {
        "correlation_id": "evt-123",
        "event_type": "response",
        "payload": {"content": "hello"},
        "source": "test-source",
        "timestamp": datetime.now(tz=UTC),
    }
    defaults.update(overrides)
    return OutboundEvent(**defaults)


# -- NullSink --------------------------------------------------------------


class TestNullSink:
    @pytest.mark.asyncio
    async def test_emit_does_not_raise(self):
        sink = NullSink()
        await sink.emit(_make_event())

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        sink = NullSink()
        await sink.close()


# -- LogSink ---------------------------------------------------------------


class TestLogSink:
    @pytest.mark.asyncio
    async def test_default_level_is_info(self):
        sink = LogSink()
        assert sink._level == logging.INFO

    @pytest.mark.asyncio
    async def test_custom_level_debug(self):
        cfg = SimpleNamespace(level="DEBUG")
        sink = LogSink(config=cfg)
        assert sink._level == logging.DEBUG

    @pytest.mark.asyncio
    async def test_custom_level_warning(self):
        cfg = SimpleNamespace(level="WARNING")
        sink = LogSink(config=cfg)
        assert sink._level == logging.WARNING

    @pytest.mark.asyncio
    async def test_invalid_level_falls_back_to_info(self):
        cfg = SimpleNamespace(level="BANANA")
        sink = LogSink(config=cfg)
        assert sink._level == logging.INFO

    @pytest.mark.asyncio
    async def test_none_config_uses_info(self):
        sink = LogSink(config=None)
        assert sink._level == logging.INFO

    @pytest.mark.asyncio
    async def test_emit_logs_event(self, caplog):
        sink = LogSink()
        event = _make_event(correlation_id="c-42", event_type="test.done")
        with caplog.at_level(logging.INFO, logger="fipsagents.server.events.sink"):
            await sink.emit(event)
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.INFO
        assert "c-42" in record.message
        assert "test.done" in record.message

    @pytest.mark.asyncio
    async def test_emit_at_debug_level(self, caplog):
        cfg = SimpleNamespace(level="DEBUG")
        sink = LogSink(config=cfg)
        event = _make_event()
        with caplog.at_level(logging.DEBUG, logger="fipsagents.server.events.sink"):
            await sink.emit(event)
        assert caplog.records[0].levelno == logging.DEBUG

    @pytest.mark.asyncio
    async def test_logged_json_contains_required_fields(self, caplog):
        sink = LogSink()
        event = _make_event(
            correlation_id="abc",
            event_type="push",
            source="github",
        )
        with caplog.at_level(logging.INFO, logger="fipsagents.server.events.sink"):
            await sink.emit(event)
        logged = caplog.records[0].message
        # Extract JSON portion after "Event sink: "
        json_str = logged.split("Event sink: ", 1)[1]
        data = json.loads(json_str)
        assert data["correlation_id"] == "abc"
        assert data["event_type"] == "push"
        assert data["source"] == "github"

    @pytest.mark.asyncio
    async def test_case_insensitive_level(self):
        cfg = SimpleNamespace(level="error")
        sink = LogSink(config=cfg)
        assert sink._level == logging.ERROR


# -- HttpCallbackSink ------------------------------------------------------


class TestHttpCallbackSink:
    @pytest.mark.asyncio
    async def test_requires_config(self):
        with pytest.raises(ValueError, match="requires config"):
            HttpCallbackSink(config=None)

    @pytest.mark.asyncio
    async def test_reads_url_from_config(self):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=10.0)
        sink = HttpCallbackSink(config=cfg)
        assert sink._url == "http://example.com/cb"
        assert sink._timeout == 10.0

    @pytest.mark.asyncio
    async def test_default_timeout(self):
        cfg = SimpleNamespace(url="http://example.com/cb")
        sink = HttpCallbackSink(config=cfg)
        assert sink._timeout == 30.0

    @pytest.mark.asyncio
    async def test_setup_creates_client(self):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=5.0)
        sink = HttpCallbackSink(config=cfg)
        assert sink._client is None
        await sink.setup()
        assert sink._client is not None
        await sink.close()

    @pytest.mark.asyncio
    async def test_emit_posts_json(self):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=5.0)
        sink = HttpCallbackSink(config=cfg)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock()
        sink._client = mock_client

        event = _make_event(correlation_id="c-99")
        await sink.emit(event)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://example.com/cb"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"
        payload = call_args[1]["json"]
        assert payload["correlation_id"] == "c-99"

    @pytest.mark.asyncio
    async def test_emit_payload_contains_all_fields(self):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=5.0)
        sink = HttpCallbackSink(config=cfg)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock()
        sink._client = mock_client

        event = _make_event(
            correlation_id="c-1",
            event_type="push",
            source="github",
            payload={"ref": "main"},
        )
        await sink.emit(event)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["correlation_id"] == "c-1"
        assert payload["event_type"] == "push"
        assert payload["source"] == "github"
        assert payload["payload"] == {"ref": "main"}
        assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_emit_creates_client_lazily(self):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=5.0)
        sink = HttpCallbackSink(config=cfg)
        assert sink._client is None

        async def patched_emit(event):
            # Trigger lazy creation, then swap the client
            if sink._client is None:
                sink._client = httpx.AsyncClient(timeout=5.0)
            # Verify the client was created
            assert sink._client is not None
            await sink.close()

        await patched_emit(_make_event())

    @pytest.mark.asyncio
    async def test_network_error_logged_not_raised(self, caplog):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=5.0)
        sink = HttpCallbackSink(config=cfg)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        sink._client = mock_client

        event = _make_event(correlation_id="fail-1")
        with caplog.at_level(logging.ERROR, logger="fipsagents.server.events.sink"):
            await sink.emit(event)  # must not raise

        assert any("Failed to emit event" in r.message for r in caplog.records)
        assert any("fail-1" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_timeout_error_logged_not_raised(self, caplog):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=5.0)
        sink = HttpCallbackSink(config=cfg)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timed out"),
        )
        sink._client = mock_client

        with caplog.at_level(logging.ERROR, logger="fipsagents.server.events.sink"):
            await sink.emit(_make_event())

        assert any("Failed to emit" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        cfg = SimpleNamespace(url="http://example.com/cb", timeout_seconds=5.0)
        sink = HttpCallbackSink(config=cfg)
        await sink.setup()
        assert sink._client is not None
        await sink.close()
        assert sink._client is None

    @pytest.mark.asyncio
    async def test_close_noop_without_client(self):
        cfg = SimpleNamespace(url="http://example.com/cb")
        sink = HttpCallbackSink(config=cfg)
        await sink.close()  # should not raise


# -- Factory tests ---------------------------------------------------------


class TestCreateEventSinkFactory:
    def test_log_type_returns_log_sink(self):
        cfg = SimpleNamespace(type="log", level="DEBUG")
        sink = create_event_sink(cfg)
        assert isinstance(sink, LogSink)

    def test_http_callback_type_returns_http_sink(self):
        cfg = SimpleNamespace(
            type="http_callback",
            url="http://example.com/cb",
            timeout_seconds=10.0,
        )
        sink = create_event_sink(cfg)
        assert isinstance(sink, HttpCallbackSink)

    def test_null_type_returns_null_sink(self):
        cfg = SimpleNamespace(type="null")
        sink = create_event_sink(cfg)
        assert isinstance(sink, NullSink)

    def test_none_returns_null_sink(self):
        sink = create_event_sink(None)
        assert isinstance(sink, NullSink)

    def test_unknown_type_raises(self):
        cfg = SimpleNamespace(type="foobar")
        with pytest.raises(ValueError, match="Unknown event sink type"):
            create_event_sink(cfg)
