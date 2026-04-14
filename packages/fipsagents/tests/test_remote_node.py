"""Tests for fipsagents.workflow.remote_node — RemoteNode HTTP delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from fipsagents.baseagent.config import BackoffConfig
from fipsagents.workflow.remote_node import RemoteNode, RemoteNodeError
from fipsagents.workflow.state import WorkflowState


# ---------------------------------------------------------------------------
# Test state
# ---------------------------------------------------------------------------


class RemoteState(WorkflowState):
    query: str = ""
    result: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_client_patch(responses: list[MagicMock]) -> MagicMock:
    """Return a patch target for httpx.AsyncClient that yields *responses* in order.

    Each call to ``client.post(...)`` consumes the next response in the list.
    The returned object is suitable for use as the ``new`` argument to
    ``patch("httpx.AsyncClient")``.
    """
    client_mock = AsyncMock()
    client_mock.post = AsyncMock(side_effect=responses)

    # Support async context manager protocol on the class itself.
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    cls_mock = MagicMock(return_value=cm)
    return cls_mock, client_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemoteNodeSuccessfulRoundTrip:
    async def test_returns_deserialized_state(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080")
        response_data = {"state": {"query": "q", "result": "done"}}

        cls_mock, _ = _make_client_patch([_mock_response(200, response_data)])
        with patch("httpx.AsyncClient", cls_mock):
            result = await node.process(RemoteState(query="q"))

        assert isinstance(result, RemoteState)
        assert result.query == "q"
        assert result.result == "done"

    async def test_empty_fields_default_correctly(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080")
        response_data = {"state": {"query": "", "result": ""}}

        cls_mock, _ = _make_client_patch([_mock_response(200, response_data)])
        with patch("httpx.AsyncClient", cls_mock):
            result = await node.process(RemoteState())

        assert result.query == ""
        assert result.result == ""


class TestRemoteNodePayload:
    async def test_sends_state_and_state_type(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080")
        response_data = {"state": {"query": "hello", "result": ""}}

        cls_mock, client_mock = _make_client_patch([_mock_response(200, response_data)])
        with patch("httpx.AsyncClient", cls_mock):
            await node.process(RemoteState(query="hello"))

        _, kwargs = client_mock.post.call_args
        payload = kwargs["json"]
        assert "state" in payload
        assert payload["state"] == {"query": "hello", "result": ""}
        assert "state_type" in payload

    async def test_state_type_is_fully_qualified(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080")
        response_data = {"state": {"query": "", "result": ""}}

        cls_mock, client_mock = _make_client_patch([_mock_response(200, response_data)])
        with patch("httpx.AsyncClient", cls_mock):
            await node.process(RemoteState())

        _, kwargs = client_mock.post.call_args
        state_type = kwargs["json"]["state_type"]
        # Must contain the module and class name.
        assert "RemoteState" in state_type
        assert "." in state_type


class TestRemoteNodeUrlConstruction:
    @pytest.mark.parametrize(
        "endpoint, path, expected_url",
        [
            ("http://agent:8080", "/process", "http://agent:8080/process"),
            ("http://agent:8080/", "/process", "http://agent:8080/process"),
            ("http://agent:8080", "/run", "http://agent:8080/run"),
            ("http://agent:8080//", "/process", "http://agent:8080/process"),
        ],
    )
    async def test_url_is_constructed_correctly(self, endpoint, path, expected_url):
        node = RemoteNode(name="n", endpoint=endpoint, path=path)
        response_data = {"state": {"query": "", "result": ""}}

        cls_mock, client_mock = _make_client_patch([_mock_response(200, response_data)])
        with patch("httpx.AsyncClient", cls_mock):
            await node.process(RemoteState())

        args, _ = client_mock.post.call_args
        assert args[0] == expected_url


class TestRemoteNodeRetry:
    async def test_retries_on_http_500(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080", retries=2)
        response_data = {"state": {"query": "", "result": "ok"}}

        responses = [
            _mock_response(500),
            _mock_response(200, response_data),
        ]
        cls_mock, client_mock = _make_client_patch(responses)
        with patch("httpx.AsyncClient", cls_mock), patch("asyncio.sleep", new_callable=AsyncMock):
            result = await node.process(RemoteState())

        assert result.result == "ok"
        assert client_mock.post.call_count == 2

    async def test_retries_on_connect_error(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080", retries=2)
        response_data = {"state": {"query": "", "result": "ok"}}

        connect_error = httpx.ConnectError("connection refused")
        responses = [connect_error, _mock_response(200, response_data)]
        cls_mock, client_mock = _make_client_patch(responses)
        with patch("httpx.AsyncClient", cls_mock), patch("asyncio.sleep", new_callable=AsyncMock):
            result = await node.process(RemoteState())

        assert result.result == "ok"
        assert client_mock.post.call_count == 2

    async def test_exhausted_retries_raises_remote_node_error(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080", retries=1)

        cls_mock, client_mock = _make_client_patch([
            _mock_response(500),
            _mock_response(500),
        ])
        with patch("httpx.AsyncClient", cls_mock), patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RemoteNodeError) as exc_info:
                await node.process(RemoteState())

        # Message should name the node and include the URL.
        msg = str(exc_info.value)
        assert "worker" in msg
        assert "http://agent:8080" in msg
        assert client_mock.post.call_count == 2  # initial + 1 retry

    async def test_zero_retries_single_attempt(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080", retries=0)

        cls_mock, client_mock = _make_client_patch([_mock_response(500)])
        with patch("httpx.AsyncClient", cls_mock):
            with pytest.raises(RemoteNodeError):
                await node.process(RemoteState())

        assert client_mock.post.call_count == 1

    async def test_zero_retries_no_sleep(self):
        node = RemoteNode(name="worker", endpoint="http://agent:8080", retries=0)

        cls_mock, _ = _make_client_patch([_mock_response(500)])
        with patch("httpx.AsyncClient", cls_mock), \
             patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            with pytest.raises(RemoteNodeError):
                await node.process(RemoteState())

        sleep_mock.assert_not_called()


class TestRemoteNodeBackoff:
    async def test_custom_backoff_delay_on_retry(self):
        """Verify that the delay on the first retry matches BackoffConfig.initial."""
        backoff = BackoffConfig(initial=0.01, max=0.1, multiplier=2.0)
        node = RemoteNode(
            name="worker",
            endpoint="http://agent:8080",
            retries=2,
            backoff=backoff,
        )
        response_data = {"state": {"query": "", "result": ""}}

        responses = [_mock_response(500), _mock_response(200, response_data)]
        cls_mock, _ = _make_client_patch(responses)

        with patch("httpx.AsyncClient", cls_mock), \
             patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await node.process(RemoteState())

        # First retry: delay = initial * multiplier^0 = 0.01
        sleep_mock.assert_called_once_with(0.01)

    async def test_backoff_multiplier_applied_on_second_retry(self):
        """Second retry delay = initial * multiplier^1."""
        backoff = BackoffConfig(initial=0.01, max=1.0, multiplier=2.0)
        node = RemoteNode(
            name="worker",
            endpoint="http://agent:8080",
            retries=3,
            backoff=backoff,
        )
        response_data = {"state": {"query": "", "result": ""}}

        responses = [
            _mock_response(500),
            _mock_response(500),
            _mock_response(200, response_data),
        ]
        cls_mock, _ = _make_client_patch(responses)

        with patch("httpx.AsyncClient", cls_mock), \
             patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await node.process(RemoteState())

        assert sleep_mock.call_count == 2
        calls = sleep_mock.call_args_list
        assert calls[0] == call(0.01)    # attempt 1: initial * 2^0
        assert calls[1] == call(0.02)    # attempt 2: initial * 2^1

    async def test_backoff_capped_at_max(self):
        """Delay never exceeds BackoffConfig.max."""
        backoff = BackoffConfig(initial=0.5, max=0.5, multiplier=2.0)
        node = RemoteNode(
            name="worker",
            endpoint="http://agent:8080",
            retries=3,
            backoff=backoff,
        )
        response_data = {"state": {"query": "", "result": ""}}

        responses = [
            _mock_response(500),
            _mock_response(500),
            _mock_response(200, response_data),
        ]
        cls_mock, _ = _make_client_patch(responses)

        with patch("httpx.AsyncClient", cls_mock), \
             patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await node.process(RemoteState())

        for sleep_call in sleep_mock.call_args_list:
            assert sleep_call.args[0] <= 0.5
