"""Tests for RemoteSubagentTransport.

Uses ``httpx.MockTransport`` to fake remote agent responses — no real
network calls are made.
"""

from __future__ import annotations

import asyncio
import json as _json

import httpx
import pytest

from fipsagents.baseagent.config import RemoteTransportConfig
from fipsagents.subagents.transport import RemoteSubagentTransport
from fipsagents.subagents.types import (
    SubagentRemoteError,
    SubagentTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(url: str, timeout: float = 60.0) -> RemoteTransportConfig:
    return RemoteTransportConfig(type="remote", url=url, timeout_seconds=timeout)


def _ok_response(
    content: str = "hello world",
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list | None = None,
    include_usage: bool = True,
) -> dict:
    """Build a well-formed OpenAI chat-completions JSON response."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    resp: dict = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "subagent",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_usage:
        resp["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return resp


def _json_handler(response_data: dict, status_code: int = 200):
    """Return an httpx MockTransport handler that always yields *response_data*."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            headers={"Content-Type": "application/json"},
            content=_json.dumps(response_data).encode(),
        )

    return handler


def _error_handler(status_code: int, detail: str = "server error"):
    """Return a handler that yields a non-2xx response with a JSON detail."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            headers={"Content-Type": "application/json"},
            content=_json.dumps({"detail": detail}).encode(),
        )

    return handler


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRemoteHappyPath:
    async def test_returns_content(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_json_handler(_ok_response("The answer is 42.")))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="What is 6 * 7?")

        assert result.content == "The answer is 42."

    async def test_returns_finish_reason(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(
            _json_handler(_ok_response("done", finish_reason="length"))
        )
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="summarise")

        assert result.finish_reason == "length"

    async def test_returns_token_counts(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(
            _json_handler(_ok_response(prompt_tokens=20, completion_tokens=8))
        )
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.tokens_used == {"input": 20, "output": 8, "cached": 0}

    async def test_agent_name_on_result(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_json_handler(_ok_response()))
        transport = RemoteSubagentTransport("my_helper", config, http_client=client)

        result = await transport.invoke(task="hi")

        assert result.agent_name == "my_helper"

    async def test_cost_usd_is_zero_in_v1(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_json_handler(_ok_response()))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.cost_usd == 0.0

    async def test_span_id_is_none_in_v1(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_json_handler(_ok_response()))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.span_id is None


# ---------------------------------------------------------------------------
# Missing usage block
# ---------------------------------------------------------------------------


class TestMissingUsage:
    async def test_no_crash_when_usage_absent(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(
            _json_handler(_ok_response(include_usage=False))
        )
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.tokens_used == {"input": 0, "output": 0, "cached": 0}

    async def test_content_still_returned_when_usage_absent(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(
            _json_handler(_ok_response("no usage here", include_usage=False))
        )
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.content == "no usage here"


# ---------------------------------------------------------------------------
# Tool calls in response
# ---------------------------------------------------------------------------


class TestToolCallsInResponse:
    async def test_tool_calls_made_reflects_count(self) -> None:
        fake_tool_calls = [
            {"id": "c1", "type": "function", "function": {"name": "fn1", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "fn2", "arguments": "{}"}},
        ]
        config = _make_config("http://agent:8080")
        client = _make_client(
            _json_handler(_ok_response(tool_calls=fake_tool_calls))
        )
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.tool_calls_made == 2

    async def test_no_tool_calls_gives_zero(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_json_handler(_ok_response()))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.tool_calls_made == 0

    async def test_empty_tool_calls_list_gives_zero(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_json_handler(_ok_response(tool_calls=[])))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        result = await transport.invoke(task="x")

        assert result.tool_calls_made == 0


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------


class TestErrorResponses:
    async def test_5xx_raises_subagent_remote_error(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_error_handler(503, "service unavailable"))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        with pytest.raises(SubagentRemoteError) as exc_info:
            await transport.invoke(task="x")

        err = exc_info.value
        assert err.status_code == 503
        assert err.agent_name == "helper"

    async def test_4xx_raises_subagent_remote_error(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_error_handler(400, "bad request"))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        with pytest.raises(SubagentRemoteError) as exc_info:
            await transport.invoke(task="x")

        assert exc_info.value.status_code == 400

    async def test_5xx_detail_in_error(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_error_handler(502, "bad gateway"))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        with pytest.raises(SubagentRemoteError) as exc_info:
            await transport.invoke(task="x")

        assert "bad gateway" in exc_info.value.detail


class TestConnectionError:
    async def test_connect_error_raises_subagent_remote_error(self) -> None:
        """A network failure converts to SubagentRemoteError(status_code=None)."""

        def failing_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        config = _make_config("http://agent:8080")
        client = _make_client(failing_handler)
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        with pytest.raises(SubagentRemoteError) as exc_info:
            await transport.invoke(task="x")

        err = exc_info.value
        assert err.status_code is None
        assert err.agent_name == "helper"
        assert "connection refused" in err.detail

    async def test_request_error_subclass_is_remote_error(self) -> None:
        """httpx.ReadError (a RequestError subclass) also maps to RemoteError."""

        def failing_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("connection reset by peer")

        config = _make_config("http://agent:8080")
        client = _make_client(failing_handler)
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        with pytest.raises(SubagentRemoteError) as exc_info:
            await transport.invoke(task="x")

        assert exc_info.value.status_code is None


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_raises_subagent_timeout_error(self) -> None:
        """A slow handler causes SubagentTimeoutError when timeout_seconds is very short."""

        async def slow_handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10)
            return httpx.Response(200, content=b"{}")

        config = _make_config("http://agent:8080", timeout=0.05)
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(slow_handler), timeout=None
        )
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        with pytest.raises(SubagentTimeoutError) as exc_info:
            await transport.invoke(task="x", timeout_seconds=0.05)

        err = exc_info.value
        assert err.agent_name == "helper"
        assert err.timeout_seconds == pytest.approx(0.05)

    async def test_timeout_error_message_contains_seconds(self) -> None:
        async def slow_handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10)
            return httpx.Response(200, content=b"{}")

        config = _make_config("http://agent:8080", timeout=0.05)
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(slow_handler), timeout=None
        )
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        with pytest.raises(SubagentTimeoutError) as exc_info:
            await transport.invoke(task="x", timeout_seconds=0.05)

        assert "0.1s" in str(exc_info.value) or "0.0s" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Trace headers pass-through
# ---------------------------------------------------------------------------


class TestTraceHeaders:
    async def test_trace_headers_passed_to_request(self) -> None:
        received_headers: dict[str, str] = {}

        def capturing_handler(request: httpx.Request) -> httpx.Response:
            received_headers.update(dict(request.headers))
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=_json.dumps(_ok_response()).encode(),
            )

        config = _make_config("http://agent:8080")
        client = _make_client(capturing_handler)
        transport = RemoteSubagentTransport("helper", config, http_client=client)
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

        await transport.invoke(
            task="x", headers={"traceparent": traceparent}
        )

        assert received_headers.get("traceparent") == traceparent

    async def test_no_headers_still_works(self) -> None:
        config = _make_config("http://agent:8080")
        client = _make_client(_json_handler(_ok_response()))
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        # Should not raise even with no headers argument.
        result = await transport.invoke(task="x")

        assert result.content is not None


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


class TestUrlHandling:
    @pytest.mark.parametrize(
        "base_url",
        [
            "http://agent:8080",
            "http://agent:8080/v1",
            "http://agent:8080/",
            "http://agent:8080/v1/",
            "http://agent:8080/v1/chat/completions",
            "http://agent:8080/v1/chat/completions/",
        ],
    )
    async def test_always_posts_to_completions_endpoint(self, base_url: str) -> None:
        received_url: list[str] = []

        def capturing_handler(request: httpx.Request) -> httpx.Response:
            received_url.append(str(request.url))
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=_json.dumps(_ok_response()).encode(),
            )

        config = _make_config(base_url)
        client = _make_client(capturing_handler)
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        await transport.invoke(task="x")

        assert len(received_url) == 1
        assert received_url[0].endswith("/v1/chat/completions"), (
            f"Expected URL ending in /v1/chat/completions, got: {received_url[0]}"
        )
        # Ensure the path is not doubled.
        assert "/v1/chat/completions/v1/chat/completions" not in received_url[0], (
            f"URL path was doubled: {received_url[0]}"
        )


# ---------------------------------------------------------------------------
# Context prepending
# ---------------------------------------------------------------------------


class TestContextPrepending:
    async def test_context_prepended_before_task(self) -> None:
        received_body: dict = {}

        def capturing_handler(request: httpx.Request) -> httpx.Response:
            received_body.update(_json.loads(request.content))
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=_json.dumps(_ok_response()).encode(),
            )

        config = _make_config("http://agent:8080")
        client = _make_client(capturing_handler)
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        await transport.invoke(task="What is the policy?", context="You are a compliance agent.")

        user_content = received_body["messages"][0]["content"]
        # Context should appear before the task, separated by a newline.
        assert user_content.startswith("You are a compliance agent.")
        assert "What is the policy?" in user_content
        context_pos = user_content.index("You are a compliance agent.")
        task_pos = user_content.index("What is the policy?")
        assert context_pos < task_pos

    async def test_no_context_sends_task_only(self) -> None:
        received_body: dict = {}

        def capturing_handler(request: httpx.Request) -> httpx.Response:
            received_body.update(_json.loads(request.content))
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=_json.dumps(_ok_response()).encode(),
            )

        config = _make_config("http://agent:8080")
        client = _make_client(capturing_handler)
        transport = RemoteSubagentTransport("helper", config, http_client=client)

        await transport.invoke(task="just the task")

        user_content = received_body["messages"][0]["content"]
        assert user_content == "just the task"
