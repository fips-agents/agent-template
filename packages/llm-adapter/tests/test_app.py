"""Integration tests for the FastAPI application routes.

The provider is mocked to avoid needing real API credentials.
The lifespan is bypassed for chat completion tests so we can inject
a mock provider directly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_adapter.models import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _aiter_from_list(items):
    for item in items:
        yield item


def _make_mock_provider():
    """Build an AsyncMock provider with sensible defaults."""
    provider = AsyncMock()
    provider.chat_completion.return_value = ChatCompletionResponse(
        id="chatcmpl-test",
        created=1234567890,
        model="test-model",
        choices=[
            Choice(
                message=ChoiceMessage(content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    # chat_completion_stream is an async generator, not a coroutine.
    # AsyncMock wraps the return value in a coroutine, which breaks
    # StreamingResponse.  Use a plain function that returns an async iterator.
    _stream_chunks = [
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
        '"created":0,"model":"m","choices":[{"index":0,'
        '"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
        '"created":0,"model":"m","choices":[{"index":0,'
        '"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
        "data: [DONE]\n\n",
    ]
    provider.chat_completion_stream = MagicMock(
        side_effect=lambda req: _aiter_from_list(_stream_chunks)
    )
    return provider


def _build_test_app(provider):
    """Create a fresh FastAPI app with routes wired to the given provider.

    This avoids triggering the real lifespan (which needs ANTHROPIC_API_KEY).
    We import the route handlers and register them on a bare app.
    """
    import llm_adapter.app as app_module

    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI):
        yield

    test_app = FastAPI(lifespan=_noop_lifespan)

    # Wire the same route handlers as the real app.
    test_app.add_api_route("/healthz", app_module.healthz, methods=["GET", "HEAD"])
    test_app.add_api_route(
        "/v1/chat/completions", app_module.chat_completions, methods=["POST"]
    )

    # Inject the provider into the module so the route handler sees it.
    app_module._provider = provider
    return test_app


@pytest.fixture
def mock_provider():
    return _make_mock_provider()


# ---------------------------------------------------------------------------
# Health check (uses the real app -- healthz doesn't need a provider)
# ---------------------------------------------------------------------------


class TestHealthz:
    def test_get_healthz(self):
        # Temporarily clear provider to avoid lifespan issues; healthz
        # doesn't touch it.
        test_app = _build_test_app(provider=None)
        client = TestClient(test_app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_head_healthz(self):
        test_app = _build_test_app(provider=None)
        client = TestClient(test_app)
        resp = client.head("/healthz")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_non_streaming(self, mock_provider):
        test_app = _build_test_app(mock_provider)
        client = TestClient(test_app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["total_tokens"] == 8

    def test_streaming(self, mock_provider):
        test_app = _build_test_app(mock_provider)
        client = TestClient(test_app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "data: [DONE]" in body

    def test_503_when_no_provider(self):
        import llm_adapter.app as app_module

        test_app = _build_test_app(provider=None)
        # Ensure _provider is None so the route returns 503.
        app_module._provider = None
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 503

    def test_provider_called_with_request_model(self, mock_provider):
        test_app = _build_test_app(mock_provider)
        client = TestClient(test_app)
        client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        call_args = mock_provider.chat_completion.call_args
        req = call_args[0][0]
        assert req.model == "claude-sonnet-4-6"

    def test_extra_fields_ignored(self, mock_provider):
        """vLLM-specific params like top_k are silently dropped."""
        test_app = _build_test_app(mock_provider)
        client = TestClient(test_app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "top_k": 50,
                "repetition_penalty": 1.1,
            },
        )
        assert resp.status_code == 200
