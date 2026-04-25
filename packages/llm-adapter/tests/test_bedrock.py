"""Tests for the Bedrock provider.

The Bedrock provider reuses all translation logic from the Anthropic
provider (tested in test_anthropic.py).  These tests verify the provider
class wiring and credential handling.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_adapter.models import ChatCompletionRequest, ChatMessage
from llm_adapter.providers.bedrock import BedrockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_anthropic_response(content_text="Hello", stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=content_text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _simple_request(**overrides):
    defaults = {
        "model": "us.anthropic.claude-sonnet-4-6-20250514-v1:0",
        "messages": [ChatMessage(role="user", content="Hello")],
        "max_tokens": 100,
    }
    defaults.update(overrides)
    return ChatCompletionRequest(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBedrockSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_bedrock_client(self):
        provider = BedrockProvider()
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}):
            with patch("anthropic.AsyncAnthropicBedrock") as mock_cls:
                mock_cls.return_value = MagicMock()
                await provider.setup()
                mock_cls.assert_called_once_with(aws_region="us-west-2")

    @pytest.mark.asyncio
    async def test_setup_default_region(self):
        provider = BedrockProvider()
        env = {k: v for k, v in os.environ.items() if k != "AWS_REGION"}
        with patch.dict(os.environ, env, clear=True):
            with patch("anthropic.AsyncAnthropicBedrock") as mock_cls:
                mock_cls.return_value = MagicMock()
                await provider.setup()
                mock_cls.assert_called_once_with(aws_region="us-east-1")


class TestBedrockChatCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self):
        provider = BedrockProvider()
        provider._client = AsyncMock()
        provider._client.messages.create.return_value = _mock_anthropic_response()

        req = _simple_request()
        resp = await provider.chat_completion(req)

        assert resp.choices[0].message.content == "Hello"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 10
        provider._client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_bedrock_model_name_passed_through(self):
        provider = BedrockProvider()
        provider._client = AsyncMock()
        provider._client.messages.create.return_value = _mock_anthropic_response()

        model_name = "us.anthropic.claude-sonnet-4-6-20250514-v1:0"
        req = _simple_request(model=model_name)
        resp = await provider.chat_completion(req)

        # Model name from request is used in the response
        assert resp.model == model_name
        # Model name is forwarded to the API
        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == model_name


class TestBedrockStreaming:
    @pytest.mark.asyncio
    async def test_streaming_delegates_to_stream_response(self):
        """Verify streaming uses the shared _stream_response function."""
        provider = BedrockProvider()

        # Build mock stream context manager yielding Anthropic events
        async def _mock_events():
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5)),
            )
            yield SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text"),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hi"),
            )
            yield SimpleNamespace(
                type="content_block_stop",
                index=0,
            )
            yield SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=1),
            )
            yield SimpleNamespace(type="message_stop")

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=_mock_events())
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        provider._client = MagicMock()
        provider._client.messages.stream.return_value = mock_stream

        req = _simple_request()
        chunks = []
        async for chunk in provider.chat_completion_stream(req):
            chunks.append(chunk)

        # Should have: role chunk, content chunk, finish chunk, usage chunk, [DONE]
        assert any('"role": "assistant"' in c for c in chunks)
        assert any('"content": "Hi"' in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"


class TestBedrockShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        provider = BedrockProvider()
        provider._client = AsyncMock()
        await provider.shutdown()
        provider._client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_client(self):
        provider = BedrockProvider()
        await provider.shutdown()  # should not raise


class TestBedrockRegistration:
    def test_registered_as_bedrock(self):
        from llm_adapter.providers import _REGISTRY

        assert "bedrock" in _REGISTRY
        assert _REGISTRY["bedrock"] is BedrockProvider
