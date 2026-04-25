"""Tests for the Azure OpenAI provider."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_adapter.models import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    ToolFunction,
)
from llm_adapter.providers.azure import (
    AzureProvider,
    _build_request_kwargs,
    _stream_response,
    _translate_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(id="chatcmpl-test", created=1234567890, choices=[choice], usage=usage)


def _mock_chunk(content=None, finish_reason=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


async def _collect_stream(events):
    async def _aiter():
        for e in events:
            yield e

    chunks = []
    async for chunk in _stream_response(_aiter(), "test-model"):
        chunks.append(chunk)
    return chunks


def _parse_sse(sse_string):
    payload = sse_string.removeprefix("data: ").strip()
    return json.loads(payload)


# ===================================================================
# _build_request_kwargs
# ===================================================================


class TestBuildRequestKwargs:
    def test_basic_request(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
        )
        kwargs = _build_request_kwargs(req)
        assert kwargs["model"] == "gpt-4o"
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"] == "hi"

    def test_temperature_forwarded(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.7,
        )
        kwargs = _build_request_kwargs(req)
        assert kwargs["temperature"] == 0.7

    def test_temperature_absent_when_none(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
        )
        kwargs = _build_request_kwargs(req)
        assert "temperature" not in kwargs

    def test_max_tokens_forwarded(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=256,
        )
        kwargs = _build_request_kwargs(req)
        assert kwargs["max_tokens"] == 256

    def test_top_p_forwarded(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
            top_p=0.9,
        )
        kwargs = _build_request_kwargs(req)
        assert kwargs["top_p"] == 0.9

    def test_tools_converted_to_dicts(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[
                Tool(
                    function=ToolFunction(
                        name="search",
                        description="Search the web",
                        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                    )
                )
            ],
        )
        kwargs = _build_request_kwargs(req)
        assert isinstance(kwargs["tools"], list)
        assert kwargs["tools"][0]["function"]["name"] == "search"

    def test_tool_choice_forwarded(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
            tool_choice="auto",
        )
        kwargs = _build_request_kwargs(req)
        assert kwargs["tool_choice"] == "auto"

    def test_messages_exclude_none_fields(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
        )
        kwargs = _build_request_kwargs(req)
        msg = kwargs["messages"][0]
        # None fields like tool_calls, tool_call_id, name should be absent.
        assert "tool_calls" not in msg
        assert "tool_call_id" not in msg
        assert "name" not in msg


# ===================================================================
# _translate_response
# ===================================================================


class TestTranslateResponse:
    def test_text_response(self):
        resp = _mock_openai_response(content="Hello world")
        result = _translate_response(resp, "gpt-4o")
        assert result.choices[0].message.content == "Hello world"
        assert result.choices[0].message.tool_calls is None

    def test_tool_calls_response(self):
        tc = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="search", arguments='{"q":"weather"}'),
        )
        resp = _mock_openai_response(content=None, finish_reason="tool_calls", tool_calls=[tc])
        result = _translate_response(resp, "gpt-4o")
        assert result.choices[0].finish_reason == "tool_calls"
        calls = result.choices[0].message.tool_calls
        assert len(calls) == 1
        assert calls[0].id == "call_abc"
        assert calls[0].function.name == "search"
        assert calls[0].function.arguments == '{"q":"weather"}'

    @pytest.mark.parametrize("reason", ["stop", "tool_calls", "length"])
    def test_finish_reason_passthrough(self, reason):
        resp = _mock_openai_response(finish_reason=reason)
        result = _translate_response(resp, "gpt-4o")
        assert result.choices[0].finish_reason == reason

    def test_usage_extraction(self):
        resp = _mock_openai_response()
        result = _translate_response(resp, "gpt-4o")
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_model_name_from_request(self):
        resp = _mock_openai_response()
        result = _translate_response(resp, "gpt-4o-deployment")
        assert result.model == "gpt-4o-deployment"


# ===================================================================
# _stream_response
# ===================================================================


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_role_chunk_first(self):
        chunks = await _collect_stream([_mock_chunk(content="hi")])
        first = _parse_sse(chunks[0])
        assert first["choices"][0]["delta"] == {"role": "assistant"}

    @pytest.mark.asyncio
    async def test_content_deltas(self):
        events = [_mock_chunk(content="Hello"), _mock_chunk(content=" world")]
        chunks = await _collect_stream(events)
        content_chunks = [
            _parse_sse(c) for c in chunks
            if not c.startswith("data: [DONE]")
            and "content" in _parse_sse(c).get("choices", [{}])[0].get("delta", {})
        ]
        assert len(content_chunks) == 2
        assert content_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert content_chunks[1]["choices"][0]["delta"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_finish_reason_chunk(self):
        events = [_mock_chunk(finish_reason="stop")]
        chunks = await _collect_stream(events)
        finish_chunks = [
            _parse_sse(c) for c in chunks
            if not c.startswith("data: [DONE]")
            and _parse_sse(c).get("choices", [{}])[0].get("finish_reason") is not None
        ]
        assert len(finish_chunks) == 1
        assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_ends_with_done(self):
        chunks = await _collect_stream([_mock_chunk(content="hi")])
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_tool_call_deltas(self):
        tc = SimpleNamespace(
            index=0,
            id="call_1",
            function=SimpleNamespace(name="search", arguments='{"q":'),
        )
        events = [_mock_chunk(tool_calls=[tc])]
        chunks = await _collect_stream(events)
        tool_chunks = [
            _parse_sse(c) for c in chunks
            if not c.startswith("data: [DONE]")
            and "tool_calls" in _parse_sse(c).get("choices", [{}])[0].get("delta", {})
        ]
        assert len(tool_chunks) == 1
        tc_data = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc_data["id"] == "call_1"
        assert tc_data["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_usage_chunk(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        usage_event = SimpleNamespace(choices=[], usage=usage)
        events = [usage_event]
        chunks = await _collect_stream(events)
        usage_chunks = [
            _parse_sse(c) for c in chunks
            if not c.startswith("data: [DONE]") and _parse_sse(c).get("usage")
        ]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["prompt_tokens"] == 10
        assert usage_chunks[0]["usage"]["completion_tokens"] == 5
        assert usage_chunks[0]["usage"]["total_tokens"] == 15


# ===================================================================
# AzureProvider setup
# ===================================================================


class TestAzureSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_azure_client(self):
        provider = AzureProvider()
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://myresource.openai.azure.com",
        }), patch("llm_adapter.providers.azure.AsyncAzureOpenAI", create=True):
            # Patch the import inside setup()
            mock_module = MagicMock()
            mock_cls_inner = MagicMock()
            mock_module.AsyncAzureOpenAI = mock_cls_inner
            with patch.dict("sys.modules", {"openai": mock_module}):
                await provider.setup()
            assert provider._client is not None
            mock_cls_inner.assert_called_once_with(
                api_key="test-key",
                azure_endpoint="https://myresource.openai.azure.com",
                api_version="2024-10-21",
            )

    @pytest.mark.asyncio
    async def test_setup_missing_api_key_raises(self):
        provider = AzureProvider()
        with patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com"}, clear=False):
            env = os.environ.copy()
            env.pop("AZURE_OPENAI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
                    await provider.setup()

    @pytest.mark.asyncio
    async def test_setup_missing_endpoint_raises(self):
        provider = AzureProvider()
        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "key"}, clear=False):
            env = os.environ.copy()
            env.pop("AZURE_OPENAI_ENDPOINT", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
                    await provider.setup()

    @pytest.mark.asyncio
    async def test_setup_default_api_version(self):
        provider = AzureProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
        }, clear=False):
            env = os.environ.copy()
            env.pop("AZURE_API_VERSION", None)
            with patch.dict(os.environ, env, clear=True), \
                 patch.dict(os.environ, {
                     "AZURE_OPENAI_API_KEY": "key",
                     "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
                 }), \
                 patch.dict("sys.modules", {"openai": mock_module}):
                await provider.setup()
                mock_module.AsyncAzureOpenAI.assert_called_once()
                call_kwargs = mock_module.AsyncAzureOpenAI.call_args
                assert call_kwargs.kwargs["api_version"] == "2024-10-21"

    @pytest.mark.asyncio
    async def test_setup_custom_api_version(self):
        provider = AzureProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "AZURE_API_VERSION": "2025-01-01",
        }), patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
            call_kwargs = mock_module.AsyncAzureOpenAI.call_args
            assert call_kwargs.kwargs["api_version"] == "2025-01-01"


# ===================================================================
# AzureProvider shutdown
# ===================================================================


class TestAzureShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        provider = AzureProvider()
        provider._client = AsyncMock()
        await provider.shutdown()
        provider._client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_client(self):
        provider = AzureProvider()
        # Should not raise.
        await provider.shutdown()


# ===================================================================
# Provider registration
# ===================================================================


class TestAzureRegistration:
    def test_registered_as_azure(self):
        from llm_adapter.providers import _REGISTRY
        assert "azure" in _REGISTRY
