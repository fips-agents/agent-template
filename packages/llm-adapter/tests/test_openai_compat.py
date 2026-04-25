"""Tests for the generic OpenAI-compatible provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_adapter.providers.openai_compat import OpenAICompatProvider


# ===================================================================
# OpenAICompatProvider setup
# ===================================================================


class TestOpenAICompatSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_client_with_correct_args(self):
        provider = OpenAICompatProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {
            "OPENAI_COMPAT_ENDPOINT": "http://localhost:8000/v1",
            "OPENAI_COMPAT_API_KEY": "my-key",
        }), patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        assert provider._client is not None
        mock_module.AsyncOpenAI.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="my-key",
        )

    @pytest.mark.asyncio
    async def test_setup_missing_endpoint_raises(self):
        provider = OpenAICompatProvider()
        env = os.environ.copy()
        env.pop("OPENAI_COMPAT_ENDPOINT", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_COMPAT_ENDPOINT"):
                await provider.setup()

    @pytest.mark.asyncio
    async def test_setup_default_api_key_is_no_key(self):
        provider = OpenAICompatProvider()
        mock_module = MagicMock()
        env = os.environ.copy()
        env.pop("OPENAI_COMPAT_API_KEY", None)
        env["OPENAI_COMPAT_ENDPOINT"] = "http://localhost:8000/v1"
        with patch.dict(os.environ, env, clear=True), \
             patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        call_kwargs = mock_module.AsyncOpenAI.call_args
        assert call_kwargs.kwargs["api_key"] == "no-key"

    @pytest.mark.asyncio
    async def test_setup_custom_api_key(self):
        provider = OpenAICompatProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {
            "OPENAI_COMPAT_ENDPOINT": "http://vllm:8000/v1",
            "OPENAI_COMPAT_API_KEY": "secret-token",
        }), patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        call_kwargs = mock_module.AsyncOpenAI.call_args
        assert call_kwargs.kwargs["api_key"] == "secret-token"


# ===================================================================
# OpenAICompatProvider shutdown
# ===================================================================


class TestOpenAICompatShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        provider = OpenAICompatProvider()
        provider._client = AsyncMock()
        await provider.shutdown()
        provider._client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_client(self):
        provider = OpenAICompatProvider()
        # Should not raise.
        await provider.shutdown()


# ===================================================================
# Provider registration
# ===================================================================


class TestOpenAICompatRegistration:
    def test_registered_as_openai_compat(self):
        from llm_adapter.providers import _REGISTRY
        assert "openai-compat" in _REGISTRY
