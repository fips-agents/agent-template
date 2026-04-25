"""Tests for the llama.cpp server provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_adapter.providers.llamacpp import LlamaCppProvider


# ===================================================================
# LlamaCppProvider setup
# ===================================================================


class TestLlamaCppSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_client_with_correct_args(self):
        provider = LlamaCppProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {
            "LLAMA_CPP_ENDPOINT": "http://localhost:8080",
            "LLAMA_CPP_API_KEY": "my-key",
        }), patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        assert provider._client is not None
        mock_module.AsyncOpenAI.assert_called_once_with(
            base_url="http://localhost:8080",
            api_key="my-key",
        )

    @pytest.mark.asyncio
    async def test_setup_missing_endpoint_raises(self):
        provider = LlamaCppProvider()
        env = os.environ.copy()
        env.pop("LLAMA_CPP_ENDPOINT", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="LLAMA_CPP_ENDPOINT"):
                await provider.setup()

    @pytest.mark.asyncio
    async def test_setup_default_api_key_is_no_key(self):
        provider = LlamaCppProvider()
        mock_module = MagicMock()
        env = os.environ.copy()
        env.pop("LLAMA_CPP_API_KEY", None)
        env["LLAMA_CPP_ENDPOINT"] = "http://localhost:8080"
        with patch.dict(os.environ, env, clear=True), \
             patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        call_kwargs = mock_module.AsyncOpenAI.call_args
        assert call_kwargs.kwargs["api_key"] == "no-key"

    @pytest.mark.asyncio
    async def test_setup_custom_api_key(self):
        provider = LlamaCppProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {
            "LLAMA_CPP_ENDPOINT": "http://localhost:8080",
            "LLAMA_CPP_API_KEY": "secret-token",
        }), patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        call_kwargs = mock_module.AsyncOpenAI.call_args
        assert call_kwargs.kwargs["api_key"] == "secret-token"


# ===================================================================
# LlamaCppProvider shutdown
# ===================================================================


class TestLlamaCppShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        provider = LlamaCppProvider()
        provider._client = AsyncMock()
        await provider.shutdown()
        provider._client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_client(self):
        provider = LlamaCppProvider()
        # Should not raise.
        await provider.shutdown()


# ===================================================================
# Provider registration
# ===================================================================


class TestLlamaCppRegistration:
    def test_registered_as_llamacpp(self):
        from llm_adapter.providers import _REGISTRY
        assert "llamacpp" in _REGISTRY
