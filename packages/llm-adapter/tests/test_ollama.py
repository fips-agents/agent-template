"""Tests for the Ollama provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_adapter.providers.ollama import OllamaProvider


# ===================================================================
# OllamaProvider setup
# ===================================================================


class TestOllamaSetup:
    @pytest.mark.asyncio
    async def test_setup_default_endpoint(self):
        provider = OllamaProvider()
        mock_module = MagicMock()
        env = os.environ.copy()
        env.pop("OLLAMA_ENDPOINT", None)
        with patch.dict(os.environ, env, clear=True), \
             patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        mock_module.AsyncOpenAI.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="no-key",
        )

    @pytest.mark.asyncio
    async def test_setup_custom_endpoint_appends_v1(self):
        provider = OllamaProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {"OLLAMA_ENDPOINT": "http://myhost:11434"}), \
             patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        call_kwargs = mock_module.AsyncOpenAI.call_args
        assert call_kwargs.kwargs["base_url"] == "http://myhost:11434/v1"

    @pytest.mark.asyncio
    async def test_setup_api_key_is_always_no_key(self):
        provider = OllamaProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {"OLLAMA_ENDPOINT": "http://localhost:11434"}), \
             patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        call_kwargs = mock_module.AsyncOpenAI.call_args
        assert call_kwargs.kwargs["api_key"] == "no-key"

    @pytest.mark.asyncio
    async def test_setup_trailing_slash_stripped_before_v1(self):
        provider = OllamaProvider()
        mock_module = MagicMock()
        with patch.dict(os.environ, {"OLLAMA_ENDPOINT": "http://localhost:11434/"}), \
             patch.dict("sys.modules", {"openai": mock_module}):
            await provider.setup()
        call_kwargs = mock_module.AsyncOpenAI.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:11434/v1"


# ===================================================================
# OllamaProvider shutdown
# ===================================================================


class TestOllamaShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        provider = OllamaProvider()
        provider._client = AsyncMock()
        await provider.shutdown()
        provider._client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_client(self):
        provider = OllamaProvider()
        # Should not raise.
        await provider.shutdown()


# ===================================================================
# Provider registration
# ===================================================================


class TestOllamaRegistration:
    def test_registered_as_ollama(self):
        from llm_adapter.providers import _REGISTRY
        assert "ollama" in _REGISTRY
