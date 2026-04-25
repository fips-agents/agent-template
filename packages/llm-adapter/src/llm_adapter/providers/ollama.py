"""Ollama provider for local model inference.

Ollama exposes an OpenAI-compatible API at ``/v1``, so this provider
is a thin specialisation of the generic OpenAI-compatible pattern with
Ollama-specific defaults and environment variables.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from llm_adapter.models import ChatCompletionRequest, ChatCompletionResponse
from llm_adapter.providers import register_provider
from llm_adapter.providers._openai_helpers import (
    _build_request_kwargs,
    _stream_response,
    _translate_response,
)
from llm_adapter.providers.base import BaseProvider

_DEFAULT_ENDPOINT = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """Ollama local inference backend.

    Credentials from environment:
      - ``OLLAMA_ENDPOINT`` — base URL of the Ollama server
        (optional, defaults to ``http://localhost:11434``)

    No API key is required; Ollama has no authentication.
    """

    def __init__(self) -> None:
        self._client: Any = None

    async def setup(self) -> None:
        """Create the async OpenAI client pointed at the Ollama endpoint."""
        from openai import AsyncOpenAI

        endpoint = os.environ.get("OLLAMA_ENDPOINT", _DEFAULT_ENDPOINT).rstrip("/")
        self._client = AsyncOpenAI(base_url=endpoint + "/v1", api_key="no-key")

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion via the Ollama endpoint."""
        kwargs = _build_request_kwargs(request)
        response = await self._client.chat.completions.create(**kwargs)
        return _translate_response(response, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Streaming chat completion yielding OpenAI-format SSE strings."""
        kwargs = _build_request_kwargs(request)
        kwargs["stream"] = True
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in _stream_response(stream, request.model):
            yield chunk

    async def shutdown(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()


register_provider("ollama", OllamaProvider)
