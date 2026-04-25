"""llama.cpp server provider.

The llama.cpp HTTP server (``llama-server``) exposes an OpenAI-compatible
``/v1/chat/completions`` endpoint, so this provider is a thin
specialisation with llama.cpp-specific environment variables.
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


class LlamaCppProvider(BaseProvider):
    """llama.cpp server backend.

    Credentials from environment:
      - ``LLAMA_CPP_ENDPOINT`` — base URL of the llama-server (required)
      - ``LLAMA_CPP_API_KEY`` — API key (optional, defaults to "no-key")

    Do NOT append ``/v1`` to the endpoint — llama-server already serves
    ``/v1/chat/completions`` from its root and the OpenAI SDK appends the
    path automatically.
    """

    def __init__(self) -> None:
        self._client: Any = None

    async def setup(self) -> None:
        """Create the async OpenAI client pointed at the llama-server endpoint."""
        from openai import AsyncOpenAI

        endpoint = os.environ.get("LLAMA_CPP_ENDPOINT")
        if not endpoint:
            raise RuntimeError("LLAMA_CPP_ENDPOINT environment variable is required")

        api_key = os.environ.get("LLAMA_CPP_API_KEY", "no-key")
        self._client = AsyncOpenAI(base_url=endpoint, api_key=api_key)

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion via the llama-server endpoint."""
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


register_provider("llamacpp", LlamaCppProvider)
