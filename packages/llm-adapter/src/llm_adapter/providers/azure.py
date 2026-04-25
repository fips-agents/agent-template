"""Azure OpenAI provider.

Azure OpenAI is OpenAI-compatible, so translation is minimal — the
adapter receives OpenAI-format requests and Azure speaks the same
format.  This provider handles Azure-specific authentication and
maps between our Pydantic models and the ``openai`` SDK objects.
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


class AzureProvider(BaseProvider):
    """Azure OpenAI backend.

    Credentials from environment:
      - ``AZURE_OPENAI_API_KEY`` — API key
      - ``AZURE_OPENAI_ENDPOINT`` — Azure endpoint URL (e.g. https://myresource.openai.azure.com)
      - ``AZURE_API_VERSION`` — API version (default: 2024-10-21)
    """

    def __init__(self) -> None:
        self._client: Any = None

    async def setup(self) -> None:
        """Create the async Azure OpenAI client."""
        from openai import AsyncAzureOpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = os.environ.get("AZURE_API_VERSION", "2024-10-21")

        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY environment variable is required")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT environment variable is required")

        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion via Azure OpenAI."""
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


register_provider("azure", AzureProvider)
