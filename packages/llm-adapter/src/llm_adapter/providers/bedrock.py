"""AWS Bedrock provider.

Reuses the Anthropic translation layer -- Bedrock Claude models use the
identical Messages API.  Only the client instantiation (IAM auth instead
of API key) differs.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from llm_adapter.models import ChatCompletionRequest, ChatCompletionResponse
from llm_adapter.providers import register_provider
from llm_adapter.providers.base import BaseProvider

# Reuse all translation logic from the Anthropic provider.
from llm_adapter.providers.anthropic import (
    _stream_response,
    _translate_request,
    _translate_response,
)


class BedrockProvider(BaseProvider):
    """AWS Bedrock backend using the Anthropic SDK's Bedrock client.

    Credentials are discovered via the standard AWS credential chain:
    ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``,
    ``AWS_REGION`` environment variables (or IAM role when running on AWS).
    """

    def __init__(self) -> None:
        self._client: Any = None

    async def setup(self) -> None:
        """Create the async Anthropic Bedrock client."""
        import anthropic

        region = os.environ.get("AWS_REGION", "us-east-1")
        self._client = anthropic.AsyncAnthropicBedrock(aws_region=region)

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion via Bedrock."""
        kwargs = _translate_request(request)
        response = await self._client.messages.create(**kwargs)
        return _translate_response(response, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Streaming chat completion yielding OpenAI-format SSE strings."""
        kwargs = _translate_request(request)
        async with self._client.messages.stream(**kwargs) as stream:
            async for chunk in _stream_response(stream, request.model):
                yield chunk

    async def shutdown(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()


register_provider("bedrock", BedrockProvider)
