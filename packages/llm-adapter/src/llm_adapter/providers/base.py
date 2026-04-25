"""Abstract base class for LLM provider backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from llm_adapter.models import ChatCompletionRequest, ChatCompletionResponse


class BaseProvider(ABC):
    """Contract that every provider backend must satisfy.

    Implementations translate between the OpenAI-compatible adapter boundary
    and the provider's native SDK / API.
    """

    @abstractmethod
    async def setup(self) -> None:
        """Initialize the provider client.  Read credentials from env."""

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion.  Returns an OpenAI-format response."""

    @abstractmethod
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Streaming chat completion.

        Yields OpenAI-format SSE strings (``data: {...}\\n\\n``).
        """

    async def shutdown(self) -> None:
        """Clean up provider resources.  Override if needed."""
