"""Azure OpenAI provider.

Azure OpenAI is OpenAI-compatible, so translation is minimal — the
adapter receives OpenAI-format requests and Azure speaks the same
format.  This provider handles Azure-specific authentication and
maps between our Pydantic models and the ``openai`` SDK objects.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator

from llm_adapter.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from llm_adapter.providers import register_provider
from llm_adapter.providers.base import BaseProvider


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _build_request_kwargs(request: ChatCompletionRequest) -> dict[str, Any]:
    """Build kwargs for openai.chat.completions.create()."""
    # Convert our Pydantic messages to dicts the SDK expects.
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
    }
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.tools:
        kwargs["tools"] = [t.model_dump() for t in request.tools]
    if request.tool_choice is not None:
        kwargs["tool_choice"] = request.tool_choice
    return kwargs


def _translate_response(
    response: Any, model: str
) -> ChatCompletionResponse:
    """Map an OpenAI SDK ChatCompletion to our Pydantic response model."""
    choice = response.choices[0]
    msg = choice.message

    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                type="function",
                function=ToolCallFunction(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in msg.tool_calls
        ]

    return ChatCompletionResponse(
        id=response.id or _make_completion_id(),
        created=response.created or int(time.time()),
        model=model,
        choices=[
            Choice(
                index=choice.index,
                message=ChoiceMessage(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=tool_calls,
                ),
                finish_reason=choice.finish_reason or "stop",
            )
        ],
        usage=Usage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        ),
    )


async def _stream_response(
    stream: Any, model: str
) -> AsyncIterator[str]:
    """Translate an OpenAI SDK async stream into SSE strings.

    The SDK yields ChatCompletionChunk objects. We serialize each
    to the ``data: {...}\\n\\n`` wire format.
    """
    completion_id = _make_completion_id()

    # Lead with role chunk.
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    try:
        async for chunk in stream:
            if not chunk.choices:
                # Usage-only chunk at the end
                if chunk.usage:
                    usage_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        },
                    }
                    yield f"data: {json.dumps(usage_data)}\n\n"
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            delta_dict: dict[str, Any] = {}
            if hasattr(delta, "content") and delta.content is not None:
                delta_dict["content"] = delta.content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                delta_dict["reasoning_content"] = delta.reasoning_content
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                delta_dict["tool_calls"] = [
                    {
                        k: v for k, v in {
                            "index": tc.index,
                            "id": tc.id if tc.id else None,
                            "type": "function" if tc.id else None,
                            "function": {
                                k2: v2 for k2, v2 in {
                                    "name": tc.function.name if tc.function and tc.function.name else None,
                                    "arguments": tc.function.arguments if tc.function and tc.function.arguments else None,
                                }.items() if v2 is not None
                            } if tc.function else None,
                        }.items() if v is not None
                    }
                    for tc in delta.tool_calls
                ]

            # Skip empty deltas that aren't finish-reason deltas
            if not delta_dict and finish_reason is None:
                continue

            out = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta_dict, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(out)}\n\n"

    except Exception as exc:
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": str(exc), "type": type(exc).__name__},
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    yield "data: [DONE]\n\n"


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
