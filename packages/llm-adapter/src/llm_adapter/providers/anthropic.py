"""Anthropic Messages API provider.

Translates OpenAI-format chat completion requests into Anthropic's native
Messages API format, and translates responses (both streaming and
non-streaming) back into the OpenAI contract that BaseAgent expects.
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
    ChatMessage,
    Choice,
    ChoiceMessage,
    Tool,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from llm_adapter.providers import register_provider
from llm_adapter.providers.base import BaseProvider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANTHROPIC_TO_OPENAI_STOP: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def _make_completion_id() -> str:
    """Return a unique completion ID in the ``chatcmpl-`` format."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


def _normalize_content(content: str | list | None) -> list[dict[str, Any]]:
    """Coerce message content into Anthropic's content-block list form."""
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content  # already a list (multimodal)


# ---------------------------------------------------------------------------
# Request translation: messages
# ---------------------------------------------------------------------------


def _build_anthropic_messages(
    messages: list[ChatMessage],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert an OpenAI message list to ``(system_prompt, anthropic_msgs)``.

    Handles system extraction, tool-result buffering (Anthropic requires
    tool results as ``user`` messages), assistant tool_calls, and a final
    merge pass to collapse consecutive same-role messages.
    """
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []
    tool_result_buffer: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        if tool_result_buffer:
            result.append({"role": "user", "content": list(tool_result_buffer)})
            tool_result_buffer.clear()

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
            continue

        if msg.role == "tool":
            tool_result_buffer.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
            )
            continue

        # user / assistant / developer -- flush any pending tool results first
        _flush_tool_results()

        if msg.role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if msg.content and isinstance(msg.content, str) and msg.content.strip():
                content_blocks.append({"type": "text", "text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        tool_input = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tool_input,
                        }
                    )
            if content_blocks:
                result.append({"role": "assistant", "content": content_blocks})
        else:
            # user or developer -- both map to Anthropic "user" role
            blocks = _normalize_content(msg.content)
            result.append({"role": "user", "content": blocks})

    # Flush any trailing tool results (most common case: agent sends tool
    # results and expects the next assistant turn).
    _flush_tool_results()

    # Merge consecutive same-role messages so Anthropic doesn't reject the
    # payload (its API forbids adjacent messages with the same role).
    merged: list[dict[str, Any]] = []
    for entry in result:
        if merged and merged[-1]["role"] == entry["role"]:
            prev = merged[-1]["content"]
            curr = entry["content"]
            # Both are lists of content blocks at this point.
            if isinstance(prev, list) and isinstance(curr, list):
                prev.extend(curr)
            else:
                merged.append(entry)
        else:
            merged.append(entry)

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, merged


# ---------------------------------------------------------------------------
# Request translation: tools
# ---------------------------------------------------------------------------


def _translate_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool definitions to Anthropic ``input_schema`` format."""
    if not tools:
        return None
    return [
        {
            "name": t.function.name,
            "description": t.function.description or "",
            "input_schema": t.function.parameters
            or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


def _translate_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Map OpenAI ``tool_choice`` to Anthropic's format.

    OpenAI values: ``"none"``, ``"auto"``, ``"required"``,
    ``{"type":"function","function":{"name":"..."}}``.

    Anthropic values: ``{"type":"auto"}``, ``{"type":"any"}``,
    ``{"type":"tool","name":"..."}``.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None  # Anthropic has no "none" — just don't send tools
        return None
    if isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        name = func.get("name") if isinstance(func, dict) else None
        if name:
            return {"type": "tool", "name": name}
    return None


# ---------------------------------------------------------------------------
# Full request translation
# ---------------------------------------------------------------------------


def _translate_request(request: ChatCompletionRequest) -> dict[str, Any]:
    """Build the kwargs dict for ``anthropic.messages.create()``."""
    system, messages = _build_anthropic_messages(request.messages)

    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens or 4096,  # Anthropic requires max_tokens
        "messages": messages,
    }

    if system:
        kwargs["system"] = system
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p

    tools = _translate_tools(request.tools)
    if tools:
        kwargs["tools"] = tools

    tool_choice = _translate_tool_choice(request.tool_choice)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    return kwargs


# ---------------------------------------------------------------------------
# Response translation: non-streaming
# ---------------------------------------------------------------------------


def _translate_response(
    response: Any, model: str
) -> ChatCompletionResponse:
    """Map an Anthropic ``Message`` object to the OpenAI response schema."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.content:
        if block.type == "text":
            content_parts.append(block.text)
        elif block.type == "thinking":
            reasoning_parts.append(block.thinking)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    type="function",
                    function=ToolCallFunction(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                )
            )

    finish_reason = _ANTHROPIC_TO_OPENAI_STOP.get(
        response.stop_reason, response.stop_reason
    )
    usage = Usage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    return ChatCompletionResponse(
        id=_make_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                message=ChoiceMessage(
                    role="assistant",
                    content="\n".join(content_parts) if content_parts else None,
                    tool_calls=tool_calls or None,
                    reasoning_content=(
                        "\n".join(reasoning_parts) if reasoning_parts else None
                    ),
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Response translation: streaming
# ---------------------------------------------------------------------------


async def _stream_response(
    stream: Any, model: str
) -> AsyncIterator[str]:
    """Translate an Anthropic event stream into OpenAI SSE chunks.

    Yields ``data: {...}\\n\\n`` strings compatible with the OpenAI streaming
    contract, terminated by ``data: [DONE]\\n\\n``.
    """
    completion_id = _make_completion_id()
    tool_call_index = 0
    block_to_tool_index: dict[int, int] = {}
    input_tokens = 0
    output_tokens = 0
    sent_role = False

    def _sse_chunk(
        delta: dict[str, Any], finish_reason: str | None = None
    ) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    try:
        async for event in stream:
            # Lead with the role chunk on first event.
            if not sent_role:
                yield _sse_chunk({"role": "assistant"})
                sent_role = True

            etype = event.type

            if etype == "message_start":
                input_tokens = event.message.usage.input_tokens

            elif etype == "content_block_start":
                block_type = event.content_block.type
                if block_type == "tool_use":
                    block_to_tool_index[event.index] = tool_call_index
                    yield _sse_chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "id": event.content_block.id,
                                    "type": "function",
                                    "function": {
                                        "name": event.content_block.name,
                                        "arguments": "",
                                    },
                                }
                            ]
                        }
                    )
                    tool_call_index += 1
                # text / thinking: deltas will follow, no chunk needed here

            elif etype == "content_block_delta":
                delta_type = event.delta.type
                if delta_type == "text_delta":
                    yield _sse_chunk({"content": event.delta.text})
                elif delta_type == "thinking_delta":
                    yield _sse_chunk({"reasoning_content": event.delta.thinking})
                elif delta_type == "input_json_delta":
                    idx = block_to_tool_index[event.index]
                    yield _sse_chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": idx,
                                    "function": {
                                        "arguments": event.delta.partial_json,
                                    },
                                }
                            ]
                        }
                    )

            elif etype == "content_block_stop":
                pass  # no action needed

            elif etype == "message_delta":
                finish_reason = _ANTHROPIC_TO_OPENAI_STOP.get(
                    event.delta.stop_reason, event.delta.stop_reason
                )
                output_tokens = event.usage.output_tokens
                yield _sse_chunk({}, finish_reason=finish_reason)

                # Usage chunk (no choices).
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            elif etype == "message_stop":
                yield "data: [DONE]\n\n"

    except Exception as exc:
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }
            ],
            "error": {"message": str(exc), "type": type(exc).__name__},
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API backend for the LLM adapter."""

    def __init__(self) -> None:
        self._client: Any = None

    async def setup(self) -> None:
        """Create the async Anthropic client from ``ANTHROPIC_API_KEY``."""
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is required"
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion via the Anthropic Messages API."""
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


# Auto-register so importing this module wires the provider into the registry.
register_provider("anthropic", AnthropicProvider)
