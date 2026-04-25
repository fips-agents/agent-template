"""Vertex AI / Gemini provider.

Translates OpenAI-format chat completion requests into Google Gemini's native
format via the ``google-genai`` SDK, and translates responses (both streaming
and non-streaming) back into the OpenAI contract that BaseAgent expects.
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

_GEMINI_TO_OPENAI_STOP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}


def _make_completion_id() -> str:
    """Return a unique completion ID in the ``chatcmpl-`` format."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _make_tool_call_id() -> str:
    """Return a unique tool call ID in the ``call_`` format."""
    return f"call_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Request translation: messages
# ---------------------------------------------------------------------------


def _build_gemini_messages(
    messages: list[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]], dict[str, str]]:
    """Convert an OpenAI message list to Gemini format.

    Returns ``(system_instruction, contents, tool_call_id_to_name)``.
    """
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []
    tool_result_buffer: list[dict[str, Any]] = []
    tool_call_id_to_name: dict[str, str] = {}

    def _flush_tool_results() -> None:
        if tool_result_buffer:
            result.append({"role": "user", "parts": list(tool_result_buffer)})
            tool_result_buffer.clear()

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
            continue

        if msg.role == "tool":
            name = tool_call_id_to_name.get(msg.tool_call_id or "", msg.name or "")
            tool_result_buffer.append(
                {
                    "function_response": {
                        "name": name,
                        "response": {"result": msg.content or ""},
                    },
                }
            )
            continue

        # user / assistant / developer -- flush any pending tool results first
        _flush_tool_results()

        if msg.role == "assistant":
            parts: list[dict[str, Any]] = []
            if msg.content and isinstance(msg.content, str) and msg.content.strip():
                parts.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    parts.append(
                        {
                            "function_call": {
                                "name": tc.function.name,
                                "args": args,
                            },
                        }
                    )
                    tool_call_id_to_name[tc.id] = tc.function.name
            if parts:
                result.append({"role": "model", "parts": parts})
        else:
            # user or developer -- both map to Gemini "user" role
            content = msg.content or ""
            text = content if isinstance(content, str) else str(content)
            result.append({"role": "user", "parts": [{"text": text}]})

    # Flush any trailing tool results.
    _flush_tool_results()

    # Merge consecutive same-role messages so Gemini doesn't reject the
    # payload (its API forbids adjacent messages with the same role).
    merged: list[dict[str, Any]] = []
    for entry in result:
        if merged and merged[-1]["role"] == entry["role"]:
            merged[-1]["parts"].extend(entry["parts"])
        else:
            merged.append(entry)

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, merged, tool_call_id_to_name


# ---------------------------------------------------------------------------
# Request translation: tools
# ---------------------------------------------------------------------------


def _translate_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool definitions to Gemini function declarations."""
    if not tools:
        return None
    declarations = []
    for t in tools:
        decl: dict[str, Any] = {"name": t.function.name}
        if t.function.description:
            decl["description"] = t.function.description
        if t.function.parameters:
            decl["parameters"] = t.function.parameters
        declarations.append(decl)
    return [{"function_declarations": declarations}]


def _translate_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Map OpenAI ``tool_choice`` to Gemini's ToolConfig format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}
        if tool_choice == "required":
            return {"function_calling_config": {"mode": "ANY"}}
        if tool_choice == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        return None
    if isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        name = func.get("name") if isinstance(func, dict) else None
        if name:
            return {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [name],
                },
            }
    return None


# ---------------------------------------------------------------------------
# Full request translation
# ---------------------------------------------------------------------------


def _translate_request(
    request: ChatCompletionRequest,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build ``(config_dict, contents)`` for Gemini generate_content."""
    system_instruction, contents, _ = _build_gemini_messages(request.messages)

    config: dict[str, Any] = {}

    if system_instruction:
        config["system_instruction"] = system_instruction
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.max_tokens is not None:
        config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None:
        config["top_p"] = request.top_p

    tools = _translate_tools(request.tools)
    if tools:
        config["tools"] = tools

    tool_config = _translate_tool_choice(request.tool_choice)
    if tool_config is not None:
        config["tool_config"] = tool_config

    return config, contents


# ---------------------------------------------------------------------------
# Response translation: non-streaming
# ---------------------------------------------------------------------------


def _translate_response(
    response: Any, model: str
) -> ChatCompletionResponse:
    """Map a Gemini response object to the OpenAI response schema."""
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    candidate = response.candidates[0]
    for part in candidate.content.parts:
        if hasattr(part, "text") and part.text is not None:
            content_parts.append(part.text)
        elif hasattr(part, "function_call") and part.function_call is not None:
            fc = part.function_call
            tool_calls.append(
                ToolCall(
                    id=_make_tool_call_id(),
                    type="function",
                    function=ToolCallFunction(
                        name=fc.name,
                        arguments=json.dumps(fc.args) if fc.args else "{}",
                    ),
                )
            )

    # Determine finish reason -- override to "tool_calls" if any calls present.
    raw_reason = str(candidate.finish_reason) if candidate.finish_reason else "STOP"
    # The enum may come as e.g. "FinishReason.STOP" or just "STOP".
    reason_key = raw_reason.rsplit(".", 1)[-1]
    if tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = _GEMINI_TO_OPENAI_STOP.get(reason_key, "stop")

    # Extract usage from response metadata.
    prompt_tokens = 0
    completion_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(meta, "candidates_token_count", 0) or 0

    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
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
    """Translate a Gemini streaming response into OpenAI SSE chunks.

    Yields ``data: {...}\\n\\n`` strings terminated by ``data: [DONE]\\n\\n``.
    Tracks previously seen text length and tool call count to emit deltas.
    """
    completion_id = _make_completion_id()
    tool_call_index = 0
    sent_role = False
    input_tokens = 0
    output_tokens = 0

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
        async for chunk in stream:
            # Lead with the role chunk on first event.
            if not sent_role:
                yield _sse_chunk({"role": "assistant"})
                sent_role = True

            candidate = chunk.candidates[0] if chunk.candidates else None
            if candidate is None:
                continue

            parts = candidate.content.parts if candidate.content else []

            # google-genai streams incremental chunks -- each chunk
            # contains only the NEW parts, so emit them directly.
            for part in parts:
                if hasattr(part, "text") and part.text is not None:
                    yield _sse_chunk({"content": part.text})
                elif hasattr(part, "function_call") and part.function_call is not None:
                    fc = part.function_call
                    tc_id = _make_tool_call_id()
                    args_str = json.dumps(fc.args) if fc.args else "{}"
                    yield _sse_chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_call_index,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": fc.name,
                                        "arguments": args_str,
                                    },
                                }
                            ]
                        }
                    )
                    tool_call_index += 1

            # Check for finish reason on this chunk.
            raw_reason = (
                str(candidate.finish_reason) if candidate.finish_reason else None
            )
            if raw_reason:
                reason_key = raw_reason.rsplit(".", 1)[-1]
                if tool_call_index > 0:
                    finish_reason = "tool_calls"
                else:
                    finish_reason = _GEMINI_TO_OPENAI_STOP.get(reason_key, "stop")

                # Skip "FINISH_REASON_UNSPECIFIED" or similar non-terminal values.
                if reason_key in _GEMINI_TO_OPENAI_STOP or tool_call_index > 0:
                    yield _sse_chunk({}, finish_reason=finish_reason)

                    # Extract usage from the chunk if available.
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        meta = chunk.usage_metadata
                        input_tokens = (
                            getattr(meta, "prompt_token_count", 0) or 0
                        )
                        output_tokens = (
                            getattr(meta, "candidates_token_count", 0) or 0
                        )

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


class VertexProvider(BaseProvider):
    """Vertex AI / Gemini backend for the LLM adapter."""

    def __init__(self) -> None:
        self._client: Any = None

    async def setup(self) -> None:
        """Create the Gemini client via ``google-genai`` with Vertex AI auth.

        Requires ``GOOGLE_CLOUD_PROJECT`` (mandatory) and optionally
        ``GOOGLE_CLOUD_LOCATION`` (defaults to ``us-central1``).
        """
        from google import genai  # noqa: E402

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT environment variable is required"
            )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        self._client = genai.Client(
            vertexai=True, project=project, location=location
        )

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion via the Gemini API."""
        config, contents = _translate_request(request)
        response = await self._client.aio.models.generate_content(
            model=request.model, contents=contents, config=config
        )
        return _translate_response(response, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Streaming chat completion yielding OpenAI-format SSE strings."""
        config, contents = _translate_request(request)
        stream = self._client.aio.models.generate_content_stream(
            model=request.model, contents=contents, config=config
        )
        async for chunk in _stream_response(stream, request.model):
            yield chunk

    async def shutdown(self) -> None:
        """No-op -- the google-genai client manages its own lifecycle."""
        pass


# Auto-register so importing this module wires the provider into the registry.
register_provider("vertex", VertexProvider)
