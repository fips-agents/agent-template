"""AWS Bedrock Converse API provider.

Supports ALL Bedrock models (Meta Llama, Mistral, DeepSeek, Qwen, Amazon
Nova, etc.) via the unified Converse API -- unlike ``bedrock.py`` which
uses the Anthropic SDK and is Claude-only.

boto3 is synchronous, so all calls are dispatched to a thread pool via
``asyncio.get_event_loop().run_in_executor()``.
"""

from __future__ import annotations

import asyncio
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

_CONVERSE_TO_OPENAI_STOP: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def _make_completion_id() -> str:
    """Return a unique completion ID in the ``chatcmpl-`` format."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Request translation: messages
# ---------------------------------------------------------------------------


def _build_converse_messages(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert OpenAI messages to ``(system_blocks, converse_messages)``.

    System messages are extracted to the top-level ``system`` parameter.
    Tool results are buffered and flushed into user messages (Converse
    requires tool results inside a ``user`` turn, same as Anthropic).
    Consecutive same-role messages are merged.
    """
    system_blocks: list[dict[str, Any]] = []
    result: list[dict[str, Any]] = []
    tool_result_buffer: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        if tool_result_buffer:
            result.append({"role": "user", "content": list(tool_result_buffer)})
            tool_result_buffer.clear()

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                system_blocks.append({"text": text})
            continue

        if msg.role == "tool":
            tool_result_buffer.append(
                {
                    "toolResult": {
                        "toolUseId": msg.tool_call_id,
                        "content": [{"text": msg.content or ""}],
                    }
                }
            )
            continue

        # user / assistant / developer -- flush pending tool results first
        _flush_tool_results()

        if msg.role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if msg.content and isinstance(msg.content, str) and msg.content.strip():
                content_blocks.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        tool_input = {}
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.id,
                                "name": tc.function.name,
                                "input": tool_input,
                            }
                        }
                    )
            if content_blocks:
                result.append({"role": "assistant", "content": content_blocks})
        else:
            # user or developer -- both map to "user"
            text = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            result.append({"role": "user", "content": [{"text": text}]})

    # Flush trailing tool results.
    _flush_tool_results()

    # Merge consecutive same-role messages.
    merged: list[dict[str, Any]] = []
    for entry in result:
        if merged and merged[-1]["role"] == entry["role"]:
            prev = merged[-1]["content"]
            curr = entry["content"]
            if isinstance(prev, list) and isinstance(curr, list):
                prev.extend(curr)
            else:
                merged.append(entry)
        else:
            merged.append(entry)

    return system_blocks, merged


# ---------------------------------------------------------------------------
# Request translation: tools
# ---------------------------------------------------------------------------


def _translate_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool definitions to Converse ``toolSpec`` format."""
    if not tools:
        return None
    return [
        {
            "toolSpec": {
                "name": t.function.name,
                "description": t.function.description or "",
                "inputSchema": {
                    "json": t.function.parameters
                    or {"type": "object", "properties": {}},
                },
            }
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Full request kwargs builder
# ---------------------------------------------------------------------------


def _build_converse_kwargs(request: ChatCompletionRequest) -> dict[str, Any]:
    """Build the kwargs dict for ``client.converse()`` / ``converse_stream()``."""
    system_blocks, messages = _build_converse_messages(request.messages)

    kwargs: dict[str, Any] = {
        "modelId": request.model,
        "messages": messages,
    }

    if system_blocks:
        kwargs["system"] = system_blocks

    inference_config: dict[str, Any] = {}
    if request.max_tokens is not None:
        inference_config["maxTokens"] = request.max_tokens
    if request.temperature is not None:
        inference_config["temperature"] = request.temperature
    if request.top_p is not None:
        inference_config["topP"] = request.top_p
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    tools = _translate_tools(request.tools)
    if tools:
        kwargs["toolConfig"] = {"tools": tools}

    return kwargs


# ---------------------------------------------------------------------------
# Response translation: non-streaming
# ---------------------------------------------------------------------------


def _translate_response(
    response: dict[str, Any], model: str
) -> ChatCompletionResponse:
    """Map a Converse API response dict to the OpenAI response schema."""
    output_msg = response["output"]["message"]
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in output_msg.get("content", []):
        if "text" in block:
            content_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                ToolCall(
                    id=tu["toolUseId"],
                    type="function",
                    function=ToolCallFunction(
                        name=tu["name"],
                        arguments=json.dumps(tu["input"]),
                    ),
                )
            )

    stop_reason = response.get("stopReason", "end_turn")
    finish_reason = _CONVERSE_TO_OPENAI_STOP.get(stop_reason, stop_reason)

    usage_data = response.get("usage", {})
    input_tokens = usage_data.get("inputTokens", 0)
    output_tokens = usage_data.get("outputTokens", 0)

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
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Response translation: streaming
# ---------------------------------------------------------------------------


def _sse_chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    """Format a single SSE frame."""
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


async def _stream_response(
    sync_stream: Any, model: str
) -> AsyncIterator[str]:
    """Translate a Converse EventStream into OpenAI SSE chunks.

    The boto3 EventStream is synchronous -- each ``next()`` may block on
    network I/O, so iteration is dispatched to a thread pool.
    """
    loop = asyncio.get_event_loop()
    completion_id = _make_completion_id()
    tool_call_index = 0
    block_to_tool_index: dict[int, int] = {}
    input_tokens = 0
    output_tokens = 0

    # Lead with role chunk.
    yield _sse_chunk(completion_id, model, {"role": "assistant"})

    # Wrap in iter() — boto3 EventStream is iterable but may not be an
    # iterator (no __next__).  The lambda must capture `it` and `sentinel`
    # as default args to avoid late-binding closure issues.
    it = iter(sync_stream)
    sentinel = object()

    while True:
        event = await loop.run_in_executor(
            None, lambda _it=it, _s=sentinel: next(_it, _s)
        )
        if event is sentinel:
            break

        if "contentBlockStart" in event:
            cbs = event["contentBlockStart"]
            idx = cbs.get("contentBlockIndex", 0)
            start = cbs.get("start", {})
            if "toolUse" in start:
                tu = start["toolUse"]
                block_to_tool_index[idx] = tool_call_index
                yield _sse_chunk(
                    completion_id,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": tool_call_index,
                                "id": tu.get("toolUseId", ""),
                                "type": "function",
                                "function": {
                                    "name": tu.get("name", ""),
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                )
                tool_call_index += 1

        elif "contentBlockDelta" in event:
            cbd = event["contentBlockDelta"]
            idx = cbd.get("contentBlockIndex", 0)
            delta = cbd.get("delta", {})
            if "text" in delta:
                yield _sse_chunk(
                    completion_id, model, {"content": delta["text"]}
                )
            elif "toolUse" in delta:
                ti = block_to_tool_index.get(idx, 0)
                partial = delta["toolUse"].get("input", "")
                yield _sse_chunk(
                    completion_id,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": ti,
                                "function": {"arguments": partial},
                            }
                        ]
                    },
                )

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
            finish_reason = _CONVERSE_TO_OPENAI_STOP.get(
                stop_reason, stop_reason
            )
            yield _sse_chunk(
                completion_id, model, {}, finish_reason=finish_reason
            )

        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
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

        # messageStart, contentBlockStop -- no action needed.

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class BedrockConverseProvider(BaseProvider):
    """AWS Bedrock Converse API backend for all Bedrock models.

    Credentials are discovered via the standard AWS credential chain:
    ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``,
    ``AWS_REGION`` environment variables (or IAM role when running on AWS).
    """

    def __init__(self) -> None:
        self._client: Any = None

    async def setup(self) -> None:
        """Create the boto3 bedrock-runtime client."""
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion via the Converse API."""
        kwargs = _build_converse_kwargs(request)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self._client.converse(**kwargs)
        )
        return _translate_response(response, request.model)

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Streaming chat completion yielding OpenAI-format SSE strings."""
        kwargs = _build_converse_kwargs(request)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self._client.converse_stream(**kwargs)
        )
        stream = response["stream"]
        async for chunk in _stream_response(stream, request.model):
            yield chunk

    async def shutdown(self) -> None:
        """No-op -- boto3 clients do not require explicit close."""


register_provider("bedrock-converse", BedrockConverseProvider)
