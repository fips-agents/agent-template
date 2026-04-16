"""Serialize a ``BaseAgent.astep_stream()`` event sequence to OpenAI Chat
Completions streaming format (SSE).

Wire format reference:
https://platform.openai.com/docs/api-reference/chat/streaming

Each SSE frame is a single ``data: <json>\\n\\n`` line. The stream ends with
``data: [DONE]\\n\\n``.  Each JSON payload has the shape::

    {
      "id": "chatcmpl-<hex>",
      "object": "chat.completion.chunk",
      "created": <unix timestamp>,
      "model": "<model name>",
      "choices": [{"index": 0, "delta": {...}, "finish_reason": null}]
    }

The public entry point is :func:`stream_events_as_sse`. It is a pure async
generator — no FastAPI, no logging, no side effects. Callers own the transport
and any logging they want to add around iteration.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    StreamEvent,
    ToolCallDelta,
    ToolResultEvent,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


def _sse_chunk(
    completion_id: str,
    model_name: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    """Serialize one OpenAI stream chunk as a single SSE ``data:`` frame."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def stream_events_as_sse(
    events: AsyncIterator[StreamEvent],
    model_name: str,
    completion_id: str | None = None,
) -> AsyncIterator[str]:
    """Translate a ``StreamEvent`` sequence into OpenAI SSE chunks.

    Args:
        events: Async iterator of ``StreamEvent`` instances, typically from
            ``BaseAgent.astep_stream()``.
        model_name: Model identifier echoed in every chunk's ``model`` field.
        completion_id: Optional completion ID. When ``None`` an ID of the form
            ``chatcmpl-<24 hex chars>`` is generated automatically.

    Yields:
        SSE-encoded strings (``data: {...}\\n\\n`` or ``data: [DONE]\\n\\n``).
        On exception from the source iterator an error chunk is yielded before
        ``[DONE]``.
    """
    if completion_id is None:
        completion_id = _make_completion_id()

    # OpenAI convention: lead with a role chunk so clients that key off the
    # first role they see don't misfire their "finalize message" logic on the
    # first content token.
    yield _sse_chunk(completion_id, model_name, {"role": "assistant"})

    # Per-index emission state: tracks which tool-call indexes have already
    # received their opening chunk (carrying id + name).
    opened_indexes: set[int] = set()

    try:
        async for event in events:
            if isinstance(event, ReasoningDelta):
                yield _sse_chunk(
                    completion_id,
                    model_name,
                    {"reasoning_content": event.content},
                )

            elif isinstance(event, ContentDelta):
                yield _sse_chunk(
                    completion_id,
                    model_name,
                    {"content": event.content},
                )

            elif isinstance(event, ToolCallDelta):
                # First delta for this index carries id + name.
                # Later deltas carry only the arguments fragment.
                # Skip deltas with neither a call_id (first) nor an
                # arguments_delta (continuation) — nothing to emit.
                if event.index not in opened_indexes and event.call_id:
                    opened_indexes.add(event.index)
                    yield _sse_chunk(
                        completion_id,
                        model_name,
                        {
                            "tool_calls": [
                                {
                                    "index": event.index,
                                    "id": event.call_id,
                                    "type": "function",
                                    "function": {
                                        "name": event.name or "",
                                        "arguments": event.arguments_delta,
                                    },
                                }
                            ]
                        },
                    )
                elif event.arguments_delta:
                    yield _sse_chunk(
                        completion_id,
                        model_name,
                        {
                            "tool_calls": [
                                {
                                    "index": event.index,
                                    "function": {
                                        "arguments": event.arguments_delta,
                                    },
                                }
                            ]
                        },
                    )

            elif isinstance(event, ToolResultEvent):
                yield _sse_chunk(
                    completion_id,
                    model_name,
                    {
                        "role": "tool",
                        "tool_call_id": event.call_id,
                        "content": event.content,
                    },
                )

            elif isinstance(event, StreamComplete):
                yield _sse_chunk(
                    completion_id,
                    model_name,
                    {},
                    finish_reason=event.finish_reason,
                )

    except Exception as exc:
        err = {"error": {"message": str(exc), "type": type(exc).__name__}}
        yield f"data: {json.dumps(err)}\n\n"

    yield "data: [DONE]\n\n"
