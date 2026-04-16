"""FastAPI server for the Demo Chat Agent.

Exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint. The agent
is stateless — the client sends the full ``messages`` array each turn;
conversation history is tracked by the UI.

Streaming emits standard OpenAI deltas with ``reasoning_content``,
``tool_calls``, ``role="tool"`` + ``tool_call_id``, and ``content`` all
in the same stream. No custom extension fields — rich clients render
phases by inspecting which delta fields each chunk carries.

Endpoints:
  GET  /healthz               -- liveness probe
  GET  /readyz                -- readiness probe
  POST /v1/chat/completions   -- OpenAI-compatible chat completions (sync + SSE)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from fipsagents.baseagent.config import parse_yaml_with_env
from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    ToolCallDelta,
    ToolResultEvent,
)

from src.agent import DemoChatAgent

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent.parent

_agent: DemoChatAgent | None = None
# A single agent instance backs all requests; streaming mutates
# self.messages so we serialize per-request access.
_agent_lock = asyncio.Lock()


def _materialize_memoryhub_config() -> None:
    """Expand ${VAR:-default} placeholders in .memoryhub.yaml in place.

    The MemoryHub SDK reads .memoryhub.yaml directly and does not
    perform env-var substitution itself.
    """
    path = APP_DIR / ".memoryhub.yaml"
    if not path.exists():
        return
    raw = path.read_text(encoding="utf-8")
    expanded = parse_yaml_with_env(raw)
    path.write_text(yaml.safe_dump(expanded, sort_keys=False), encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _materialize_memoryhub_config()
    _agent = DemoChatAgent(
        config_path=APP_DIR / "agent.yaml",
        base_dir=APP_DIR,
    )
    await _agent.setup()
    logger.info("Demo Chat Agent ready")
    try:
        yield
    finally:
        await _agent.shutdown()


app = FastAPI(title="Demo Chat Agent", version="0.2.0", lifespan=lifespan)


# -- OpenAI request schema --------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


# -- Helpers ----------------------------------------------------------------

def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


def _messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert incoming Pydantic messages back to OpenAI-shaped dicts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _sync_response(model_name: str, content: str) -> dict[str, Any]:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _sse_chunk(
    completion_id: str,
    model_name: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    """Serialize an OpenAI stream chunk as a single SSE ``data:`` line."""
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


# -- Endpoints --------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if _agent is None:
        return JSONResponse({"status": "not ready"}, status_code=503)
    return {"status": "ready"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    model_name = req.model or _agent.config.model.name
    incoming = _messages_to_dicts(req.messages)

    if not req.stream:
        async with _agent_lock:
            _agent.messages = list(incoming)
            result = await _agent.run()
        return JSONResponse(_sync_response(model_name, str(result)))

    return StreamingResponse(
        _stream_openai_deltas(incoming, model_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx/haproxy buffering
        },
    )


async def _stream_openai_deltas(
    incoming: list[dict[str, Any]],
    model_name: str,
) -> AsyncIterator[str]:
    """Drive the agent's event stream and serialize to OpenAI SSE chunks.

    Maps each framework ``StreamEvent`` to a standard OpenAI delta. No
    extension fields — clients that understand reasoning/tool roles
    render them; simpler clients still see the assistant content.
    """
    completion_id = _completion_id()

    # Open with an empty assistant role chunk. Some clients key off the
    # first role they see; sending this up front avoids their
    # "finalize message" logic firing on the first content chunk.
    yield _sse_chunk(completion_id, model_name, {"role": "assistant"})

    async with _agent_lock:
        assert _agent is not None
        _agent.messages = list(incoming)

        # Per-index tool-call emission state: tracks whether we've sent
        # the opening chunk (id/name) for a given tool_call index yet.
        opened_indexes: set[int] = set()

        try:
            async for event in _agent.astep_stream(max_iterations=10):
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
                    # First delta for this index carries id+name; later
                    # deltas carry arguments token-by-token.
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
                    # Emit tool result as a standard ``tool``-role
                    # message chunk. This uses only OpenAI-spec fields.
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
                    # Emit metrics as a usage-like trailing chunk for
                    # rich clients. Standard OpenAI ``usage`` lives at
                    # the chunk top level (not in delta), but sticking
                    # to delta-only for now keeps parsers happy.
                    yield _sse_chunk(
                        completion_id,
                        model_name,
                        {},
                        finish_reason=event.finish_reason,
                    )
                    logger.info(
                        "Stream complete: finish=%s model_calls=%d tool_calls=%d "
                        "ttft_content=%s total=%.2fs",
                        event.finish_reason,
                        event.metrics.model_calls,
                        event.metrics.tool_calls,
                        event.metrics.time_to_first_content,
                        event.metrics.total_time,
                    )
        except Exception as exc:
            logger.exception("Stream errored")
            err = {"error": {"message": str(exc), "type": type(exc).__name__}}
            yield f"data: {json.dumps(err)}\n\n"

    yield "data: [DONE]\n\n"
