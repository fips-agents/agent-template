"""Request/response models and helpers for the OpenAI-compatible server."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fipsagents.baseagent.events import StreamMetrics


# Session ID format: 1-128 alphanumeric characters, hyphens, or underscores.
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


# ---------------------------------------------------------------------------
# Request / response schema
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Request body for POST /v1/sessions."""

    session_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, v: str | None) -> str | None:
        if v is not None and not _SESSION_ID_RE.match(v):
            raise ValueError(
                "session_id must be 1-128 characters: "
                "letters, digits, hyphens, or underscores"
            )
        return v


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
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    # vLLM-specific parameters — forwarded via extra_body.
    top_k: int | None = None
    repetition_penalty: float | None = None
    reasoning_effort: str | None = None
    # Session persistence (extension field, not part of OpenAI API).
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation persistence. "
        "If provided but no session exists, one is created automatically.",
    )

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, v: str | None) -> str | None:
        if v is not None and not _SESSION_ID_RE.match(v):
            raise ValueError(
                "session_id must be 1-128 characters: "
                "letters, digits, hyphens, or underscores"
            )
        return v


class CreateFeedbackRequest(BaseModel):
    """Request body for POST /v1/feedback."""

    trace_id: str
    rating: int
    session_id: str | None = None
    comment: str | None = None
    correction: str | None = None
    model_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    turn_index: int | None = Field(default=None, ge=0)
    agent_type: str | None = None

    @field_validator("rating")
    @classmethod
    def _validate_rating(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("rating must be 1 (thumbs-up) or -1 (thumbs-down)")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _extract_overrides(req: ChatCompletionRequest) -> dict[str, Any]:
    """Extract non-None sampling parameters from the request.

    Standard OpenAI parameters go at the top level. vLLM-specific parameters
    (top_k, repetition_penalty, reasoning_effort) are placed inside
    ``extra_body`` so the openai SDK forwards them without validation errors.
    """
    overrides: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    # Standard OpenAI parameters.
    for key in (
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "logprobs", "top_logprobs",
    ):
        val = getattr(req, key, None)
        if val is not None:
            overrides[key] = val

    # vLLM-specific parameters — must go via extra_body.
    for key in ("top_k", "repetition_penalty", "reasoning_effort"):
        val = getattr(req, key, None)
        if val is not None:
            extra_body[key] = val

    if extra_body:
        overrides["extra_body"] = extra_body

    return overrides


def _sync_response(
    model_name: str,
    content: str,
    *,
    metrics: StreamMetrics | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    m = metrics or StreamMetrics()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens,
        },
        "stream_metrics": {
            "time_to_first_reasoning": m.time_to_first_reasoning,
            "time_to_first_content": m.time_to_first_content,
            "total_time": m.total_time,
            "inter_token_latencies": m.inter_token_latencies,
            "model_calls": m.model_calls,
            "tool_calls": m.tool_calls,
        },
    }
