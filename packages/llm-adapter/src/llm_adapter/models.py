"""OpenAI-compatible request / response models for the adapter boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Request models — what BaseAgent sends to the adapter
# ---------------------------------------------------------------------------


class ToolFunction(BaseModel):
    """Function definition inside a tool object."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    """A tool offered to the model."""

    type: Literal["function"] = "function"
    function: ToolFunction


class ToolCallFunction(BaseModel):
    """The function payload within a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A tool invocation returned by the model."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str
    content: str | list[Any] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Incoming request following the OpenAI chat completions contract.

    ``extra="ignore"`` silently drops vLLM-specific parameters such as
    ``top_k``, ``repetition_penalty``, ``reasoning_effort``, and
    ``extra_body`` that BaseAgent may forward.
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Response models — what the adapter returns to BaseAgent
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    """Token usage counters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    """The assistant message within a response choice."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChoiceMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Response following the OpenAI chat completions contract."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
