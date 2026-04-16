"""Typed events emitted by ``BaseAgent.astep_stream``.

A streaming agent run produces a sequence of these events. Server code
serializes them to whatever wire format the consumer expects:

- The standard ``/v1/chat/completions`` SSE shape uses only standard
  OpenAI delta fields (``reasoning_content``, ``tool_calls``,
  ``role="tool"`` + ``tool_call_id``, ``content``). No custom fields
  required.
- A future ``/v1/responses`` endpoint can serialize the same event
  stream to the OpenAI Responses API event protocol used by LlamaStack.

Events are intentionally framework-internal. Consumers depend on this
typed surface, not on litellm chunk shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class ReasoningDelta:
    """Incremental chunk of model reasoning ("thinking")."""

    content: str


@dataclass
class ToolCallDelta:
    """Incremental chunk of a tool-call decision streamed from the model.

    The first delta for a given ``index`` carries ``call_id`` and
    ``name``. Subsequent deltas for the same ``index`` only carry
    ``arguments_delta`` — the JSON arguments string streamed
    token-by-token. Consumers should accumulate ``arguments_delta`` per
    ``index`` until the model finishes the tool-call decision.
    """

    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class ToolResultEvent:
    """Result of executing a tool the model decided to call.

    Emitted after the agent runs the tool. ``call_id`` matches the
    ``call_id`` from the originating ``ToolCallDelta`` so consumers can
    pair decisions with results.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class ContentDelta:
    """Incremental chunk of the user-visible assistant response."""

    content: str


@dataclass
class StreamMetrics:
    """Per-stream timing and token counts.

    Captured incrementally during the stream and finalized in
    ``StreamComplete``. Times are seconds since the stream began.
    Counts come from the provider's usage block when available;
    otherwise they remain ``None``.
    """

    time_to_first_reasoning: float | None = None
    time_to_first_content: float | None = None
    total_time: float = 0.0
    inter_token_latencies: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model_calls: int = 0
    tool_calls: int = 0


@dataclass
class StreamComplete:
    """Terminal event for a streaming agent run."""

    finish_reason: str
    metrics: StreamMetrics


# Discriminated union of every event a stream can emit.
StreamEvent = Union[
    ReasoningDelta,
    ToolCallDelta,
    ToolResultEvent,
    ContentDelta,
    StreamComplete,
]
