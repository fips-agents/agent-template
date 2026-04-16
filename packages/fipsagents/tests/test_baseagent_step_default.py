"""Test ``BaseAgent.step()``'s default implementation.

Subclasses that override only ``astep_stream`` should get a working sync
``step`` for free — the default concatenates ``ContentDelta`` events and
ignores reasoning, tool-call, tool-result, and completion events.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from fipsagents.baseagent import BaseAgent, StepOutcome
from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    StreamEvent,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)


class _StubAgent(BaseAgent):
    """Minimal BaseAgent subclass that yields a fixed event stream.

    Skips the normal ``__init__`` so we don't need a full config/llm/tools
    setup — ``step`` only calls ``astep_stream``, which we override.
    """

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def astep_stream(  # type: ignore[override]
        self, *, max_iterations: int = 10
    ) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_default_step_concatenates_content_deltas():
    agent = _StubAgent(
        [
            ContentDelta(content="Hello "),
            ContentDelta(content="world"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
    )
    result = await agent.step()
    assert result.outcome == StepOutcome.DONE
    assert result.result == "Hello world"


@pytest.mark.asyncio
async def test_default_step_ignores_reasoning_and_tool_events():
    agent = _StubAgent(
        [
            ReasoningDelta(content="thinking about it"),
            ToolCallDelta(index=0, call_id="call_1", name="search",
                          arguments_delta='{"q":"hi"}'),
            ToolResultEvent(call_id="call_1", name="search", content="result"),
            ContentDelta(content="final answer"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
    )
    result = await agent.step()
    assert result.result == "final answer"


@pytest.mark.asyncio
async def test_default_step_empty_stream_returns_empty_string():
    agent = _StubAgent(
        [StreamComplete(finish_reason="stop", metrics=StreamMetrics())]
    )
    result = await agent.step()
    assert result.outcome == StepOutcome.DONE
    assert result.result == ""


@pytest.mark.asyncio
async def test_subclass_can_override_step_and_bypass_default():
    """An explicit ``step`` override still works — the default only kicks in
    when the subclass does NOT override step()."""

    class _ExplicitStepAgent(_StubAgent):
        async def step(self):  # type: ignore[override]
            from fipsagents.baseagent import StepResult
            return StepResult.done("bypassed")

    agent = _ExplicitStepAgent([ContentDelta(content="ignored")])
    result = await agent.step()
    assert result.result == "bypassed"
