"""Gemma 4 Tool Calling — validate native tool use via vLLM.

Exercises Gemma 4 E4B-it's native tool calling capability. vLLM's
gemma4 tool-call parser translates the model's native <|tool_call>
tokens into standard OpenAI tool_calls, so this agent uses the same
BaseAgent ReAct loop as any other model.

The interesting part is not the agent code (it's minimal) but the
deployment: a model with non-standard tool syntax served through
vLLM's parser layer, consumed transparently via LiteLLM.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.events import StreamEvent, ToolCallDelta

logger = logging.getLogger(__name__)


class GemmaToolAgent(BaseAgent):
    """Chat agent that validates Gemma 4 tool calling."""

    async def astep_stream(
        self, *, max_iterations: int = 10
    ) -> AsyncIterator[StreamEvent]:
        """Stream with tool-call logging for validation."""
        async for event in super().astep_stream(max_iterations=max_iterations):
            if isinstance(event, ToolCallDelta) and event.name:
                logger.info(
                    "Tool call: %s (call_id=%s)",
                    event.name,
                    event.call_id,
                )
            yield event
