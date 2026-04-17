"""Demo Chat Agent — minimal ReAct loop for the FIPS-Agents Q&A demo.

Takes an open-ended user message, optionally calls tools, and returns a
natural-language response. MemoryHub is wired via the SDK path
(``self.memory``): the base class injects a stable memory prefix at
setup time, and this subclass writes the user turn back to memory after
each response.

Only overrides ``astep_stream``. The sync ``step`` path inherits
``BaseAgent``'s default implementation, which consumes the same stream
and concatenates content deltas — so streaming and sync clients share
the identical tool-dispatch / memory-write behavior.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.events import StreamComplete, StreamEvent

logger = logging.getLogger(__name__)

MEMORY_SCOPE = "project"
# MemoryHub SDK calls this project_id, not scope_id. Writing to a new
# project auto-enrolls.
MEMORY_SCOPE_ID = "demo-chat-agent"


class DemoChatAgent(BaseAgent):
    """Conversational agent with tool use + MemoryHub-backed memory."""

    # -- Helpers ------------------------------------------------------------

    def _latest_user_message(self) -> str | None:
        return next(
            (m["content"] for m in reversed(self.messages) if m["role"] == "user"),
            None,
        )

    async def _persist_user_turn(self, content: str) -> None:
        try:
            await self.memory.write(
                content=content,
                scope=MEMORY_SCOPE,
                project_id=MEMORY_SCOPE_ID,
                metadata={"source": "demo-chat-agent"},
            )
        except Exception:
            logger.exception("Memory write failed (continuing)")

    # -- Streaming path (canonical — sync ``step`` consumes this) -----------

    async def astep_stream(
        self, *, max_iterations: int = 10
    ) -> AsyncIterator[StreamEvent]:
        """Streaming ReAct loop with post-turn memory write-back."""
        latest_user = self._latest_user_message()

        async for event in super().astep_stream(max_iterations=max_iterations):
            yield event
            if isinstance(event, StreamComplete) and latest_user:
                await self._persist_user_turn(latest_user)
