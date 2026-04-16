"""Demo Chat Agent — minimal ReAct loop for the FIPS-Agents Q&A demo.

Takes an open-ended user message, optionally calls tools, and returns a
natural-language response. MemoryHub is wired via the SDK path
(``self.memory``): the agent searches memory before responding and
writes facts back after each turn.

Only overrides ``astep_stream``. The sync ``step`` path inherits
``BaseAgent``'s default implementation, which consumes the same stream
and concatenates content deltas — so sync and streaming clients share
the identical memory-recall / tool-dispatch / memory-write behavior.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.events import StreamComplete, StreamEvent

logger = logging.getLogger(__name__)

# Cap how many memories we inject as context per turn. Too many and we
# blow the context window with noise; too few and recall feels random.
MAX_MEMORY_RESULTS = 5

MEMORY_SCOPE = "project"
# MemoryHub SDK calls this project_id, not scope_id. Writing to a new
# project auto-enrolls.
MEMORY_SCOPE_ID = "demo-chat-agent"


class DemoChatAgent(BaseAgent):
    """Conversational agent with tool use + MemoryHub-backed memory."""

    # -- Shared pre/post-step helpers ---------------------------------------

    def _ensure_system_prompt(self) -> None:
        if not any(m["role"] == "system" for m in self.messages):
            self.messages.insert(
                0, {"role": "system", "content": self.build_system_prompt()}
            )

    def _latest_user_message(self) -> str | None:
        return next(
            (m["content"] for m in reversed(self.messages) if m["role"] == "user"),
            None,
        )

    async def _inject_memory_recall(self, query: str) -> None:
        """Search MemoryHub and inject results as a system note before
        the latest user message."""
        # raw_results=True bypasses a SDK<->server schema mismatch where
        # stub entries from the server lack the ``content`` field the
        # SDK's SearchResult Pydantic model requires.
        try:
            memories = await self.memory.search(
                query,
                max_results=MAX_MEMORY_RESULTS,
                scope=MEMORY_SCOPE,
                project_id=MEMORY_SCOPE_ID,
                raw_results=True,
            )
        except Exception:
            logger.exception("Memory search failed; continuing without recall")
            return
        if not memories:
            return
        recall = self._format_memories(memories)
        logger.info("Recalled %d memories for turn", len(memories))
        # Insert as a system message right before the latest user
        # message — keeps recall in the high-attention zone.
        user_idx = next(
            (
                i
                for i, m in enumerate(self.messages)
                if m["role"] == "user" and m["content"] == query
            ),
            None,
        )
        if user_idx is None:
            return
        self.messages.insert(user_idx, {"role": "system", "content": recall})

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

    @staticmethod
    def _format_memories(memories: list[dict]) -> str:
        """Render search results as a compact system note for the model."""
        lines = ["Relevant memories from prior conversations:"]
        for m in memories:
            content = m.get("content") or m.get("stub") or ""
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)

    # -- Streaming path (canonical — sync ``step`` consumes this) -----------

    async def astep_stream(
        self, *, max_iterations: int = 10
    ) -> AsyncIterator[StreamEvent]:
        """Streaming variant of ``step`` with the same memory behavior.

        Adds system prompt + memory recall before delegating to
        ``BaseAgent.astep_stream``, and writes the user turn to memory
        after the stream terminates.
        """
        if not self.messages:
            return

        self._ensure_system_prompt()

        latest_user = self._latest_user_message()
        if latest_user:
            await self._inject_memory_recall(latest_user)

        async for event in super().astep_stream(max_iterations=max_iterations):
            yield event
            if isinstance(event, StreamComplete) and latest_user:
                # Write to memory only after we've emitted the terminal
                # event so the client gets its response as fast as
                # possible.
                await self._persist_user_turn(latest_user)
