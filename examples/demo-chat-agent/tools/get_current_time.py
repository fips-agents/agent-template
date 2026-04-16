"""Trivial example tool used for the demo.

Shows the @tool decorator pattern and lets the audience see a visible
tool call fire in the conversation history. Replace with something
useful the moment this stops being a teaching artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fipsagents.baseagent import tool


@tool(
    description="Return the current UTC date and time in ISO-8601 format.",
    visibility="llm_only",
)
async def get_current_time() -> str:
    """Return the current UTC timestamp.

    Returns:
        ISO-8601 UTC timestamp, e.g. ``2026-04-16T14:05:09+00:00``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
