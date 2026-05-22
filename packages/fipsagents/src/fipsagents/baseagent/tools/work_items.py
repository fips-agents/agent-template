"""Factory for work-item coordination stock tools.

Call :func:`make_work_item_tools` once per agent instance during setup.
The returned list of callables is decorated with ``@tool`` and ready to pass to
``ToolRegistry.register``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fipsagents.baseagent.events import (
    WorkItemCheckedOut,
    WorkItemCompleted,
    WorkItemReleased,
)
from fipsagents.baseagent.tools import tool
from fipsagents.baseagent.tools._stock import StockToolSpec
from fipsagents.server.work_items import HandoffNote

logger = logging.getLogger("fipsagents.work_items_tool")


def _handoff_to_dict(note: HandoffNote) -> dict[str, Any]:
    """Serialize HandoffNote to dict."""
    return {
        "accomplished": note.accomplished,
        "attempted": note.attempted,
        "remaining": note.remaining,
        "blockers": note.blockers,
        "artifacts": note.artifacts,
        "context": note.context,
    }


def make_work_item_tools(agent: object) -> list:
    """Build the work-item coordination tools for this agent.

    Returns:
        List of 5 ``@tool``-decorated async functions ready for
        ``ToolRegistry.register``.
    """

    def _get_store():
        """Retrieve the work-item store from agent, raising if not configured."""
        store = getattr(agent, "_work_item_store", None)
        if store is None:
            raise RuntimeError("Work item store not configured")
        return store

    def _get_actor_id() -> str:
        """Retrieve the actor ID from agent attributes."""
        return getattr(agent, "_work_item_actor_id", None) or "unknown"

    def _emit(event):
        """Append *event* to ``agent._work_item_events`` defensively."""
        buf = getattr(agent, "_work_item_events", None)
        if buf is not None:
            buf.append(event)

    @tool(
        description=(
            "List available work items from the pool that match your capabilities. "
            "Returns items ordered by priority (highest first)."
        ),
        visibility="llm_only",
        name="check_available_work",
    )
    async def check_available_work(max_results: int = 5) -> str:
        """List work items available for checkout.

        Args:
            max_results: Maximum number of items to return.

        Returns:
            JSON array of work items with id, title, description, priority,
            and handoff_note.
        """
        store = _get_store()
        caps = None
        cfg = getattr(agent, "config", None)
        if cfg and hasattr(cfg, "server") and hasattr(cfg.server, "work_items"):
            from fipsagents.server.work_items import Capability

            caps = [
                Capability(name=c.name, value=c.value)
                for c in cfg.server.work_items.capabilities
            ]
        items = await store.list_available(capabilities=caps, max_results=max_results)
        return json.dumps(
            [
                {
                    "id": item.id,
                    "title": item.title,
                    "description": item.description,
                    "priority": item.priority,
                    "handoff_note": (
                        _handoff_to_dict(item.handoff_note)
                        if item.handoff_note
                        else None
                    ),
                }
                for item in items
            ]
        )

    @tool(
        description=(
            "Check out a work item from the pool and claim it for processing. "
            "Only one agent can hold a work item at a time. "
            "The lease auto-expires if not renewed."
        ),
        visibility="llm_only",
        name="checkout_work_item",
    )
    async def checkout_work_item(
        item_id: str, lease_duration_seconds: int = 300
    ) -> str:
        """Check out and claim a work item.

        Args:
            item_id: ID of the work item to check out.
            lease_duration_seconds: How long to hold the lease before auto-expire.

        Returns:
            JSON object with full work item details including acceptance_criteria
            and handoff_note.
        """
        store = _get_store()
        actor = _get_actor_id()
        item = await store.checkout(
            item_id, actor, lease_duration_seconds=lease_duration_seconds
        )
        _emit(WorkItemCheckedOut(item_id=item.id, actor_id=actor, title=item.title))
        result = {
            "id": item.id,
            "title": item.title,
            "description": item.description,
            "acceptance_criteria": item.acceptance_criteria,
            "handoff_note": (
                _handoff_to_dict(item.handoff_note) if item.handoff_note else None
            ),
            "lease_expires_at": item.lease_expires_at,
        }
        return json.dumps(result)

    @tool(
        description=(
            "Mark a checked-out work item as complete. "
            "Provide a summary of what was accomplished."
        ),
        visibility="llm_only",
        name="complete_work_item",
    )
    async def complete_work_item(
        item_id: str,
        result_summary: str,
        accomplished: list[str],
        review_required: bool = False,
    ) -> str:
        """Complete a work item.

        Args:
            item_id: ID of the work item to complete.
            result_summary: Summary of what was accomplished.
            accomplished: List of specific accomplishments.
            review_required: Whether human review is needed before final acceptance.

        Returns:
            JSON object with item id, status, and title.
        """
        store = _get_store()
        actor = _get_actor_id()
        handoff = HandoffNote(accomplished=accomplished)
        item = await store.complete(
            item_id,
            result={"summary": result_summary},
            handoff_note=handoff,
            review_required=review_required,
        )
        _emit(WorkItemCompleted(item_id=item.id, actor_id=actor, title=item.title))
        return json.dumps(
            {"id": item.id, "status": item.status.value, "title": item.title}
        )

    @tool(
        description=(
            "Release a work item back to the pool with a structured handoff note "
            "for the next agent."
        ),
        visibility="llm_only",
        name="release_work_item",
    )
    async def release_work_item(
        item_id: str,
        accomplished: list[str],
        remaining: list[str],
        blockers: list[str] | None = None,
        context: str = "",
    ) -> str:
        """Release a work item back to the pool.

        Args:
            item_id: ID of the work item to release.
            accomplished: What was completed during this checkout.
            remaining: What still needs to be done.
            blockers: Issues preventing further progress.
            context: Additional context for the next agent.

        Returns:
            JSON object with item id, status, and title.
        """
        store = _get_store()
        actor = _get_actor_id()
        handoff = HandoffNote(
            accomplished=accomplished,
            remaining=remaining,
            blockers=blockers or [],
            context=context,
        )
        item = await store.release(item_id, handoff_note=handoff)
        _emit(WorkItemReleased(item_id=item.id, actor_id=actor, title=item.title))
        return json.dumps(
            {"id": item.id, "status": item.status.value, "title": item.title}
        )

    @tool(
        description=(
            "Update progress on a checked-out work item. "
            "Implicitly renews the lease."
        ),
        visibility="llm_only",
        name="update_work_progress",
    )
    async def update_work_progress(
        item_id: str,
        status_message: str,
        accomplished_so_far: list[str] | None = None,
    ) -> str:
        """Update progress on a work item.

        Args:
            item_id: ID of the work item.
            status_message: Current status message.
            accomplished_so_far: Optional list of what has been done so far.

        Returns:
            JSON object with item id, status, and updated_at timestamp.
        """
        store = _get_store()
        progress = {"status_message": status_message}
        if accomplished_so_far:
            progress["accomplished_so_far"] = accomplished_so_far
        item = await store.update_progress(item_id, progress=progress)
        return json.dumps(
            {
                "id": item.id,
                "status": item.status.value,
                "updated_at": item.updated_at,
            }
        )

    return [
        check_available_work,
        checkout_work_item,
        complete_work_item,
        release_work_item,
        update_work_progress,
    ]


STOCK_TOOL_SPEC = StockToolSpec(
    factory=make_work_item_tools,
    condition=lambda agent: (
        hasattr(agent, "config")
        and hasattr(getattr(agent, "config", None), "server")
        and getattr(
            getattr(getattr(agent, "config", None), "server", None),
            "work_items",
            None,
        )
        is not None
        and getattr(
            getattr(getattr(agent, "config", None), "server", None),
            "work_items",
            None,
        ).enabled
    ),
)
