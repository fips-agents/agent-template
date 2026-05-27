"""Tests for capability auto-discovery from MCP servers and skills."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fipsagents.baseagent.skills import Skill, SkillLoader
from fipsagents.server.work_items import (
    Capability,
    WorkItem,
    WorkItemStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_stub(
    *,
    mcp_clients: list[tuple] | None = None,
    skills: dict[str, Skill] | None = None,
    config_capabilities: list[Capability] | None = None,
):
    """Build a minimal object with the attributes _discover_capabilities reads.

    Uses a real SkillLoader and attaches the BaseAgent method so we can
    call it without requiring full setup().
    """
    from fipsagents.baseagent.agent import BaseAgent

    agent = MagicMock(spec=[])  # empty spec — we set everything explicitly
    agent._mcp_clients = mcp_clients or []

    loader = SkillLoader()
    if skills:
        loader._skills = dict(skills)
    agent.skills = loader

    agent._discovered_capabilities = []

    # Wire config.server.work_items.capabilities when provided.
    if config_capabilities is not None:
        wi_cfg = MagicMock()
        wi_cfg.capabilities = config_capabilities
        server = MagicMock()
        server.work_items = wi_cfg
        cfg = MagicMock()
        cfg.server = server
        agent.config = cfg
    else:
        agent.config = None

    # Bind the real method to our stub.
    agent._discover_capabilities = BaseAgent._discover_capabilities.__get__(
        agent, type(agent)
    )
    return agent


# ---------------------------------------------------------------------------
# _discover_capabilities() unit tests
# ---------------------------------------------------------------------------


class TestDiscoverCapabilities:
    def test_mcp_capabilities(self):
        agent = _make_agent_stub(
            mcp_clients=[
                (MagicMock(), "search"),
                (MagicMock(), "memory"),
            ],
        )
        agent._discover_capabilities()

        names = [c.name for c in agent._discovered_capabilities]
        assert "mcp:search" in names
        assert "mcp:memory" in names
        assert all(c.value == 1.0 for c in agent._discovered_capabilities)

    def test_skill_capabilities(self):
        skills = {
            "code-review": Skill(name="code-review", description="Review code"),
            "testing": Skill(name="testing", description="Write tests"),
        }
        agent = _make_agent_stub(skills=skills)
        agent._discover_capabilities()

        names = [c.name for c in agent._discovered_capabilities]
        assert "skill:code-review" in names
        assert "skill:testing" in names

    def test_config_capabilities_merged(self):
        agent = _make_agent_stub(
            mcp_clients=[(MagicMock(), "search")],
            config_capabilities=[
                Capability(name="python", value=0.9),
                Capability(name="mcp:search", value=0.5),  # duplicate
            ],
        )
        agent._discover_capabilities()

        names = [c.name for c in agent._discovered_capabilities]
        assert names.count("mcp:search") == 1
        assert "python" in names

        # mcp:search should keep the auto-discovered value (1.0), not the
        # config's 0.5, because auto-discovered is added first.
        search_cap = next(
            c for c in agent._discovered_capabilities if c.name == "mcp:search"
        )
        assert search_cap.value == 1.0

    def test_empty_when_no_subsystems(self):
        agent = _make_agent_stub()
        agent._discover_capabilities()
        assert agent._discovered_capabilities == []

    def test_combined_mcp_and_skills(self):
        skills = {"analysis": Skill(name="analysis", description="Analyze code")}
        agent = _make_agent_stub(
            mcp_clients=[(MagicMock(), "tavily")],
            skills=skills,
        )
        agent._discover_capabilities()

        names = [c.name for c in agent._discovered_capabilities]
        assert names == ["mcp:tavily", "skill:analysis"]


# ---------------------------------------------------------------------------
# check_available_work passes discovered capabilities
# ---------------------------------------------------------------------------


class TestCheckAvailableWorkCapabilities:
    @pytest.mark.asyncio
    async def test_passes_discovered_caps(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = MagicMock()
        agent._work_item_store = AsyncMock()
        agent._work_item_store.list_available = AsyncMock(return_value=[])
        agent._work_item_actor_id = "actor-1"
        agent._work_item_events = []
        agent._discovered_capabilities = [
            Capability(name="mcp:search", value=1.0),
            Capability(name="skill:testing", value=1.0),
        ]
        agent.config.server.work_items.capabilities = []

        tools = make_work_item_tools(agent)
        check_tool = next(
            t
            for t in tools
            if getattr(t, "__base_agent_tool__").name == "check_available_work"
        )

        await check_tool(max_results=3)

        agent._work_item_store.list_available.assert_called_once_with(
            capabilities=agent._discovered_capabilities, max_results=3
        )

    @pytest.mark.asyncio
    async def test_passes_none_when_no_caps(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = MagicMock()
        agent._work_item_store = AsyncMock()
        agent._work_item_store.list_available = AsyncMock(return_value=[])
        agent._work_item_actor_id = "actor-1"
        agent._work_item_events = []
        agent._discovered_capabilities = []
        agent.config.server.work_items.capabilities = []

        tools = make_work_item_tools(agent)
        check_tool = next(
            t
            for t in tools
            if getattr(t, "__base_agent_tool__").name == "check_available_work"
        )

        await check_tool(max_results=5)

        # Empty list is falsy, so should pass None.
        agent._work_item_store.list_available.assert_called_once_with(
            capabilities=None, max_results=5
        )


# ---------------------------------------------------------------------------
# checkout / complete / release track _checked_out_work_item
# ---------------------------------------------------------------------------


class TestCheckedOutWorkItemTracking:
    def _make_tools_agent(self):
        """Create a mock agent for tool tests."""
        agent = MagicMock()
        agent._work_item_store = AsyncMock()
        agent._work_item_actor_id = "actor-1"
        agent._work_item_events = []
        agent._discovered_capabilities = []
        agent._checked_out_work_item = None
        agent.config.server.work_items.capabilities = []
        return agent

    @pytest.mark.asyncio
    async def test_checkout_sets_checked_out_item(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = self._make_tools_agent()
        item = WorkItem(
            id="wi_1",
            title="Test",
            status=WorkItemStatus.checked_out,
            lease_expires_at="2099-01-01T00:00:00Z",
        )
        agent._work_item_store.checkout = AsyncMock(return_value=item)

        tools = make_work_item_tools(agent)
        checkout_tool = next(
            t
            for t in tools
            if getattr(t, "__base_agent_tool__").name == "checkout_work_item"
        )

        await checkout_tool(item_id="wi_1", lease_duration_seconds=300)
        assert agent._checked_out_work_item is item

    @pytest.mark.asyncio
    async def test_complete_clears_checked_out_item(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = self._make_tools_agent()
        item = WorkItem(
            id="wi_1",
            title="Test",
            status=WorkItemStatus.completed,
        )
        agent._work_item_store.complete = AsyncMock(return_value=item)
        agent._checked_out_work_item = WorkItem(id="wi_1", title="Test")

        tools = make_work_item_tools(agent)
        complete_tool = next(
            t
            for t in tools
            if getattr(t, "__base_agent_tool__").name == "complete_work_item"
        )

        await complete_tool(
            item_id="wi_1",
            result_summary="Done",
            accomplished=["task 1"],
        )
        assert agent._checked_out_work_item is None

    @pytest.mark.asyncio
    async def test_release_clears_checked_out_item(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = self._make_tools_agent()
        item = WorkItem(
            id="wi_1",
            title="Test",
            status=WorkItemStatus.available,
        )
        agent._work_item_store.release = AsyncMock(return_value=item)
        agent._checked_out_work_item = WorkItem(id="wi_1", title="Test")

        tools = make_work_item_tools(agent)
        release_tool = next(
            t
            for t in tools
            if getattr(t, "__base_agent_tool__").name == "release_work_item"
        )

        await release_tool(
            item_id="wi_1",
            accomplished=["step 1"],
            remaining=["step 2"],
        )
        assert agent._checked_out_work_item is None
