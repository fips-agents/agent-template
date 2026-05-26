"""Tests for the work-item coordination layer."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fipsagents.baseagent.events import (
    TrustLevelChanged,
    WorkItemCheckedOut,
    WorkItemReleased,
)
from fipsagents.server.work_items import (
    Capability,
    HandoffNote,
    NullWorkItemStore,
    WorkItem,
    WorkItemStatus,
    create_work_item_store,
)


# ---------------------------------------------------------------------------
# NullWorkItemStore
# ---------------------------------------------------------------------------


class TestNullWorkItemStore:
    @pytest.mark.asyncio
    async def test_list_available_returns_empty(self):
        store = NullWorkItemStore()
        items = await store.list_available()
        assert items == []

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        store = NullWorkItemStore()
        assert await store.get("any") is None

    @pytest.mark.asyncio
    async def test_checkout_raises(self):
        store = NullWorkItemStore()
        with pytest.raises(NotImplementedError):
            await store.checkout("wi_123", "actor1")

    @pytest.mark.asyncio
    async def test_complete_raises(self):
        store = NullWorkItemStore()
        with pytest.raises(NotImplementedError):
            await store.complete("wi_123")

    @pytest.mark.asyncio
    async def test_release_raises(self):
        store = NullWorkItemStore()
        with pytest.raises(NotImplementedError):
            await store.release("wi_123")

    @pytest.mark.asyncio
    async def test_fail_raises(self):
        store = NullWorkItemStore()
        with pytest.raises(NotImplementedError):
            await store.fail("wi_123", error="test error")

    @pytest.mark.asyncio
    async def test_expire_leases_returns_empty(self):
        store = NullWorkItemStore()
        expired = await store.expire_leases()
        assert expired == []

    @pytest.mark.asyncio
    async def test_create_returns_item(self):
        store = NullWorkItemStore()
        item = WorkItem(id="wi_123", title="Test")
        result = await store.create(item)
        assert result == item


# ---------------------------------------------------------------------------
# Create WorkItemStore Factory
# ---------------------------------------------------------------------------


class TestCreateWorkItemStore:
    def test_none_returns_null(self):
        store = create_work_item_store(None)
        assert isinstance(store, NullWorkItemStore)

    def test_empty_string_returns_null(self):
        store = create_work_item_store("")
        assert isinstance(store, NullWorkItemStore)

    def test_sqlite_returns_sqlite(self, tmp_path):
        from fipsagents.server.work_item_stores.sqlite import SqliteWorkItemStore
        store = create_work_item_store("sqlite", sqlite_path=str(tmp_path / "test.db"))
        assert isinstance(store, SqliteWorkItemStore)

    def test_postgres_requires_url(self):
        with pytest.raises(ValueError, match="database_url"):
            create_work_item_store("postgres")

    def test_postgres_returns_postgres_store(self):
        from fipsagents.server.work_item_stores.postgres import PostgresWorkItemStore
        store = create_work_item_store(
            "postgres", database_url="postgresql://localhost/test",
        )
        assert isinstance(store, PostgresWorkItemStore)


# ---------------------------------------------------------------------------
# SqliteWorkItemStore
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store(tmp_path):
    from fipsagents.server.work_item_stores.sqlite import SqliteWorkItemStore
    s = SqliteWorkItemStore(str(tmp_path / "test.db"))
    yield s
    await s.close()


class TestSqliteWorkItemStore:
    @pytest.mark.asyncio
    async def test_create_and_get(self, store):
        item = WorkItem(
            id="wi_test_1",
            title="Test item",
            description="A test work item",
        )
        await store.create(item)

        retrieved = await store.get("wi_test_1")
        assert retrieved is not None
        assert retrieved.id == "wi_test_1"
        assert retrieved.title == "Test item"
        assert retrieved.description == "A test work item"

    @pytest.mark.asyncio
    async def test_create_generates_id(self, store):
        item = WorkItem(id="", title="Auto-ID item")
        created = await store.create(item)

        assert created.id.startswith("wi_")

        retrieved = await store.get(created.id)
        assert retrieved is not None
        assert retrieved.title == "Auto-ID item"

    @pytest.mark.asyncio
    async def test_list_available_empty(self, store):
        items = await store.list_available()
        assert items == []

    @pytest.mark.asyncio
    async def test_list_available_filters_status(self, store):
        await store.create(WorkItem(id="wi_available", title="Available", status=WorkItemStatus.available))
        await store.create(WorkItem(id="wi_completed", title="Done", status=WorkItemStatus.completed))
        await store.create(WorkItem(id="wi_failed", title="Failed", status=WorkItemStatus.failed))

        items = await store.list_available()
        assert len(items) == 1
        assert items[0].id == "wi_available"

    @pytest.mark.asyncio
    async def test_list_available_orders_by_priority(self, store):
        await store.create(WorkItem(id="wi_low", title="Low", priority=1))
        await store.create(WorkItem(id="wi_high", title="High", priority=10))
        await store.create(WorkItem(id="wi_medium", title="Medium", priority=5))

        items = await store.list_available()
        assert len(items) == 3
        assert items[0].id == "wi_high"  # Priority 10
        assert items[1].id == "wi_medium"  # Priority 5
        assert items[2].id == "wi_low"  # Priority 1

    @pytest.mark.asyncio
    async def test_checkout_sets_status_and_lease(self, store):
        await store.create(WorkItem(id="wi_checkout_test", title="Checkout test"))

        item = await store.checkout("wi_checkout_test", "actor1", lease_duration_seconds=60)

        assert item.status == WorkItemStatus.checked_out
        assert item.assignee == "actor1"
        assert item.lease_expires_at is not None

        # Verify lease is in the future
        expires_at = datetime.fromisoformat(item.lease_expires_at)
        now = datetime.now(timezone.utc)
        assert expires_at > now

    @pytest.mark.asyncio
    async def test_checkout_unavailable_raises(self, store):
        await store.create(WorkItem(id="wi_already_out", title="Already out"))
        await store.checkout("wi_already_out", "actor1")

        with pytest.raises(ValueError, match="not available"):
            await store.checkout("wi_already_out", "actor2")

    @pytest.mark.asyncio
    async def test_checkout_appends_attempt(self, store):
        await store.create(WorkItem(id="wi_attempt_test", title="Attempt test"))

        item = await store.checkout("wi_attempt_test", "actor1")
        assert len(item.attempt_history) == 1
        assert item.attempt_history[0].actor_id == "actor1"
        assert item.attempt_history[0].started_at is not None
        assert item.attempt_history[0].ended_at is None

    @pytest.mark.asyncio
    async def test_renew_lease(self, store):
        await store.create(WorkItem(id="wi_renew", title="Renew test"))
        item1 = await store.checkout("wi_renew", "actor1", lease_duration_seconds=60)
        original_expires = item1.lease_expires_at

        # Wait a moment to ensure timestamp changes
        await asyncio.sleep(0.01)

        item2 = await store.renew_lease("wi_renew", "actor1", lease_duration_seconds=120)
        assert item2.lease_expires_at != original_expires

        # New expiry should be later
        expires1 = datetime.fromisoformat(original_expires)
        expires2 = datetime.fromisoformat(item2.lease_expires_at)
        assert expires2 > expires1

    @pytest.mark.asyncio
    async def test_renew_lease_wrong_actor_raises(self, store):
        await store.create(WorkItem(id="wi_wrong_actor", title="Wrong actor"))
        await store.checkout("wi_wrong_actor", "actor1")

        with pytest.raises(ValueError, match="checked out by"):
            await store.renew_lease("wi_wrong_actor", "actor2")

    @pytest.mark.asyncio
    async def test_complete_sets_status(self, store):
        await store.create(WorkItem(id="wi_complete", title="Complete test"))
        await store.checkout("wi_complete", "actor1")

        item = await store.complete("wi_complete")
        assert item.status == WorkItemStatus.completed
        assert item.assignee is None
        assert item.lease_expires_at is None

    @pytest.mark.asyncio
    async def test_complete_with_review(self, store):
        await store.create(WorkItem(id="wi_review", title="Review test"))
        await store.checkout("wi_review", "actor1")

        item = await store.complete("wi_review", review_required=True)
        assert item.status == WorkItemStatus.review_pending
        assert item.assignee is None

    @pytest.mark.asyncio
    async def test_release_returns_to_available(self, store):
        await store.create(WorkItem(id="wi_release", title="Release test"))
        await store.checkout("wi_release", "actor1")

        handoff = HandoffNote(
            accomplished=["Step 1", "Step 2"],
            remaining=["Step 3", "Step 4"],
        )
        item = await store.release("wi_release", handoff_note=handoff)

        assert item.status == WorkItemStatus.available
        assert item.assignee is None
        assert item.lease_expires_at is None

    @pytest.mark.asyncio
    async def test_release_preserves_handoff(self, store):
        await store.create(WorkItem(id="wi_handoff", title="Handoff test"))
        await store.checkout("wi_handoff", "actor1")

        handoff = HandoffNote(
            accomplished=["Done this"],
            remaining=["Do that"],
            context="Important context",
        )
        item = await store.release("wi_handoff", handoff_note=handoff)

        assert item.handoff_note is not None
        assert item.handoff_note.accomplished == ["Done this"]
        assert item.handoff_note.remaining == ["Do that"]
        assert item.handoff_note.context == "Important context"

        # Verify it persists
        retrieved = await store.get("wi_handoff")
        assert retrieved.handoff_note is not None
        assert retrieved.handoff_note.context == "Important context"

    @pytest.mark.asyncio
    async def test_fail_marks_failed(self, store):
        await store.create(WorkItem(id="wi_fail", title="Fail test"))
        await store.checkout("wi_fail", "actor1")

        item = await store.fail("wi_fail", error="Something went wrong", retry=False)
        assert item.status == WorkItemStatus.failed

    @pytest.mark.asyncio
    async def test_fail_with_retry_resets(self, store):
        await store.create(WorkItem(id="wi_retry", title="Retry test"))
        await store.checkout("wi_retry", "actor1")

        item = await store.fail("wi_retry", error="Temporary failure", retry=True)
        assert item.status == WorkItemStatus.available
        assert item.assignee is None

    @pytest.mark.asyncio
    async def test_accept_review(self, store):
        await store.create(WorkItem(id="wi_accept", title="Accept test"))
        await store.checkout("wi_accept", "actor1")
        await store.complete("wi_accept", review_required=True)

        item = await store.accept("wi_accept")
        assert item.status == WorkItemStatus.completed

    @pytest.mark.asyncio
    async def test_reject_review(self, store):
        await store.create(WorkItem(id="wi_reject", title="Reject test"))
        await store.checkout("wi_reject", "actor1")
        await store.complete("wi_reject", review_required=True)

        item = await store.reject("wi_reject", reason="Not good enough")
        assert item.status == WorkItemStatus.available
        assert item.handoff_note is not None
        assert "Rejected" in item.handoff_note.context
        assert "Not good enough" in item.handoff_note.context

    @pytest.mark.asyncio
    async def test_expire_leases(self, store):
        await store.create(WorkItem(id="wi_expire", title="Expire test"))

        # Checkout with very short lease
        await store.checkout("wi_expire", "actor1", lease_duration_seconds=1)

        # Wait for lease to expire
        await asyncio.sleep(1.1)

        expired = await store.expire_leases()
        assert len(expired) == 1
        assert expired[0].id == "wi_expire"
        assert expired[0].status == WorkItemStatus.available

    @pytest.mark.asyncio
    async def test_capability_matching(self, store):
        # Create items with different capability requirements
        await store.create(WorkItem(
            id="wi_python",
            title="Python task",
            required_capabilities=[Capability(name="python", value=0.8)],
        ))
        await store.create(WorkItem(
            id="wi_rust",
            title="Rust task",
            required_capabilities=[Capability(name="rust", value=0.7)],
        ))
        await store.create(WorkItem(
            id="wi_both",
            title="Multi-language task",
            required_capabilities=[
                Capability(name="python", value=0.5),
                Capability(name="rust", value=0.5),
            ],
        ))

        # Agent with only Python capability
        python_caps = [Capability(name="python", value=0.9)]
        items = await store.list_available(capabilities=python_caps)
        ids = {item.id for item in items}
        assert "wi_python" in ids
        assert "wi_rust" not in ids
        assert "wi_both" not in ids

        # Agent with both capabilities
        both_caps = [
            Capability(name="python", value=0.8),
            Capability(name="rust", value=0.8),
        ]
        items = await store.list_available(capabilities=both_caps)
        ids = {item.id for item in items}
        assert "wi_python" in ids
        assert "wi_rust" in ids
        assert "wi_both" in ids


# ---------------------------------------------------------------------------
# Capability Matching Logic
# ---------------------------------------------------------------------------


class TestCapabilityMatching:
    def _matches(self, required, offered):
        """Helper using SqliteWorkItemStore's matching logic."""
        from fipsagents.server.work_item_stores.sqlite import SqliteWorkItemStore
        store = SqliteWorkItemStore(":memory:")
        return store._matches_capabilities(required, offered)

    def test_empty_required_matches_everything(self):
        assert self._matches([], [])
        assert self._matches([], [Capability(name="python", value=1.0)])

    def test_basic_name_match(self):
        required = [Capability(name="python", value=0.5)]
        offered = [Capability(name="python", value=0.5)]
        assert self._matches(required, offered)

    def test_value_threshold(self):
        required = [Capability(name="python", value=0.7)]
        offered = [Capability(name="python", value=0.8)]
        assert self._matches(required, offered)

    def test_value_below_threshold_fails(self):
        required = [Capability(name="python", value=0.9)]
        offered = [Capability(name="python", value=0.5)]
        assert not self._matches(required, offered)

    def test_conjunction(self):
        required = [
            Capability(name="python", value=0.7),
            Capability(name="rust", value=0.6),
        ]

        # Both satisfied
        offered = [
            Capability(name="python", value=0.8),
            Capability(name="rust", value=0.7),
        ]
        assert self._matches(required, offered)

        # Only one satisfied
        offered_partial = [Capability(name="python", value=0.8)]
        assert not self._matches(required, offered_partial)


# ---------------------------------------------------------------------------
# Work Item Stock Tools
# ---------------------------------------------------------------------------


def _make_mock_agent(*, work_items_enabled: bool = True):
    """Create a mock agent for testing stock tool registration."""
    agent = MagicMock()

    # Config structure
    agent.config.server.work_items.enabled = work_items_enabled
    agent.config.server.work_items.capabilities = []

    # Work item infrastructure
    agent._work_item_store = AsyncMock()
    agent._work_item_actor_id = "test-actor"
    agent._work_item_events = []

    return agent


class TestWorkItemStockTools:
    def test_condition_false_when_disabled(self):
        from fipsagents.baseagent.tools.work_items import STOCK_TOOL_SPEC

        agent = _make_mock_agent(work_items_enabled=False)
        assert not STOCK_TOOL_SPEC.condition(agent)

    def test_condition_true_when_enabled(self):
        from fipsagents.baseagent.tools.work_items import STOCK_TOOL_SPEC

        agent = _make_mock_agent(work_items_enabled=True)
        assert STOCK_TOOL_SPEC.condition(agent)

    def test_factory_returns_list(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = _make_mock_agent()
        tools = make_work_item_tools(agent)

        assert isinstance(tools, list)
        assert len(tools) == 5

    def test_tool_names(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = _make_mock_agent()
        tools = make_work_item_tools(agent)

        names = {getattr(tool, "__base_agent_tool__").name for tool in tools}
        assert names == {
            "check_available_work",
            "checkout_work_item",
            "complete_work_item",
            "release_work_item",
            "update_work_progress",
        }

    @pytest.mark.asyncio
    async def test_check_available_work_tool(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = _make_mock_agent()
        agent._work_item_store.list_available = AsyncMock(return_value=[
            WorkItem(
                id="wi_1",
                title="Test item",
                description="Test description",
                priority=5,
            )
        ])

        tools = make_work_item_tools(agent)
        check_tool = next(t for t in tools if getattr(t, "__base_agent_tool__").name == "check_available_work")

        result = await check_tool(max_results=5)
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["id"] == "wi_1"
        assert data[0]["title"] == "Test item"

    @pytest.mark.asyncio
    async def test_checkout_work_item_emits_event(self):
        from fipsagents.baseagent.tools.work_items import make_work_item_tools

        agent = _make_mock_agent()
        agent._work_item_store.checkout = AsyncMock(return_value=WorkItem(
            id="wi_1",
            title="Test",
            description="Desc",
            acceptance_criteria=["AC1", "AC2"],
            lease_expires_at="2026-05-21T12:00:00Z",
        ))

        tools = make_work_item_tools(agent)
        checkout_tool = next(t for t in tools if getattr(t, "__base_agent_tool__").name == "checkout_work_item")

        await checkout_tool(item_id="wi_1", lease_duration_seconds=300)

        assert len(agent._work_item_events) == 1
        event = agent._work_item_events[0]
        assert isinstance(event, WorkItemCheckedOut)
        assert event.item_id == "wi_1"
        assert event.actor_id == "test-actor"

    @pytest.mark.asyncio
    async def test_complete_work_item_drains_trust_events(self):
        """Promotion events from TrustManager surface in _self_healing_events."""
        from fipsagents.baseagent.tools.work_items import make_work_item_tools
        from fipsagents.baseagent.trust import TrustManager

        agent = _make_mock_agent()
        agent._self_healing_events = []

        # Use a real TrustManager with a low promotion threshold so a single
        # completion triggers a level 0 -> 1 transition.
        agent._trust_manager = TrustManager(thresholds=(1.0, 50.0, 200.0, 500.0))

        agent._work_item_store.complete = AsyncMock(return_value=WorkItem(
            id="wi_promo", title="Promo task", status=WorkItemStatus.completed,
        ))

        tools = make_work_item_tools(agent)
        complete_tool = next(
            t for t in tools
            if getattr(t, "__base_agent_tool__").name == "complete_work_item"
        )

        await complete_tool(
            item_id="wi_promo",
            result_summary="done",
            accomplished=["everything"],
        )

        # Trust event should have been drained into _self_healing_events.
        assert len(agent._self_healing_events) == 1
        evt = agent._self_healing_events[0]
        assert isinstance(evt, TrustLevelChanged)
        assert evt.from_level == 0
        assert evt.to_level == 1

    @pytest.mark.asyncio
    async def test_release_work_item_records_trust_failure(self):
        """Releasing a work item records a trust failure and drains events."""
        from fipsagents.baseagent.tools.work_items import make_work_item_tools
        from fipsagents.baseagent.trust import TrustManager

        agent = _make_mock_agent()
        agent._self_healing_events = []

        # Start at level 1 with score just above demotion threshold (5.0)
        # so a single failure (-5.0) triggers demotion.
        agent._trust_manager = TrustManager(
            thresholds=(10.0, 50.0, 200.0, 500.0),
        )
        # Manually set level and score for the test scenario.
        agent._trust_manager._state.level = 1
        agent._trust_manager._state.score = 5.0

        agent._work_item_store.release = AsyncMock(return_value=WorkItem(
            id="wi_rel", title="Released task", status=WorkItemStatus.available,
        ))

        tools = make_work_item_tools(agent)
        release_tool = next(
            t for t in tools
            if getattr(t, "__base_agent_tool__").name == "release_work_item"
        )

        await release_tool(
            item_id="wi_rel",
            accomplished=["step 1"],
            remaining=["step 2", "step 3"],
        )

        # Trust failure should have been recorded.
        assert agent._trust_manager._state.failures == 1

        # Demotion event should be in _self_healing_events.
        trust_events = [
            e for e in agent._self_healing_events
            if isinstance(e, TrustLevelChanged)
        ]
        assert len(trust_events) == 1
        assert trust_events[0].from_level == 1
        assert trust_events[0].to_level == 0


# ---------------------------------------------------------------------------
# Stock Tool Discovery (List Return)
# ---------------------------------------------------------------------------


class TestDiscoverStockListSupport:
    def test_single_return_still_works(self):
        from fipsagents.baseagent.tools._registry import ToolRegistry
        from fipsagents.baseagent.tools import tool

        registry = ToolRegistry()

        # Mock agent with a single-function stock tool spec
        agent = MagicMock()
        agent.config.server.work_items.enabled = False  # Disable real work items

        # Create a mock module with single-function factory
        @tool(description="Test tool", visibility="llm_only")
        async def single_test_tool():
            return "result"

        mock_spec = MagicMock()
        mock_spec.condition = lambda a: True
        mock_spec.factory = lambda a: single_test_tool

        # Manually register via the pattern
        result = mock_spec.factory(agent)
        tools_to_register = result if isinstance(result, list) else [result]

        assert len(tools_to_register) == 1
        registry.register(tools_to_register[0])
        assert registry.get("single_test_tool") is not None

    def test_list_return_registers_all(self):
        from fipsagents.baseagent.tools._registry import ToolRegistry
        from fipsagents.baseagent.tools import tool

        registry = ToolRegistry()

        # Mock agent
        agent = MagicMock()

        # Create a mock module with list-returning factory
        @tool(description="Tool 1", visibility="llm_only", name="tool_one")
        async def tool1():
            return "one"

        @tool(description="Tool 2", visibility="llm_only", name="tool_two")
        async def tool2():
            return "two"

        mock_spec = MagicMock()
        mock_spec.condition = lambda a: True
        mock_spec.factory = lambda a: [tool1, tool2]

        # Manually register via the pattern
        result = mock_spec.factory(agent)
        tools_to_register = result if isinstance(result, list) else [result]

        assert len(tools_to_register) == 2
        for tool_fn in tools_to_register:
            registry.register(tool_fn)

        assert registry.get("tool_one") is not None
        assert registry.get("tool_two") is not None
