"""Tests for session schema migration and state management."""
import pytest
from fipsagents.server.sessions import SqliteSessionStore


@pytest.fixture
async def sqlite_store():
    store = SqliteSessionStore(":memory:")
    yield store
    await store.close()


class TestSqliteMigration:
    @pytest.mark.asyncio
    async def test_new_columns_exist_after_init(self, sqlite_store):
        """New columns are created during table init."""
        sid = await sqlite_store.create("test_session")
        state = await sqlite_store.get_state(sid)
        # All state fields should exist with null/empty defaults
        assert state.get("pending_question") is None
        assert state.get("permission_scope_active") is None
        assert state.get("parent_session_id") is None
        assert state.get("forked_at_message_id") is None


class TestUpdateState:
    @pytest.mark.asyncio
    async def test_update_and_get_pending_question(self, sqlite_store):
        sid = await sqlite_store.create("s1")
        await sqlite_store.update_state(sid, pending_question="q_123")
        state = await sqlite_store.get_state(sid)
        assert state["pending_question"] == "q_123"

    @pytest.mark.asyncio
    async def test_clear_pending_question(self, sqlite_store):
        sid = await sqlite_store.create("s1")
        await sqlite_store.update_state(sid, pending_question="q_123")
        await sqlite_store.update_state(sid, pending_question=None)
        state = await sqlite_store.get_state(sid)
        assert state["pending_question"] is None

    @pytest.mark.asyncio
    async def test_update_compaction_state(self, sqlite_store):
        sid = await sqlite_store.create("s1")
        await sqlite_store.update_state(
            sid,
            compaction_state={"last_compacted_at": "2026-05-18T12:00:00Z", "compaction_count": 1},
        )
        state = await sqlite_store.get_state(sid)
        assert state["compaction_state"]["compaction_count"] == 1

    @pytest.mark.asyncio
    async def test_unknown_fields_ignored(self, sqlite_store):
        sid = await sqlite_store.create("s1")
        result = await sqlite_store.update_state(sid, nonexistent_field="value")
        assert result is False

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_empty(self, sqlite_store):
        state = await sqlite_store.get_state("nonexistent")
        assert state == {}


class TestForkLineage:
    @pytest.mark.asyncio
    async def test_create_with_fork_lineage(self, sqlite_store):
        parent = await sqlite_store.create("parent")
        child = await sqlite_store.create(
            "child",
            parent_session_id=parent,
            forked_at_message_id="msg_abc123",
        )
        state = await sqlite_store.get_state(child)
        assert state["parent_session_id"] == parent
        assert state["forked_at_message_id"] == "msg_abc123"

    @pytest.mark.asyncio
    async def test_create_with_permission_scope(self, sqlite_store):
        sid = await sqlite_store.create(
            "s1",
            permission_scope_active="static",
        )
        state = await sqlite_store.get_state(sid)
        assert state["permission_scope_active"] == "static"
