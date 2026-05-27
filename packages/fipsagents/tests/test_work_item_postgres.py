"""Tests for PostgresWorkItemStore.

Integration tests require TEST_POSTGRES_URL in the environment and
the ``asyncpg`` package. Unit tests for the factory and import paths
run unconditionally.
"""
from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio

from fipsagents.server.work_items import (
    Capability,
    HandoffNote,
    WorkItem,
    WorkItemStatus,
    create_work_item_store,
)


# ---------------------------------------------------------------------------
# Import / factory unit tests (always run)
# ---------------------------------------------------------------------------


class TestPostgresImport:
    def test_import_class(self):
        from fipsagents.server.work_item_stores.postgres import PostgresWorkItemStore
        assert PostgresWorkItemStore is not None

    def test_import_from_package(self):
        from fipsagents.server.work_item_stores import PostgresWorkItemStore
        assert PostgresWorkItemStore is not None


class TestPostgresFactory:
    def test_factory_creates_postgres_store(self):
        from fipsagents.server.work_item_stores.postgres import PostgresWorkItemStore
        store = create_work_item_store(
            "postgres", database_url="postgresql://localhost/test",
        )
        assert isinstance(store, PostgresWorkItemStore)

    def test_factory_raises_without_url(self):
        with pytest.raises(ValueError, match="database_url"):
            create_work_item_store("postgres")

    def test_factory_raises_with_empty_url(self):
        with pytest.raises(ValueError, match="database_url"):
            create_work_item_store("postgres", database_url="")


# ---------------------------------------------------------------------------
# Integration tests (require live Postgres)
# ---------------------------------------------------------------------------

asyncpg = pytest.importorskip("asyncpg")

_SKIP_REASON = "No TEST_POSTGRES_URL set"
_HAS_PG = bool(os.environ.get("TEST_POSTGRES_URL"))


@pytest_asyncio.fixture
async def pg_store():
    """Create a PostgresWorkItemStore, ensure a clean table, and tear down."""
    if not _HAS_PG:
        pytest.skip(_SKIP_REASON)

    from fipsagents.server.work_item_stores.postgres import PostgresWorkItemStore

    url = os.environ["TEST_POSTGRES_URL"]
    store = PostgresWorkItemStore(url)

    # Ensure table exists, then truncate for a clean slate.
    pool = await store._get_pool()
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE work_items")

    yield store

    # Clean up after test.
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS work_items")
    await store.close()


@pytest.mark.skipif(not _HAS_PG, reason=_SKIP_REASON)
class TestPostgresWorkItemStore:
    @pytest.mark.asyncio
    async def test_create_and_get(self, pg_store):
        item = WorkItem(id="wi_pg_1", title="PG item", description="desc")
        await pg_store.create(item)

        retrieved = await pg_store.get("wi_pg_1")
        assert retrieved is not None
        assert retrieved.id == "wi_pg_1"
        assert retrieved.title == "PG item"

    @pytest.mark.asyncio
    async def test_create_generates_id(self, pg_store):
        item = WorkItem(id="", title="Auto-ID")
        created = await pg_store.create(item)
        assert created.id.startswith("wi_")

        retrieved = await pg_store.get(created.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, pg_store):
        assert await pg_store.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_available_empty(self, pg_store):
        items = await pg_store.list_available()
        assert items == []

    @pytest.mark.asyncio
    async def test_list_available_filters_status(self, pg_store):
        await pg_store.create(WorkItem(id="wi_avail", title="A"))
        await pg_store.create(WorkItem(
            id="wi_done", title="B", status=WorkItemStatus.completed,
        ))

        items = await pg_store.list_available()
        assert len(items) == 1
        assert items[0].id == "wi_avail"

    @pytest.mark.asyncio
    async def test_list_available_orders_by_priority(self, pg_store):
        await pg_store.create(WorkItem(id="wi_lo", title="Lo", priority=1))
        await pg_store.create(WorkItem(id="wi_hi", title="Hi", priority=10))
        await pg_store.create(WorkItem(id="wi_mid", title="Mid", priority=5))

        items = await pg_store.list_available()
        assert [it.id for it in items] == ["wi_hi", "wi_mid", "wi_lo"]

    @pytest.mark.asyncio
    async def test_list_available_by_parent(self, pg_store):
        await pg_store.create(WorkItem(id="wi_c1", title="C1", parent_id="p1"))
        await pg_store.create(WorkItem(id="wi_c2", title="C2", parent_id="p2"))

        items = await pg_store.list_available(parent_id="p1")
        assert len(items) == 1
        assert items[0].id == "wi_c1"

    @pytest.mark.asyncio
    async def test_checkout_sets_status_and_lease(self, pg_store):
        await pg_store.create(WorkItem(id="wi_co", title="Checkout"))

        item = await pg_store.checkout("wi_co", "actor1", lease_duration_seconds=60)
        assert item.status == WorkItemStatus.checked_out
        assert item.assignee == "actor1"
        assert item.lease_expires_at is not None

    @pytest.mark.asyncio
    async def test_checkout_appends_attempt(self, pg_store):
        await pg_store.create(WorkItem(id="wi_att", title="Attempt"))
        item = await pg_store.checkout("wi_att", "actor1")

        assert len(item.attempt_history) == 1
        assert item.attempt_history[0].actor_id == "actor1"

    @pytest.mark.asyncio
    async def test_checkout_unavailable_raises(self, pg_store):
        await pg_store.create(WorkItem(id="wi_busy", title="Busy"))
        await pg_store.checkout("wi_busy", "actor1")

        with pytest.raises(ValueError, match="not available"):
            await pg_store.checkout("wi_busy", "actor2")

    @pytest.mark.asyncio
    async def test_checkout_not_found_raises(self, pg_store):
        with pytest.raises(ValueError, match="not found"):
            await pg_store.checkout("nonexistent", "actor1")

    @pytest.mark.asyncio
    async def test_renew_lease(self, pg_store):
        await pg_store.create(WorkItem(id="wi_ren", title="Renew"))
        item1 = await pg_store.checkout("wi_ren", "actor1", lease_duration_seconds=60)
        original = item1.lease_expires_at

        await asyncio.sleep(0.01)
        item2 = await pg_store.renew_lease("wi_ren", "actor1", lease_duration_seconds=120)
        assert item2.lease_expires_at != original

    @pytest.mark.asyncio
    async def test_renew_wrong_actor_raises(self, pg_store):
        await pg_store.create(WorkItem(id="wi_wa", title="WA"))
        await pg_store.checkout("wi_wa", "actor1")

        with pytest.raises(ValueError, match="checked out by"):
            await pg_store.renew_lease("wi_wa", "actor2")

    @pytest.mark.asyncio
    async def test_complete(self, pg_store):
        await pg_store.create(WorkItem(id="wi_cmp", title="Complete"))
        await pg_store.checkout("wi_cmp", "actor1")

        item = await pg_store.complete("wi_cmp")
        assert item.status == WorkItemStatus.completed
        assert item.assignee is None
        assert item.lease_expires_at is None

    @pytest.mark.asyncio
    async def test_complete_with_review(self, pg_store):
        await pg_store.create(WorkItem(id="wi_rev", title="Review"))
        await pg_store.checkout("wi_rev", "actor1")

        item = await pg_store.complete("wi_rev", review_required=True)
        assert item.status == WorkItemStatus.review_pending

    @pytest.mark.asyncio
    async def test_release_returns_to_available(self, pg_store):
        await pg_store.create(WorkItem(id="wi_rel", title="Release"))
        await pg_store.checkout("wi_rel", "actor1")

        handoff = HandoffNote(accomplished=["A"], remaining=["B"])
        item = await pg_store.release("wi_rel", handoff_note=handoff)
        assert item.status == WorkItemStatus.available
        assert item.handoff_note.accomplished == ["A"]

    @pytest.mark.asyncio
    async def test_fail_marks_failed(self, pg_store):
        await pg_store.create(WorkItem(id="wi_fl", title="Fail"))
        await pg_store.checkout("wi_fl", "actor1")

        item = await pg_store.fail("wi_fl", error="broke")
        assert item.status == WorkItemStatus.failed

    @pytest.mark.asyncio
    async def test_fail_with_retry(self, pg_store):
        await pg_store.create(WorkItem(id="wi_rty", title="Retry"))
        await pg_store.checkout("wi_rty", "actor1")

        item = await pg_store.fail("wi_rty", error="temp", retry=True)
        assert item.status == WorkItemStatus.available

    @pytest.mark.asyncio
    async def test_accept(self, pg_store):
        await pg_store.create(WorkItem(id="wi_acc", title="Accept"))
        await pg_store.checkout("wi_acc", "actor1")
        await pg_store.complete("wi_acc", review_required=True)

        item = await pg_store.accept("wi_acc")
        assert item.status == WorkItemStatus.completed

    @pytest.mark.asyncio
    async def test_reject(self, pg_store):
        await pg_store.create(WorkItem(id="wi_rej", title="Reject"))
        await pg_store.checkout("wi_rej", "actor1")
        await pg_store.complete("wi_rej", review_required=True)

        item = await pg_store.reject("wi_rej", reason="nope")
        assert item.status == WorkItemStatus.available
        assert "Rejected" in item.handoff_note.context
        assert "nope" in item.handoff_note.context

    @pytest.mark.asyncio
    async def test_expire_leases(self, pg_store):
        await pg_store.create(WorkItem(id="wi_exp", title="Expire"))
        await pg_store.checkout("wi_exp", "actor1", lease_duration_seconds=1)

        await asyncio.sleep(1.1)
        expired = await pg_store.expire_leases()
        assert len(expired) == 1
        assert expired[0].id == "wi_exp"
        assert expired[0].status == WorkItemStatus.available

    @pytest.mark.asyncio
    async def test_capability_matching(self, pg_store):
        await pg_store.create(WorkItem(
            id="wi_py", title="Python",
            required_capabilities=[Capability(name="python", value=0.8)],
        ))
        await pg_store.create(WorkItem(
            id="wi_rs", title="Rust",
            required_capabilities=[Capability(name="rust", value=0.7)],
        ))

        py_caps = [Capability(name="python", value=0.9)]
        items = await pg_store.list_available(capabilities=py_caps)
        ids = {it.id for it in items}
        assert "wi_py" in ids
        assert "wi_rs" not in ids

    @pytest.mark.asyncio
    async def test_update_progress(self, pg_store):
        await pg_store.create(WorkItem(id="wi_prg", title="Progress"))
        await pg_store.checkout("wi_prg", "actor1")

        item = await pg_store.update_progress(
            "wi_prg", progress={"step": 2, "total": 5},
        )
        assert item.lease_expires_at is not None

    @pytest.mark.asyncio
    async def test_handoff_roundtrip(self, pg_store):
        """Verify handoff note survives create -> checkout -> release -> get."""
        await pg_store.create(WorkItem(id="wi_hrt", title="Handoff RT"))
        await pg_store.checkout("wi_hrt", "actor1")
        handoff = HandoffNote(
            accomplished=["did X"],
            remaining=["do Y"],
            context="important",
            artifacts={"log": "/tmp/log.txt"},
        )
        await pg_store.release("wi_hrt", handoff_note=handoff)

        retrieved = await pg_store.get("wi_hrt")
        assert retrieved.handoff_note is not None
        assert retrieved.handoff_note.accomplished == ["did X"]
        assert retrieved.handoff_note.artifacts == {"log": "/tmp/log.txt"}

    @pytest.mark.asyncio
    async def test_close_idempotent(self, pg_store):
        await pg_store.close()
        await pg_store.close()  # should not raise
