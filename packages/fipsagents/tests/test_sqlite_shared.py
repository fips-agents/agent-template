"""Tests for shared SQLite connection manager."""

import pytest
import pytest_asyncio

from fipsagents.server.sqlite import SqliteConnectionManager
from fipsagents.server.sessions import SqliteSessionStore
from fipsagents.server.tracing import SqliteTraceStore, Span, Trace


class TestSqliteConnectionManager:
    @pytest_asyncio.fixture
    async def manager(self):
        mgr = SqliteConnectionManager()
        yield mgr
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_acquire_returns_connection(self, manager, tmp_path):
        conn = await manager.acquire(str(tmp_path / "test.db"))
        assert conn is not None

    @pytest.mark.asyncio
    async def test_acquire_deduplicates_by_path(self, manager, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn1 = await manager.acquire(db_path)
        conn2 = await manager.acquire(db_path)
        assert conn1 is conn2

    @pytest.mark.asyncio
    async def test_acquire_deduplicates_relative_and_absolute(self, manager, tmp_path):
        """abspath resolution maps relative and absolute to the same entry."""
        import os

        abs_path = str(tmp_path / "test.db")
        # Build a relative path that resolves to the same file.
        rel_path = os.path.relpath(abs_path)
        conn1 = await manager.acquire(abs_path)
        conn2 = await manager.acquire(rel_path)
        assert conn1 is conn2

    @pytest.mark.asyncio
    async def test_close_all_closes_connections(self, manager, tmp_path):
        conn = await manager.acquire(str(tmp_path / "test.db"))
        await manager.close_all()
        # After close_all, acquiring again should give a new connection
        conn2 = await manager.acquire(str(tmp_path / "test.db"))
        assert conn2 is not conn


class TestSharedConnection:
    @pytest.mark.asyncio
    async def test_two_stores_share_connection(self, tmp_path):
        mgr = SqliteConnectionManager()
        conn = await mgr.acquire(str(tmp_path / "shared.db"))

        session_store = SqliteSessionStore(str(tmp_path / "shared.db"), connection=conn)
        trace_store = SqliteTraceStore(str(tmp_path / "shared.db"), connection=conn)

        # Session store works
        sid = await session_store.create("test-session")
        assert sid == "test-session"
        await session_store.save("test-session", [{"role": "user", "content": "hi"}])
        msgs = await session_store.load("test-session")
        assert len(msgs) == 1

        # Trace store works using the same connection
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        trace = Trace(
            trace_id="tr-1",
            started_at=now,
            ended_at=now,
            model="test",
            spans=[Span(trace_id="tr-1", span_id="sp-1", name="request")],
        )
        await trace_store.save_trace(trace)
        loaded = await trace_store.get_trace("tr-1")
        assert loaded is not None
        assert loaded.trace_id == "tr-1"

        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_managed_store_close_is_noop(self, tmp_path):
        mgr = SqliteConnectionManager()
        conn = await mgr.acquire(str(tmp_path / "shared.db"))

        store = SqliteSessionStore(str(tmp_path / "shared.db"), connection=conn)
        await store.create("s1")
        await store.close()  # Should NOT close the connection

        # Connection should still work via the manager
        store2 = SqliteSessionStore(str(tmp_path / "shared.db"), connection=conn)
        sid = await store2.create("s2")
        assert sid == "s2"

        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_unmanaged_store_still_works(self, tmp_path):
        """Backward compat: stores without managed connection work as before."""
        store = SqliteSessionStore(str(tmp_path / "compat.db"))
        sid = await store.create()
        assert sid.startswith("sess_")
        await store.close()
