"""Tests for session persistence backends."""


import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone

from fipsagents.server.sessions import (
    NullSessionStore,
    SqliteSessionStore,
    create_session_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_store(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "test.db"))
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# NullSessionStore
# ---------------------------------------------------------------------------


class TestNullSessionStore:
    @pytest.mark.asyncio
    async def test_create_generates_id(self):
        store = NullSessionStore()
        sid = await store.create()
        assert sid.startswith("sess_")

    @pytest.mark.asyncio
    async def test_create_uses_provided_id(self):
        store = NullSessionStore()
        sid = await store.create("my-session")
        assert sid == "my-session"

    @pytest.mark.asyncio
    async def test_load_returns_none(self):
        store = NullSessionStore()
        assert await store.load("anything") is None

    @pytest.mark.asyncio
    async def test_save_is_noop(self):
        store = NullSessionStore()
        # Should not raise
        await store.save("s1", [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_delete_returns_false(self):
        store = NullSessionStore()
        assert await store.delete("nonexistent") is False

    @pytest.mark.asyncio
    async def test_exists_returns_false(self):
        store = NullSessionStore()
        assert await store.exists("anything") is False

    @pytest.mark.asyncio
    async def test_update_returns_false(self):
        store = NullSessionStore()
        assert await store.update("anything", cost_data={"x": 1}) is False
        assert await store.update("anything") is False


# ---------------------------------------------------------------------------
# SqliteSessionStore
# ---------------------------------------------------------------------------


class TestSqliteSessionStore:
    @pytest.mark.asyncio
    async def test_create_and_load(self, sqlite_store):
        sid = await sqlite_store.create()
        messages = await sqlite_store.load(sid)
        assert messages == []

    @pytest.mark.asyncio
    async def test_save_and_load(self, sqlite_store):
        sid = await sqlite_store.create()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        await sqlite_store.save(sid, msgs)
        loaded = await sqlite_store.load(sid)
        assert loaded == msgs

    @pytest.mark.asyncio
    async def test_save_preserves_created_at(self, sqlite_store):
        sid = await sqlite_store.create()
        db = await sqlite_store._get_db()

        cursor = await db.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?", (sid,)
        )
        row = await cursor.fetchone()
        original_created = row[0]

        await sqlite_store.save(sid, [{"role": "user", "content": "update"}])

        cursor = await db.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?", (sid,)
        )
        row = await cursor.fetchone()
        assert row[0] == original_created, (
            f"created_at changed from {original_created!r} to {row[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store):
        sid = await sqlite_store.create()
        assert await sqlite_store.delete(sid) is True
        assert await sqlite_store.load(sid) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, sqlite_store):
        assert await sqlite_store.delete("no-such-session") is False

    @pytest.mark.asyncio
    async def test_exists(self, sqlite_store):
        sid = await sqlite_store.create()
        assert await sqlite_store.exists(sid) is True
        await sqlite_store.delete(sid)
        assert await sqlite_store.exists(sid) is False

    @pytest.mark.asyncio
    async def test_session_continuity(self, sqlite_store):
        """Save messages, load, append new ones, save again, verify full history."""
        sid = await sqlite_store.create()
        batch1 = [{"role": "user", "content": "first"}]
        await sqlite_store.save(sid, batch1)

        loaded = await sqlite_store.load(sid)
        loaded.append({"role": "assistant", "content": "second"})
        await sqlite_store.save(sid, loaded)

        final = await sqlite_store.load(sid)
        assert final == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]

    @pytest.mark.asyncio
    async def test_delete_before(self, sqlite_store):
        """delete_before removes old sessions but keeps recent ones."""
        old_sid = await sqlite_store.create("old-session")
        new_sid = await sqlite_store.create("new-session")

        # Backdate the old session's updated_at
        db = await sqlite_store._get_db()
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (old_time, old_sid),
        )
        await db.commit()

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        deleted = await sqlite_store.delete_before(cutoff)

        assert deleted == 1
        assert await sqlite_store.exists(old_sid) is False
        assert await sqlite_store.exists(new_sid) is True

    @pytest.mark.asyncio
    async def test_create_duplicate_id_raises(self, sqlite_store):
        """Creating a session with an existing ID raises IntegrityError."""
        import aiosqlite
        await sqlite_store.create("dup-id")
        with pytest.raises(aiosqlite.IntegrityError):
            await sqlite_store.create("dup-id")

    @pytest.mark.asyncio
    async def test_close_and_reopen(self, tmp_path):
        """Data persists across close/reopen cycles."""
        db_path = str(tmp_path / "persist.db")

        store = SqliteSessionStore(db_path)
        sid = await store.create("persist-me")
        msgs = [{"role": "user", "content": "remember this"}]
        await store.save(sid, msgs)
        await store.close()

        store2 = SqliteSessionStore(db_path)
        loaded = await store2.load(sid)
        await store2.close()

        assert loaded == msgs

    # -- update() / cost_data ------------------------------------------------

    @staticmethod
    async def _read_cost_data(store, session_id):
        """Read raw cost_data JSON via direct DB query."""
        import json as _json

        db = await store._get_db()
        cursor = await db.execute(
            "SELECT cost_data FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return _json.loads(row[0]) if row else None

    @pytest.mark.asyncio
    async def test_update_merges_cost_data(self, sqlite_store):
        """Successive update() calls shallow-merge with write-wins."""
        sid = await sqlite_store.create()

        assert await sqlite_store.update(sid, cost_data={"a": 1}) is True
        assert await sqlite_store.update(sid, cost_data={"b": 2, "a": 5}) is True

        merged = await self._read_cost_data(sqlite_store, sid)
        assert merged == {"a": 5, "b": 2}

    @pytest.mark.asyncio
    async def test_update_missing_session(self, sqlite_store):
        """update() on a nonexistent session returns False."""
        assert await sqlite_store.update("doesnotexist", cost_data={"x": 1}) is False

    @pytest.mark.asyncio
    async def test_update_none_returns_existence(self, sqlite_store):
        """cost_data=None means: just confirm whether the session exists."""
        sid = await sqlite_store.create()
        assert await sqlite_store.update(sid) is True
        assert await sqlite_store.update("missing-session") is False

    @pytest.mark.asyncio
    async def test_save_preserves_cost_data(self, sqlite_store):
        """save() must not clobber cost_data accumulated via update()."""
        sid = await sqlite_store.create()
        await sqlite_store.update(sid, cost_data={"tokens": 100, "usd": 0.01})

        await sqlite_store.save(sid, [{"role": "user", "content": "first"}])
        await sqlite_store.save(sid, [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ])

        cost = await self._read_cost_data(sqlite_store, sid)
        assert cost == {"tokens": 100, "usd": 0.01}

    @pytest.mark.asyncio
    async def test_migration_adds_cost_data_column(self, tmp_path):
        """A pre-existing DB without cost_data is migrated transparently."""
        import aiosqlite

        db_path = str(tmp_path / "legacy.db")

        # Create the old schema by hand (no cost_data column).
        async with aiosqlite.connect(db_path) as legacy:
            await legacy.execute(
                "CREATE TABLE sessions ("
                "  session_id TEXT PRIMARY KEY, "
                "  messages TEXT NOT NULL, "
                "  created_at TEXT NOT NULL, "
                "  updated_at TEXT NOT NULL"
                ")"
            )
            await legacy.execute(
                "INSERT INTO sessions (session_id, messages, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("legacy-1", "[]", "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
            )
            await legacy.commit()

        # Open via SqliteSessionStore -- _ensure_table() should add cost_data.
        store = SqliteSessionStore(db_path)
        try:
            assert await store.update("legacy-1", cost_data={"a": 1}) is True
            cost = await self._read_cost_data(store, "legacy-1")
            assert cost == {"a": 1}
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateSessionStore:
    def test_null(self):
        store = create_session_store(None)
        assert isinstance(store, NullSessionStore)

    def test_sqlite(self):
        store = create_session_store("sqlite")
        assert isinstance(store, SqliteSessionStore)

    def test_postgres_requires_url(self):
        with pytest.raises(ValueError, match="database_url"):
            create_session_store("postgres")
