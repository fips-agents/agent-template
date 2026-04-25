"""Session persistence backends.

Stores and retrieves conversation message histories keyed by session ID.
The server loads messages before processing a request and saves after
the response completes. When no session store is configured, the
``NullSessionStore`` provides backward-compatible ephemeral behavior.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _generate_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:16]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore(ABC):
    """Pluggable session persistence backend."""

    @abstractmethod
    async def create(self, session_id: str | None = None) -> str:
        """Create a session. Generate ID if not provided."""

    @abstractmethod
    async def load(self, session_id: str) -> list[dict] | None:
        """Load messages for a session. None if not found."""

    @abstractmethod
    async def save(self, session_id: str, messages: list[dict]) -> None:
        """Persist the full message history for a session."""

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Remove a session. Return True if it existed."""

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""

    @abstractmethod
    async def delete_before(self, cutoff: datetime) -> int:
        """Delete sessions not updated since *cutoff*. Return count deleted."""

    async def close(self) -> None:
        """Release resources. Default is a no-op."""


class NullSessionStore(SessionStore):
    """No persistence -- every request is ephemeral."""

    async def create(self, session_id: str | None = None) -> str:
        sid = session_id or _generate_session_id()
        logger.debug("NullSessionStore: created (ephemeral) %s", sid)
        return sid

    async def load(self, session_id: str) -> list[dict] | None:
        return None

    async def save(self, session_id: str, messages: list[dict]) -> None:
        pass

    async def delete(self, session_id: str) -> bool:
        return False

    async def exists(self, session_id: str) -> bool:
        return False

    async def delete_before(self, cutoff: datetime) -> int:
        return 0


class SqliteSessionStore(SessionStore):
    """Single-file session persistence via aiosqlite."""

    _CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    messages    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)"""

    def __init__(self, db_path: str = "./agent.db", *, connection: Any = None) -> None:
        self._db_path = db_path
        self._db: Any = connection  # Pre-set if managed
        self._managed = connection is not None
        self._initialized = False

    async def _get_db(self) -> Any:
        if self._db is None:
            import aiosqlite

            self._db = await aiosqlite.connect(self._db_path)
        if not self._initialized:
            await self._ensure_table()
        return self._db

    async def _ensure_table(self) -> None:
        db = self._db
        await db.execute(self._CREATE_TABLE)
        await db.commit()
        self._initialized = True

    async def create(self, session_id: str | None = None) -> str:
        sid = session_id or _generate_session_id()
        now = _utc_now_iso()
        db = await self._get_db()
        await db.execute(
            "INSERT INTO sessions (session_id, messages, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (sid, "[]", now, now),
        )
        await db.commit()
        logger.debug("SqliteSessionStore: created %s", sid)
        return sid

    async def load(self, session_id: str) -> list[dict] | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT messages FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        logger.debug("SqliteSessionStore: loaded %s", session_id)
        return json.loads(row[0])

    async def save(self, session_id: str, messages: list[dict]) -> None:
        now = _utc_now_iso()
        db = await self._get_db()
        await db.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, messages, created_at, updated_at) "
            "VALUES (?, ?, COALESCE("
            "  (SELECT created_at FROM sessions WHERE session_id = ?), ?"
            "), ?)",
            (session_id, json.dumps(messages), session_id, now, now),
        )
        await db.commit()
        logger.debug("SqliteSessionStore: saved %s (%d messages)", session_id, len(messages))

    async def delete(self, session_id: str) -> bool:
        db = await self._get_db()
        cursor = await db.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def exists(self, session_id: str) -> bool:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        return await cursor.fetchone() is not None

    async def delete_before(self, cutoff: datetime) -> int:
        db = await self._get_db()
        cursor = await db.execute(
            "DELETE FROM sessions WHERE updated_at < ?",
            (cutoff.isoformat(),),
        )
        await db.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.debug("SqliteSessionStore: housekeeping removed %d sessions", deleted)
        return deleted

    async def close(self) -> None:
        if self._db is not None and not self._managed:
            await self._db.close()
            self._db = None
            self._initialized = False


class PostgresSessionStore(SessionStore):
    """Enterprise session persistence via asyncpg."""

    _CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    messages    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)"""
    _CREATE_INDEX = (
        "CREATE INDEX IF NOT EXISTS idx_sessions_updated "
        "ON sessions (updated_at)"
    )

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._pool: Any = None  # asyncpg.Pool
        self._initialized = False

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._database_url)
        if not self._initialized:
            await self._ensure_table()
        return self._pool

    async def _ensure_table(self) -> None:
        pool = self._pool
        async with pool.acquire() as conn:
            await conn.execute(self._CREATE_TABLE)
            await conn.execute(self._CREATE_INDEX)
        self._initialized = True

    async def create(self, session_id: str | None = None) -> str:
        sid = session_id or _generate_session_id()
        now = datetime.now(timezone.utc)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, messages, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4)",
                sid, json.dumps([]), now, now,
            )
        logger.debug("PostgresSessionStore: created %s", sid)
        return sid

    async def load(self, session_id: str) -> list[dict] | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT messages FROM sessions WHERE session_id = $1",
                session_id,
            )
        if row is None:
            return None
        logger.debug("PostgresSessionStore: loaded %s", session_id)
        messages = row["messages"]
        # asyncpg auto-decodes JSONB to Python objects
        if isinstance(messages, str):
            return json.loads(messages)
        return messages

    async def save(self, session_id: str, messages: list[dict]) -> None:
        now = datetime.now(timezone.utc)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, messages, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (session_id) DO UPDATE "
                "SET messages = EXCLUDED.messages, updated_at = EXCLUDED.updated_at",
                session_id, json.dumps(messages), now, now,
            )
        logger.debug(
            "PostgresSessionStore: saved %s (%d messages)", session_id, len(messages),
        )

    async def delete(self, session_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sessions WHERE session_id = $1",
                session_id,
            )
        # asyncpg returns "DELETE N"
        return not result.endswith("0")

    async def exists(self, session_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM sessions WHERE session_id = $1",
                session_id,
            )
        return row is not None

    async def delete_before(self, cutoff: datetime) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sessions WHERE updated_at < $1",
                cutoff,
            )
        # asyncpg returns "DELETE N"
        deleted = int(result.split()[-1])
        if deleted:
            logger.debug("PostgresSessionStore: housekeeping removed %d sessions", deleted)
        return deleted

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False


def create_session_store(
    backend: str | None,
    *,
    sqlite_path: str = "./agent.db",
    database_url: str = "",
    sqlite_connection: Any = None,
) -> SessionStore:
    """Create a session store from config values."""
    if backend == "sqlite":
        return SqliteSessionStore(sqlite_path, connection=sqlite_connection)
    elif backend == "postgres":
        if not database_url:
            raise ValueError("PostgresSessionStore requires database_url")
        return PostgresSessionStore(database_url)
    return NullSessionStore()
