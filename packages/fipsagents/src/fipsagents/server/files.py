"""File persistence backends.

Stores metadata (filename, MIME type, size, SHA-256, extracted text) and
raw bytes for files uploaded to the agent. The ``POST /v1/files``
endpoint persists uploads via this module; ``ChatCompletionRequest``'s
``file_ids`` field resolves to extracted text injected into message
context before BaseAgent processes the request.

Two-tier separation: metadata lives in a relational store (SQLite or
Postgres), bytes live in object storage (local filesystem for dev,
S3-compatible for production). For dev parity, ``SqliteFileStore``
owns both — metadata in SQLite plus bytes in a local directory sharded
by ``file_id`` prefix. UUID-based keys are used everywhere; the
user-supplied filename is metadata only and never appears in a
storage path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _generate_file_id() -> str:
    return f"file_{uuid.uuid4().hex[:24]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bytes_path(bytes_dir: str, file_id: str) -> str:
    """Sharded path under *bytes_dir* keyed by *file_id*.

    Two-character prefix shard keeps a single directory from growing
    unbounded. ``file_<32 hex>`` → ``<bytes_dir>/fi/file_<32 hex>``.
    """
    shard = file_id[:2] if len(file_id) >= 2 else "00"
    return os.path.join(bytes_dir, shard, file_id)


# Module-level cache so we only complain about a missing libmagic once
# per process instead of on every upload.
_magic_unavailable_logged = False


def detect_mime(data: bytes) -> str | None:
    """Sniff the MIME type of *data* via libmagic, or None on failure.

    Returns the detected MIME (e.g. ``application/pdf``) when both
    python-magic and libmagic are available; logs a one-time warning
    and returns None when either is missing. The caller is expected to
    fall back to the client-supplied ``Content-Type`` in that case.

    The first call constructs a ``Magic`` instance which loads the
    libmagic database; subsequent calls reuse it via a module-level
    cache.
    """
    global _magic_unavailable_logged
    try:
        magic_mod = _get_magic_module()
    except ImportError:
        if not _magic_unavailable_logged:
            logger.warning(
                "detect_mime: python-magic / libmagic not available; "
                "falling back to client-supplied Content-Type. Install "
                "python-magic and libmagic for content-based MIME "
                "validation (fipsagents[files] + system libmagic).",
            )
            _magic_unavailable_logged = True
        return None
    try:
        return magic_mod.from_buffer(data)
    except Exception as exc:
        logger.warning("detect_mime: libmagic raised %s: %s", type(exc).__name__, exc)
        return None


def _get_magic_module():
    """Return a cached ``magic.Magic(mime=True)`` instance.

    Split out for test injection — callers can monkey-patch this to
    simulate libmagic being absent without uninstalling the library.
    """
    cached = getattr(_get_magic_module, "_cached", None)
    if cached is not None:
        return cached
    import magic  # type: ignore[import-not-found]

    m = magic.Magic(mime=True)
    _get_magic_module._cached = m  # type: ignore[attr-defined]
    return m


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


ParseStatus = Literal["pending", "processing", "completed", "failed", "skipped"]


@dataclass
class FileRecord:
    """A file uploaded to the agent.

    ``user_id`` mirrors :class:`FeedbackRecord` semantics: the
    gateway-issued ``X-Auth-Subject`` header value, or ``"anonymous"``
    when unauthenticated. ``session_id`` is optional — files can exist
    independently of a session.

    ``parse_status`` lifecycle:

    - ``pending``    — bytes uploaded, parsing not yet attempted (default)
    - ``processing`` — parse in flight
    - ``completed``  — ``extracted_text`` is populated
    - ``failed``     — ``parse_error`` is populated
    - ``skipped``    — file type intentionally not parsed (binary, unknown)
    """

    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    user_id: str = "anonymous"
    session_id: str | None = None
    extracted_text: str | None = None
    parse_status: ParseStatus = "pending"
    parse_error: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    deleted_at: str | None = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FileStore(ABC):
    """Pluggable file persistence backend (metadata + bytes)."""

    @abstractmethod
    async def save(self, record: FileRecord, data: bytes) -> str:
        """Persist *record* metadata and *data* bytes atomically.

        ``record.size_bytes`` and ``record.sha256`` MUST match
        ``len(data)`` and the SHA-256 of *data*; the implementation may
        verify and raise ``ValueError`` on mismatch.

        Returns the ``file_id``.
        """

    @abstractmethod
    async def get_metadata(self, file_id: str) -> FileRecord | None:
        """Retrieve metadata. Returns None if not found or soft-deleted."""

    @abstractmethod
    async def get_bytes(self, file_id: str) -> bytes | None:
        """Retrieve raw bytes. Returns None if not found."""

    @abstractmethod
    async def get_extracted_text(self, file_id: str) -> str | None:
        """Retrieve extracted text (parser output). None if not parsed."""

    @abstractmethod
    async def update_extracted_text(
        self,
        file_id: str,
        *,
        extracted_text: str | None = None,
        parse_status: ParseStatus | None = None,
        parse_error: str | None = None,
    ) -> bool:
        """Update parse-result fields. Returns True if the file existed."""

    @abstractmethod
    async def list_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[FileRecord]:
        """Return files attached to *session_id*, newest first."""

    @abstractmethod
    async def delete(self, file_id: str) -> bool:
        """Remove file metadata and bytes. Returns True if it existed."""

    @abstractmethod
    async def delete_before(self, cutoff: datetime) -> int:
        """Delete files created before *cutoff*. Returns count deleted."""

    async def close(self) -> None:
        """Release resources. Default is a no-op."""


# ---------------------------------------------------------------------------
# Null backend
# ---------------------------------------------------------------------------


class NullFileStore(FileStore):
    """No persistence — uploads are accepted but immediately discarded."""

    async def save(self, record: FileRecord, data: bytes) -> str:
        logger.debug("NullFileStore: discarded %s (%d bytes)", record.file_id, len(data))
        return record.file_id

    async def get_metadata(self, file_id: str) -> FileRecord | None:
        return None

    async def get_bytes(self, file_id: str) -> bytes | None:
        return None

    async def get_extracted_text(self, file_id: str) -> str | None:
        return None

    async def update_extracted_text(
        self,
        file_id: str,
        *,
        extracted_text: str | None = None,
        parse_status: ParseStatus | None = None,
        parse_error: str | None = None,
    ) -> bool:
        return False

    async def list_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[FileRecord]:
        return []

    async def delete(self, file_id: str) -> bool:
        return False

    async def delete_before(self, cutoff: datetime) -> int:
        return 0


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SqliteFileStore(FileStore):
    """SQLite metadata + sharded local-filesystem bytes.

    Suitable for development and single-replica edge deployments. For
    production, pair Postgres metadata with an S3-compatible bytes
    backend (MinIO).
    """

    _CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS files (
    file_id          TEXT PRIMARY KEY,
    session_id       TEXT,
    user_id          TEXT NOT NULL DEFAULT 'anonymous',
    filename         TEXT NOT NULL,
    mime_type        TEXT NOT NULL,
    size_bytes       INTEGER NOT NULL,
    sha256           TEXT NOT NULL,
    extracted_text   TEXT,
    parse_status     TEXT NOT NULL DEFAULT 'pending',
    parse_error      TEXT,
    bytes_path       TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    deleted_at       TEXT
)"""
    _CREATE_INDEX_SESSION = (
        "CREATE INDEX IF NOT EXISTS idx_files_session ON files (session_id)"
    )
    _CREATE_INDEX_CREATED = (
        "CREATE INDEX IF NOT EXISTS idx_files_created ON files (created_at)"
    )

    def __init__(
        self,
        db_path: str = "./agent.db",
        *,
        bytes_dir: str = "./files",
        connection: Any = None,
    ) -> None:
        self._db_path = db_path
        self._bytes_dir = bytes_dir
        self._db: Any = connection
        self._managed = connection is not None
        self._initialized = False

    async def _get_db(self) -> Any:
        if self._db is None:
            import aiosqlite

            self._db = await aiosqlite.connect(self._db_path)
        if not self._initialized:
            await self._ensure_schema()
        return self._db

    async def _ensure_schema(self) -> None:
        db = self._db
        await db.execute(self._CREATE_TABLE)
        await db.execute(self._CREATE_INDEX_SESSION)
        await db.execute(self._CREATE_INDEX_CREATED)
        await db.commit()
        os.makedirs(self._bytes_dir, exist_ok=True)
        self._initialized = True

    async def save(self, record: FileRecord, data: bytes) -> str:
        if record.size_bytes != len(data):
            raise ValueError(
                f"size_bytes mismatch: record says {record.size_bytes}, "
                f"data is {len(data)} bytes"
            )
        actual_sha = _sha256(data)
        if record.sha256 and record.sha256 != actual_sha:
            raise ValueError(
                f"sha256 mismatch for {record.file_id}: "
                f"record says {record.sha256}, data hashes to {actual_sha}"
            )
        # Trust caller-provided sha256 if present, else fill it in.
        sha = record.sha256 or actual_sha

        path = _bytes_path(self._bytes_dir, record.file_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write to a temp file then rename for atomicity.
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)

        db = await self._get_db()
        await db.execute(
            "INSERT INTO files ("
            "  file_id, session_id, user_id, filename, mime_type, "
            "  size_bytes, sha256, extracted_text, parse_status, "
            "  parse_error, bytes_path, created_at, deleted_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.file_id,
                record.session_id,
                record.user_id,
                record.filename,
                record.mime_type,
                record.size_bytes,
                sha,
                record.extracted_text,
                record.parse_status,
                record.parse_error,
                path,
                record.created_at,
                record.deleted_at,
            ),
        )
        await db.commit()
        logger.debug(
            "SqliteFileStore: saved %s (%d bytes, sha %s..)",
            record.file_id, record.size_bytes, sha[:8],
        )
        return record.file_id

    async def get_metadata(self, file_id: str) -> FileRecord | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT file_id, session_id, user_id, filename, mime_type, "
            "       size_bytes, sha256, extracted_text, parse_status, "
            "       parse_error, created_at, deleted_at "
            "FROM files WHERE file_id = ? AND deleted_at IS NULL",
            (file_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return FileRecord(
            file_id=row[0],
            session_id=row[1],
            user_id=row[2],
            filename=row[3],
            mime_type=row[4],
            size_bytes=row[5],
            sha256=row[6],
            extracted_text=row[7],
            parse_status=row[8],
            parse_error=row[9],
            created_at=row[10],
            deleted_at=row[11],
        )

    async def get_bytes(self, file_id: str) -> bytes | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT bytes_path FROM files "
            "WHERE file_id = ? AND deleted_at IS NULL",
            (file_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        path = row[0]
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            logger.warning(
                "SqliteFileStore: metadata for %s exists but bytes missing at %s",
                file_id, path,
            )
            return None

    async def get_extracted_text(self, file_id: str) -> str | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT extracted_text FROM files "
            "WHERE file_id = ? AND deleted_at IS NULL",
            (file_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row[0]

    async def update_extracted_text(
        self,
        file_id: str,
        *,
        extracted_text: str | None = None,
        parse_status: ParseStatus | None = None,
        parse_error: str | None = None,
    ) -> bool:
        if extracted_text is None and parse_status is None and parse_error is None:
            return await self._exists(file_id)

        sets: list[str] = []
        params: list[Any] = []
        if extracted_text is not None:
            sets.append("extracted_text = ?")
            params.append(extracted_text)
        if parse_status is not None:
            sets.append("parse_status = ?")
            params.append(parse_status)
        if parse_error is not None:
            sets.append("parse_error = ?")
            params.append(parse_error)
        params.append(file_id)

        db = await self._get_db()
        cursor = await db.execute(
            f"UPDATE files SET {', '.join(sets)} "
            "WHERE file_id = ? AND deleted_at IS NULL",
            tuple(params),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def _exists(self, file_id: str) -> bool:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT 1 FROM files WHERE file_id = ? AND deleted_at IS NULL",
            (file_id,),
        )
        return await cursor.fetchone() is not None

    async def list_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[FileRecord]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT file_id, session_id, user_id, filename, mime_type, "
            "       size_bytes, sha256, extracted_text, parse_status, "
            "       parse_error, created_at, deleted_at "
            "FROM files "
            "WHERE session_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at DESC "
            "LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            FileRecord(
                file_id=r[0],
                session_id=r[1],
                user_id=r[2],
                filename=r[3],
                mime_type=r[4],
                size_bytes=r[5],
                sha256=r[6],
                extracted_text=r[7],
                parse_status=r[8],
                parse_error=r[9],
                created_at=r[10],
                deleted_at=r[11],
            )
            for r in rows
        ]

    async def delete(self, file_id: str) -> bool:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT bytes_path FROM files "
            "WHERE file_id = ? AND deleted_at IS NULL",
            (file_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        path = row[0]
        await db.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        await db.commit()
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        # Best-effort: remove the shard dir if it's now empty.
        shard_dir = os.path.dirname(path)
        try:
            if shard_dir and shard_dir != self._bytes_dir:
                os.rmdir(shard_dir)
        except OSError:
            pass
        return True

    async def delete_before(self, cutoff: datetime) -> int:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT file_id, bytes_path FROM files WHERE created_at < ?",
            (cutoff.isoformat(),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return 0
        for _, path in rows:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        await db.execute(
            "DELETE FROM files WHERE created_at < ?",
            (cutoff.isoformat(),),
        )
        await db.commit()
        deleted = len(rows)
        if deleted:
            logger.debug(
                "SqliteFileStore: housekeeping removed %d files", deleted,
            )
        return deleted

    async def close(self) -> None:
        if self._db is not None and not self._managed:
            await self._db.close()
            self._db = None
            self._initialized = False


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------


class PostgresFileStore(FileStore):
    """Enterprise file persistence — Postgres metadata + local-FS bytes.

    Mirrors :class:`PostgresSessionStore` exactly: asyncpg pool managed
    lazily, schema created on first access, ``IF NOT EXISTS`` everywhere
    so re-runs are safe. The bytes layout is identical to
    :class:`SqliteFileStore` — sharded local filesystem under
    ``bytes_dir`` — because the S3-compatible bytes backend is a
    follow-up PR. Production deployments using Postgres should mount
    ``bytes_dir`` on a PVC sized for the expected upload volume.
    """

    _CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS files (
    file_id          TEXT PRIMARY KEY,
    session_id       TEXT,
    user_id          TEXT NOT NULL DEFAULT 'anonymous',
    filename         TEXT NOT NULL,
    mime_type        TEXT NOT NULL,
    size_bytes       BIGINT NOT NULL,
    sha256           TEXT NOT NULL,
    extracted_text   TEXT,
    parse_status     TEXT NOT NULL DEFAULT 'pending',
    parse_error      TEXT,
    bytes_path       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at       TIMESTAMPTZ
)"""
    _CREATE_INDEX_SESSION = (
        "CREATE INDEX IF NOT EXISTS idx_files_session ON files (session_id)"
    )
    _CREATE_INDEX_CREATED = (
        "CREATE INDEX IF NOT EXISTS idx_files_created ON files (created_at)"
    )

    def __init__(self, database_url: str, *, bytes_dir: str = "./files") -> None:
        self._database_url = database_url
        self._bytes_dir = bytes_dir
        self._pool: Any = None  # asyncpg.Pool
        self._initialized = False

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._database_url)
        if not self._initialized:
            await self._ensure_schema()
        return self._pool

    async def _ensure_schema(self) -> None:
        pool = self._pool
        async with pool.acquire() as conn:
            await conn.execute(self._CREATE_TABLE)
            await conn.execute(self._CREATE_INDEX_SESSION)
            await conn.execute(self._CREATE_INDEX_CREATED)
        os.makedirs(self._bytes_dir, exist_ok=True)
        self._initialized = True

    async def save(self, record: FileRecord, data: bytes) -> str:
        if record.size_bytes != len(data):
            raise ValueError(
                f"size_bytes mismatch: record says {record.size_bytes}, "
                f"data is {len(data)} bytes"
            )
        actual_sha = _sha256(data)
        if record.sha256 and record.sha256 != actual_sha:
            raise ValueError(
                f"sha256 mismatch for {record.file_id}: "
                f"record says {record.sha256}, data hashes to {actual_sha}"
            )
        sha = record.sha256 or actual_sha

        path = _bytes_path(self._bytes_dir, record.file_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)

        # FileRecord.created_at is a string for SQLite-friendly storage;
        # parse it back to a datetime for TIMESTAMPTZ. Default to now()
        # when the field is empty (tests that build a record without
        # touching the default factory).
        created_at = _parse_iso(record.created_at) or datetime.now(timezone.utc)
        deleted_at = _parse_iso(record.deleted_at)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO files ("
                "  file_id, session_id, user_id, filename, mime_type, "
                "  size_bytes, sha256, extracted_text, parse_status, "
                "  parse_error, bytes_path, created_at, deleted_at"
                ") VALUES "
                "($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
                record.file_id,
                record.session_id,
                record.user_id,
                record.filename,
                record.mime_type,
                record.size_bytes,
                sha,
                record.extracted_text,
                record.parse_status,
                record.parse_error,
                path,
                created_at,
                deleted_at,
            )
        logger.debug(
            "PostgresFileStore: saved %s (%d bytes, sha %s..)",
            record.file_id, record.size_bytes, sha[:8],
        )
        return record.file_id

    async def get_metadata(self, file_id: str) -> FileRecord | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT file_id, session_id, user_id, filename, mime_type, "
                "       size_bytes, sha256, extracted_text, parse_status, "
                "       parse_error, created_at, deleted_at "
                "FROM files WHERE file_id = $1 AND deleted_at IS NULL",
                file_id,
            )
        if row is None:
            return None
        return FileRecord(
            file_id=row["file_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            filename=row["filename"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            extracted_text=row["extracted_text"],
            parse_status=row["parse_status"],
            parse_error=row["parse_error"],
            created_at=_iso(row["created_at"]),
            deleted_at=_iso(row["deleted_at"]),
        )

    async def get_bytes(self, file_id: str) -> bytes | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT bytes_path FROM files "
                "WHERE file_id = $1 AND deleted_at IS NULL",
                file_id,
            )
        if row is None:
            return None
        path = row["bytes_path"]
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            logger.warning(
                "PostgresFileStore: metadata for %s exists but bytes missing at %s",
                file_id, path,
            )
            return None

    async def get_extracted_text(self, file_id: str) -> str | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT extracted_text FROM files "
                "WHERE file_id = $1 AND deleted_at IS NULL",
                file_id,
            )
        if row is None:
            return None
        return row["extracted_text"]

    async def update_extracted_text(
        self,
        file_id: str,
        *,
        extracted_text: str | None = None,
        parse_status: ParseStatus | None = None,
        parse_error: str | None = None,
    ) -> bool:
        if extracted_text is None and parse_status is None and parse_error is None:
            return await self._exists(file_id)

        # Build SET clauses with positional placeholders ($2, $3, ...).
        sets: list[str] = []
        params: list[Any] = []
        if extracted_text is not None:
            sets.append(f"extracted_text = ${len(params) + 2}")
            params.append(extracted_text)
        if parse_status is not None:
            sets.append(f"parse_status = ${len(params) + 2}")
            params.append(parse_status)
        if parse_error is not None:
            sets.append(f"parse_error = ${len(params) + 2}")
            params.append(parse_error)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE files SET {', '.join(sets)} "
                "WHERE file_id = $1 AND deleted_at IS NULL",
                file_id, *params,
            )
        return not result.endswith("0")

    async def _exists(self, file_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM files WHERE file_id = $1 AND deleted_at IS NULL",
                file_id,
            )
        return row is not None

    async def list_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[FileRecord]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT file_id, session_id, user_id, filename, mime_type, "
                "       size_bytes, sha256, extracted_text, parse_status, "
                "       parse_error, created_at, deleted_at "
                "FROM files "
                "WHERE session_id = $1 AND deleted_at IS NULL "
                "ORDER BY created_at DESC "
                "LIMIT $2 OFFSET $3",
                session_id, limit, offset,
            )
        return [
            FileRecord(
                file_id=r["file_id"],
                session_id=r["session_id"],
                user_id=r["user_id"],
                filename=r["filename"],
                mime_type=r["mime_type"],
                size_bytes=r["size_bytes"],
                sha256=r["sha256"],
                extracted_text=r["extracted_text"],
                parse_status=r["parse_status"],
                parse_error=r["parse_error"],
                created_at=_iso(r["created_at"]),
                deleted_at=_iso(r["deleted_at"]),
            )
            for r in rows
        ]

    async def delete(self, file_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT bytes_path FROM files "
                "WHERE file_id = $1 AND deleted_at IS NULL",
                file_id,
            )
            if row is None:
                return False
            path = row["bytes_path"]
            await conn.execute(
                "DELETE FROM files WHERE file_id = $1",
                file_id,
            )
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        shard_dir = os.path.dirname(path)
        try:
            if shard_dir and shard_dir != self._bytes_dir:
                os.rmdir(shard_dir)
        except OSError:
            pass
        return True

    async def delete_before(self, cutoff: datetime) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT file_id, bytes_path FROM files WHERE created_at < $1",
                cutoff,
            )
            if not rows:
                return 0
            for r in rows:
                try:
                    os.remove(r["bytes_path"])
                except FileNotFoundError:
                    pass
            await conn.execute(
                "DELETE FROM files WHERE created_at < $1",
                cutoff,
            )
        deleted = len(rows)
        if deleted:
            logger.debug(
                "PostgresFileStore: housekeeping removed %d files", deleted,
            )
        return deleted

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False


def _parse_iso(value: str | None) -> datetime | None:
    """Parse a FileRecord ISO timestamp string into a UTC datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _iso(value: datetime | None) -> str | None:
    """Render a Postgres TIMESTAMPTZ back to FileRecord's ISO string."""
    if value is None:
        return None
    return value.isoformat()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_file_store(
    backend: str | None,
    *,
    sqlite_path: str = "./agent.db",
    database_url: str = "",
    bytes_dir: str = "./files",
    sqlite_connection: Any = None,
) -> FileStore:
    """Create a file store from config values.

    Supported backends:
      - ``sqlite``   — :class:`SqliteFileStore` (single-replica, dev / edge)
      - ``postgres`` — :class:`PostgresFileStore` (metadata in PG, bytes
                       on local FS; production deployments must mount
                       ``bytes_dir`` on a PVC)
      - ``http``     — platform-routed (not yet implemented)
      - ``None``     — :class:`NullFileStore` (accepted-then-discarded)

    The S3-compatible bytes backend lands in a follow-up PR; until then
    Postgres deployments share the SQLite layout for raw bytes.
    """
    if backend == "sqlite":
        return SqliteFileStore(
            sqlite_path,
            bytes_dir=bytes_dir,
            connection=sqlite_connection,
        )
    if backend == "postgres":
        if not database_url:
            raise ValueError("PostgresFileStore requires database_url")
        return PostgresFileStore(database_url, bytes_dir=bytes_dir)
    if backend == "http":
        raise NotImplementedError(
            "FileStore backend 'http' is not yet implemented; "
            "use 'sqlite', 'postgres', or leave unset (Null)."
        )
    return NullFileStore()
