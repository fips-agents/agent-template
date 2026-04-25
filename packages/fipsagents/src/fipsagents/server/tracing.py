"""Trace data model and persistence backends.

A trace represents one request through the agent. It contains spans
representing operations: model calls, tool executions, memory lookups.
The ``TraceCollector`` (in ``collector.py``) builds traces from
``StreamEvent``s; this module provides the data model and storage.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Dedicated logger for structured trace output (used by NullTraceStore).
_trace_logger = logging.getLogger("fipsagents.tracing")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single operation within a trace.

    ``start_time`` / ``end_time`` are :func:`time.monotonic` values --
    relative, not wall clock. They exist for computing durations. The
    parent :class:`Trace` carries wall-clock timestamps in ISO 8601.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str = ""  # e.g. "request", "step:1", "model_call", "tool:search"
    start_time: float = 0.0
    end_time: float | None = None
    status: str = "ok"  # "ok" | "error"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


@dataclass
class TraceSummary:
    """Lightweight trace summary for list responses."""

    trace_id: str
    started_at: str  # ISO 8601
    ended_at: str | None
    model: str | None = None
    session_id: str | None = None
    status: str = "ok"
    duration_ms: float | None = None
    span_count: int = 0
    tool_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class Trace:
    """Complete trace with all spans."""

    trace_id: str
    started_at: str  # ISO 8601
    ended_at: str | None = None
    model: str | None = None
    session_id: str | None = None
    status: str = "ok"
    spans: list[Span] = field(default_factory=list)

    def to_summary(self) -> TraceSummary:
        """Create a lightweight summary from this trace."""
        tool_calls = sum(1 for s in self.spans if s.name.startswith("tool:"))
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        duration_ms: float | None = None

        # Aggregate token counts across all model_call spans.
        for s in self.spans:
            if s.name == "model_call":
                pt = s.attributes.get("prompt_tokens")
                ct = s.attributes.get("completion_tokens")
                if pt is not None:
                    prompt_tokens = (prompt_tokens or 0) + pt
                if ct is not None:
                    completion_tokens = (completion_tokens or 0) + ct

        # Duration from root span (first span without a parent).
        root_spans = [s for s in self.spans if s.parent_span_id is None]
        if root_spans and root_spans[0].duration_ms is not None:
            duration_ms = root_spans[0].duration_ms

        return TraceSummary(
            trace_id=self.trace_id,
            started_at=self.started_at,
            ended_at=self.ended_at,
            model=self.model,
            session_id=self.session_id,
            status=self.status,
            duration_ms=duration_ms,
            span_count=len(self.spans),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TraceStore(ABC):
    """Pluggable trace persistence backend."""

    @abstractmethod
    async def save_trace(self, trace: Trace) -> None:
        """Persist a completed trace."""

    @abstractmethod
    async def get_trace(self, trace_id: str) -> Trace | None:
        """Retrieve a trace by ID."""

    @abstractmethod
    async def list_traces(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[TraceSummary]:
        """List recent traces (summary only)."""

    @abstractmethod
    async def delete_before(self, cutoff: datetime) -> int:
        """Remove old traces. Return count deleted."""

    async def close(self) -> None:
        """Release resources. Default is a no-op."""


# ---------------------------------------------------------------------------
# Null (structured logging, default)
# ---------------------------------------------------------------------------


class NullTraceStore(TraceStore):
    """No persistence -- traces are logged as structured JSON then discarded."""

    async def save_trace(self, trace: Trace) -> None:
        _trace_logger.debug(
            "trace %s: %s",
            trace.trace_id,
            json.dumps(asdict(trace), default=str),
        )

    async def get_trace(self, trace_id: str) -> Trace | None:
        return None

    async def list_traces(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[TraceSummary]:
        return []

    async def delete_before(self, cutoff: datetime) -> int:
        return 0


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class SqliteTraceStore(TraceStore):
    """Single-file trace persistence via aiosqlite."""

    _CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    model       TEXT,
    session_id  TEXT,
    status      TEXT NOT NULL DEFAULT 'ok',
    spans       TEXT NOT NULL,
    summary     TEXT
)"""
    _CREATE_INDEX = (
        "CREATE INDEX IF NOT EXISTS idx_traces_started "
        "ON traces (started_at)"
    )

    def __init__(self, db_path: str = "./agent.db") -> None:
        self._db_path = db_path
        self._db: Any = None  # aiosqlite.Connection, typed loosely to keep import lazy
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
        await db.execute(self._CREATE_INDEX)
        await db.commit()
        self._initialized = True

    async def save_trace(self, trace: Trace) -> None:
        db = await self._get_db()
        spans_json = json.dumps([asdict(s) for s in trace.spans], default=str)
        summary_json = json.dumps(asdict(trace.to_summary()), default=str)
        await db.execute(
            "INSERT OR REPLACE INTO traces "
            "(trace_id, started_at, ended_at, model, session_id, status, spans, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace.trace_id,
                trace.started_at,
                trace.ended_at,
                trace.model,
                trace.session_id,
                trace.status,
                spans_json,
                summary_json,
            ),
        )
        await db.commit()
        logger.debug("SqliteTraceStore: saved trace %s (%d spans)", trace.trace_id, len(trace.spans))

    async def get_trace(self, trace_id: str) -> Trace | None:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT trace_id, started_at, ended_at, model, session_id, status, spans "
            "FROM traces WHERE trace_id = ?",
            (trace_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        spans_raw = json.loads(row[6])
        spans = [
            Span(
                trace_id=s["trace_id"],
                span_id=s["span_id"],
                parent_span_id=s.get("parent_span_id"),
                name=s.get("name", ""),
                start_time=s.get("start_time", 0.0),
                end_time=s.get("end_time"),
                status=s.get("status", "ok"),
                attributes=s.get("attributes", {}),
                events=s.get("events", []),
            )
            for s in spans_raw
        ]

        return Trace(
            trace_id=row[0],
            started_at=row[1],
            ended_at=row[2],
            model=row[3],
            session_id=row[4],
            status=row[5],
            spans=spans,
        )

    async def list_traces(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[TraceSummary]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT summary FROM traces ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        summaries: list[TraceSummary] = []
        for (summary_json,) in rows:
            if summary_json is None:
                continue
            d = json.loads(summary_json)
            summaries.append(TraceSummary(
                trace_id=d["trace_id"],
                started_at=d["started_at"],
                ended_at=d.get("ended_at"),
                model=d.get("model"),
                session_id=d.get("session_id"),
                status=d.get("status", "ok"),
                duration_ms=d.get("duration_ms"),
                span_count=d.get("span_count", 0),
                tool_calls=d.get("tool_calls", 0),
                prompt_tokens=d.get("prompt_tokens"),
                completion_tokens=d.get("completion_tokens"),
            ))
        return summaries

    async def delete_before(self, cutoff: datetime) -> int:
        db = await self._get_db()
        cursor = await db.execute(
            "DELETE FROM traces WHERE started_at < ?",
            (cutoff.isoformat(),),
        )
        await db.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.debug("SqliteTraceStore: housekeeping removed %d traces", deleted)
        return deleted

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_trace_store(
    backend: str | None,
    *,
    sqlite_path: str = "./agent.db",
    database_url: str = "",
) -> TraceStore:
    """Create a trace store from config values."""
    if backend == "sqlite":
        return SqliteTraceStore(sqlite_path)
    if backend == "postgres":
        logger.warning(
            "PostgresTraceStore is not yet implemented; "
            "traces will use NullTraceStore (structured logging only)"
        )
    return NullTraceStore()
