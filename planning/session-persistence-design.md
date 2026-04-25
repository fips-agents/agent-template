# Session Persistence Design (#78)

## Problem

Agent conversations are ephemeral — lost when the pod restarts or when
the next request overwrites `agent.messages`. Enterprise deployments
need conversation continuity, and even dev/edge environments benefit
from session history for debugging.

## Design principle

When LlamaStack or another orchestration layer is present, it handles
session management. The agent framework only needs its own persistence
when those services aren't available — and in those cases, the
environment is typically small, so the solution must be lightweight.

## Session lifecycle (REST)

```
POST   /v1/sessions                → create session, return session_id
POST   /v1/chat/completions        → session_id in body (optional)
GET    /v1/sessions/{session_id}   → retrieve message history
DELETE /v1/sessions/{session_id}   → clean up
```

- `POST /v1/sessions` accepts an optional `session_id` in the body.
  If omitted, the server generates one (e.g., `sess_<uuid>`).
- `POST /v1/chat/completions` with `session_id`: server loads prior
  messages, appends incoming messages, runs the agent, persists the
  full history after the response completes.
- `POST /v1/chat/completions` without `session_id`: ephemeral,
  current behavior — fully backward-compatible.

## Architecture

```
                 ┌─────────────────────────────┐
                 │     OpenAIChatServer         │
                 │                              │
  request ──────►  session_store.load(sid)      │
                 │  agent.messages = loaded     │
                 │  agent.astep_stream()        │
                 │  session_store.save(sid, msgs)│
                 │                              │
                 └─────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         NullStore    SqliteStore   PostgresStore
        (ephemeral)   (edge/dev)   (enterprise)
```

The session store lives in the **server module**, not BaseAgent.
BaseAgent continues to work with `self.messages` and has no concept
of sessions. The server layer handles load-before / save-after
around each request.

## Storage interface

```python
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
    async def delete(self, session_id: str) -> None:
        """Remove a session and its history."""

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
```

## Backends

### NullSessionStore (default)

No persistence. `create()` generates an ID, `load()` always returns
None, `save()` is a no-op. This is the current behavior — zero
overhead, fully backward-compatible.

### SqliteSessionStore

Single-file persistence for edge/dev/MicroShift. Schema:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    messages     TEXT NOT NULL,   -- JSON array
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
```

- Messages stored as JSON text (SQLite has no native JSONB but
  json_extract works for queries if needed).
- File location configurable, defaults to `./sessions.db`.
- Uses `aiosqlite` for async access.

### PostgresSessionStore

Full persistence for enterprise. Schema:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    messages     JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_sessions_updated ON sessions (updated_at);
```

- Connection via `DATABASE_URL` env var.
- Uses `asyncpg` (already an optional dependency in fipsagents).

## Configuration

Session persistence shares a storage layer with tracing (#81). One
backend config, per-feature enable flags, shared connection.

```yaml
# agent.yaml
server:
  host: "0.0.0.0"
  port: 8080
  storage:
    backend: null             # null | sqlite | postgres
    # SQLite-specific:
    sqlite_path: "./agent.db"
    # Postgres-specific (or use DATABASE_URL env var):
    database_url: "${DATABASE_URL:-}"
  sessions:
    enabled: false            # opt-in
    max_age_hours: 168        # auto-delete sessions older than 7 days (0 = no expiry)
```

When `storage.backend` is `null` (default), sessions are not
persisted even if `sessions.enabled` is true — the NullSessionStore
handles creates/loads as no-ops. Setting `storage.backend: sqlite`
or `postgres` enables actual persistence. This lets the same config
structure serve both sessions and traces (see tracing-design.md).

## Server integration

In `OpenAIChatServer`:

1. During lifespan startup: instantiate the session store from config.
2. `POST /v1/sessions`: call `store.create()`, return the session_id.
3. `POST /v1/chat/completions`: if `session_id` in request body:
   - `stored = await store.load(session_id)`
   - If stored: prepend stored messages before incoming messages
   - After response: `await store.save(session_id, agent.messages)`
4. During lifespan shutdown: close the store connection.

## Request model changes

Add optional `session_id` to `ChatCompletionRequest`:

```python
class ChatCompletionRequest(BaseModel):
    # ... existing fields ...
    session_id: str | None = None
```

This is an extension field — OpenAI's API doesn't have it, but
conforming clients ignore unknown fields.

## What this does NOT cover

- **LlamaStack session management**: When LlamaStack is the
  orchestration layer, it owns sessions. The agent doesn't persist
  locally — it receives the full context in each request from
  LlamaStack. No session store needed.
- **Multi-replica session affinity**: If multiple replicas serve the
  same agent, SQLite won't work (file-local). PostgreSQL handles this
  naturally. For SQLite, the agent must be single-replica.
- **Message compaction / summarization**: Out of scope. The store
  persists raw messages. Context window management is the agent's
  responsibility.

## Implementation order

1. `SessionStore` ABC + `NullSessionStore` (default, no behavior change)
2. Server integration (routes, load/save hooks)
3. `SqliteSessionStore` backend
4. `PostgresSessionStore` backend
5. Housekeeping (max_age_hours cleanup)
6. Helm chart: optional PostgreSQL sidecar or external connection
