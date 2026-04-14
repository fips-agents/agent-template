## Custom Memory Backends

When the built-in backends (MemoryHub, SQLite, PGVector) don't fit your infrastructure, you can implement and register your own. A custom backend is appropriate when you want to use Redis, DynamoDB, Elasticsearch, an internal REST API, or any other store. The interface is four async methods; the factory wires the rest.

## The Interface

All backends implement `MemoryClientBase` from `fipsagents.baseagent.memory`:

```python
class MemoryClientBase:
    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]: ...
    async def write(self, content: str, **kwargs: Any) -> dict[str, Any] | None: ...
    async def update(self, memory_id: str, content: str, **kwargs: Any) -> dict[str, Any] | None: ...
    async def report_contradiction(self, memory_id: str, description: str) -> None: ...
```

**`search(query, **kwargs) -> list[dict]`**
Returns matching memories. Each dict must have at minimum `id` and `content`; `metadata`, `created_at`, and `updated_at` are conventional. The `kwargs` may include `limit` (int, default 10). Return `[]` on error — never raise.

**`write(content, **kwargs) -> dict | None`**
Creates a new memory. Generate a unique `id` (e.g., `uuid4()`). The `kwargs` may include `metadata` (dict), `scope` (str), and `weight` (float). Return a dict with at minimum `id`, `content`, and `created_at`. Return `None` on error — never raise.

**`update(memory_id, content, **kwargs) -> dict | None`**
Updates an existing memory's content. The `kwargs` may include `metadata`. Return a dict with at minimum `id`, `content`, and `updated_at`. Return `None` if the id doesn't exist, and `None` on error — never raise.

**`report_contradiction(memory_id, description) -> None`**
Signals that a memory contradicts observed behavior. Logging a warning is an acceptable implementation. Must not raise.

## The Implicit Contract

The type signatures above don't capture everything:

**Never raise.** Every method must catch its own exceptions and degrade gracefully. A flaky memory backend must never crash the agent — that's the core promise of the memory system. Log warnings, return empty results or `None`.

**All methods are async.** If your backing store is synchronous (e.g., `redis-py` sync client, `sqlite3`), wrap blocking calls with `asyncio.to_thread()`. See `SQLiteMemoryClient` in `memory_sqlite.py` for the pattern.

**Thread safety.** Methods may be called from multiple coroutines concurrently. Use connection pools or async locks as appropriate for your backing store.

**Idempotent setup.** If your backend needs schema creation or index initialization, make it idempotent (`CREATE TABLE IF NOT EXISTS`, `PUT IF NOT EXISTS`). The agent may restart, and `setup()` will be called again.

**Optional `setup()`.** If your class defines an `async def setup(self)` method, the factory calls it after instantiation. Use this for async initialization: connecting to a pool, creating tables, warming a cache. There is no teardown hook — design your backend to handle process exit gracefully.

**Return dicts, not models.** All return values are plain dicts. This keeps the interface simple and avoids coupling agent code to your backend's internal models.

## Minimal Example: InMemoryClient

A complete implementation that stores memories in a Python dict. Useful for testing and prototyping:

```python
"""In-memory backend for testing and prototyping."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fipsagents.baseagent.memory import MemoryClientBase

logger = logging.getLogger(__name__)


class InMemoryClient(MemoryClientBase):
    """Stores memories in a Python dict. Data is lost on process exit."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        limit = int(kwargs.get("limit", 10))
        query_lower = query.lower()
        matches = [
            m for m in self._store.values()
            if query_lower in m["content"].lower()
        ]
        return matches[:limit]

    async def write(self, content: str, **kwargs: Any) -> dict[str, Any] | None:
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": memory_id,
            "content": content,
            "metadata": kwargs.get("metadata"),
            "created_at": now,
            "updated_at": now,
        }
        self._store[memory_id] = entry
        return entry

    async def update(
        self, memory_id: str, content: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        if memory_id not in self._store:
            return None
        self._store[memory_id]["content"] = content
        self._store[memory_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "metadata" in kwargs:
            self._store[memory_id]["metadata"] = kwargs["metadata"]
        return self._store[memory_id]

    async def report_contradiction(
        self, memory_id: str, description: str
    ) -> None:
        logger.warning("Contradiction: memory %s — %s", memory_id, description)
```

## Registering Your Backend

### Option A: `backend: custom` in agent.yaml (recommended)

For backends specific to one project, set `backend_class` to the fully-qualified dotted import path of your class:

```yaml
memory:
  backend: custom
  backend_class: myproject.memory.InMemoryClient
```

The factory imports the module, checks that the class is a `MemoryClientBase` subclass, instantiates it with no arguments, and calls `setup()` if defined. If loading fails at any point, the agent falls back to `NullMemoryClient` and logs the error.

Requirements:
- The module must be on `sys.path` or installed as a package
- The class must subclass `MemoryClientBase`
- The constructor must accept zero arguments

### Option B: Add a named backend to the package

For backends you want to share across projects or contribute to fipsagents:

1. Create `packages/fipsagents/src/fipsagents/baseagent/memory_<name>.py`
2. Export a `create_<name>_client(config_path: Path) -> MemoryClientBase` factory function (see `create_sqlite_client` for the pattern)
3. Add a dispatch case in `memory.py`'s `create_memory_client()`
4. Add the backend name to `MemoryConfig.backend`'s `Literal` type in `config.py`

## Testing Your Backend

Copy the following contract test class into your test suite and provide a `memory_client` fixture. The tests verify that your implementation satisfies the full interface contract, including the implicit requirements (graceful degradation, `None` for missing ids, etc.).

```python
"""Reusable contract tests for memory backends.

Usage:

    from fipsagents.testing.memory_contract import MemoryContractTests

    class TestMyBackend(MemoryContractTests):
        @pytest.fixture
        async def memory_client(self):
            return MyMemoryClient()
"""

from __future__ import annotations

import pytest


class MemoryContractTests:
    """Mixin of tests that verify a MemoryClientBase implementation.

    Subclass this and provide a ``memory_client`` fixture that returns
    your backend instance.
    """

    @pytest.fixture
    async def memory_client(self):
        raise NotImplementedError("Provide a memory_client fixture")

    @pytest.mark.asyncio
    async def test_search_returns_list(self, memory_client):
        result = await memory_client.search("anything")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_write_returns_dict_with_id(self, memory_client):
        result = await memory_client.write("test content")
        assert result is not None
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result

    @pytest.mark.asyncio
    async def test_write_then_search_finds_it(self, memory_client):
        written = await memory_client.write("unique canary value")
        results = await memory_client.search("canary")
        assert any(r["id"] == written["id"] for r in results)

    @pytest.mark.asyncio
    async def test_update_existing_returns_dict(self, memory_client):
        written = await memory_client.write("original")
        result = await memory_client.update(written["id"], "revised")
        assert result is not None
        assert result["content"] == "revised"

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, memory_client):
        result = await memory_client.update("nonexistent-id", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_report_contradiction_does_not_raise(self, memory_client):
        written = await memory_client.write("a fact")
        await memory_client.report_contradiction(written["id"], "observed contrary")

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, memory_client):
        for i in range(5):
            await memory_client.write(f"memory about topic {i}")
        results = await memory_client.search("topic", limit=2)
        assert len(results) <= 2
```

This is provided as a code example to copy — it is not currently shipped as an importable module. A future version of fipsagents may publish it as `fipsagents.testing.memory_contract`.

## Available Backends Reference

| Backend | Config file | Dependencies | Search type | Best for |
|---------|-------------|--------------|-------------|----------|
| `memoryhub` | `.memoryhub.yaml` | `memoryhub` | Full (MemoryHub server) | Production with MemoryHub |
| `sqlite` | `.memory-sqlite.yaml` | None (stdlib) | Keyword (FTS5) | Local dev, testing |
| `pgvector` | `.memory-pgvector.yaml` | `asyncpg`, `pgvector` | Semantic (vector cosine) | Production without MemoryHub |
| `custom` | — | Your choice | Your choice | Custom infrastructure |
| `null` | — | None | None (disabled) | Explicitly disable memory |
