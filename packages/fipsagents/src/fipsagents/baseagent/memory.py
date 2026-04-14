"""Pluggable memory backends for BaseAgent.

Provides a common ``MemoryClientBase`` interface and multiple backend
implementations.  ``NullMemoryClient`` is the silent no-op fallback used
when no backend is configured or any backend fails to initialise.

``create_memory_client`` is the factory entry point.  It accepts an
optional ``MemoryConfig`` object to select a specific backend; without
one it auto-detects by looking for ``.memoryhub.yaml`` (backward compat).

The factory **never** raises — the agent always gets a usable client.

Supported backends:
  - ``memoryhub``  — MemoryHub SDK (auto-detected or explicit)
  - ``sqlite``     — Local SQLite with FTS5 (via ``memory_sqlite`` module)
  - ``pgvector``   — PostgreSQL + pgvector (via ``memory_pgvector`` module)
  - ``custom``     — Any ``MemoryClientBase`` subclass at a dotted import path
  - ``null``       — Explicitly disabled
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fipsagents.baseagent.config import MemoryConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol — the interface both clients satisfy
# ---------------------------------------------------------------------------


class MemoryClientBase:
    """Base class defining the memory client interface.

    Both ``MemoryClient`` and ``NullMemoryClient`` expose these async
    methods so agent code never needs to check which implementation it has.
    """

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search memories by query string."""
        raise NotImplementedError

    async def write(self, content: str, **kwargs: Any) -> dict[str, Any] | None:
        """Write a new memory entry."""
        raise NotImplementedError

    async def update(
        self, memory_id: str, content: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Update an existing memory entry."""
        raise NotImplementedError

    async def report_contradiction(
        self, memory_id: str, description: str
    ) -> None:
        """Report a contradiction against an existing memory."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Null implementation — the "off" state
# ---------------------------------------------------------------------------


class NullMemoryClient(MemoryClientBase):
    """No-op memory client returned when MemoryHub is not configured.

    Every method succeeds silently and returns empty results, so agent
    code can call memory operations without guarding on configuration.
    """

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def write(self, content: str, **kwargs: Any) -> dict[str, Any] | None:
        return None

    async def update(
        self, memory_id: str, content: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None

    async def report_contradiction(
        self, memory_id: str, description: str
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Real implementation — wraps the MemoryHub SDK
# ---------------------------------------------------------------------------


class MemoryClient(MemoryClientBase):
    """Async wrapper around the MemoryHub SDK.

    Instantiated only when ``.memoryhub.yaml`` exists and the ``memoryhub``
    package is importable.  All methods catch SDK/network errors and degrade
    gracefully (log + return empty) so a flaky MemoryHub server never
    crashes the agent.

    Parameters
    ----------
    sdk:
        An initialised MemoryHub SDK client instance (the object returned
        by ``memoryhub.MemoryHubClient(...)`` or equivalent).
    """

    def __init__(self, sdk: Any) -> None:
        self._sdk = sdk

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        try:
            # SDK v0.5.0 uses .search(); earlier versions used .search_memory()
            _search = getattr(self._sdk, "search", None) or self._sdk.search_memory
            result = await _search(query=query, **kwargs)
            if isinstance(result, list):
                return result
            # SDK v0.5.0 returns SearchResult with .results attribute
            results = getattr(result, "results", None)
            if results is not None:
                # Convert Pydantic models to dicts if needed
                return [
                    r.model_dump() if hasattr(r, "model_dump") else r
                    for r in results
                ]
            # Fallback for older SDK versions
            return getattr(result, "memories", [])
        except Exception:
            logger.warning(
                "MemoryHub search failed for query %r — returning empty results",
                query,
                exc_info=True,
            )
            return []

    async def write(self, content: str, **kwargs: Any) -> dict[str, Any] | None:
        try:
            # SDK v0.5.0 uses .write(); earlier versions used .write_memory()
            _write = getattr(self._sdk, "write", None) or self._sdk.write_memory
            result = await _write(content=content, **kwargs)
            if isinstance(result, dict):
                return result
            # SDK v0.5.0 returns WriteResult Pydantic model
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return None
        except Exception:
            logger.warning(
                "MemoryHub write failed — memory not persisted",
                exc_info=True,
            )
            return None

    async def update(
        self, memory_id: str, content: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        try:
            # SDK v0.5.0 uses .update(); earlier versions used .update_memory()
            _update = getattr(self._sdk, "update", None) or self._sdk.update_memory
            result = await _update(memory_id=memory_id, content=content, **kwargs)
            return result if isinstance(result, dict) else None
        except Exception:
            logger.warning(
                "MemoryHub update failed for memory %s — not persisted",
                memory_id,
                exc_info=True,
            )
            return None

    async def report_contradiction(
        self, memory_id: str, description: str
    ) -> None:
        try:
            await self._sdk.report_contradiction(
                memory_id=memory_id, description=description
            )
        except Exception:
            logger.warning(
                "MemoryHub report_contradiction failed for memory %s",
                memory_id,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def create_memory_client(
    config_path: str | Path = ".memoryhub.yaml",
    *,
    config: MemoryConfig | None = None,
) -> MemoryClientBase:
    """Create the appropriate memory client based on configuration.

    When *config* is provided and ``config.backend`` is set, dispatches
    directly to the named backend.  Otherwise falls back to auto-detection
    via the ``.memoryhub.yaml`` file (backward compatible).

    This function **never** raises — the agent always gets a usable client.

    Parameters
    ----------
    config_path:
        Path to the ``.memoryhub.yaml`` file.  Used only when *config* is
        not provided (legacy call sites).
    config:
        Optional ``MemoryConfig`` from ``agent.yaml``.  When present,
        ``config.backend`` and ``config.config_path`` drive selection.

    Returns
    -------
    MemoryClientBase:
        A live backend client or ``NullMemoryClient``.
    """
    # Resolve effective backend and config path.
    backend = config.backend if config else None
    effective_path = Path(config.config_path if config else config_path)

    # Explicit dispatch when backend is set.
    if backend == "null":
        logger.debug("Memory backend explicitly set to 'null' — disabled")
        return NullMemoryClient()

    if backend == "memoryhub":
        return await _create_memoryhub_client(effective_path)

    if backend == "sqlite":
        return await _create_sqlite_client(effective_path)

    if backend == "pgvector":
        return await _create_pgvector_client(effective_path)

    if backend == "custom":
        if not config or not config.backend_class:
            logger.error(
                "Memory backend is 'custom' but no backend_class specified "
                "in memory config"
            )
            return NullMemoryClient()
        return await _create_custom_client(config.backend_class)

    # No explicit backend — auto-detect (backward compat).
    return await _create_memoryhub_client(effective_path)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


async def _create_memoryhub_client(path: Path) -> MemoryClientBase:
    """Create a MemoryHub-backed memory client."""
    if not path.exists():
        logger.debug(
            "No MemoryHub config at %s — memory integration disabled", path
        )
        return NullMemoryClient()

    # Lazy import — memoryhub is an optional dependency.
    try:
        import memoryhub  # noqa: F811
    except ImportError:
        logger.warning(
            "memoryhub package is not installed but .memoryhub.yaml exists at "
            "%s — falling back to NullMemoryClient.  Install with: "
            "pip install memoryhub",
            path,
        )
        return NullMemoryClient()

    try:
        import yaml

        raw = path.read_text(encoding="utf-8")
        hub_config = yaml.safe_load(raw) or {}

        # Read the API key from the conventional location if not in config.
        api_key = hub_config.get("api_key")
        if not api_key:
            key_path = Path.home() / ".config" / "memoryhub" / "api-key"
            if key_path.exists():
                api_key = key_path.read_text(encoding="utf-8").strip()

        # Build the SDK client — exact kwargs depend on the memoryhub SDK.
        sdk_kwargs: dict[str, Any] = {}
        if api_key:
            sdk_kwargs["api_key"] = api_key
        server_url = hub_config.get("server_url") or hub_config.get("url")
        if server_url:
            sdk_kwargs["server_url"] = server_url

        sdk = memoryhub.MemoryHubClient(**sdk_kwargs)

        # SDK v0.5.0 registers via __aenter__ (auto-calls register_session).
        # Older SDKs may expose register_session directly.
        if hasattr(sdk, "__aenter__"):
            await sdk.__aenter__()
        elif hasattr(sdk, "register_session"):
            await sdk.register_session(api_key=api_key)

        logger.info("MemoryHub integration enabled (config: %s)", path)
        return MemoryClient(sdk=sdk)

    except Exception:
        logger.warning(
            "Failed to initialise MemoryHub from %s — falling back to "
            "NullMemoryClient.  The agent will run without memory.",
            path,
            exc_info=True,
        )
        return NullMemoryClient()


async def _create_sqlite_client(config_path: Path) -> MemoryClientBase:
    """Create a SQLite-backed memory client."""
    try:
        from fipsagents.baseagent.memory_sqlite import create_sqlite_client

        return await create_sqlite_client(config_path)
    except ImportError:
        logger.error(
            "SQLite memory backend requested but memory_sqlite module "
            "not found — falling back to NullMemoryClient"
        )
        return NullMemoryClient()


async def _create_pgvector_client(config_path: Path) -> MemoryClientBase:
    """Create a PGVector-backed memory client."""
    try:
        from fipsagents.baseagent.memory_pgvector import create_pgvector_client

        return await create_pgvector_client(config_path)
    except ImportError:
        logger.error(
            "PGVector memory backend requested but memory_pgvector module "
            "not found — falling back to NullMemoryClient. "
            "Install with: pip install fipsagents[pgvector]"
        )
        return NullMemoryClient()


async def _create_custom_client(dotted_path: str) -> MemoryClientBase:
    """Import and instantiate a custom MemoryClientBase subclass."""
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if not (isinstance(cls, type) and issubclass(cls, MemoryClientBase)):
            logger.error(
                "Custom memory backend class %s is not a MemoryClientBase "
                "subclass — falling back to NullMemoryClient",
                dotted_path,
            )
            return NullMemoryClient()
        instance = cls()
        # If the custom class has an async setup method, call it.
        if hasattr(instance, "setup") and callable(instance.setup):
            await instance.setup()
        return instance
    except Exception:
        logger.warning(
            "Failed to load custom memory backend from %s — "
            "falling back to NullMemoryClient",
            dotted_path,
            exc_info=True,
        )
        return NullMemoryClient()
