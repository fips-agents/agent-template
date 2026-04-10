"""Optional MemoryHub integration for BaseAgent.

Provides ``MemoryClient`` for programmatic read/write of agent memories
through the MemoryHub SDK, and ``NullMemoryClient`` as a silent no-op
fallback when MemoryHub is not configured or unavailable.

The ``create_memory_client`` factory handles detection, lazy import, and
graceful degradation so that agent code can unconditionally call
``self.memory.search(...)`` without caring whether MemoryHub is active.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
            result = await self._sdk.search_memory(query=query, **kwargs)
            if isinstance(result, list):
                return result
            # Some SDK versions return a wrapper object with a .memories attr
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
            result = await self._sdk.write_memory(content=content, **kwargs)
            return result if isinstance(result, dict) else None
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
            result = await self._sdk.update_memory(
                memory_id=memory_id, content=content, **kwargs
            )
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
) -> MemoryClientBase:
    """Create the appropriate memory client based on configuration.

    Detection logic:

    1. If *config_path* does not exist on disk, return ``NullMemoryClient``.
    2. Try to ``import memoryhub``.  If the package is missing, log a
       warning and return ``NullMemoryClient``.
    3. Read the YAML config, build a real SDK client, and return a
       ``MemoryClient`` wrapper.  If anything fails during SDK
       initialisation (bad config, unreachable server, etc.), log a
       warning and return ``NullMemoryClient``.

    This function **never** raises — the agent always gets a usable client.

    Parameters
    ----------
    config_path:
        Path to the ``.memoryhub.yaml`` file.  Defaults to the
        conventional location at the project root.

    Returns
    -------
    MemoryClientBase:
        Either a live ``MemoryClient`` or a ``NullMemoryClient``.
    """
    path = Path(config_path)

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
        config = yaml.safe_load(raw) or {}

        # Read the API key from the conventional location if not in config
        api_key = config.get("api_key")
        if not api_key:
            key_path = Path.home() / ".config" / "memoryhub" / "api-key"
            if key_path.exists():
                api_key = key_path.read_text(encoding="utf-8").strip()

        # Build the SDK client — exact kwargs depend on the memoryhub SDK
        sdk_kwargs: dict[str, Any] = {}
        if api_key:
            sdk_kwargs["api_key"] = api_key
        server_url = config.get("server_url") or config.get("url")
        if server_url:
            sdk_kwargs["server_url"] = server_url

        sdk = memoryhub.MemoryHubClient(**sdk_kwargs)

        # If the SDK supports async session registration, do it now.
        # register_session takes only api_key per the MemoryHub loading rule.
        if hasattr(sdk, "register_session"):
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
