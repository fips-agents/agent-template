"""Provider registry with auto-import of built-in backends."""

from __future__ import annotations

from llm_adapter.providers.base import BaseProvider

_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a provider class under the given name."""
    _REGISTRY[name] = cls


def get_provider(name: str) -> BaseProvider:
    """Instantiate and return a provider by name.

    Raises ``ValueError`` if the name is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY) or ["(none registered)"]
        raise ValueError(f"Unknown provider: {name!r}. Available: {available}")
    return _REGISTRY[name]()


# Auto-import providers to trigger registration.
from llm_adapter.providers import anthropic as _anthropic  # noqa: F401, E402
