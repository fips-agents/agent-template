"""Message compaction backends.

Compactors reduce the length of a conversation's message history by
summarising or pruning older turns.  The server invokes the compactor
before the model call when the message list exceeds a configured
threshold.  ``NullCompactor`` (default) is a no-op -- fully
backward-compatible.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompactionState:
    """Serialisable state tracking compaction history for a session."""

    last_compacted_at: str | None = None
    last_compacted_message_id: str | None = None
    compaction_count: int = 0


@dataclass
class CompactionResult:
    """Result of a compaction attempt."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    original_count: int = 0
    compacted_count: int = 0
    skipped: bool = False
    skip_reason: str | None = None


class Compactor(ABC):
    """Pluggable message compaction backend."""

    @abstractmethod
    async def should_compact(
        self,
        messages: list[dict[str, Any]],
        *,
        state: CompactionState | None = None,
    ) -> bool:
        """Return True if the message list should be compacted."""

    @abstractmethod
    async def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        state: CompactionState | None = None,
    ) -> CompactionResult:
        """Compact the message list. Must preserve ``id`` fields on
        surviving messages."""

    async def close(self) -> None:
        """Release resources. Default no-op."""


class NullCompactor(Compactor):
    """No compaction -- messages pass through unchanged."""

    async def should_compact(
        self,
        messages: list[dict[str, Any]],
        *,
        state: CompactionState | None = None,
    ) -> bool:
        return False

    async def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        state: CompactionState | None = None,
    ) -> CompactionResult:
        return CompactionResult(
            messages=messages,
            original_count=len(messages),
            compacted_count=len(messages),
            skipped=True,
            skip_reason="null_compactor",
        )


def create_compactor(
    backend: str | None = None,
    **kwargs: Any,
) -> Compactor:
    """Create a compactor from config. Only ``null`` for now; #166 adds ``llm``."""
    if backend is None or backend == "null":
        return NullCompactor()
    raise ValueError(f"Unknown compactor backend: {backend!r}")
