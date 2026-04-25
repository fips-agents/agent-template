"""Adapter configuration loaded from environment variables."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, field_validator


class AdapterConfig(BaseModel):
    """Runtime configuration for the LLM adapter sidecar.

    All values are read from environment variables via the ``from_env()``
    classmethod.  No pydantic-settings dependency is required.
    """

    provider: Literal["anthropic", "bedrock", "bedrock-converse", "azure", "openai-compat", "ollama"] = "anthropic"
    port: int = 8081
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return upper

    @classmethod
    def from_env(cls) -> AdapterConfig:
        """Build a config instance from ``ADAPTER_*`` / ``LOG_LEVEL`` env vars."""
        return cls(
            provider=os.environ.get("ADAPTER_PROVIDER", "anthropic"),
            port=int(os.environ.get("ADAPTER_PORT", "8081")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
