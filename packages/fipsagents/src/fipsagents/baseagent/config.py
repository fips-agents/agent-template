"""Agent configuration with YAML parsing and environment variable substitution.

Loads ``agent.yaml``, resolves ``${VAR:-default}`` placeholders against the
current environment, and validates the result into typed Pydantic models.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""


# ---------------------------------------------------------------------------
# Environment variable substitution
# ---------------------------------------------------------------------------

# Matches ${VAR}, ${VAR:-default}, or ${VAR-default}
_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::?-(?P<default>[^}]*))?\}"
)


def substitute_env_vars(
    value: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> str:
    """Replace ``${VAR:-default}`` tokens in *value* with environment values.

    Parameters
    ----------
    value:
        The string that may contain ``${VAR:-default}`` placeholders.
    env:
        Environment mapping.  Defaults to ``os.environ``.
    strict:
        When *True*, raise ``ConfigError`` for any variable that has neither
        an environment value nor a default.  When *False* (the default), the
        raw placeholder is left in place so it surfaces clearly in logs.
    """
    env = env if env is not None else os.environ

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        result = env.get(name)
        if result is not None:
            return result
        if default is not None:
            return default
        if strict:
            raise ConfigError(
                f"Environment variable ${{{name}}} is required but not set "
                f"and has no default value"
            )
        return match.group(0)  # leave placeholder intact

    return _ENV_PATTERN.sub(_replace, value)


def _substitute_recursive(
    obj: Any,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> Any:
    """Walk an arbitrary structure and substitute env vars in all strings."""
    if isinstance(obj, str):
        return substitute_env_vars(obj, env=env, strict=strict)
    if isinstance(obj, dict):
        return {
            k: _substitute_recursive(v, env=env, strict=strict)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_substitute_recursive(v, env=env, strict=strict) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """LLM provider and generation settings."""

    endpoint: str | None = None
    name: str = "meta-llama/Llama-3.3-70B-Instruct"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class McpServerConfig(BaseModel):
    """Connection details for a single MCP server.

    Exactly one transport must be specified:

    - **HTTP** (streamable-http): set ``url``.
    - **stdio** (subprocess): set ``command`` (and optionally ``args``,
      ``env``, ``cwd``).
    """

    # HTTP transport
    url: str | None = None

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None

    @model_validator(mode="after")
    def _require_exactly_one_transport(self) -> "McpServerConfig":
        has_url = self.url is not None
        has_command = self.command is not None
        if not has_url and not has_command:
            raise ValueError(
                "McpServerConfig requires either 'url' (HTTP) or "
                "'command' (stdio), got neither"
            )
        if has_url and has_command:
            raise ValueError(
                "McpServerConfig cannot have both 'url' and 'command' "
                "— pick one transport"
            )
        return self


class ToolsConfig(BaseModel):
    """Settings for local tool discovery."""

    local_dir: str = "./tools"
    visibility_default: Literal["agent_only", "llm_only", "both"] = "agent_only"


class PromptsConfig(BaseModel):
    """Settings for prompt template discovery."""

    dir: str = "./prompts"


class BackoffConfig(BaseModel):
    """Exponential backoff parameters for the agent loop."""

    initial: float = Field(default=1.0, gt=0.0)
    max: float = Field(default=30.0, gt=0.0)
    multiplier: float = Field(default=2.0, gt=1.0)

    @model_validator(mode="after")
    def _max_ge_initial(self) -> "BackoffConfig":
        if self.max < self.initial:
            raise ValueError(
                f"backoff.max ({self.max}) must be >= backoff.initial ({self.initial})"
            )
        return self


class LoopConfig(BaseModel):
    """Agent loop execution parameters."""

    max_iterations: int = Field(default=100, gt=0)
    backoff: BackoffConfig = Field(default_factory=BackoffConfig)

    @field_validator("max_iterations", mode="before")
    @classmethod
    def _coerce_max_iterations(cls, v: Any) -> Any:
        """Allow ``max_iterations`` to arrive as a string (from env var substitution)."""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(
                    f"loop.max_iterations must be an integer, got '{v}'"
                ) from None
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(
                f"logging.level must be one of {sorted(allowed)}, got '{v}'"
            )
        return upper


class MemoryConfig(BaseModel):
    """Memory backend settings.

    Controls which memory backend the agent uses.  When ``backend`` is
    unset (the default), the factory auto-detects by looking for
    ``.memoryhub.yaml`` — preserving backward compatibility.

    Supported backends:
      - ``memoryhub`` — MemoryHub SDK (requires ``memoryhub`` package)
      - ``markdown``  — Human-readable markdown file(s) (zero dependencies)
      - ``sqlite``    — Local SQLite with FTS5 (zero dependencies)
      - ``pgvector``  — PostgreSQL + pgvector (requires ``asyncpg``)
      - ``custom``    — Bring your own: set ``backend_class`` to a dotted
                        import path for a ``MemoryClientBase`` subclass
      - ``null``      — Explicitly disable memory

    Prefix injection:
      - ``prefix_role``      — Role for the memory prefix message: ``system``
                               (default, universal) or ``developer``
                               (harmony-format models like gpt-oss).
      - ``max_prefix_chars`` — Maximum character length for the memory prefix.
                               Prevents large backends from dumping their
                               entire store.  0 disables the limit.
    """

    backend: Literal["memoryhub", "markdown", "sqlite", "pgvector", "llamastack", "custom", "null"] | None = None
    config_path: str = ".memoryhub.yaml"
    backend_class: str | None = None
    prefix_role: Literal["system", "developer"] = "system"
    max_prefix_chars: int = 8000

    @field_validator("backend", mode="before")
    @classmethod
    def _coerce_empty_backend(cls, v: Any) -> Any:
        """Coerce empty strings to None (from ``${MEMORY_BACKEND:-}``)."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class ToolInspectionConfig(BaseModel):
    """Tool call inspection settings."""

    enabled: bool = True
    mode: Literal["enforce", "observe"] | None = None  # None = inherit from security.mode


class GuardrailsConfig(BaseModel):
    """Code execution guardrails settings."""

    mode: Literal["enforce", "observe"] | None = None


class SecurityConfig(BaseModel):
    """Security settings controlling inspection and audit behavior."""

    mode: Literal["enforce", "observe"] = "enforce"
    tool_inspection: ToolInspectionConfig = Field(
        default_factory=ToolInspectionConfig
    )
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)


class NodeConfig(BaseModel):
    """Configuration for a single workflow node's deployment topology."""

    type: Literal["local", "remote"] = "local"
    endpoint: str | None = None
    path: str = "/process"
    timeout: float = 30.0
    retries: int = 2

    @model_validator(mode="after")
    def _validate_remote_has_endpoint(self) -> "NodeConfig":
        if self.type == "remote" and not self.endpoint:
            raise ValueError("Remote nodes require an 'endpoint'")
        return self


class AgentConfig(BaseModel):
    """Top-level agent configuration, corresponding to ``agent.yaml``."""

    model: LLMConfig = Field(default_factory=LLMConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    nodes: dict[str, NodeConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def parse_yaml_with_env(
    raw: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Parse a YAML string after resolving ``${VAR:-default}`` placeholders.

    Parameters
    ----------
    raw:
        Raw YAML content (may contain env var placeholders).
    env:
        Custom environment mapping.  Defaults to ``os.environ``.
    strict:
        Raise on unresolved variables that have no default.

    Returns
    -------
    dict:
        The parsed, substituted YAML as a plain dictionary.

    Raises
    ------
    ConfigError:
        On YAML syntax errors or (when *strict*) unresolved variables.
    """
    substituted = substitute_env_vars(raw, env=env, strict=strict)
    try:
        data = yaml.safe_load(substituted)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in agent config: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"agent.yaml must be a YAML mapping at the top level, "
            f"got {type(data).__name__}"
        )
    return data


def load_config(
    path: str | Path = "agent.yaml",
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> AgentConfig:
    """Load and validate agent configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    env:
        Custom environment mapping.  Defaults to ``os.environ``.
    strict:
        Raise on unresolved environment variables that lack defaults.

    Returns
    -------
    AgentConfig:
        Fully validated configuration.

    Raises
    ------
    ConfigError:
        When the file cannot be read, the YAML is invalid, or
        validation fails.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise ConfigError(
            f"Configuration file not found: {filepath.resolve()}\n"
            f"Create an agent.yaml or pass an explicit path to load_config()."
        )
    try:
        raw = filepath.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read {filepath}: {exc}") from exc

    data = parse_yaml_with_env(raw, env=env, strict=strict)

    try:
        return AgentConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"Invalid agent configuration: {exc}") from exc


def load_config_from_string(
    raw: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
) -> AgentConfig:
    """Load and validate agent configuration from a YAML string.

    Useful for testing or when the config is assembled programmatically.
    """
    data = parse_yaml_with_env(raw, env=env, strict=strict)
    try:
        return AgentConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"Invalid agent configuration: {exc}") from exc
