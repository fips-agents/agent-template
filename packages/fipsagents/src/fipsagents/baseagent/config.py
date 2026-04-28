"""Agent configuration with YAML parsing and environment variable substitution.

Loads ``agent.yaml``, resolves ``${VAR:-default}`` placeholders against the
current environment, and validates the result into typed Pydantic models.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

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
# Adapter sidecar constants
# ---------------------------------------------------------------------------

_ADAPTER_PORT: int = 8081
_ADAPTER_ENDPOINT: str = f"http://localhost:{_ADAPTER_PORT}/v1"
_OFF_PLATFORM_PROVIDERS: frozenset[str] = frozenset({"anthropic", "bedrock", "azure"})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """LLM provider and generation settings."""

    provider: Literal["openai", "anthropic", "bedrock", "azure"] = "openai"
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
    system: str = "system"


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
      - ``injection_mode``   — Where to place retrieved memories:
                               ``prefix`` (default) inserts a separate message
                               before the user turn.  ``user_turn`` appends
                               memories to the user message inside XML tags,
                               which small models (8K-16K) treat as
                               higher-salience context.
      - ``injection_tag``    — XML tag name wrapping user-turn memories
                               (default ``user_memories``).  Only used when
                               ``injection_mode`` is ``user_turn``.

    Budget presets:
      - ``budget``           — Shorthand that sets defaults for
                               ``max_prefix_chars``, ``max_results``, and
                               ``min_weight`` based on model tier:

                               =======  ================  ===========  ==========
                               Budget   max_prefix_chars  max_results  min_weight
                               =======  ================  ===========  ==========
                               small    500               5            0.7
                               medium   4000              20           0.5
                               large    8000              50           0.3
                               =======  ================  ===========  ==========

                               Explicit field values always override the preset.
                               ``custom`` and ``None`` use field defaults.
      - ``max_results``      — Maximum number of memories to retrieve.
      - ``min_weight``       — Minimum weight threshold for retrieved memories.
                               Results below this weight are filtered out.

    Loading:
      - ``loading_pattern``  — When to retrieve memories.  ``eager``
                               (default when unset) loads at setup time.
                               ``lazy``, ``lazy_with_rebias``, and ``jit``
                               defer to after the first user message.
                               When set, overrides the pattern from
                               ``.memoryhub.yaml``.  Required for
                               file-based backends that want deferred loading.
    """

    _BUDGET_PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "small": {"max_prefix_chars": 500, "max_results": 5, "min_weight": 0.7},
        "medium": {"max_prefix_chars": 4000, "max_results": 20, "min_weight": 0.5},
        "large": {"max_prefix_chars": 8000, "max_results": 50, "min_weight": 0.3},
    }

    backend: Literal["memoryhub", "markdown", "sqlite", "pgvector", "llamastack", "custom", "null"] | None = None
    config_path: str = ".memoryhub.yaml"
    backend_class: str | None = None
    prefix_role: Literal["system", "developer"] = "system"
    max_prefix_chars: int = 8000
    injection_mode: Literal["prefix", "user_turn"] = "prefix"
    injection_tag: str = "user_memories"
    budget: Literal["small", "medium", "large", "custom"] | None = None
    max_results: int = 50
    min_weight: float = 0.0
    loading_pattern: Literal["eager", "lazy", "lazy_with_rebias", "jit"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_budget_presets(cls, data: Any) -> Any:
        """Fill in budget-controlled fields that the user didn't set."""
        if not isinstance(data, dict):
            return data
        budget = data.get("budget")
        if budget and budget in cls._BUDGET_PRESETS:
            for key, val in cls._BUDGET_PRESETS[budget].items():
                data.setdefault(key, val)
        return data

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


class AgentIdentity(BaseModel):
    """Agent name, description, and version for logging and API endpoints."""

    name: str = "agent"
    description: str = ""
    version: str = "0.1.0"


class StorageConfig(BaseModel):
    """Shared storage backend for sessions and traces.

    When ``backend`` is ``null`` (default), no persistence — features
    degrade gracefully to no-ops. ``sqlite`` uses a single file for
    both sessions and traces. ``postgres`` uses a shared connection pool.
    ``http`` delegates to a sibling ``fipsagents-platform`` service over
    REST; ``platform_url`` is required and ``platform_token`` is an
    optional static bearer token for service-to-service flows
    (per-request tokens forwarded from the inbound ``Authorization``
    header take precedence when present).
    """

    backend: Literal["sqlite", "postgres", "http"] | None = None
    sqlite_path: str = "./agent.db"
    database_url: str = ""
    platform_url: str = ""
    platform_token: str = ""

    @field_validator("backend", mode="before")
    @classmethod
    def _coerce_empty_backend(cls, v: Any) -> Any:
        """Coerce empty strings to None (from ``${STORAGE_BACKEND:-}``)."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("database_url", mode="before")
    @classmethod
    def _coerce_empty_url(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            return ""
        return v


class _PerStoreBackendMixin(BaseModel):
    """Per-store override for ``StorageConfig.backend``.

    When ``None`` the store inherits ``storage.backend``.  Allows mixing
    backends — eg ``feedback.backend: http`` while sessions/traces stay
    on local SQLite.
    """

    backend: Literal["sqlite", "postgres", "http"] | None = None

    @field_validator("backend", mode="before")
    @classmethod
    def _coerce_empty_backend(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class SessionsConfig(_PerStoreBackendMixin):
    """Session persistence settings."""

    enabled: bool = False
    max_age_hours: int = Field(default=168, ge=0)


class TracesConfig(_PerStoreBackendMixin):
    """Trace collection settings."""

    enabled: bool = False
    max_age_hours: int = Field(default=168, ge=0)
    sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    exporter: Literal["store", "otel"] | None = None
    otel_endpoint: str | None = None
    service_name: str = "fipsagents"


class MetricsConfig(BaseModel):
    """Prometheus metrics settings.

    ``token_label_mode`` controls the label cardinality of
    ``agent_tokens_total``.  Each step up adds one dimension at the cost
    of more time-series stored by Prometheus.

    - ``model`` (default) — current behaviour, only ``model`` and
      ``direction`` labels.  Bounded by the model catalog.
    - ``tenant`` — also adds ``tenant_id`` (typically gateway-stamped via
      the ``X-Tenant`` header). Bounded by the tenant count, suitable for
      most enterprise deployments.
    - ``session`` — also adds ``session_id``.  **High cardinality**: one
      time-series per session per direction per model.  Only enable when
      you have an external aggregation step (eg Prometheus federation,
      Mimir) that can absorb the volume; otherwise prefer
      ``GET /v1/sessions/{id}/usage`` for per-session totals.
    """

    enabled: bool = False
    token_label_mode: Literal["model", "tenant", "session"] = "model"


class FeedbackConfig(_PerStoreBackendMixin):
    """Feedback collection settings."""

    enabled: bool = False
    max_age_hours: int = Field(default=720, ge=0)


class ScannerConfig(BaseModel):
    """Virus-scanning sidecar settings.

    The scanner runs between MIME sniffing and parsing on every
    upload. It speaks HTTP to a sidecar that wraps ClamAV (or any
    other engine that matches the contract): POST the file bytes,
    expect a JSON body of ``{"infected": bool, "viruses": [str]}`` or
    a 200/422 status.

    ``fail_mode`` controls behavior when the scanner sidecar is
    unreachable or errors:

    - ``open`` (default) — accept the upload and log a warning. Right
      for non-production / dev environments where occasional sidecar
      hiccups should not break the API.
    - ``closed`` — reject the upload with HTTP 503. Right for
      production where every file must be scanned before storage.

    When ``url`` is empty (default), no scanner is configured and the
    upload path runs without virus checks.
    """

    url: str = ""
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    fail_mode: Literal["open", "closed"] = "open"


class FilesConfig(_PerStoreBackendMixin):
    """File upload settings.

    ``bytes_dir`` is only used by ``SqliteFileStore`` for local-FS bytes
    storage in dev mode; production deployments using S3-compatible bytes
    storage will ignore it. ``allowed_mime_types`` is enforced by the
    ``POST /v1/files`` endpoint when present (an empty list disables the
    allowlist).
    """

    enabled: bool = False
    max_file_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    bytes_dir: str = "./files"
    allowed_mime_types: list[str] = Field(default_factory=list)
    max_age_hours: int = Field(default=720, ge=0)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)


class PricingRate(BaseModel):
    """Per-token / per-request pricing for a single model.

    All rates are USD. Token rates are quoted per 1,000 tokens to match
    public model-pricing tables (OpenAI, Anthropic, Bedrock). For
    self-hosted vLLM deployments without dollar billing, leave the
    defaults at zero -- :func:`fipsagents.server.pricing.compute_cost`
    will return ``0.0`` and the new ``/usage`` endpoint will surface a
    no-op cost line.

    ``cached_input_per_1k`` covers prompt-cache hits when the provider
    returns a ``prompt_tokens_details.cached_tokens`` count; OpenAI's
    semantics treat cached tokens as a subset of ``prompt_tokens`` billed
    at a reduced rate. ``None`` means "no cached-tier discount" and the
    full ``input_per_1k`` rate applies.
    """

    input_per_1k: float = Field(default=0.0, ge=0.0)
    output_per_1k: float = Field(default=0.0, ge=0.0)
    cached_input_per_1k: float | None = Field(default=None, ge=0.0)
    per_request: float = Field(default=0.0, ge=0.0)


class PricingConfig(BaseModel):
    """Token cost lookup table.

    ``default`` applies to any model not listed in ``models``. ``models``
    keys must match the model identifier exactly as it appears on
    completion requests (typically ``model.name`` from ``agent.yaml``).
    """

    default: PricingRate = Field(default_factory=PricingRate)
    models: dict[str, PricingRate] = Field(default_factory=dict)


class BudgetLimits(BaseModel):
    """Soft (warn) and hard (enforce) USD limits.

    ``warn_usd`` is logged when the running total crosses it. ``limit_usd``
    triggers :class:`fipsagents.server.budget.BudgetExceededError` (HTTP 402)
    when ``budget.mode`` is ``enforce``. Setting either to ``0`` (the default)
    disables that threshold.
    """

    warn_usd: float = Field(default=0.0, ge=0.0)
    limit_usd: float = Field(default=0.0, ge=0.0)


class BudgetConfig(BaseModel):
    """Per-session and per-tenant cost budgets.

    Per-session budgets read cumulative ``cost_data`` from the session
    store and convert to USD via :class:`PricingConfig`.  Per-tenant
    budgets aggregate session deltas in-process — accurate for
    single-replica deployments and for "this agent's view" of cross-session
    tenant cost.  Multi-replica tenant aggregation requires a separate
    cross-agent service and is out of scope here.

    ``mode``:

    - ``observe`` — log warnings + limit crossings, never raise.
    - ``enforce`` (default) — raise ``BudgetExceededError`` on hard limit.
    """

    mode: Literal["observe", "enforce"] = "enforce"
    per_session: BudgetLimits = Field(default_factory=BudgetLimits)
    per_tenant: BudgetLimits = Field(default_factory=BudgetLimits)

    def is_active(self) -> bool:
        """True if any limit is configured (warn or hard, session or tenant)."""
        return any(
            v > 0.0 for v in (
                self.per_session.warn_usd,
                self.per_session.limit_usd,
                self.per_tenant.warn_usd,
                self.per_tenant.limit_usd,
            )
        )


class ServerConfig(BaseModel):
    """HTTP server binding and feature configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8080, gt=0, le=65535)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    traces: TracesConfig = Field(default_factory=TracesConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @field_validator("port", mode="before")
    @classmethod
    def _coerce_port(cls, v: Any) -> Any:
        """Coerce string port values from env-var substitution."""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(
                    f"server.port must be an integer, got '{v}'"
                ) from None
        return v


class AgentConfig(BaseModel):
    """Top-level agent configuration, corresponding to ``agent.yaml``."""

    agent: AgentIdentity = Field(default_factory=AgentIdentity)
    model: LLMConfig = Field(default_factory=LLMConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
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
