# CLAUDE.md

This is the agent-template project -- a monorepo of agent templates for the `fips-agents` CLI. It scaffolds production-ready AI agents for OpenShift.

## Project Status

Both templates are implemented. The agent-loop template (`templates/agent-loop/`) handles single-agent loops. The workflow template (`templates/workflow/`) handles multi-node directed graphs with typed state. BaseAgent is extracted into a shared pip-installable package at `packages/fipsagents/`.

## Key Documents

Read these before making any architectural decisions:

- `docs/architecture.md` -- The authoritative design document. Covers BaseAgent, tool planes, skills, config, deployment, MemoryHub integration. All decisions here are final unless explicitly changed.
- `planning/requirements.md` -- What the system must do.
- `planning/scope.md` -- What is and is not in scope.
- `planning/constraints.md` -- Non-negotiable technical constraints.

## Architecture Decisions (Quick Reference)

These are settled. Do not revisit without explicit discussion.

- **BaseAgent** is pure Python, async throughout, no framework dependencies (no LangChain, no LangGraph)
- **openai** (async SDK) is the LLM client -- connects to any OpenAI-compatible endpoint (vLLM, LlamaStack, llm-d)
- **LLM adapter** at `packages/llm-adapter/` is a sidecar FastAPI service translating OpenAI-compatible requests to 8 provider APIs (Anthropic, Bedrock, Bedrock Converse, Azure OpenAI, OpenAI-compatible, Ollama, llama.cpp, Vertex AI/Gemini). Scaffolded as source code into the project (not a pip dependency). Packaging decision: stays as a scaffolded sidecar (Option B). If we ever publish to PyPI as `fipsagents-llm-adapter` (Option A), the scaffolded-source path must remain first-class -- same dual-path model as BaseAgent. Rationale: fips-agents is building blocks, not a framework; scaffolded source minimizes dependency surface and gives developers full control.
- **FastMCP v3** is the MCP client -- not v2
- **Two tool planes**: agent-code tools (plane 1, invisible to LLM) and LLM-callable tools (plane 2). Both go through BaseAgent for logging/RBAC/retry. Visibility per tool: `agent_only`, `llm_only`, `both`.
- **@tool decorator** for local tools, same convention as FastMCP. Auto-discovered from `tools/` directory.
- **Prompts** are Markdown with YAML frontmatter, one file per prompt in `prompts/`
- **Skills** follow the agentskills.io spec exactly -- directory per skill, SKILL.md with frontmatter, progressive disclosure
- **Rules** are plain Markdown files in `rules/`, no frontmatter
- **agent.yaml** with `${VAR:-default}` env var substitution for configuration
- **Immutable container images** -- code, tools, prompts, skills, rules all baked in. Only env-specific config is external.
- **Pluggable memory backends** -- memoryhub, markdown, sqlite, pgvector, llamastack, custom, or null. `self.memory` is always a `MemoryClientBase` (never None). MemoryHub adds MCP path for LLM-driven memory tools. `build_memory_prefix()` injects a stable memory block at setup time (role configurable via `memory.prefix_role`: `system` or `developer`). `_inject_deferred_memory()` runs at the top of `astep_stream()` for deferred patterns (`lazy`, `lazy_with_rebias`, `jit`) — extracts the last user message as a search query, calls `self.memory.search()`, and injects results. Small models (8K-16K context) ignore system-prompt memories — they need `injection_mode: user_turn` which appends memories to the user message inside `<injection_tag>` XML tags. The server strips echoed tags from non-streaming responses defensively.
- **Memory budget presets** -- `budget: small | medium | large | custom` in `MemoryConfig` sets defaults for `max_prefix_chars`, `max_results`, and `min_weight`. Small = 500 chars / 5 results / min_weight 0.7 (8K-16K models). Medium = 4K / 20 / 0.5 (32K-128K). Large = 8K / 50 / 0.3 (128K+). Explicit field values always override the preset. `_apply_budget_presets` is a `model_validator(mode="before")` using `data.setdefault()`.
- **`loading_pattern`** in `MemoryConfig` -- controls when memories are retrieved: `eager` (setup time, default), `lazy` (after first user message), `lazy_with_rebias`, `jit`. Config-level takes precedence over `.memoryhub.yaml` SDK pattern. Required for file-based backends (markdown, sqlite) that want deferred loading since they have no `project_config`.
- **`astep_stream()` accepts `**model_kwargs`** -- forwarded to `call_model_stream_raw()`. The server's `_extract_overrides()` splits `ChatCompletionRequest` parameters into standard OpenAI params and vLLM-specific `extra_body` params (top_k, repetition_penalty, reasoning_effort).
- **Helm chart** bundles only the agent. Infrastructure (vLLM, LlamaStack, PGVector) is pre-deployed via rh-ai-quickstart/ai-architecture-charts.
- **Red Hat UBI** base images for all containers
- **`call_model_validated()`** is a first-class BaseAgent method -- call model, validate with a tool, retry with backoff
- **fipsagents** is the shared pip-installable package at `packages/fipsagents/`. Both templates depend on it. Import as `from fipsagents.baseagent import BaseAgent`. Workflow classes are also in the package: `from fipsagents.workflow import Graph, WorkflowRunner, BaseNode, AgentNode`.
- **WorkflowNode** protocol (`typing.Protocol`) -- structural subtyping, no inheritance coupling. Both BaseNode and AgentNode satisfy it.
- **BaseNode** for lightweight workflow nodes (routing, gating). **AgentNode** for full-agent workflow nodes (LLM, tools, MCP). **RemoteNode** for nodes that delegate to already-deployed agents via HTTP POST.
- **NodeConfig** in `AgentConfig` maps node names to deployment topology (`local` or `remote`). `WorkflowRunner` auto-wraps remote nodes transparently -- the graph definition stays topology-agnostic.
- **Workflow state** is a typed Pydantic model with `extra="forbid"`. Data only -- execution metadata stays in structured logs.
- **@node decorator** marks classes for workflow registration, mirroring the @tool pattern.
- **SecurityConfig** in `AgentConfig` -- global `mode` (`enforce`/`observe`) with per-layer override (`tool_inspection.mode`, `guardrails.mode`). `ToolInspector` scans tool call arguments for secrets, C2 patterns, and prompt injection before execution. Audit findings log to `fipsagents.security.audit`. Wired in `setup()` step 4b.
- **Session persistence** is server-layer only — `SessionStore` ABC with `NullSessionStore` (default, ephemeral), `SqliteSessionStore` (edge/dev), `PostgresSessionStore` (enterprise). BaseAgent has no concept of sessions; the server handles load-before / save-after around each request. REST endpoints: `POST /v1/sessions`, `GET /v1/sessions/{id}`, `DELETE /v1/sessions/{id}`. Optional `session_id` field on `ChatCompletionRequest`.
- **Tracing** is server-layer only — `TraceCollector` wraps `astep_stream()` as a pure observer, building span trees from `StreamEvent`s without modifying them. `TraceStore` ABC with `NullTraceStore` (structured JSON logging, default), `SqliteTraceStore` (edge/dev), and `PostgresTraceStore` (enterprise). Query endpoints: `GET /v1/traces`, `GET /v1/traces/{id}`. Sampling rate configurable.
- **Shared storage layer** — `ServerConfig` has `StorageConfig` (`backend: null | sqlite | postgres`), `SessionsConfig` (`enabled`, `max_age_hours`), `TracesConfig` (`enabled`, `max_age_hours`, `sampling_rate`, `exporter`, `otel_endpoint`, `service_name`), `MetricsConfig` (`enabled`). Both sessions and traces share the storage backend. When backend is `null`, both features degrade to no-ops (fully backward-compatible). `SqliteConnectionManager` in `sqlite.py` deduplicates connections by resolved path when both stores use SQLite.
- **Prometheus metrics** — `MetricsCollector` in `metrics.py` follows the TraceCollector observer pattern. Records `agent_requests_total`, `agent_request_duration_seconds`, `agent_model_call_duration_seconds`, `agent_tool_call_total`, `agent_tokens_total`. Exposed at `GET /metrics` in Prometheus text format. Optional `[metrics]` extra (`prometheus_client`). `NullMetricsCollector` when disabled.
- **OTEL trace export** — `OTELTraceStore` in `otel.py` wraps an inner `TraceStore` with OpenTelemetry span export via OTLP. Span IDs are deterministically hashed (SHA-256) from internal string IDs. Monotonic-to-wallclock conversion anchored on `Trace.started_at`. Optional `[otel]` extra (`opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`). Configure via `traces.exporter: otel` in agent.yaml.
- **Distributed trace propagation** — W3C Trace Context (`traceparent` header) extracted from incoming requests and injected into outgoing `RemoteNode` HTTP calls. `propagation.py` provides `extract_trace_context()` and `inject_trace_context()`. `TraceCollector` accepts `parent_trace_id`/`parent_span_id` to join distributed traces. `RemoteNode.set_trace_context()` injects headers.
- **Server module structure** — `fipsagents.server` is a proper package: `app.py` (OpenAIChatServer), `models.py` (request/response schemas), `sessions.py` (session stores), `tracing.py` (trace model + stores), `collector.py` (TraceCollector), `metrics.py` (Prometheus metrics), `otel.py` (OTEL export), `propagation.py` (W3C Trace Context), `sqlite.py` (shared connection manager). `__init__.py` re-exports `OpenAIChatServer`, `ChatCompletionRequest`, `ChatMessage`.
- **`probe_role_support()`** is a diagnostic utility in `fipsagents.baseagent.diagnostics` -- probes whether a deployed model supports a given message role (e.g. `developer`). Template inspection (best-effort, checks vLLM model metadata) + canary completion (prompt token delta). Not on the hot path.
- **`ThinkTagParser`** in `fipsagents.baseagent.reasoning` -- streaming parser that separates `<think>…</think>` blocks from content deltas. Auto-enabled for Granite and DeepSeek models (via `create_reasoning_parser(model_name)`). Wired in `setup()` step 11 and `astep_stream`. Falls back gracefully when vLLM's `--reasoning-parser` already handles extraction server-side.
- **`McpServerConfig`** supports two YAML-configurable transports: HTTP (`url`) and stdio (`command`/`args`/`env`/`cwd`). Pydantic validator enforces exactly one. `connect_mcp()` also accepts FastMCP server objects for in-process transport (programmatic, not YAML).
- **`connect_mcp()` discovers all three MCP capability types**: tools (registered in ToolRegistry), prompts (`_mcp_prompts` dict, rendered via `get_mcp_prompt()`), and resources (`_mcp_resources` dict, read via `read_resource()`). Resource templates stored separately in `_mcp_resource_templates`. MCP prompts are kept separate from local prompts (different lifecycle). Resources are agent-plane by default. Resource subscriptions are not implemented.
- **MCP integration test harness** at `packages/fipsagents/tests/integration/mcp/` -- pytest-based, mark-driven (`local_tool`, `mcp_http`, `mcp_stdio`, `llamastack`, `kagenti`). Tests every dispatch path with real MCP servers where available, graceful skip when infrastructure is unavailable.
- **Tool calling model requirements**: gpt-oss-20b generates proper OpenAI-compatible `tool_calls`. Granite 3.3 8B does NOT -- it writes Python code instead of using the tool calling protocol. When building agents that depend on tool calling, verify the model supports it. This is a model capability gap, not a LlamaStack or BaseAgent issue.

## Repository Structure

```
agent-template/
  docs/                    # User-facing: architecture, problem, vision
  planning/                # In-flight: requirements, scope, constraints
  packages/
    fipsagents/            # Shared BaseAgent package (pip-installable)
  sandbox/                 # EXTRACTED to fips-agents/code-sandbox (pointer README remains)
  examples/                # Runnable demos (shared-memory, code-sandbox-agent, document-analysis)
  templates/
    agent-loop/            # Single-agent loop template
    workflow/              # Multi-node workflow template
```

The template directory (what gets cloned by fips-agents) will contain:

```
.claude/commands/          # Slash commands: plan-agent, create-agent, etc.
.claude/rules/             # AI assistant rules
AGENTS.md                  # Open standard
agent.yaml                 # Config with env var substitution
prompts/                   # Markdown + YAML frontmatter
tools/                     # @tool decorated Python files
skills/                    # agentskills.io spec directories
rules/                     # Plain Markdown
evals/                     # Harness-agnostic eval cases
src/fipsagents/baseagent/  # BaseAgent package (installed via fipsagents pip package)
src/agent.py               # ~20-30 line subclass
Containerfile              # Red Hat UBI base
chart/                     # Helm chart
pyproject.toml
Makefile
```

## Development Conventions

- Python async throughout -- every I/O operation is async
- Tools use `@tool` decorator with visibility parameter
- One tool per file in `tools/`, one prompt per file in `prompts/`, one skill per directory in `skills/`
- Keep files under 512 lines
- Use pydantic for config validation and structured output schemas
- pytest for testing
- No mocking to hide errors -- let broken things stay visibly broken

## Dependencies

- openai -- LLM client (async SDK)
- fastmcp (v3) -- MCP client
- memoryhub -- optional, MemoryHub memory backend
- asyncpg -- optional, PGVector memory backend (`pip install fipsagents[pgvector]`)
- aiosqlite -- optional, SQLite session/trace backends (``pip install fipsagents[server]``)
- pydantic -- config and schema validation
- httpx -- async HTTP
- python-frontmatter -- parsing prompt/skill files

## Slash Commands (for scaffolded agents)

These live in `.claude/commands/` within the template:

- `/plan-agent` -- Design the agent before writing code. Produces AGENT_PLAN.md.
- `/create-agent` -- Scaffold agent from AGENT_PLAN.md.
- `/exercise-agent` -- Test agent behavior through role-play scenarios.
- `/deploy-agent` -- Build container and deploy to OpenShift.
- `/add-tool` -- Add a new tool with @tool decorator.
- `/add-skill` -- Add a new skill directory (agentskills.io spec).
- `/add-memory` -- Wire MemoryHub integration via memoryhub config init.

## Infrastructure Context

Agents consume services from rh-ai-quickstart/ai-architecture-charts:
- vLLM for inference
- LlamaStack for orchestration/guardrails (treated as an external endpoint)
- PGVector for vector storage
- MinIO for object storage

The agent talks to these through configured URLs in agent.yaml. It does not deploy or manage them.

## Common Mistakes to Avoid

- Do not import LlamaStack libraries into agent code -- LlamaStack is an external endpoint
- Do not import openai directly -- use BaseAgent's `call_model*()` methods
- Do not put tool dispatch logic in agent subclasses -- use `self.use_tool()`
- Do not hardcode model names or endpoints -- use agent.yaml with env var substitution
- Do not create ConfigMaps for prompts -- prompts are baked into the image for traceability
- Do not skip the `visibility` parameter on tools -- every tool must declare its plane
- Do not assume OpenShift route timeouts are sufficient for multi-agent chains -- default is 30s. Add `oc annotate route <name> haproxy.router.openshift.io/timeout=180s` for routes serving agents that delegate to other agents.
- Do not use MCP tools for memory with small models (8K-16K context) -- tool schemas alone consume ~4K tokens, overflowing the context. Use the framework's `self.memory` SDK connector instead (zero tool token cost). MCP memory tools are viable only on models with 32K+ context.
- Do not rely on system-prompt placement for memory grounding with small models -- Granite 3.3 8B and similar models treat system-prompt content as suggestions. Use `injection_mode: user_turn` in agent.yaml to append memories to user messages where the model treats them as high-salience context.
- Do not hardcode memory retrieval limits -- use `budget: small` (or medium/large) in agent.yaml to get sensible defaults for `max_prefix_chars`, `max_results`, and `min_weight` based on the model's context window. Explicit values override the preset.
- Do not forget `loading_pattern` for file-based backends -- markdown and sqlite backends have no `.memoryhub.yaml` and no `project_config`, so deferred loading patterns only work when `loading_pattern` is set explicitly in agent.yaml.
- Do not put session or trace logic in BaseAgent -- sessions and traces are server-layer concerns. BaseAgent works with `self.messages` and emits `StreamEvent`s; the server wraps those with persistence and observation.
- Sessions support two creation modes: explicit (`POST /v1/sessions`) and auto-create-on-first-use (pass a `session_id` on `ChatCompletionRequest`). The `save()` method uses upsert semantics — if the session doesn't exist, it is created automatically. `POST /v1/sessions` is optional but recommended when you need to control the session ID or check for duplicates.
