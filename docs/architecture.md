# Architecture

agent-template is a monorepo of agent templates for the `fips-agents` CLI. It scaffolds production-ready AI agents that deploy to OpenShift, communicate with LLMs through the OpenAI SDK, and let developers focus on the work that actually differentiates their agent: prompts, tools, model selection, and evals.

This document describes the system architecture, core abstractions, and the reasoning behind each design decision.

## System Context

The agent template sits in a specific layer of a broader stack. Understanding this layering is essential because the template deliberately excludes concerns handled elsewhere.

**Infrastructure layer** is provided by `rh-ai-quickstart/ai-architecture-charts` -- composable Helm charts that deploy vLLM, LlamaStack, PGVector, MinIO, and other services onto OpenShift. This project does not deploy or manage any of that infrastructure. Agents built from this template consume those services through well-defined APIs.

**The fips-agents CLI** clones this repository when a developer runs `fips-agents create agent my-agent`, following the same pattern established by `fips-agents/mcp-server-template`. The CLI selects a template variant, copies it into a new project directory, and hands off to the developer.

**LlamaStack**, when used, is treated as an external endpoint. The agent speaks OpenAI-compatible chat completions to whatever URL is configured. LlamaStack's guardrails, tracing, and routing are its own concern -- the agent neither knows nor cares what sits behind the endpoint.

## Template Variants

The repository contains two template directories and a shared package:

**`packages/fipsagents/`** is the shared BaseAgent framework, distributed as a pip-installable Python package (`fipsagents` on PyPI). Both templates depend on it. Extracting BaseAgent into a shared package eliminates code duplication and ensures a single source of truth for the core agent abstraction.

**`templates/agent-loop/`** scaffolds a single-agent loop: read context, call model, act on response, repeat. This covers the majority of agent use cases. The developer subclasses BaseAgent, implements `step()`, and gets LLM communication, tool dispatch, MCP connections, and all other common concerns for free.

**`templates/workflow/`** scaffolds a directed graph of nodes with typed state. Nodes are either lightweight `BaseNode` instances (for routing, transformation, gating) or `AgentNode` instances (BaseAgent subclasses with full LLM/tools/MCP capabilities). The `WorkflowRunner` manages graph traversal, node lifecycle, per-node retry, error edges, and structured logging. State is a Pydantic model that flows through the graph -- execution metadata stays in logs, not on state.

Both templates share BaseAgent via the `fipsagents` package, follow the same directory conventions (tools, prompts, skills, rules, evals), and use the same deployment model (immutable container images on OpenShift).

## BaseAgent

BaseAgent is the core abstraction. It is pure Python, async throughout, and carries no framework dependencies -- no LangChain, no LangGraph. A typical agent subclass is 20-30 lines of code because BaseAgent handles every common concern: LLM communication, tool dispatch, MCP connections, prompt loading, memory access, skill management, configuration, and lifecycle.

### LLM Client

All LLM communication goes through the OpenAI async SDK, which connects to any OpenAI-compatible endpoint (vLLM, LlamaStack, llm-d). Switching endpoints is a configuration change -- update the model name and endpoint URL in `agent.yaml` -- not a code change.

**Important:** The OpenAI SDK requires `OPENAI_API_KEY` to be set even when connecting to unauthenticated endpoints (e.g., a vLLM instance with no auth). Set it to any non-empty string (e.g., `OPENAI_API_KEY=not-required`) in the agent's environment. Without this, the SDK raises `AuthenticationError` before the request is sent.

BaseAgent exposes five methods for model interaction:

`call_model(messages, **kwargs)` makes a standard chat completion call and returns the response. This is the workhorse for most interactions.

`call_model_json(messages, schema, **kwargs)` requests structured output conforming to a Pydantic schema and returns a parsed, validated object. It handles the provider-specific details of requesting JSON mode or structured output.

`call_model_stream(messages, **kwargs)` returns an async generator that yields content-delta strings as they arrive, for use cases where latency to first token matters but only the user-visible text is needed.

`call_model_stream_raw(messages, **kwargs)` is the richer sibling: it yields the full provider chunk for each delta so callers can inspect `content`, `role`, `tool_calls`, `reasoning_content`, and any other fields the provider emits. Used internally by `astep_stream` (see below) and appropriate for any caller that needs to surface tool decisions or thinking as separate phases. `call_model_stream` is implemented in terms of `call_model_stream_raw`.

`call_model_validated(messages, validator_tool, **kwargs)` is a first-class pattern, not an afterthought. It calls the model, validates the output by invoking a tool (which can be a schema check, a domain-specific validator, or anything else registered in the tool system), and retries with exponential backoff if validation fails. This pattern recurs constantly in production agents -- extracting structured data from unstructured responses, ensuring outputs meet domain constraints -- and deserves dedicated support rather than being reimplemented in every agent.

### Streaming Agent Loop

`BaseAgent.astep_stream()` is the streaming counterpart to `step()`. It drives the full ReAct loop (model call → tool execution → model call → ...) in streaming mode and yields a typed event stream from `fipsagents.baseagent.events`:

- `ReasoningDelta(content)` -- incremental thinking chunk (maps from `delta.reasoning_content`; gpt-oss-20b, o1, o3 emit this natively)
- `ToolCallDelta(index, call_id, name, arguments_delta)` -- streaming tool-call decision, with arguments arriving token-by-token
- `ToolResultEvent(call_id, name, content, is_error)` -- result of a tool the agent just executed, paired to the originating `call_id`
- `ContentDelta(content)` -- incremental user-visible response chunk
- `StreamComplete(finish_reason, metrics)` -- terminal event carrying `StreamMetrics` (TTFT, ITL samples, total time, model/tool call counts, token usage)

Tool dispatch inside the streaming loop flows through the same registry as non-streaming, so the event stream is source-agnostic: tools from MCP servers and local `@tool` functions produce identical event shapes. This is load-bearing for the framework's OpenAI-compatibility story -- server code serializing `astep_stream` to `/v1/chat/completions` SSE can use only standard OpenAI delta fields (`reasoning_content`, `tool_calls`, `role:"tool"` + `tool_call_id`, `content`) with no custom extensions. Dumb OpenAI clients see the assistant content; rich clients render thinking, tool execution, and response as separate phases by inspecting which fields each delta carries.

### Two Tool Planes

Tools are the primary way agents interact with the world, and this template makes a critical architectural distinction between two fundamentally different calling patterns.

**Plane 1: Agent-code tools.** These are called by the agent's Python code directly. The LLM never sees them. Examples include `validate_schema`, `send_email`, `open_door` -- structured actions where the agent code decides what to call and when. The agent invokes them through `self.use_tool("send_email", to="...", subject="...")` without knowing implementation details.

**Plane 2: LLM-callable tools.** These are surfaced to the LLM as part of the tool-calling protocol during chat completions. The LLM decides when and how to call them based on conversation context. MCP-discovered tools land here by default.

Both planes flow through BaseAgent's tool infrastructure. Logging, RBAC, retry logic, and rate limiting apply uniformly regardless of which plane initiated the call. This is non-negotiable -- you cannot have tools that bypass access control just because the LLM called them instead of agent code.

Each tool declares a visibility attribute controlling which plane(s) can access it: `agent_only` (plane 1 only), `llm_only` (plane 2 only), or `both`. The relevant API methods are:

- `use_tool(name, **kwargs)` -- call any tool from agent code (plane 1)
- `register_tool(tool, visibility)` -- register a tool with its visibility setting
- `list_tools()` -- all tools with metadata
- `get_llm_tools()` -- tool definitions formatted for the LLM tool-calling API (only `llm_only` and `both`)
- `handle_tool_call(tool_call)` -- routes LLM-initiated tool calls through the same infrastructure

### Tool Definition

Tools use the `@tool` decorator, following the same convention as FastMCP. They are auto-discovered from the `tools/` directory at startup:

```python
from fipsagents.baseagent.tools import tool

@tool(visibility="agent_only")
async def validate_schema(data: dict, schema_name: str) -> bool:
    """Validate data against a named schema."""
    ...
```

The tool name comes from the function name, the description from the docstring, and parameters from type hints. No registration boilerplate.

### MCP Integration

BaseAgent includes a built-in MCP client (FastMCP v3) for connecting to remote tool servers. `connect_mcp(target)` accepts three transport types:

- **str** -- URL for streamable-http (e.g., `"https://mcp-server/mcp/"`).
- **McpServerConfig** -- HTTP via `url` field, or stdio subprocess via `command`/`args`/`env`/`cwd` fields. Configured in `agent.yaml` under `mcp_servers`.
- **FastMCP object** -- in-process transport, no subprocess or network (useful for testing and co-located servers).

Discovered tools are registered with `llm_only` visibility by default -- the assumption being that MCP tools are designed for LLM-driven invocation. They participate in the same logging, RBAC, and rate-limiting infrastructure as local tools.

Beyond tools, `connect_mcp()` also discovers **prompts** and **resources** from each server:

- **MCP prompts** are stored in `self._mcp_prompts` (keyed by name). Call `get_mcp_prompt(name, arguments={...})` to render a prompt through the originating server. `list_mcp_prompts()` returns metadata for all discovered prompts. MCP prompts are kept separate from local prompts (Markdown + YAML files in `prompts/`) -- they have different lifecycles and rendering mechanisms.
- **MCP resources** are stored in `self._mcp_resources` (keyed by URI). Call `read_resource(uri)` to fetch content on demand. `list_mcp_resources()` and `list_mcp_resource_templates()` return metadata. Resources are agent-plane by default -- agent subclasses choose which resources to surface to the LLM.
- Resource subscriptions (real-time update notifications) are not implemented. Servers that don't expose prompts or resources are handled gracefully -- discovery errors are logged at DEBUG level and don't affect tool registration.

### Platform Mode (OGX delegation)

Platform mode is an opt-in switch that delegates LLM orchestration to OGX (the rebrand of LlamaStack) via its `/v1/responses` endpoint instead of `/v1/chat/completions`. Set `platform.enabled: true` in `agent.yaml` and OGX takes over MCP tool calls, shield enforcement, and the inference loop server-side -- the agent makes a single `call_model_responses()` per turn rather than running its own ReAct loop.

When `platform.enabled` is true:

- The framework skips the `connect_mcp()` startup loop. Entries under `platform.mcp` are passed to OGX on every Responses request; OGX handles the MCP transport (currently SSE).
- Shields listed in `platform.guardrails` are enforced server-side. Each entry is a shield ID registered in OGX's stack YAML (e.g. `code-scanner`, `llama-guard`).
- The legacy `mcp_servers:` block is ignored. Misconfigurations (both blocks set) surface a structured log line at setup, not silent precedence.

Each `platform.mcp` entry takes `name` (becomes `server_label` on the wire) plus exactly one of `connector_id` (references a connector pre-registered in OGX's stack YAML) or `url` (inline `server_url`). An optional `authorization` field forwards a bearer token for OAuth-protected MCP servers. The framework validates the exactly-one-of constraint at config-load time.

Three new framework methods support the Responses surface:

- `call_model_responses(input, *, tools=None, guardrails=None, **kw) -> PlatformResponse` -- non-streaming. Returns a wrapper exposing `content` (joined `output_text` parts), `refusal` (set when a guardrail fires), `usage`, and `response_id`. Defaults `tools` to the configured `platform.mcp` entries and `guardrails` to `platform.guardrails`; either can be overridden per-call.
- `call_model_responses_stream(input, ...) -> AsyncIterator[StreamEvent]` -- streaming. Maps OGX's Responses event protocol onto the existing `StreamEvent` taxonomy: `response.output_text.delta` becomes `ContentDelta`, and a `refusal` in the terminal `response.completed` payload becomes `GuardrailFiredEvent` followed by `StreamComplete(finish_reason="guardrail")`. Tool-call events (`response.tool_call.*`) are not yet mapped -- a follow-up will wire them once OGX-orchestrated MCP tool calls can be exercised end-to-end.
- `moderate(content, *, model=None) -> ModerationResult` -- wraps `/v1/moderations`. Observability-only; never blocks. Defaults `model` to the first entry in `platform.guardrails` (in OGX, shield IDs and moderation model IDs share a namespace).

Two OGX-side quirks worth knowing:

**No dedicated guardrail event type.** OGX signals a fired shield structurally: `output[*].content[*].type == "refusal"` on the terminal payload. There is no machine-readable shield ID -- the violation type is embedded in the refusal text (e.g. `"(flagged for: insecure-eval-use)"`). The framework parses this heuristically into `GuardrailFiredEvent.shield_id`, falling back to a comma-joined list of configured shields when the pattern isn't present.

**Late-firing output shields can leak pre-shield content during streaming.** When OGX's post-generation shield triggers (model generated unsafe content from a benign prompt), the unsafe `output_text.delta` events have already been streamed; the framework passes them through to consumers. The terminal `response.completed` payload then replaces the streamed text with the refusal, and the framework emits `GuardrailFiredEvent` + `finish_reason="guardrail"`. This matches how hosted providers like Anthropic and OpenAI handle late-firing safety filters. Consumers that need post-shield content only should buffer until `StreamComplete`.

The `guardrails` field is an OGX extension to the Responses API and is not recognized by the OpenAI Python SDK's typed kwargs. The framework routes it through `extra_body` (the SDK's escape hatch for provider extensions). Caller-supplied `extra_body` keys are merged, not overwritten.

Platform mode is decoupled from observability (#81). When that issue lands, OTLP wiring will apply uniformly to both chat-completions and Responses-based agents.

### Prompts

`load_prompt(name, **variables)` loads a prompt from the `prompts/` directory, performs variable substitution, and returns the rendered text. `list_prompts()` returns available prompts with their metadata. The prompt format is described in its own section below.

### Memory

`self.memory` is always a `MemoryClientBase` instance -- either a live backend or `NullMemoryClient` (silent no-op). Agent code can unconditionally call `self.memory.search(...)` without checking configuration. The backend is selected via `memory.backend` in `agent.yaml`; when unset, the factory auto-detects `.memoryhub.yaml` for backward compatibility. Memory integration is described in detail in its own section.

### Skills

`self.skills` is a dictionary of skill stubs loaded at startup. `load_skill(name)` activates a skill (loading its full content), and `unload_skill(name)` deactivates it. This progressive-disclosure pattern is described in the Skills section.

### Lifecycle

Every BaseAgent subclass follows the same lifecycle:

`setup()` loads configuration, connects to MCP servers, discovers local tools and prompts, initializes memory (if configured), and loads skill stubs. This runs once at startup.

`step()` is one iteration of agent logic. The default implementation consumes `astep_stream()` and concatenates `ContentDelta` content into a `StepResult.done`, so most subclasses override only `astep_stream` and get a working sync path for free -- any pre/post-turn hooks (system prompt injection, memory recall, memory write) live in a single place and both sync and streaming clients share identical behavior. Override `step()` directly only when a subclass needs sync-specific behavior that doesn't map cleanly onto events.

`teardown()` disconnects MCP servers and performs cleanup. This runs once at shutdown.

`run()` is the loop driver. It calls `setup()`, then calls `step()` repeatedly until the agent signals completion or hits the configured maximum iteration count. Built-in protective patterns -- max iterations, exponential backoff on errors, rate limiting -- prevent runaway behavior.

### Conversation State

`messages` holds the current conversation history. `add_message()` appends to it. `clear_messages()` resets it. These are deliberately simple because conversation management strategies vary widely between agents, and BaseAgent should not impose a particular approach.

## HTTP Server

Most FIPS-Agents chat deployments sit behind an OpenAI-compatible HTTP endpoint so a UI, gateway, or another agent can call them through the ecosystem-standard `/v1/chat/completions` contract. `fipsagents.server.OpenAIChatServer` is the canonical implementation: a FastAPI app that takes a `BaseAgent` subclass and exposes `/v1/chat/completions` (sync + SSE), `/v1/agent-info` (model config, system prompt, and LLM-visible tools for UI settings panels), `/healthz`, and `/readyz` with no hand-written HTTP glue.

```python
from fipsagents.server import OpenAIChatServer
from myagent import MyAgent

server = OpenAIChatServer(MyAgent, config_path="agent.yaml")
app = server.app  # for uvicorn / gunicorn

if __name__ == "__main__":
    server.run()  # convenience wrapper around uvicorn.run
```

The class owns the agent lifecycle via FastAPI lifespan -- `MyAgent.setup()` on startup, `shutdown()` on teardown -- and serializes per-request access through an `asyncio.Lock` so concurrent requests don't interleave writes to the shared `agent.messages`. Streaming delegates to `fipsagents.serialization.openai_sse:stream_events_as_sse` (see below); the agent subclass itself is fully unaware of HTTP.

### Opt-in extra, not a core dependency

`fipsagents` core has no FastAPI dependency. `OpenAIChatServer` lives behind the `[server]` optional-dependencies extra:

```toml
pip install 'fipsagents[server]'   # pulls in fastapi + uvicorn[standard]
```

Agents that don't expose HTTP -- workflow nodes, batch jobs, evaluation harnesses -- pay no FastAPI install cost. Importing `fipsagents.server` without the extra installed raises a clear `ImportError` pointing at the install command.

### Not a plugin system

There is deliberately **one** HTTP server class and **one** wire-format serializer in the package today. Agents needing a different wire format (WebSocket push, Anthropic Messages API, OpenAI Responses API) either write their own server in their own repo or wait for issue-tracked follow-ups to add a sibling function. There is no registry, no strategy class, no `BaseServer` abstract -- the test for any new serializer is whether it slots in as a plain async function with the same type signature, not whether it registers into some lookup. This keeps the framework's public surface small enough to read top to bottom.

## Streaming Serialization

The streaming wire format -- translating `StreamEvent` sequences to bytes on the wire -- is its own concern, split out from the HTTP server so the same serializer can be reused by WebSocket handlers, test harnesses, or alternative transports.

```python
from fipsagents.serialization.openai_sse import stream_events_as_sse

async for chunk in stream_events_as_sse(agent.astep_stream(), model_name):
    yield chunk  # yields SSE frames ending with "data: [DONE]\n\n"
```

`stream_events_as_sse` is a pure async generator: no FastAPI, no logging, no side effects. It accepts any `AsyncIterator[StreamEvent]`, maps each event to an OpenAI chat-completion-chunk delta using only standard OpenAI wire fields (`reasoning_content`, `tool_calls`, `role:"tool"` + `tool_call_id`, `content`), and terminates with `[DONE]`. On exception from the source iterator it emits an error chunk before `[DONE]` so clients always see a clean termination.

After the terminal `StreamComplete`, the serializer emits one additional chunk with `choices: []` and a top-level `usage` object -- matching OpenAI's `stream_options: {include_usage: true}` behaviour so token counts are visible to standard clients. The same chunk also carries a sibling `stream_metrics` object with TTFT, time-to-first-content, total time, inter-token latencies, and model/tool call counters drawn from `StreamMetrics`. Conforming OpenAI clients ignore the extension; dashboards and eval harnesses that know to look for it get richer instrumentation without a second endpoint. The sync (`stream: false`) response body carries the same `usage` + `stream_metrics` at the top level.

### Adding new wire formats

Wire formats follow the same convention: one module, one pure function, one explicit import path.

```
fipsagents.serialization.openai_sse:stream_events_as_sse                            # OpenAI Chat Completions
fipsagents.serialization.anthropic_messages:stream_events_as_anthropic_messages      # Anthropic Messages (#41)
fipsagents.serialization.responses_api:stream_events_as_responses                    # future (#35)
```

The type signature `(events: AsyncIterator[StreamEvent], model_name: str, ...) -> AsyncIterator[str]` is the contract. No base class, no registry -- grep for the function name to know what exists.

## Observability

Sessions, traces, and metrics are server-layer concerns -- BaseAgent has no concept of any of them. The server wraps each request with persistence and observation; the agent works with `self.messages` and emits `StreamEvent`s. All observability features share a storage backend configured once in `agent.yaml` and degrade to silent no-ops when disabled.

### Storage Backend

`ServerConfig` has a `StorageConfig` that selects a single backend (`null`, `sqlite`, or `postgres`) shared by sessions and traces:

```yaml
server:
  storage:
    backend: sqlite                    # null | sqlite | postgres
    sqlite_path: ./agent.db
    # postgres_dsn: postgresql://...   # for backend: postgres
  sessions:
    enabled: true
    max_age_hours: 168
  traces:
    enabled: true
    max_age_hours: 72
    sampling_rate: 1.0
  metrics:
    enabled: false
```

When backend is `null`, both sessions and traces degrade to no-ops (fully backward-compatible). `SqliteConnectionManager` (in `fipsagents.server.sqlite`) deduplicates connections by resolved file path, so sessions and traces sharing the same SQLite database get a single connection pool rather than two.

### Sessions

`SessionStore` is an ABC with four implementations: `NullSessionStore` (default, ephemeral -- messages live only for the request lifetime), `SqliteSessionStore` (edge/dev), `PostgresSessionStore` (enterprise), and `HttpSessionStore` (delegates to a sibling `fipsagents-platform` service over REST -- see [Cross-Agent Platform Service](#cross-agent-platform-service)). The server loads conversation history before each request and saves it after, so the agent subclass never touches persistence directly.

The ABC also exposes `update(session_id, *, cost_data=None) -> bool` and `get_cost_data(session_id) -> dict` for the per-turn cost-tracking accumulator wired into `OpenAIChatServer`. Each turn's `prompt_tokens` / `completion_tokens` (extracted from the terminal `StreamComplete` event) are accumulated and persisted via `update()`; failures are caught and logged so cost-tracking issues never break the chat response. SQLite stores `cost_data` as a `TEXT` column (JSON-encoded, application-side shallow merge); Postgres uses native `JSONB ||` merge. `HttpSessionStore.get_cost_data()` reads from the platform's `GET /v1/sessions/{id}/cost_data` endpoint; older platform deployments (≤ 0.2.0) 404 cleanly and the agent silently degrades to per-turn-delta writes.

REST endpoints: `POST /v1/sessions`, `GET /v1/sessions/{id}`, `GET /v1/sessions/{id}/usage` (computed dollar view, see [Cost Tracking](#cost-tracking)), `PATCH /v1/sessions/{id}` (partial update for `cost_data`), `DELETE /v1/sessions/{id}`.

### Cost Tracking

Token cost tracking has two layers:

1. **Raw counters.** Per-turn `prompt_tokens` / `completion_tokens` are accumulated into `cost_data` on the session record. This works regardless of whether the deployment has a configured pricing table -- self-hosted vLLM deployments without dollar billing still get token totals.
2. **Computed dollars.** `PricingConfig` in `agent.yaml` maps each model to a `PricingRate` (USD per 1k tokens for input, output, and optional cached tokens; plus an optional flat per-request fee). `compute_cost()` in `fipsagents.server.pricing` is a pure helper that takes a model name + token counts + pricing and returns rounded USD. It backs the `GET /v1/sessions/{id}/usage` endpoint, which returns a single computed-cost JSON shape so callers (gateway, UI, BudgetEnforcer) don't need to replicate the rate table.

`PricingConfig` falls back to a `default` rate when a model is not listed in the per-model table, so untyped agent.yaml stays valid. Pricing for cached tokens follows OpenAI semantics: cached tokens are a subset of input tokens billed at a discounted rate; if no `cached_input_per_1k` is set, the full input rate applies.

`BudgetEnforcer` (in `fipsagents.server.budget`) follows the same observer pattern as `MetricsCollector` and `TraceCollector` — it hooks the chat-completion code path with a pre-request check and a post-stream record. Per-session limits read cumulative `cost_data` from the session store on every turn, so they work across restarts and replicas (any agent that loads the same session sees the same total). Per-tenant limits aggregate session-cost deltas in-process, so they accurately represent "this agent process's view" of cross-session tenant cost; multi-replica tenant aggregation requires a separate cross-agent service and is out of scope.

When a hard limit is hit, `BudgetEnforcer` raises `BudgetExceededError`, which the server maps to **HTTP 402 Payment Required** with a structured body (`error`, `scope`, `identifier`, `current_usd`, `limit_usd`). Soft warnings emit a single `WARNING` log line per crossing per scope and then go quiet. `budget.mode: observe` downgrades hard-limit raising to log-only — useful for measuring impact before turning enforcement on.

`TraceCollector` stamps OTEL GenAI semantic-convention attributes on its spans alongside the legacy attribute names: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.system` on the request span, and `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` / `gen_ai.response.model` on the model_call span. Existing trace consumers that read `prompt_tokens` / `completion_tokens` keep working; OTEL backends (Tempo, Honeycomb, Grafana Cloud) get the standard keys they expect for free.

`CreateSessionRequest` is a Pydantic model that validates session IDs: alphanumeric characters, hyphens, and underscores only, 1-128 characters. The same validation applies to the optional `session_id` field on `ChatCompletionRequest`. Sessions support two creation modes: explicit creation via `POST /v1/sessions`, and auto-create-on-first-use -- when a `session_id` is passed on a chat completion request, the `save()` method uses upsert semantics and creates the session if it doesn't exist. The explicit endpoint is optional but recommended when you need to control the ID or check for duplicates.

### Traces

`TraceCollector` wraps `astep_stream()` as a pure observer, building span trees from `StreamEvent`s without modifying them. It produces `Trace` objects containing a tree of `Span`s (model calls, tool executions, full request lifecycle). `TraceStore` is an ABC with three implementations:

- `NullTraceStore` -- structured JSON logging, no persistence (default)
- `SqliteTraceStore` -- edge and development use
- `PostgresTraceStore` -- enterprise use; JSONB spans, `TIMESTAMPTZ` timestamps, `asyncpg` connection pool. Mirrors the `PostgresSessionStore` implementation pattern.

Query endpoints: `GET /v1/traces`, `GET /v1/traces/{id}`. Sampling rate is configurable via `server.traces.sampling_rate`.

### Prometheus Metrics

`MetricsCollector` follows the same observer pattern as `TraceCollector` -- it wraps request handling and records counters and histograms without modifying the agent's behavior. Five metrics are tracked:

- `agent_requests_total` -- request count by status
- `agent_request_duration_seconds` -- end-to-end request latency histogram
- `agent_model_call_duration_seconds` -- per-model-call latency histogram
- `agent_tool_call_total` -- tool invocation count by tool name
- `agent_tokens_total` -- token consumption by direction (prompt/completion); optional `tenant_id` and `session_id` label dimensions controlled by `metrics.token_label_mode`

Exposed at `GET /metrics` in Prometheus text format. Requires the `[metrics]` optional extra (`prometheus_client`). When metrics are disabled, `NullMetricsCollector` is a silent no-op.

`token_label_mode` selects which dimensions are attached to `agent_tokens_total`. Each step up adds one label at the cost of more time-series stored by Prometheus:

- `model` (default) — only `model` and `direction`. Bounded by the model catalog.
- `tenant` — also adds `tenant_id` (gateway-stamped via the `X-Tenant` header; missing-header default is `"default"`). Suitable for most enterprise deployments.
- `session` — also adds `session_id`. **High cardinality**: one time-series per session per direction per model. Only enable when an external aggregation step (Prometheus federation, Mimir) can absorb the volume; otherwise prefer `GET /v1/sessions/{id}/usage` for per-session totals.

```yaml
server:
  metrics:
    enabled: true
    token_label_mode: tenant  # or "model" (default), or "session"
```

```toml
pip install 'fipsagents[metrics]'
```

### OTEL Trace Export

`OTELTraceStore` wraps an inner `TraceStore` (typically `SqliteTraceStore` or `PostgresTraceStore`) and translates internal `Span` objects to OpenTelemetry spans exported via OTLP. Span IDs are deterministically hashed (SHA-256 of the internal string ID) so traces are reproducible. Monotonic timestamps from `StreamMetrics` are converted to wall-clock times anchored on `Trace.started_at`.

```yaml
server:
  traces:
    enabled: true
    exporter: otel
    otel_endpoint: http://otel-collector:4317
    service_name: my-agent
```

```toml
pip install 'fipsagents[otel]'  # opentelemetry-sdk + opentelemetry-exporter-otlp-proto-grpc
```

### Distributed Trace Propagation

Multi-agent deployments (workflow graphs with `RemoteNode`) propagate trace context across HTTP boundaries using the W3C Trace Context specification. `propagation.py` provides two functions:

`extract_trace_context(headers)` pulls `traceparent` from incoming request headers and returns a `(trace_id, parent_span_id)` tuple. `TraceCollector` accepts these as `parent_trace_id` and `parent_span_id` to join the distributed trace.

`inject_trace_context(headers, trace_id, span_id)` sets the `traceparent` header on outgoing requests. `RemoteNode.set_trace_context()` calls this before delegating to a downstream agent, so the downstream agent's traces are children of the calling node's span.

The result is a single trace tree spanning all agents in a workflow, viewable in any W3C-compliant trace backend (Jaeger, Tempo, the OTEL collector).

## Prompts

Prompts are Markdown files with YAML frontmatter, stored one-per-file in the `prompts/` directory:

```markdown
---
name: summarize
description: Summarize a document for the user
model: default
temperature: 0.3
variables:
  - name: document
    required: true
  - name: max_length
    default: "500 words"
---

You are a document summarizer. Summarize the following document
in {max_length} or less.

## Document

{document}
```

The frontmatter carries metadata -- name, description, model preferences, temperature, and variable declarations. The body is the prompt template with `{variable_name}` substitution. This format keeps prompts human-readable and editable without touching Python code, while the frontmatter provides enough structure for tooling and documentation generation.

Prompt changes are code changes. They go through PR review, CI, and image builds like any other source file.

## Skills

Skills follow the agentskills.io specification. Each skill lives in its own directory under `skills/` with a `SKILL.md` file (YAML frontmatter describing the skill, Markdown body with full instructions) and optional subdirectories for scripts, references, and assets:

```
skills/
  example-skill/
    SKILL.md
    scripts/
    references/
    assets/
```

The key design principle is progressive disclosure to manage context budgets. At startup, BaseAgent loads only the frontmatter from each SKILL.md -- roughly 100 tokens per skill, enough to know what each skill does. When a skill is activated via `load_skill(name)`, the full Markdown body is loaded into context. Resources from subdirectories are loaded on demand. This layered approach lets an agent have dozens of available skills without burning its entire context window at startup.

Skills replace the concept of commands entirely. Rather than hardcoded command handlers, skills provide a flexible, declarative way to extend agent capabilities.

## Rules

Rules are plain Markdown files in the `rules/` directory. No frontmatter -- the filename is the identifier. They contain imperative, actionable guidance that is loaded at startup and injected into the agent's context.

Rules differ from prompts in intent: prompts are templates for specific interactions, while rules are persistent behavioral constraints that apply across all interactions. Keeping them in separate files (rather than embedding them in a system prompt) makes them individually reviewable and independently deployable.

## Configuration

Agent configuration lives in `agent.yaml` with environment variable substitution using `${VAR:-default}` syntax:

```yaml
model:
  endpoint: ${MODEL_ENDPOINT:-http://llamastack:8321/v1}
  name: ${MODEL_NAME:-meta-llama/Llama-3.3-70B-Instruct}
  temperature: 0.7
  max_tokens: 4096

mcp_servers:
  - url: ${MCP_WEATHER_URL:-http://weather-mcp:8080/mcp}

tools:
  local_dir: ./tools
  visibility_default: agent_only

prompts:
  dir: ./prompts

loop:
  max_iterations: ${MAX_ITERATIONS:-100}
  backoff:
    initial: 1.0
    max: 30.0
    multiplier: 2.0

logging:
  level: ${LOG_LEVEL:-INFO}
```

The env var substitution pattern provides clean separation between the configuration structure (which is baked into the container image) and environment-specific values (which come from OpenShift ConfigMaps and Secrets at deploy time). Defaults ensure the agent can run locally without any external configuration, while every value can be overridden for production.

## Memory Integration

Memory is optional and pluggable. The `memory.backend` field in `agent.yaml` selects which backend to use:

| Backend | Config file | Dependencies | Search type | Best for |
|---------|-------------|--------------|-------------|----------|
| `memoryhub` | `.memoryhub.yaml` | `memoryhub` | Full (server-side) | Production with MemoryHub |
| `markdown` | `.memory-markdown.yaml` | None (stdlib) | Case-insensitive substring | Human-curated, git-committed memory |
| `sqlite` | `.memory-sqlite.yaml` | None (stdlib) | Keyword (FTS5) | Local dev, testing |
| `pgvector` | `.memory-pgvector.yaml` | `asyncpg`, `pgvector` | Semantic (vector cosine) | Production without MemoryHub |
| `llamastack` | `.memory-llamastack.yaml` | None (`httpx` in core) | Semantic (vector similarity) | Already-on-LlamaStack deployments |
| `custom` | -- | Your choice | Your choice | Custom infrastructure |
| `null` | -- | None | None (disabled) | Explicitly disable memory |

When `backend` is unset, the factory auto-detects `.memoryhub.yaml` for backward compatibility. All backends implement `MemoryClientBase` with four async methods: `search()`, `write()`, `update()`, and `report_contradiction()`. When no backend is configured (or any backend fails to initialise), `self.memory` is a `NullMemoryClient` -- a silent no-op that returns empty results so agent code never needs to guard on configuration.

### Picking a backend

Memory implementations cluster into maturity levels; most teams should start at the simplest level that addresses their actual failure modes. The first decision is whether the agent needs memory at all:

```
Does this agent need memory across sessions?
├─ No  → backend: null (or just leave it unset).
└─ Yes → How will you curate it?
   ├─ I'll read and edit the memory file by hand, and commit it to git
   │  ├─ One topic       → backend: markdown, file: ./memory.md      (Level 1)
   │  └─ Multiple areas  → backend: markdown, dir:  ./memories        (Level 2)
   ├─ Agent needs searchable memory on one host, I won't curate by hand
   │                     → backend: sqlite                            (Level 3)
   ├─ Multiple agents share memory, or I need vector similarity
   │  ├─ Already using LlamaStack for inference?
   │  │                  → backend: llamastack                        (Level 4a)
   │  └─ Otherwise       → backend: pgvector                          (Level 4b)
   └─ Regulated environment: audit trails, RBAC, retention, deletion-with-evidence
                           → backend: memoryhub                       (Level 5)
```

Each jump is a real jump, not a sliding scale of features. If you find yourself asking "should the markdown backend have search ranking?" the answer is usually "no, move to SQLite."

### The prefix-cache pattern

Regardless of backend, *how* an agent injects memory into the context affects prefix-cache hit rates at the model endpoint. Modern inference servers (vLLM, OpenAI, Anthropic) cache prefixes across requests; a turn whose first N tokens match the previous turn pays zero time-to-first-token for those tokens.

Cache-friendly ordering looks like:

1. System prompt (stable across turns)
2. Memory block (stable across turns — inject once at session start, not re-query per turn)
3. Conversation history (the changing part)

`BaseAgent.build_memory_prefix()` is the hook for this. Called once during `setup()`, the default implementation runs `self.memory.search("")` and joins the `content` fields with `---` separators, truncating at `config.memory.max_prefix_chars` (default 8 000; 0 disables the limit). The result is injected as a message at index 1 in `self.messages`, immediately after the system prompt:

```python
# After setup(), self.messages looks like:
[
    {"role": "system",    "content": "<system prompt>"},
    {"role": "<prefix_role>", "content": "<memory prefix>"},  # only if non-empty
]
# Conversation turns append after this — the prefix never shifts.
```

The message role is controlled by `config.memory.prefix_role` (default `"system"`). Models that support the OpenAI harmony format (gpt-oss-20b, o-series) can set this to `"developer"` to place memories in the harmony hierarchy (`system > developer > user`). See [#49](https://github.com/fips-agents/agent-template/issues/49) for a planned probe to detect model support at runtime.

Subclasses override `build_memory_prefix()` to customise the query, formatting, or to return `None` unconditionally when they prefer per-turn recall. Agents that need fresher memory mid-session call `self.memory.search()` directly from their `astep_stream` override -- the prefix is a session-level stable cache, not a replacement for dynamic retrieval.

The markdown backend's `search(query="")` is designed to pair well with this: it returns every section or file in stable file order, so the prefix is deterministic across restarts. Other backends (SQLite, PGVector, MemoryHub) can use the same pattern; results are retrieved once at session start and pinned for the session's lifetime.

### Deferred loading and injection modes

The prefix-cache pattern works well for frontier models with large context windows.
Small models (8K-16K context, e.g. Granite 3.3 8B) have two problems: (1) a
system-level memory prefix consumes a significant fraction of their context, and (2)
they tend to treat system-prompt content as suggestions rather than instructions.

`_inject_deferred_memory()` addresses both issues. It runs at the top of
`astep_stream()`, before the first model call, and is controlled by two config fields:

**`injection_mode`** controls *where* memories land:

- `prefix` (default) -- inserts a new message before the user turn, same
  behavior as the setup-time prefix but deferred to after the user message arrives.
- `user_turn` -- appends memories to the user message inside
  `<injection_tag>...</injection_tag>` XML tags. Small models treat user-message
  content as higher-salience context, so this path yields better recall.

**`loading_pattern`** controls *when* memories are retrieved:

- `eager` (default) -- at `setup()` time via `build_memory_prefix()`. Best for
  prefix-cache hit rates with frontier models.
- `lazy` -- after the first user message. The user's text becomes the search query.
- `lazy_with_rebias` -- lazy, plus re-retrieves when the topic shifts.
- `jit` -- retrieves every turn (not yet fully implemented; currently behaves like lazy).

For MemoryHub backends, the loading pattern from `.memoryhub.yaml` is used unless
`loading_pattern` is set explicitly in `agent.yaml` (config-level takes precedence).
For file-based backends (markdown, sqlite), set `loading_pattern` in `agent.yaml` directly.

**`budget`** is a shorthand that sets `max_prefix_chars`, `max_results`, and `min_weight`
based on model tier:

| Budget | max_prefix_chars | max_results | min_weight | Target models |
|--------|-----------------|-------------|------------|---------------|
| `small` | 500 | 5 | 0.7 | Granite 8B, similar 8K-16K models |
| `medium` | 4000 | 20 | 0.5 | Granite 70B, 32K-128K models |
| `large` | 8000 | 50 | 0.3 | GPT-OSS 20B, 128K+ models |

Explicit field values always override the budget preset. Example `agent.yaml`:

```yaml
memory:
  budget: small
  injection_mode: user_turn
  loading_pattern: lazy
```

This configures the agent for a small model: deferred loading, user-turn injection,
and a tight memory budget (500 chars, 5 results, min weight 0.7).

**The SDK path** exposes `self.memory` for programmatic access from agent code. This is for cases where the agent logic itself needs to read or write memories -- caching intermediate results, maintaining state across iterations, or implementing retrieval patterns that the LLM shouldn't control directly.

**The MCP path** (MemoryHub only) makes MemoryHub's tools available to the LLM through the standard MCP client. The LLM can read and write memories as part of its tool-calling workflow.

Custom backends can be registered via `backend: custom` with a `backend_class` dotted import path in `agent.yaml`. See `docs/custom-memory-backend.md` for the full guide.

For multi-agent deployments with MemoryHub, multiple agents can connect to the same instance. Scope-based visibility (user, project, role, organization, enterprise) and RBAC control which agents can see which memories, enabling shared-memory architectures without coupling the agents to each other.

## File Uploads

`POST /v1/files` accepts multipart uploads, persists each file via the configured `FileStore`, and exposes them to subsequent `/v1/chat/completions` requests via two distinct paths depending on what the caller wants the model to see:

- **Extracted text** — pass `file_ids: ["file_..."]` on the request body. The framework reads each file's extracted text (Docling for PDFs, native UTF-8 for plain text) and injects it as a system message just before the user's turn, or runs chunked retrieval against pgvector when chunking is enabled (see ADR-0002).
- **Image bytes** — reference `file_id:<id>` from an `image_url` content block on the user message (`{"type": "image_url", "image_url": {"url": "file_id:..."}}`). `OpenAIChatServer._resolve_image_file_ids` walks user messages, fetches bytes from the configured `BytesStore`, sniffs the MIME type via libmagic, and rewrites the URL in place to `data:{mime};base64,…` before forwarding to the model. The same upload can be referenced via `file_ids` (text) and via `file_id:` (image bytes) on different requests; they are independent paths and never overlap.

Uploads are an opt-in feature — set `server.files.enabled: true` and install the `[files]` extra to pull in Docling (text extraction) and `python-magic` (content-based MIME sniffing).

### Storage layout (per ADR-0001)

`FileStore` owns metadata; bytes are delegated to a separate `BytesStore`. The two compose:

- **Metadata backends** — `SqliteFileStore` (single-replica dev) or `PostgresFileStore` (production). Both persist `FileRecord` rows: id, filename, MIME, size, status, extracted text, plus chunking lifecycle columns (see below).
- **Bytes backends** — `LocalFsBytesStore` (sharded local filesystem at `bytes_dir`, single-replica only) or `S3BytesStore` (S3-compatible: AWS S3, MinIO, GCS S3-mode, R2, B2 — multi-replica safe, requires the `[s3]` extra).

The split means "Postgres metadata + S3 bytes" is one config block, not a separate `FileStore` class. `DELETE /v1/files/{id}` cascades through both stores plus the `ChunkStore` (when chunking is enabled) before the metadata row is removed.

### Optional virus scanning

When `scanner.url` is configured, every upload is POSTed to a ClamAV-fronting HTTP sidecar before persistence. The sidecar exposes a `{infected, viruses}` JSON contract. `fail_mode: open` (dev default) accepts uploads when the scanner is unreachable; `fail_mode: closed` (production-recommended) returns 503.

### Large-file chunking + retrieval (per ADR-0002)

Without chunking, every referenced file's full extracted text is injected into the prompt. That works for small files but blows the context window on long PDFs. Chunking turns file references into a per-query retrieval — the canonical RAG path scoped to the request's `file_ids`.

When `server.files.chunking.enabled: true`, the upload pipeline forks at the size threshold:

1. **Small files** (`<= small_file_threshold_tokens`) — full text inlined, identical to the 0.17.0 behaviour.
2. **Large files** (`> small_file_threshold_tokens`) — `app.py::_chunk_uploaded_file` runs asynchronously after the upload responds. The `Chunker` splits text into overlapping windows, each chunk is embedded via the configured OpenAI-compatible embedding endpoint, and the result is written to the `ChunkStore`. `chunk_status` and `chunk_count` columns on `FileRecord` track lifecycle (`pending` → `in_progress` → `completed` / `failed`).

At chat-completion time, `app.py::_resolve_file_attachments` runs a three-way fork:

- **Chunked + ready** — the user's last message becomes the retrieval query, `chunk_store.search()` returns the top-K nearest chunks for each `file_id`, and only those chunks are injected.
- **Chunked, not ready** (warm-up window) — falls back to full-text injection so the request still works.
- **Chunking disabled** — full-text path, unchanged from 0.17.0.

`Chunker` has two implementations: `RecursiveTokenChunker` (token-based splitter, default) and `DoclingChunker` (heading-aware Markdown splitter, auto-selected when the `[files]` extra is installed). `tiktoken` is a soft dep — when unavailable, the chunker falls back to a `len(text)//4` character heuristic, which keeps FIPS-only builds working without a hard dependency on a non-FIPS tokenizer.

`ChunkStore` mirrors the `BytesStore`/`FileStore` ABC pattern: `NullChunkStore` (default, no-op) and `PgvectorChunkStore` (Postgres + pgvector, requires `[chunking]` extra). The pgvector store enforces per-file scoping at retrieval — `file_id` is part of the query predicate, not just a filter — so a chunk uploaded by user A is never surfaced to user B's `file_ids`. That scoping is the auth boundary; there is no separate access-control layer between `chunk_store.search()` and the embedding result.

`ChunkingConfig` carries budget presets parallel to `MemoryConfig`. `budget: small` (chunk 400 tokens / top-K 3 / threshold 2K) suits small-context models; `budget: medium` (600 / 5 / 4K, default sizing) suits 32K–128K models; `budget: large` (800 / 8 / 8K) suits 128K+ models. Explicit per-tier knobs (`chunk_size_tokens`, `chunk_overlap_tokens`, `small_file_threshold_tokens`, `retrieval_top_k`, `retrieval_min_score`) override the preset. `backend: "null"` (the default) preserves the 0.17.0 full-text behaviour; `backend: "pgvector"` requires `database_url` and `embedding_url`.

See [ADR-0002](adr/0002-large-file-chunking-pgvector.md) for the full design discussion: alternatives considered, the per-file-scope auth argument, why the warm-up window falls back to full-text rather than blocking, and what's deferred (reranking, hybrid BM25+vector search, native `HybridChunker` with `DoclingDocument`).

## Reasoning Extraction

Some models emit chain-of-thought reasoning in the `reasoning_content` delta field (gpt-oss-20b, o-series). Others embed it in the content stream as `<think>…</think>` XML blocks (Granite 3.3, DeepSeek). Without extraction, think tags leak into the user-visible response.

`astep_stream` handles both paths:

1. **Native reasoning** — `delta.reasoning_content` is emitted as `ReasoningDelta` directly. No extraction needed.
2. **Think-tag extraction** — `ThinkTagParser` (in `fipsagents.baseagent.reasoning`) is a streaming state machine that separates `<think>` blocks from content. It handles tags split across chunk boundaries, multiple blocks per response, and unclosed blocks. Content outside think tags emits as `ContentDelta`; content inside emits as `ReasoningDelta`.

The parser is auto-enabled at `setup()` step 11 based on model name (`granite` or `deepseek` substring match via `create_reasoning_parser()`). When vLLM is started with `--reasoning-parser granite`, it does the extraction server-side and populates `reasoning_content` directly — in that case the parser is a harmless no-op since content won't contain the tags.

Only `ContentDelta` text is appended to the assistant message in conversation history. Reasoning is surfaced to streaming consumers (UI collapsed panels, metrics) but never stored in `self.messages`.

## Deployment Model

The deployment model is built on a principle of immutable container images. Everything that defines an agent's behavior -- code, tools, prompts, skills, rules -- is baked into the image. The only external inputs are environment-specific configuration values injected through OpenShift ConfigMaps and Secrets (endpoint URLs, credentials, tuning parameters).

This means prompt and tool changes follow the same path as code changes: PR review, CI validation, image build, deployment. Every deployed state is traceable to a single image tag, which maps to a git commit. There are no surprises from runtime configuration drift.

### What Ships in the Image

The Containerfile (Red Hat UBI base) packages the complete agent:

- Python source (BaseAgent + agent subclass)
- `tools/` directory with all tool implementations
- `prompts/` directory with all prompt templates
- `skills/` directory with all skill definitions
- `rules/` directory with all rule files
- `agent.yaml` with defaults
- `.memoryhub.yaml` (if configured)
- All Python dependencies

### What Lives Outside the Image

- `agent.yaml` overrides via ConfigMap (endpoint URLs, model names, tuning parameters)
- Credentials via Secrets (API keys, tokens)
- Infrastructure services (vLLM, LlamaStack, PGVector, MemoryHub) -- deployed separately

### Helm Chart

The Helm chart bundles only the agent itself: a Deployment, Service, ConfigMap, and optional Route. When the code execution sandbox is enabled (`sandbox.enabled: true`), the chart adds a sidecar container, an emptyDir volume for temporary code files, and a `SANDBOX_URL` environment variable pointing the agent at the sidecar. The chart does not deploy any infrastructure. The expectation is that vLLM, LlamaStack, PGVector, and other services are already running, deployed by `rh-ai-quickstart/ai-architecture-charts` or equivalent.

This separation keeps the agent chart simple and avoids version-coupling between the agent and its infrastructure. An agent upgrade does not force an infrastructure upgrade, and vice versa.

## AI-Assisted Development Experience

The `.claude/` directory in each scaffolded project drives the developer experience when working with Claude Code (or similar AI coding assistants). This is not an incidental feature -- it is a core part of the template's value proposition.

### Slash Commands

The template provides a progression of slash commands that guide developers through the agent lifecycle:

`/plan-agent` helps the developer think through what the agent should do, what tools it needs, and what prompts it requires before writing code.

`/create-agent` scaffolds the agent subclass, initial tools, and prompts based on the plan.

`/exercise-agent` runs the agent through test scenarios to validate behavior.

`/deploy-agent` builds the container image and deploys to OpenShift.

Three additional commands support incremental development: `/add-tool`, `/add-skill`, and `/add-memory` each guide the developer through adding that specific capability to an existing agent.

### AGENTS.md

Each scaffolded project includes an `AGENTS.md` file following the open standard convention. This file describes the agent's capabilities, tools, and interaction patterns in a format that other agents and tooling can consume.

## Evals

The `evals/` directory is scaffolded with a harness-agnostic format: an `evals.yaml` file defining test cases, a `run_evals.py` runner, a `fixtures/` directory for test data, and a README explaining the approach. The template does not build a full eval framework -- it provides the structure and supports plugging in external harnesses. The intent is that eval definitions live alongside agent code and go through the same review process.

## Template Directory Layout

```
my-agent/
  .claude/
    commands/
      plan-agent.md
      create-agent.md
      exercise-agent.md
      deploy-agent.md
      add-tool.md
      add-skill.md
      add-memory.md
    rules/
    CLAUDE.md
  AGENTS.md
  agent.yaml
  .memoryhub.yaml              # Optional
  prompts/
    system.md
  tools/
    example_tool.py
  skills/
    example-skill/
      SKILL.md
      scripts/
      references/
      assets/
  rules/
    example_rule.md
  evals/
    README.md
    evals.yaml
    run_evals.py
    fixtures/
  src/
    fipsagents/
      baseagent/
        __init__.py
        agent.py
        tools.py
        prompts.py
        skills.py
        rules.py
        config.py
        memory.py
        llm.py
    agent.py
  Containerfile
  chart/
    Chart.yaml
    values.yaml
    templates/
  pyproject.toml
  Makefile
```

The `src/fipsagents/baseagent/` package contains the framework (installed via the `fipsagents` pip package). `src/agent.py` is the developer's subclass -- the only file most developers need to edit for a basic agent. Each concern (tools, prompts, skills, rules, config, memory, LLM) has its own module within the baseagent package, keeping files small and focused.

## Dependencies

The dependency footprint is deliberately minimal:

- **openai** -- LLM client (async SDK) for OpenAI-compatible endpoints (vLLM, LlamaStack, llm-d)
- **FastMCP v3** -- MCP client for remote tool server integration
- **memoryhub SDK** -- optional; MemoryHub programmatic access (one of several pluggable memory backends)
- **asyncpg** -- optional; PGVector memory backend (`pip install fipsagents[pgvector]`)
- **FastAPI + uvicorn** -- optional; OpenAI-compatible HTTP server (`pip install fipsagents[server]`)
- **pydantic** -- configuration validation and structured output schemas
- **httpx** -- async HTTP (also used internally by FastMCP)
- **python-frontmatter** -- parsing YAML frontmatter in prompt and skill files

Everything else comes from the Python standard library. There are no agent framework dependencies. This is intentional: frameworks impose opinions about control flow, state management, and composition that conflict with keeping the BaseAgent abstraction simple and the developer's subclass small.

## Workflow Template

The workflow framework lives in the `fipsagents` package (`packages/fipsagents/src/fipsagents/workflow/`) and implements a state-graph execution model for composing multiple agents and lightweight nodes into directed workflows. The workflow template (`templates/workflow/`) imports from `fipsagents.workflow` and provides a thin re-export shim at `src/workflow/` for backwards compatibility with existing scaffolded projects.

### Core Abstractions

**WorkflowNode** is a `typing.Protocol` defining the minimal contract: `async def process(self, state: T) -> T` and a `name` attribute. Both BaseNode and AgentNode satisfy this protocol through structural subtyping -- no inheritance coupling.

**BaseNode** is a minimal node class for routing, transformation, validation, and gating. It has a logger and a name but no LLM, tools, or MCP. Use it when a node's logic is pure Python without model calls.

**AgentNode** bridges BaseAgent into the workflow context. It extends BaseAgent, implements `step()` as a guard (raises NotImplementedError if called outside a workflow), and provides `process(state) -> state` as the method developers override. A workflow AgentNode has full access to `self.call_model()`, `self.use_tool()`, `self.memory`, `self.prompts`, and all other BaseAgent capabilities.

**WorkflowState** is a Pydantic BaseModel with `extra="forbid"` that developers subclass to define typed state. State carries only data -- execution metadata (timings, node history, retry counts) belongs in structured logs, not on the state object. This separation was a deliberate design decision based on prior experience with state objects that accumulated metadata and became unmanageable.

**Graph** provides a fluent API for wiring nodes and edges: `add_node()`, `add_edge()`, `add_conditional_edge()`, `add_error_edge()`, `set_entry_point()`. All mutation methods return `self` for chaining. The graph validates structural integrity before execution.

**WorkflowRunner** traverses the graph, passing state between nodes. It manages AgentNode lifecycle (calling `setup()` and `shutdown()` on all AgentNodes), applies per-node retry logic, routes to error edges when retries are exhausted, enforces a max-steps guard, and emits structured log events at every node transition.

The `@node` decorator marks classes for workflow registration, mirroring the `@tool` decorator pattern from BaseAgent.

### Current Scope (v1)

- Linear chains: A → B → C → END
- Conditional routing: A → (if condition) B else C
- Error edges: if node X fails after retries, route to node Y
- Per-node retry with configurable retry count
- Structured logging at every node transition
- Typed Pydantic state with extra-field rejection

### Deferred to v2

- Fan-out/fan-in (parallel node execution)
- Cycles (loop back to previous node) -- max-steps guard is already in place
- Checkpointing and resume
- Subgraph composition
- Event-driven wait (same shape as HITL -- a node's `process()` can await anything; this is an implementation detail, not a different paradigm)

## Code Execution Sandbox

Agents sometimes need to execute LLM-generated Python code -- solving math problems, transforming data, or validating logic. Running arbitrary code in the agent process is unacceptable, so the template provides an optional sandbox sidecar that agents opt into by adding a single tool.

### Architecture

The sandbox runs as a sidecar container in the same pod as the agent. The `code_executor` tool (a standard `@tool(visibility="llm_only")` in the agent's `tools/` directory) sends code to the sidecar over localhost. The sidecar validates the code, executes it in an isolated subprocess, and returns stdout/stderr/exit_code.

This is a tool, not a framework feature. Agents that don't need code execution don't carry the sidecar. Agents that do need it add the tool file and set `sandbox.enabled: true` in their Helm values.

### Pre-execution Guardrails

Before any code runs, an AST-based validator walks the parse tree and collects all violations in a single pass. Two checks are applied:

**Import allowlist.** Only 17 safe standard-library modules are permitted: math, statistics, itertools, functools, re, datetime, collections, json, csv, string, textwrap, decimal, fractions, random, operator, typing. Any other import is rejected with a message naming the blocked module.

**Pattern scanner.** The AST visitor blocks dangerous calls (`eval`, `exec`, `compile`, `open`, `__import__`, `getattr`, `setattr`, `delattr`, `breakpoint`, `input`), dangerous module attribute access (`subprocess.*`, `socket.*`, `importlib.*`, `os.system`, `os.popen`), and dangerous dunder attribute access (`__subclasses__`, `__globals__`, `__builtins__`).

All violations are returned at once so the LLM can fix everything in a single retry rather than playing whack-a-mole with one error at a time.

### Runtime Isolation

Code that passes guardrails is written to a temporary file and executed via `python3 -I` (isolated mode: no user site-packages, PYTHON* environment variables ignored) in a separate subprocess. The sidecar enforces a configurable timeout (default 10 seconds, max 30) and kills the process on expiry. Output is capped at 50 KB per stream.

The sidecar container runs with `readOnlyRootFilesystem: true` and an emptyDir mount at `/tmp` (10 Mi limit) for temporary code files. It drops all Linux capabilities and runs as non-root.

### Limitations (v1)

AST guardrails are a defense-in-depth layer, not a hard security boundary. Python's dynamic nature means a sufficiently creative attacker can find bypass vectors. The real security comes from layering: AST validation teaches the LLM what is allowed, while container-level constraints (non-root, read-only filesystem, dropped capabilities, resource limits) provide the actual enforcement. Issue #26 tracks v2 hardening, including running the sandbox in a separate pod with a deny-all-egress NetworkPolicy.

## Cross-Agent Platform Service

The v0.12.0 enterprise feature track added four stateful surfaces to BaseAgent's server layer — sessions (`/v1/sessions`), traces (`/v1/traces`), feedback (`/v1/feedback`), and metrics (`/metrics`). All four follow the same pattern: BaseAgent owns a pluggable store (Null / SQLite / Postgres) and the server exposes REST endpoints. This was the right call when most deployments had one or two agents.

In multi-agent deployments — say ten agents fronted by a single gateway and UI — per-agent ownership chafes: ten Postgres pools, ten schema-migration loops, fan-out for cross-agent queries ("show me all thumbs-down feedback this week"), schema-as-de-facto-API once N agents write the same table, and no auth boundary between agents that share storage. Sessions are conceptually cross-agent in the first place: a user talks to "the system," not to agent #4.

### Decision: remote-store adapter, with a sibling platform service

We extend the existing pluggable-store pattern with HTTP implementations rather than changing where state ownership lives. Concretely:

- **`HttpFeedbackStore`, `HttpSessionStore`, `HttpTraceStore`** are added to `fipsagents.server` alongside the Null/SQLite/Postgres implementations. They satisfy the same ABCs and speak HTTP to a central service.
- **`fips-agents/fipsagents-platform`** is a new sibling repo: a FastAPI service that owns one Postgres pool, one schema, one migration loop, and re-exposes the same store semantics as REST. It is itself implemented on top of the SQLite/Postgres ABCs — not a parallel implementation.
- **`gateway-template` gets a routing mode.** When the platform service is deployed, the gateway routes `/v1/sessions/*`, `/v1/feedback/*`, and `/v1/traces/*` directly to it rather than fanning out to per-agent endpoints. Per-agent endpoints become a fallback for single-agent deployments.
- **BaseAgent's HTTP endpoints stay** for backward compatibility. When an agent is configured with an `HttpXStore`, those endpoints become pass-throughs (or are turned off via config). Existing v0.12.0 deployments do not break.

### Why this shape

This is the least disruptive path that addresses the multi-agent rough edges without forcing a topology on small deployments:

- **Backward compatible.** Single-agent deployments keep their current SQLite/Postgres setup.
- **Multi-agent path is clean.** One new service, deployment-time config flag, switch one agent at a time by editing its `agent.yaml`.
- **Cost Tracking (#104) v1 + v2 have shipped on this foundation.** Per-session token usage attaches to session records via `SessionStore.update()` (added in fipsagents 0.14.0). Whether the session store is `SqliteSessionStore` or `HttpSessionStore`, the BaseAgent-side code is identical — the data lands wherever sessions land. SQLite/Postgres backends get cumulative semantics for free; HTTP-backed deployments converge on cumulative semantics via the platform's `GET /v1/sessions/{id}/cost_data` endpoint (closed in 0.14.2 + platform 0.2.1). `PricingConfig` and the computed-cost `GET /v1/sessions/{id}/usage` endpoint layer dollar amounts on top. `BudgetEnforcer` reads those dollars to enforce per-session and per-tenant limits — see the [Cost Tracking](#cost-tracking) section.
- **Auth boundary becomes possible.** The platform service has its own auth surface (JWT against the same Keycloak as the gateway), so per-agent service-account tokens can gate writes.
- **OTEL is not forced.** A future iteration can have the platform's `TraceStore` ship to OTEL internally, but adopters do not have to deploy a collector in step one.

### Tradeoffs

- **"Schema as contract" does not disappear** — it moves from raw Postgres to the platform's HTTP API. This is better (versioned, HTTP semantics for migration) but not free; we need an API stability policy.
- **New ops surface** — Helm chart, readiness probe, Postgres dependency. Worth it iff the deployment actually runs ≥2 agents.
- **Test infrastructure doubles** — every store ABC needs both in-process and HTTP-roundtrip tests.

### Out of scope for the initial extraction

- **Auth boundary between agents writing shared storage.** Needed if multi-tenant is in scope; otherwise punt and revisit when the second tenant arrives.
- **Cross-agent session continuity** (a user's conversation following them across agents). Protocol question, not a storage question.
- **Migrating traces to OTEL.** Already partially solved by `OTELTraceStore`; the platform's `TraceStore` can adopt it later without a topology change.
