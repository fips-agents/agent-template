# Architecture

agent-template is a monorepo of agent templates for the `fips-agents` CLI. It scaffolds production-ready AI agents that deploy to OpenShift, communicate with LLMs through litellm's 100+ provider integrations, and let developers focus on the work that actually differentiates their agent: prompts, tools, model selection, and evals.

This document describes the system architecture, core abstractions, and the reasoning behind each design decision.

## System Context

The agent template sits in a specific layer of a broader stack. Understanding this layering is essential because the template deliberately excludes concerns handled elsewhere.

**Infrastructure layer** is provided by `rh-ai-quickstart/ai-architecture-charts` -- composable Helm charts that deploy vLLM, LlamaStack, PGVector, MinIO, and other services onto OpenShift. This project does not deploy or manage any of that infrastructure. Agents built from this template consume those services through well-defined APIs.

**The fips-agents CLI** clones this repository when a developer runs `fips-agents create agent my-agent`, following the same pattern established by `redhat-ai-americas/mcp-server-template`. The CLI selects a template variant, copies it into a new project directory, and hands off to the developer.

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

All LLM communication goes through litellm, which provides a unified OpenAI-compatible interface to 100+ providers (vLLM, LlamaStack, Anthropic, OpenAI, Azure, Bedrock, and others). Switching providers is a configuration change -- update the model string prefix and endpoint URL -- not a code change.

**Important:** litellm's OpenAI provider requires `OPENAI_API_KEY` to be set even when connecting to unauthenticated endpoints (e.g., a vLLM instance with no auth). Set it to any non-empty string (e.g., `OPENAI_API_KEY=not-required`) in the agent's environment. Without this, litellm raises `AuthenticationError` before the request is sent.

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

Most FIPS-Agents chat deployments sit behind an OpenAI-compatible HTTP endpoint so a UI, gateway, or another agent can call them through the ecosystem-standard `/v1/chat/completions` contract. `fipsagents.server.OpenAIChatServer` is the canonical implementation: a FastAPI app that takes a `BaseAgent` subclass and exposes `/v1/chat/completions` (sync + SSE), `/healthz`, and `/readyz` with no hand-written HTTP glue.

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

Each jump is a real jump, not a sliding scale of features. If you find yourself asking "should the markdown backend have search ranking?" the answer is usually "no, move to SQLite." The primer at `research/agent-memory-primer.md` goes into more detail on when each level is the right choice.

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

The message role is controlled by `config.memory.prefix_role` (default `"system"`). Models that support the OpenAI harmony format (gpt-oss-20b, o-series) can set this to `"developer"` to place memories in the harmony hierarchy (`system > developer > user`). See [#49](https://github.com/redhat-ai-americas/agent-template/issues/49) for a planned probe to detect model support at runtime.

Subclasses override `build_memory_prefix()` to customise the query, formatting, or to return `None` unconditionally when they prefer per-turn recall. Agents that need fresher memory mid-session call `self.memory.search()` directly from their `astep_stream` override -- the prefix is a session-level stable cache, not a replacement for dynamic retrieval.

The markdown backend's `search(query="")` is designed to pair well with this: it returns every section or file in stable file order, so the prefix is deterministic across restarts. Other backends (SQLite, PGVector, MemoryHub) can use the same pattern; results are retrieved once at session start and pinned for the session's lifetime.

**The SDK path** exposes `self.memory` for programmatic access from agent code. This is for cases where the agent logic itself needs to read or write memories -- caching intermediate results, maintaining state across iterations, or implementing retrieval patterns that the LLM shouldn't control directly.

**The MCP path** (MemoryHub only) makes MemoryHub's tools available to the LLM through the standard MCP client. The LLM can read and write memories as part of its tool-calling workflow.

Custom backends can be registered via `backend: custom` with a `backend_class` dotted import path in `agent.yaml`. See `docs/custom-memory-backend.md` for the full guide.

For multi-agent deployments with MemoryHub, multiple agents can connect to the same instance. Scope-based visibility (user, project, role, organization, enterprise) and RBAC control which agents can see which memories, enabling shared-memory architectures without coupling the agents to each other.

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

- **litellm** -- LLM client providing the OpenAI-compatible interface to 100+ providers
- **FastMCP v3** -- MCP client for remote tool server integration
- **memoryhub SDK** -- optional; MemoryHub programmatic access (one of several pluggable memory backends)
- **asyncpg** -- optional; PGVector memory backend (`pip install fipsagents[pgvector]`)
- **FastAPI + uvicorn** -- optional; OpenAI-compatible HTTP server (`pip install fipsagents[server]`)
- **pydantic** -- configuration validation and structured output schemas
- **httpx** -- async HTTP (also used internally by litellm and FastMCP)
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
