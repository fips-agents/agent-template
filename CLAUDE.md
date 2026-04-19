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
- **litellm** is the LLM client -- provides OpenAI-compatible interface to 100+ providers
- **FastMCP v3** is the MCP client -- not v2
- **Two tool planes**: agent-code tools (plane 1, invisible to LLM) and LLM-callable tools (plane 2). Both go through BaseAgent for logging/RBAC/retry. Visibility per tool: `agent_only`, `llm_only`, `both`.
- **@tool decorator** for local tools, same convention as FastMCP. Auto-discovered from `tools/` directory.
- **Prompts** are Markdown with YAML frontmatter, one file per prompt in `prompts/`
- **Skills** follow the agentskills.io spec exactly -- directory per skill, SKILL.md with frontmatter, progressive disclosure
- **Rules** are plain Markdown files in `rules/`, no frontmatter
- **agent.yaml** with `${VAR:-default}` env var substitution for configuration
- **Immutable container images** -- code, tools, prompts, skills, rules all baked in. Only env-specific config is external.
- **Pluggable memory backends** -- memoryhub, markdown, sqlite, pgvector, llamastack, custom, or null. `self.memory` is always a `MemoryClientBase` (never None). MemoryHub adds MCP path for LLM-driven memory tools. `build_memory_prefix()` injects a stable memory block at setup time (role configurable via `memory.prefix_role`: `system` or `developer`).
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
  sandbox/                 # Code execution sandbox sidecar (FastAPI, UBI)
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

- litellm -- LLM client
- fastmcp (v3) -- MCP client
- memoryhub -- optional, MemoryHub memory backend
- asyncpg -- optional, PGVector memory backend (`pip install fipsagents[pgvector]`)
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
- Do not use the openai SDK directly -- use litellm for provider portability
- Do not put tool dispatch logic in agent subclasses -- use `self.use_tool()`
- Do not hardcode model names or endpoints -- use agent.yaml with env var substitution
- Do not create ConfigMaps for prompts -- prompts are baked into the image for traceability
- Do not skip the `visibility` parameter on tools -- every tool must declare its plane
