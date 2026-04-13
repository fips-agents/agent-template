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

**`packages/base-agent/`** is the shared BaseAgent framework, distributed as a pip-installable Python package. Both templates depend on it. Extracting BaseAgent into a shared package eliminates code duplication and ensures a single source of truth for the core agent abstraction.

**`templates/agent-loop/`** scaffolds a single-agent loop: read context, call model, act on response, repeat. This covers the majority of agent use cases. The developer subclasses BaseAgent, implements `step()`, and gets LLM communication, tool dispatch, MCP connections, and all other common concerns for free.

**`templates/workflow/`** scaffolds a directed graph of nodes with typed state. Nodes are either lightweight `BaseNode` instances (for routing, transformation, gating) or `AgentNode` instances (BaseAgent subclasses with full LLM/tools/MCP capabilities). The `WorkflowRunner` manages graph traversal, node lifecycle, per-node retry, error edges, and structured logging. State is a Pydantic model that flows through the graph -- execution metadata stays in logs, not on state.

Both templates share BaseAgent via the `base-agent` package, follow the same directory conventions (tools, prompts, skills, rules, evals), and use the same deployment model (immutable container images on OpenShift).

## BaseAgent

BaseAgent is the core abstraction. It is pure Python, async throughout, and carries no framework dependencies -- no LangChain, no LangGraph. A typical agent subclass is 20-30 lines of code because BaseAgent handles every common concern: LLM communication, tool dispatch, MCP connections, prompt loading, memory access, skill management, configuration, and lifecycle.

### LLM Client

All LLM communication goes through litellm, which provides a unified OpenAI-compatible interface to 100+ providers (vLLM, LlamaStack, Anthropic, OpenAI, Azure, Bedrock, and others). Switching providers is a configuration change -- update the model string prefix and endpoint URL -- not a code change.

BaseAgent exposes four methods for model interaction:

`call_model(messages, **kwargs)` makes a standard chat completion call and returns the response. This is the workhorse for most interactions.

`call_model_json(messages, schema, **kwargs)` requests structured output conforming to a Pydantic schema and returns a parsed, validated object. It handles the provider-specific details of requesting JSON mode or structured output.

`call_model_stream(messages, **kwargs)` returns an async generator that yields response chunks as they arrive, for use cases where latency to first token matters.

`call_model_validated(messages, validator_tool, **kwargs)` is a first-class pattern, not an afterthought. It calls the model, validates the output by invoking a tool (which can be a schema check, a domain-specific validator, or anything else registered in the tool system), and retries with exponential backoff if validation fails. This pattern recurs constantly in production agents -- extracting structured data from unstructured responses, ensuring outputs meet domain constraints -- and deserves dedicated support rather than being reimplemented in every agent.

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
from base_agent.tools import tool

@tool(visibility="agent_only")
async def validate_schema(data: dict, schema_name: str) -> bool:
    """Validate data against a named schema."""
    ...
```

The tool name comes from the function name, the description from the docstring, and parameters from type hints. No registration boilerplate.

### MCP Integration

BaseAgent includes a built-in MCP client (FastMCP v3) for connecting to remote tool servers. `connect_mcp(url)` connects to an MCP server, discovers its tools, and registers them with `llm_only` visibility by default -- the assumption being that MCP tools are designed for LLM-driven invocation. `disconnect_mcp(url)` handles clean disconnection. Discovered tools participate in the same logging, RBAC, and rate-limiting infrastructure as local tools.

### Prompts

`load_prompt(name, **variables)` loads a prompt from the `prompts/` directory, performs variable substitution, and returns the rendered text. `list_prompts()` returns available prompts with their metadata. The prompt format is described in its own section below.

### Memory

When a `.memoryhub.yaml` file is present, `self.memory` provides the MemoryHub SDK client for programmatic read/write from agent code. When absent, `self.memory` is `None` and the agent works without it. Memory integration is described in detail in its own section.

### Skills

`self.skills` is a dictionary of skill stubs loaded at startup. `load_skill(name)` activates a skill (loading its full content), and `unload_skill(name)` deactivates it. This progressive-disclosure pattern is described in the Skills section.

### Lifecycle

Every BaseAgent subclass follows the same lifecycle:

`setup()` loads configuration, connects to MCP servers, discovers local tools and prompts, initializes memory (if configured), and loads skill stubs. This runs once at startup.

`step()` is the abstract method that subclasses implement. It contains the agent's core logic -- one iteration of whatever the agent does.

`teardown()` disconnects MCP servers and performs cleanup. This runs once at shutdown.

`run()` is the loop driver. It calls `setup()`, then calls `step()` repeatedly until the agent signals completion or hits the configured maximum iteration count. Built-in protective patterns -- max iterations, exponential backoff on errors, rate limiting -- prevent runaway behavior.

### Conversation State

`messages` holds the current conversation history. `add_message()` appends to it. `clear_messages()` resets it. These are deliberately simple because conversation management strategies vary widely between agents, and BaseAgent should not impose a particular approach.

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

## MemoryHub Integration

MemoryHub integration is optional and first-class. When a developer wants memory capabilities, they run `memoryhub config init`, which generates two files: `.memoryhub.yaml` (behavioral configuration, committed to source) and a rules file for memory-related agent behavior.

BaseAgent detects `.memoryhub.yaml` during `setup()` and wires up two access paths:

**The MCP path** makes MemoryHub's 15 tools available to the LLM through the standard MCP client. The LLM can read and write memories as part of its tool-calling workflow.

**The SDK path** exposes `self.memory` for programmatic access from agent code. This is for cases where the agent logic itself needs to read or write memories -- caching intermediate results, maintaining state across iterations, or implementing retrieval patterns that the LLM shouldn't control directly.

When `.memoryhub.yaml` is absent, `self.memory` is `None` and MemoryHub tools are not registered. The agent works identically in either case; memory is purely additive.

For multi-agent deployments, multiple agents can connect to the same MemoryHub instance. Scope-based visibility (user, project, role, organization, enterprise) and RBAC control which agents can see which memories, enabling shared-memory architectures without coupling the agents to each other.

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

The Helm chart bundles only the agent itself: a Deployment, Service, ConfigMap, and optional Route. It does not deploy any infrastructure. The expectation is that vLLM, LlamaStack, PGVector, and other services are already running, deployed by `rh-ai-quickstart/ai-architecture-charts` or equivalent.

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
    base_agent/
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

The `src/base_agent/` package contains the framework. `src/agent.py` is the developer's subclass -- the only file most developers need to edit for a basic agent. Each concern (tools, prompts, skills, rules, config, memory, LLM) has its own module within base_agent, keeping files small and focused.

## Dependencies

The dependency footprint is deliberately minimal:

- **litellm** -- LLM client providing the OpenAI-compatible interface to 100+ providers
- **FastMCP v3** -- MCP client for remote tool server integration
- **memoryhub SDK** -- optional; MemoryHub programmatic access
- **pydantic** -- configuration validation and structured output schemas
- **httpx** -- async HTTP (also used internally by litellm and FastMCP)
- **python-frontmatter** -- parsing YAML frontmatter in prompt and skill files

Everything else comes from the Python standard library. There are no agent framework dependencies. This is intentional: frameworks impose opinions about control flow, state management, and composition that conflict with keeping the BaseAgent abstraction simple and the developer's subclass small.

## Workflow Template

The workflow template (`templates/workflow/`) implements a state-graph execution model for composing multiple agents and lightweight nodes into directed workflows.

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
