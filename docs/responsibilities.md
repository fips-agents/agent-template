# Responsibilities and Non-Goals

This document is for contributors. It describes what `agent-template`
is, what it is not, and which adjacent layer owns what. Read it before
proposing a feature — it will save you time.

The boundaries below are deliberate. They are not permanent. If you
have a concrete use case the current boundaries do not serve, open a
discussion issue and make the case. The point of writing them down is
so we can argue with them, not so we can hide behind them.

## What this project is

`agent-template` is a general-purpose harness for building production
AI agents that run on Red Hat OpenShift. It scaffolds an immutable
container image with code, tools, prompts, skills, and rules baked in,
and ships it through a Helm chart that consumes platform services
(inference, memory, identity, tracing) provided by adjacent layers.

It is not a coding agent. It is not an IDE. It is not a chat
application. It is not a multi-tenant control plane. It is one tier in
a stack — the tier that runs the agent loop.

## What each adjacent layer owns

The fips-agents stack is intentionally layered. When you find yourself
about to add a feature, ask whether the concern belongs here or in one
of the layers below. If it belongs elsewhere, the right move is usually
to wire `agent-template` to consume the existing service rather than
to reimplement it.

### OGX (rebrand of LlamaStack) — orchestration, safety, observability

OGX owns:

- Server-side orchestration via the Responses API (`/v1/responses`):
  MCP tool dispatch, shield enforcement, the inference loop.
- Shields and guardrails (Llama Guard, Prompt Guard, custom shields).
  Registered in OGX `config.yaml`; agents reference them by ID.
- Moderation (`/v1/moderations`) and structured safety logs.
- OpenTelemetry trace emission for the orchestration it performs.
- The model registry — vLLM, llm-d, and any other inference provider
  is abstracted behind OGX from the agent's perspective.

`agent-template` consumes OGX as an HTTP endpoint. We do not
reimplement Llama Guard. We do not maintain our own shield registry.
We do not orchestrate MCP server-side when OGX is present (platform
mode delegates that to OGX). When platform mode is disabled, the
framework orchestrates MCP client-side via FastMCP v3 — that path
exists because not every deployment runs OGX, not because we want to
duplicate OGX's capabilities.

### kagenti — identity, RBAC, multi-tenant boundaries

kagenti owns:

- Identity (who is this caller?).
- RBAC (what is this caller allowed to do?).
- Multi-tenant isolation at the service-mesh level.
- Per-tenant network policy and authn/authz.

`agent-template` accepts identity. It does not manage it. The
permission policy planned for this project (allow/deny/ask per tool
call) is about *capability gating* — which tools the LLM may invoke,
with which arguments, under which scope. It layers on top of the
identity kagenti provides. It does not replace kagenti's RBAC.

If you find yourself proposing a multi-tenant feature, the answer is
almost always "kagenti owns this." The exceptions — per-session
isolation of conversation history, per-tool permission scoping — are
narrow and well-defined.

### Memory backends — persistent agent memory

The framework defines a `MemoryClientBase` ABC. Implementations:

- `memoryhub` — MemoryHub backend (centralized, governed, contradiction-tracked).
- `markdown` — file-based, Git-friendly, dev/edge.
- `sqlite` — local persistence, single-file.
- `pgvector` — Postgres + pgvector, vector search.
- `llamastack` — delegate to OGX/LlamaStack memory.
- `custom` — bring your own implementation.
- `null` — no memory, zero overhead.

We use the abstraction. We do not reinvent it. MemoryHub is one
backend among many — when discussing memory features, phrase them
generically against the ABC, not against MemoryHub specifically.
Memory CRUD, contradiction tracking, scoping (user / project /
organizational / enterprise) are properties of the backend, not of
this project.

### OpenShift — platform, builds, deployment, monitoring

OpenShift owns:

- The cluster, namespaces, and project boundaries.
- Secret management (Secrets, ESO, Vault integration).
- Network policy and service mesh.
- Build pipelines (BuildConfig, Tekton).
- Built-in monitoring (Prometheus, Grafana, Alertmanager).
- Workspace lifecycle (a "workspace" in our world is an OpenShift
  project — we do not need a parallel concept).

`agent-template` deploys as a tenant of OpenShift. The Helm chart
declares what the agent needs; OpenShift provides it. We do not
manage cluster-level resources. We do not ship a control plane.

### Inference layer — the model itself

vLLM, llm-d, llama.cpp, Ollama, OpenAI-compatible cloud endpoints — these
host the model. The framework speaks OpenAI-compatible HTTP via the
async `openai` SDK. When the inference provider is not natively
OpenAI-compatible, the LLM adapter sidecar at `packages/llm-adapter/`
translates.

We are an OpenAI-compatible client. We do not host models. We do not
fine-tune. We do not benchmark inference performance. We do not
distribute model weights.

### Adjacent fips-agents components — the rest of the stack

`agent-template` is one repo in a stack of cooperating components.
Knowing what each one owns prevents adding things to the wrong place.

- **`gateway-template`** — Go HTTP gateway. Sits between `ui-template`
  and one or more agents. Handles authentication, rate limiting,
  request shaping, multi-agent routing. Coding-agent-style features
  that conflate UI and agent (slash-command parsing, transcript
  rendering, command-history UX) belong here, not in BaseAgent.
- **`ui-template`** — chat UI. Renders conversations, streams events,
  handles user input. All rendering decisions live here. If you find
  yourself thinking about colors, keybindings, or panel layouts, you
  are in the wrong repo.
- **`code-sandbox`** — sandboxed code execution as an MCP server. If
  an agent needs to run code (Python, shell), it calls this MCP
  server through the standard MCP transport. The sandbox owns the
  isolation boundary, resource limits, and language runtimes.
- **`mcp-server-template`** — scaffolding for new MCP servers. When
  you want to expose a new capability set (FHIR tools, OData, runbook
  search), the right move is usually a new MCP server here, not new
  built-in tools in this repo.
- **`fipsagents-platform`** — REST veneer over the `fipsagents.server`
  ABCs for cross-agent shared state (sessions, traces, files,
  metrics) when multiple agents need to share. Single-agent
  deployments do not need it.
- **`fips-agents-cli`** (`fips-agents` command) — the scaffolding CLI.
  It clones this template, the workflow template, and others, then
  customizes them for a specific agent. It does not ship runtime
  code; it ships scaffolding.

When proposing a feature, ask: does this belong in BaseAgent (the
agent loop), in `fipsagents.server` (HTTP / persistence / observation),
in the gateway (request shaping, multi-agent routing), in the UI
(rendering), in an MCP server (a capability set), or in an extension
(a domain-specific bundle of tools / prompts / skills)? Most "I want
to add X" answers are not "X goes in BaseAgent."

## Explicit non-goals

The list below is what we have actively decided not to build into this
project. Each item has a rationale; the rationale is the load-bearing
part — if your situation invalidates it, that is a real argument, not
just a preference clash.

- **TUI, themes, custom keybindings, IDE/editor integration.** UI
  decisions live in `ui-template`. This project ships HTTP servers
  (OpenAI-compatible chat completions, plus a few adjacent endpoints).
  How that gets rendered to a human is not our concern. Adding
  terminal UI here would either fork into a competing UI surface or
  duplicate work `ui-template` already does.
- **Language Server Protocol integration, glob/grep/edit/apply_patch
  coding tools, formatters, terminal/PTY emulators.** These are the
  vocabulary of a coding agent. A general-purpose harness should not
  ship them. They belong in a code-tools extension that downstream
  projects opt into. Bundling them into the core would constrain
  every non-coding agent (customer service, ops, healthcare,
  document processing) with capabilities they neither want nor can
  audit.
- **Git shadow-repo snapshots for file-edit revert.** A coding-agent
  feature that doesn't generalize. The general-purpose subset is
  session fork and revert at the data layer (planned), which works
  for any session regardless of whether it touched files.
- **Cross-device session sync via event sourcing.** This drives
  "phone-controls-IDE" use cases that are not the deployed-on-OpenShift
  shape we target. It also imposes a substantial complexity tax
  (event log, conflict resolution, replay semantics) for a benefit
  that vanishes in a server-deployed agent.
- **Public share-this-conversation URLs.** A compliance footgun in
  regulated environments (healthcare, finance, regulated industrial).
  Tenant-owned export — the operator can pull a session and share it
  through tenant-controlled channels — is the right answer. Public
  unauthenticated URLs are not.
- **Workspaces, project registry, user dashboards, control plane.**
  kagenti owns multi-tenant boundaries; OpenShift owns workspace
  lifecycle. We do not need a parallel concept of "workspace" in
  `agent-template`. If a feature needs the idea of "what is the
  current project", it should derive that from kagenti's identity
  context, not from a registry we maintain.
- **Agent Client Protocol (ACP) or any IDE-driver protocol.** ACP is
  designed for IDEs to drive coding agents. Our equivalent ecosystem
  play is OpenAI-compatible HTTP — every gateway, UI, automation
  tool, and notebook in the world speaks it. We are not adding a
  second protocol to chase IDE integration that we are not the right
  layer to provide.
- **npm-style runtime plugin systems / pip-install-at-startup
  extensibility.** Incompatible with the immutable-image model.
  Extension points are ABC-based: pluggable memory backends,
  pluggable session/trace stores, custom tool implementations
  registered via `@tool`, custom node implementations registered via
  `@node`. Anything dynamic enough to need runtime install belongs
  in an MCP server, not in the agent's process.
- **Multi-language agent loops (TypeScript, Go, Java BaseAgent
  ports).** Python is the language of the agent. Other languages
  consume it via OpenAI-compatible HTTP. Maintaining parallel
  framework implementations across languages is a tax we are not
  paying.

## Domain-specific capabilities go in extensions

A general-purpose template cannot be all things. Healthcare needs FHIR
(patient / encounter / observation tools, terminology services,
deidentification utilities). Industrial needs OData and OPC UA.
Software engineering needs a code-tools extension (glob, scoped
read/write, AST navigation, sandboxed execution). Retail needs
catalog and inventory tools. Finance needs compliance helpers.

None of these belong in the main template. They belong in extensions —
domain-specific bundles that contribute tools, prompts, skills,
optional memory backends, and configuration to an agent at scaffold
time.

The shape of the extension model is currently being designed. See the
"template extensions / add-ons" architecture discussion issue for the
open questions: layout (separate template repos, packaged
distributions, submodules), discovery (declared in `agent.yaml`),
surface (what an extension contributes), versioning, namespacing, and
deployment.

The point of getting extensions right early: every domain ends up
needing the same shape. Without a first-class model, every team forks
the template, copies the same five tools into it, and we end up with
N divergent forks instead of N domain extensions composing on a
shared core.

## How to challenge a non-goal

The boundaries above are explicit so you can argue with them. If you
have a concrete use case the current scope does not serve:

1. Open a discussion issue (label `enhancement`, mention the relevant
   non-goal in the body).
2. State the use case in plain language. Lead with the user, not the
   feature.
3. Explain why one of the adjacent layers (OGX, kagenti, gateway, UI,
   an MCP server, an extension) cannot serve the need.
4. Propose where in the stack the capability would live if added.

The most persuasive arguments are concrete: a real downstream agent
that hits a wall, a specific compliance requirement, a real-world
integration that the current scope blocks. Hypotheticals carry less
weight than scars.
