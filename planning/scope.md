# Scope

## In Scope

**BaseAgent class and supporting modules.** The complete base class with LLM communication (via litellm), two tool planes with visibility control, MCP client (FastMCP v3), prompt loading, skill management (agentskills.io), rules loading, configuration management, optional MemoryHub integration (dual-path: MCP and SDK), and lifecycle methods with built-in protective patterns.

**Agent loop template.** The priority build. Scaffolds a single-agent loop with a working example that can run locally and deploy to OpenShift. This is the first thing that ships.

**Agentic workflow template.** Designed during ideation as a LangGraph-like state graph without the package dependency. The design is documented but implementation is deferred until the agent loop template is proven in production.

**Directory scaffolding.** The scaffolded project includes properly structured directories for prompts (Markdown with YAML frontmatter), tools (decorated Python functions), skills (agentskills.io spec), rules (plain Markdown), and evals (harness-agnostic format with a lightweight local runner).

**AI-assisted development experience.** The `.claude/` directory with slash commands (`/plan-agent`, `/create-agent`, `/exercise-agent`, `/deploy-agent`, `/add-tool`, `/add-skill`, `/add-memory`), rules, and CLAUDE.md that guide developers through the agent lifecycle.

**Helm chart.** A minimal chart that deploys the agent (Deployment, Service, ConfigMap, optional Route) without deploying any infrastructure services.

**Documentation.** Architecture design document, infrastructure setup guidance pointing to rh-ai-quickstart, and developer workflow documentation.

## Out of Scope

**Infrastructure deployment.** vLLM, LlamaStack, PGVector, MinIO, and other infrastructure services are deployed by rh-ai-quickstart/ai-architecture-charts. This project does not deploy, manage, or version-couple with those services.

**LlamaStack internals.** The agent treats LlamaStack as an external endpoint. It does not import LlamaStack libraries, configure OTel instrumentation, or manage guardrail policies. Those are LlamaStack's concern.

**Central tool registry (Tool Hub).** The vision for an enterprise tool registry with per-tool RBAC, quarantine capabilities, and forensic tracing is captured as a [side quest](../research/side-quests/tool-hub.md) for a future project. The current template's BaseAgent design supports it later -- every tool call goes through centralized infrastructure -- but the registry itself is not built here.

**Full eval framework.** The template scaffolds an evals directory and provides a lightweight runner, but it does not build a comprehensive evaluation framework. It supports plugging in external harnesses like Inspect AI or OpenAI evals.

**UI or frontend.** Agents built from this template are backend services. Any UI is a separate project.

**Multi-agent orchestration.** Beyond what the workflow template eventually provides (a state graph of agent nodes), this project does not build higher-level orchestration systems for coordinating independent agents. Multi-agent coordination through shared memory (MemoryHub) is supported, but an orchestration layer is not.

**TypeScript support.** Templates are Python-only. The Markdown-with-YAML-frontmatter prompt format is noted as harder to consume from TypeScript, but TypeScript agent templates are not in scope.

**MemoryHub deployment.** MemoryHub is a separate service. This project integrates with it but does not deploy or manage it.
