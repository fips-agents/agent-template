# Next Steps

## Resolved Across All Sessions

### Session 1 (2026-04-09)
- Monorepo with two template directories
- BaseAgent owns all common concerns
- LlamaStack is external (agent speaks OpenAI-compatible API)
- rh-ai-quickstart/ai-architecture-charts is the infra layer
- Helm chart bundles just the agent
- Prompts as Markdown with YAML frontmatter
- Immutable container images

### Session 2 (2026-04-09)
- Full BaseAgent API surface defined
- Two tool planes (agent-code vs LLM-callable) with visibility control
- Agent config as YAML with env var substitution
- AGENTS.md follows open standard (minimal for most agents)
- Skills follow agentskills.io spec
- Evals scaffold with harness-agnostic format
- MemoryHub as optional first-class integration (dual-path: MCP + SDK)
- Tool Hub captured as side quest

### Session 3 (2026-04-09)
- Template is a public repo cloned by fips-agents CLI (same pattern as mcp-server-template)
- Slash commands: /plan-agent, /create-agent, /exercise-agent, /deploy-agent, /add-tool, /add-skill, /add-memory
- Tool auto-discovery via @tool decorator (FastMCP convention)
- Rules as plain markdown in rules/ directory
- litellm as the LLM client layer (100+ providers, no framework lock-in)
- BaseAgent is pure Python, async, no framework dependencies
- Workflow manager: LangGraph concepts without the package (deferred build)
- Full template repo structure defined including .claude/ directory

## Remaining Open Questions

- **Slash command content** — What goes in each .claude/commands/*.md file? Need to write the actual command prompts.
- **CLAUDE.md content** — What guidance does the template's CLAUDE.md provide? Needs to cover the full agent development workflow.
- **BaseAgent package** — Does base_agent live in the template repo (copied into each agent) or as a pip-installable package that agents depend on? If it's copied, every agent has its own copy. If it's a package, agents share one version.
- **Example agent** — What's the scaffolded example agent? A simple ReAct loop? A hello-world that calls one tool?
- **Makefile targets** — What make commands does the template provide? (install, run-local, test, build, deploy, etc.)
- **Testing strategy** — How do we test BaseAgent itself? How do we test scaffolded agents?

## Ready for Proposal

All major architectural decisions are resolved. The remaining questions are implementation details that belong in a `/propose` technical proposal.

When ready:
- `/propose` — Create a detailed technical proposal with implementation plan
