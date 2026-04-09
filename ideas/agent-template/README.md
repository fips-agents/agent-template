# agent-template

A monorepo of agent templates for the `fips-agents` CLI. Scaffolds production-ready AI agents that deploy to OpenShift, talk to LLMs via litellm (100+ providers), and let developers focus on prompts, tools, model selection, and evals instead of boilerplate.

## Current Status

Ideation complete (3 sessions, 2026-04-09). Ready for `/propose`.

## Key Decisions

### Architecture
- **Monorepo** with two template directories: `templates/agent-loop/` (priority) and `templates/agentic-workflow/` (deferred)
- **BaseAgent class** — pure Python, async, no framework dependencies. Owns model calling (via litellm), two tool planes, prompt loading, MCP client (FastMCP v3), MemoryHub SDK, skills, lifecycle.
- **Agent subclasses are ~20-30 lines** — implement only the interesting parts
- **litellm** for LLM calls — portable across vLLM, LlamaStack, Anthropic, OpenAI, Azure, Bedrock, 100+ providers
- **LlamaStack is external** — the agent points at an endpoint via litellm, doesn't know what's behind it

### Developer Experience
- **Template is a public repo** cloned by `fips-agents` CLI (same pattern as mcp-server-template)
- **Slash commands guide the workflow:** `/plan-agent` → `/create-agent` → `/exercise-agent` → `/deploy-agent`
- **Utility commands:** `/add-tool`, `/add-skill`, `/add-memory`
- `.claude/` directory with commands, rules, and CLAUDE.md drives the AI-assisted development experience

### Tools
- **@tool decorator** (FastMCP convention) — auto-discovered from `tools/` directory
- **Two tool planes** — agent-code tools (plane 1, invisible to LLM) and LLM-callable tools (plane 2)
- **Visibility control:** `agent_only`, `llm_only`, `both` — both planes go through BaseAgent for logging/RBAC/retry
- **MCP tools** via FastMCP v3 client — discovered and registered automatically

### Content & Config
- **Prompts as Markdown with YAML frontmatter** — one file per prompt
- **Skills follow agentskills.io spec** — directory per skill, progressive disclosure
- **Rules as plain markdown** — one file per rule in `rules/`
- **agent.yaml** with env var substitution — same config works locally and on OpenShift
- **Immutable container images** — code, tools, prompts, skills, rules all baked in

### Integrations
- **MemoryHub** — optional first-class integration (dual-path: MCP for LLM, SDK for agent code). Configured via `memoryhub config init`.
- **rh-ai-quickstart/ai-architecture-charts** — assumed infra layer (vLLM, LlamaStack, PGVector, etc.)
- **Helm chart** bundles just the agent — infra services are pre-deployed

### Patterns
- **Protective patterns built in** — max iterations, exponential backoff/retry, rate limiting
- **call_model_validated()** — first-class pattern for output validation with retry
- **Workflow manager** (deferred) — LangGraph concepts without the package dependency

## Template Structure

```
my-agent/
├── .claude/
│   ├── commands/                # Slash commands for workflow
│   │   ├── plan-agent.md
│   │   ├── create-agent.md
│   │   ├── exercise-agent.md
│   │   ├── deploy-agent.md
│   │   ├── add-tool.md
│   │   ├── add-skill.md
│   │   └── add-memory.md
│   ├── rules/                   # AI assistant rules
│   └── CLAUDE.md
├── AGENTS.md                    # Open standard, minimal
├── agent.yaml                   # Operational config
├── .memoryhub.yaml              # Optional (memoryhub config init)
├── prompts/
│   └── system.md
├── tools/
│   └── example_tool.py          # @tool decorator
├── skills/
│   └── example-skill/
│       └── SKILL.md             # agentskills.io spec
├── rules/
│   └── example_rule.md
├── evals/
│   ├── README.md
│   ├── evals.yaml
│   ├── run_evals.py
│   └── fixtures/
├── src/
│   ├── base_agent/              # BaseAgent package
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── tools.py
│   │   ├── prompts.py
│   │   ├── skills.py
│   │   ├── rules.py
│   │   ├── config.py
│   │   ├── memory.py
│   │   └── llm.py
│   └── agent.py                 # ~20-30 line subclass
├── Containerfile
├── chart/                       # Helm chart
├── pyproject.toml
└── Makefile
```

## Related Projects

- [rh-ai-quickstart/ai-architecture-charts](https://github.com/rh-ai-quickstart/ai-architecture-charts) — Infra layer
- [redhat-ai-americas/memory-hub](https://github.com/redhat-ai-americas/memory-hub) — Optional enterprise memory layer
- [redhat-ai-americas/mcp-server-template](https://github.com/redhat-ai-americas/mcp-server-template) — Sister template (MCP servers)
- [BerriAI/litellm](https://github.com/BerriAI/litellm) — LLM client layer
- [agentskills.io](https://agentskills.io/specification) — Skills specification
- [agents.md](https://agents.md/) — AGENTS.md open standard

## Side Quests

- [Tool Hub](side-quests/tool-hub.md) — Enterprise tool registry with RBAC and quarantine
