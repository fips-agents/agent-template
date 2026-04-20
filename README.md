# agent-template

Agent templates for the `fips-agents` CLI. Scaffolds production-ready AI agents that deploy to OpenShift, communicate with LLMs via litellm (100+ providers), and let developers focus on prompts, tools, model selection, and evals.

## Status

Both templates (agent-loop and workflow) are implemented, along with the shared `fipsagents` package (on PyPI) and an optional code execution sandbox sidecar.

## How It Works

A developer runs `fips-agents create agent my-agent`, selects a template variant (agent loop or agentic workflow), and gets a project that compiles, runs locally, and deploys to OpenShift via Helm. The scaffolded project includes AI-assisted slash commands (`/plan-agent`, `/create-agent`, `/exercise-agent`, `/deploy-agent`) that guide development from design through deployment.

The core abstraction is BaseAgent -- a pure Python async class that handles LLM communication (via litellm), tool dispatch across two planes (agent-code and LLM-callable), MCP client connections (FastMCP v3), prompt loading, skill management (agentskills.io spec), and lifecycle. A typical agent subclass is 20-30 lines.

## Documentation

- [docs/](docs/) -- Architecture, design decisions, problem statement, and vision.
- [planning/](planning/) -- Requirements, scope, constraints, and next steps.
- [fips-agents/code-sandbox](https://github.com/fips-agents/code-sandbox) -- Code execution sandbox sidecar (extracted to standalone repo).

## Infrastructure

Agents built from this template run on OpenShift and consume services deployed by [rh-ai-quickstart/ai-architecture-charts](https://github.com/rh-ai-quickstart/ai-architecture-charts) (vLLM, LlamaStack, PGVector, etc.). The Helm chart in each scaffolded agent bundles only the agent itself.

## Related Projects

- [fips-agents/code-sandbox](https://github.com/fips-agents/code-sandbox) -- Code execution sandbox sidecar
- [redhat-ai-americas/mcp-server-template](https://github.com/redhat-ai-americas/mcp-server-template) -- Sister template for MCP servers
- [redhat-ai-americas/memory-hub](https://github.com/redhat-ai-americas/memory-hub) -- Optional enterprise memory layer
- [BerriAI/litellm](https://github.com/BerriAI/litellm) -- LLM client layer
- [agentskills.io](https://agentskills.io/specification) -- Skills specification
- [agents.md](https://agents.md/) -- AGENTS.md open standard
