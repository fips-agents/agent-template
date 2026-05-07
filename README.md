# agent-template

Agent templates for the `fips-agents` CLI. Scaffolds production-ready AI agents that deploy to Red Hat AI, communicate with LLMs via the OpenAI async SDK (any OpenAI-compatible endpoint), and let developers focus on prompts, tools, model selection, and evals.

## Status

Both templates (agent-loop and workflow) are implemented, along with the shared `fipsagents` package (on PyPI) and an optional code execution sandbox sidecar.

## How It Works

A developer runs `fips-agents create agent my-agent`, selects a template variant (agent loop or agentic workflow), and gets a project that compiles, runs locally, and deploys to Red Hat AI via Helm. The scaffolded project includes AI-assisted slash commands (`/plan-agent`, `/create-agent`, `/exercise-agent`, `/deploy-agent`) that guide development from design through deployment.

The core abstraction is BaseAgent -- a pure Python async class that handles LLM communication (via the OpenAI async SDK), tool dispatch across two planes (agent-code and LLM-callable), MCP client connections (FastMCP v3), prompt loading, skill management (agentskills.io spec), and lifecycle. A typical agent subclass is 20-30 lines.

## Documentation

- [docs/](docs/) -- Architecture, design decisions, problem statement, and vision.
- [planning/](planning/) -- Requirements, scope, constraints, and next steps.
- [fips-agents/code-sandbox](https://github.com/fips-agents/code-sandbox) -- Code execution sandbox sidecar (extracted to standalone repo).

## Infrastructure

Agents built from this template run on Red Hat AI and consume services deployed by [rh-ai-quickstart/ai-architecture-charts](https://github.com/rh-ai-quickstart/ai-architecture-charts) (vLLM, LlamaStack, PGVector, etc.). The Helm chart in each scaffolded agent bundles only the agent itself.

## Related Projects

- [fips-agents/gateway-template](https://github.com/fips-agents/gateway-template) -- OpenAI-compatible Go gateway that fronts agent deployments
- [fips-agents/ui-template](https://github.com/fips-agents/ui-template) -- Minimal chat UI that talks to the gateway
- [fips-agents/fipsagents-platform](https://github.com/fips-agents/fipsagents-platform) -- Cross-agent platform service for centralized feedback / sessions / traces (multi-agent topologies)
- [fips-agents/code-sandbox](https://github.com/fips-agents/code-sandbox) -- Code execution sandbox sidecar
- [fips-agents/mcp-server-template](https://github.com/fips-agents/mcp-server-template) -- Sister template for MCP servers
- [redhat-ai-americas/memory-hub](https://github.com/redhat-ai-americas/memory-hub) -- Optional enterprise memory layer
- [agentskills.io](https://agentskills.io/specification) -- Skills specification
- [agents.md](https://agents.md/) -- AGENTS.md open standard
