# Problem Space

Building AI agents that run on OpenShift is still too much boilerplate. Developers spend more time on infrastructure wiring — model client setup, tool registration, MCP connections, Helm charts, tracing config — than on the parts that actually matter: prompt engineering, model selection, RAG connections, and evals.

## What Exists Today

- **rh-ai-quickstart/ai-architecture-charts** solves the infra layer (vLLM, LlamaStack, PGVector) but has no developer-facing scaffolding or agent abstractions.
- **red-hat-data-services/agentic-starter-kits** provides sample agents but they're a zoo of different patterns, not a reusable scaffold.
- **LangGraph templates** (`langgraph new`) offer scaffolding but no OpenShift deployment story, no FIPS awareness, no LlamaStack integration.
- **OpenAI Agents SDK** has clean abstractions but is OpenAI-optimized and has no Kubernetes deployment story.

## The Gap

Nobody has built: CLI scaffolding + BaseAgent pattern + OpenAI-compatible API portability + OpenShift Helm deployment + AGENTS.md/skills/rules/commands as a single cohesive tool. The infra exists. The framework abstractions exist. The glue layer for enterprise developers is missing.

## Who Feels This

- Developers building AI agents on OpenShift AI who want to focus on agent logic, not plumbing
- Teams that need agents to be portable across LlamaStack, raw vLLM, and cloud providers
- Organizations using `fips-agents` CLI that need a standard way to create new agents
