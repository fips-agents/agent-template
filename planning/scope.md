# Scope

## In Scope

- BaseAgent class with full API surface (model calling, tool planes, MCP client, memory, skills, lifecycle)
- Agent loop template (priority build)
- Agentic workflow template (design now, build later)
- Prompt, tool, skill, rule, and eval directory scaffolding
- Skills following agentskills.io spec
- AGENTS.md following open standard
- Helm chart for agent-only deployment
- Optional MemoryHub integration (dual-path: MCP + SDK)
- Documentation pointing to rh-ai-quickstart for infra
- MCP client integration via FastMCP v3
- OpenAI-compatible LLM client in BaseAgent
- Immutable container image pattern (everything baked in)
- Protective patterns (max iterations, backoff/retry, rate limiting)

## Out of Scope

- Infrastructure deployment (vLLM, LlamaStack, PGVector, etc.) — handled by rh-ai-quickstart/ai-architecture-charts
- LlamaStack library imports or direct OTel instrumentation — LlamaStack is an external endpoint
- Central tool registry (Tool Hub) — captured as side quest for future project
- Building a full eval framework — scaffold and support external harnesses
- UI/frontend for agents
- Multi-agent orchestration beyond what the workflow template eventually provides
- TypeScript support (Python only)
- MemoryHub deployment — it's a separate service, we just integrate with it
