# Research

## GitHub Projects

### rh-ai-quickstart/ai-architecture-charts
- 14 stars, active (356 commits)
- Composable Helm charts for LlamaStack, vLLM, PGVector, MinIO, MCP servers, Llama Guard on OpenShift
- **Role in our project:** The assumed infra layer. We build on top of this.

### red-hat-data-services/agentic-starter-kits
- 9 stars, updated Feb 2026
- Eight agent implementations across LangGraph, LlamaIndex, CrewAI, AutoGen, vanilla Python
- **Role in our project:** Reference implementations. Our template provides the structured scaffold these lack.

### rh-ai-kickstart/llama-stack-ReAct
- 1 star, 31 commits
- ReAct agent + LlamaStack + vLLM on OpenShift with single-command Helm deploy
- **Role in our project:** Reference architecture for LlamaStack + vLLM + Helm pattern.

### redhat-ai-americas/mcp-server-template
- The template pattern we're following for developer experience
- Uses `.claude/commands/` for slash commands: /plan-tools → /create-tools → /exercise-tools → /deploy-mcp
- fips-agents CLI clones this repo; developers follow the slash command workflow
- **Role in our project:** The model for our template's developer experience. We replicate this pattern with /plan-agent → /create-agent → /exercise-agent → /deploy-agent.

### redhat-ai-americas/memory-hub
- Centralized memory system for AI agents on OpenShift AI
- FastMCP 3 server with 15 tools, multi-tier scoping, semantic search, RBAC, audit trail
- Python SDK for programmatic access, CLI for terminal use
- `memoryhub config init` wizard for agent integration
- **Role in our project:** Optional first-class integration. Dual-path: MCP for LLM, SDK for agent code.

### fastapi-langgraph-agent-production-ready-template
- 2.1k stars, 489 forks
- FastAPI + LangGraph + PostgreSQL/pgvector + Prometheus/Grafana + JWT auth
- **Role in our project:** Shows what the community wants but lacks enterprise features. We fill that gap.

### LangGraph (langchain-ai/langgraph)
- 10k+ stars
- State graph abstraction for agent workflows: nodes, edges, conditional routing, checkpointing
- **Role in our project:** Conceptual model for our workflow manager (deferred build). Same ideas, no package dependency.

### openai/openai-agents-python
- Very high activity
- Clean Agent class + Runner abstraction, multi-agent handoffs
- **Role in our project:** Validates the BaseAgent pattern. Their Agent class is similar in spirit.

## Libraries & Tools

### litellm (BerriAI/litellm)
- 42.7k stars, 36,703 commits, actively maintained
- OpenAI-compatible interface for 100+ LLM providers
- Supports tool calling, structured output, streaming, batch processing
- Provider portability via model string prefix (e.g., "anthropic/claude-sonnet-4-20250514", "openai/gpt-4")
- **Role in our project:** The LLM client layer in BaseAgent. One dependency for full provider portability.

### FastMCP v3
- Latest version of the FastMCP framework
- Client and server for MCP protocol
- Streamable-HTTP transport (SSE deprecated)
- **Role in our project:** MCP client in BaseAgent for tool discovery and execution.

### agentskills.io
- Open specification for agent skills
- Directory-per-skill with SKILL.md (YAML frontmatter + markdown body)
- Progressive disclosure: stubs at startup, full load on activation, resources on demand
- **Role in our project:** The skills format we follow exactly.

### agents.md
- Open standard for agent project files (like README but for agents)
- Stewarded by Agentic AI Foundation under Linux Foundation
- Just markdown, no required fields, adopted by 60k+ projects
- **Role in our project:** The AGENTS.md format we follow.

## Ecosystem Notes

- LlamaStack 0.3.x has native OpenTelemetry support — spans for inference, tool calls, vector DB ops are automatic
- MCP servers don't auto-receive trace context; manual propagation header injection needed
- No public FIPS-aware agent scaffolding tool exists — this is a genuine gap
- The fips-agents CLI uses a clone-template model (clone public repo, follow slash command workflow)
