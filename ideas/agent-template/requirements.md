# Requirements (High Level)

## Must Have

- BaseAgent class that handles all common agent concerns
- Agent subclasses should be ~20-30 lines for typical use cases
- OpenAI-compatible API for LLM communication (portable across LlamaStack, vLLM, cloud)
- Two tool planes: agent-code tools (plane 1) and LLM-callable tools (plane 2), both through BaseAgent
- Tool visibility control: agent_only, llm_only, both
- MCP client built-in (FastMCP v3) for remote tool discovery and use
- Local tool loading from a tools/ directory (one file per tool, auto-discovered)
- Prompt loading from a prompts/ directory (Markdown with YAML frontmatter)
- Skills following agentskills.io spec (directory per skill, progressive disclosure)
- Rules as separate files in rules/ directory
- Helm chart for OpenShift deployment (Deployment, Service, ConfigMap, optional Route)
- AGENTS.md following the open standard convention
- Scaffolded evals/ directory with harness-agnostic format and external harness support
- Immutable container images (code, tools, prompts, skills, rules all baked in)
- Max iterations, exponential backoff/retry, rate limiting as built-in protective patterns
- call_model_validated() as a first-class pattern for output validation with retry
- Works with rh-ai-quickstart/ai-architecture-charts as the infra layer
- Two template directories: agent-loop (priority) and agentic-workflow (planned)

## Should Have

- Optional MemoryHub integration via memoryhub config init (dual-path: MCP for LLM, SDK for agent code)
- Example prompts, tools, skills, rules, and evals in the scaffold
- Clear documentation pointing to rh-ai-quickstart for infra setup
- Configuration via YAML with env var substitution for environment portability
- Red Hat UBI base image in the Containerfile

## Nice to Have

- Pre-built health check / readiness probe endpoints
- Example AGENTS.md with common patterns
- Integration test scaffold that can run against a local or remote LLM
- Context budget management for skill loading
