# Requirements

These are high-level, declarative requirements. They describe what the system must do, not how it should be implemented. For implementation details, see [docs/architecture.md](../docs/architecture.md).

## Must Have

**BaseAgent class.** A single base class that handles all common agent concerns: LLM communication, tool dispatch, MCP client connections, prompt loading, skill management, configuration, and lifecycle. Agent subclasses should be approximately 20-30 lines for typical use cases, implementing only the agent's unique behavior.

**Provider-portable LLM communication.** Agents must communicate with LLMs through an OpenAI-compatible interface (via litellm) that works against LlamaStack, raw vLLM, Anthropic, OpenAI, Azure, Bedrock, and other providers. Switching providers must be a configuration change, not a code change.

**Two tool planes with visibility control.** Tools must support two distinct invocation paths: agent-code tools (called by Python code, invisible to the LLM) and LLM-callable tools (surfaced to the LLM through the tool-calling protocol). Both paths must go through BaseAgent's infrastructure for logging, RBAC, and retry. Each tool must declare its visibility: `agent_only`, `llm_only`, or `both`.

**Built-in MCP client.** BaseAgent must include an MCP client (FastMCP v3) that connects to remote tool servers, discovers tools, and registers them alongside local tools with a unified invocation interface.

**Local tool auto-discovery.** Tools defined as decorated Python functions in a `tools/` directory must be auto-discovered and registered at startup without manual registration code.

**Prompt loading from files.** Prompts stored as Markdown files with YAML frontmatter in a `prompts/` directory must be loadable by name with variable substitution.

**Skills following agentskills.io spec.** Skills must follow the agentskills.io specification: one directory per skill with a SKILL.md file, progressive disclosure (stubs at startup, full content on activation), and optional scripts/references/assets subdirectories.

**Rules as separate files.** Behavioral rules must be stored as individual Markdown files in a `rules/` directory, loaded at startup and injected into the agent's context.

**Helm chart for OpenShift deployment.** Each scaffolded agent must include a Helm chart that deploys the agent as a Deployment, Service, and ConfigMap with an optional Route. The chart must not deploy infrastructure services.

**AGENTS.md following the open standard.** Each scaffolded agent must include an AGENTS.md file following the agents.md open standard convention.

**Scaffolded evals directory.** Each agent must include an `evals/` directory with a harness-agnostic eval case format, a lightweight local runner, and support for integration with external eval harnesses.

**Immutable container images.** Code, tools, prompts, skills, and rules must all be baked into the container image. The only external inputs at runtime are environment-specific configuration values (endpoint URLs, credentials, tuning parameters) and infrastructure services.

**Built-in protective patterns.** Max iterations, exponential backoff/retry on model and tool calls, and rate limiting must be built into BaseAgent, not left to individual agent implementations.

**First-class validated output.** `call_model_validated()` must be a built-in BaseAgent method that calls the model, validates output through a tool, and retries with backoff on validation failure.

**Compatibility with rh-ai-quickstart.** Agents must work with services deployed by rh-ai-quickstart/ai-architecture-charts as the assumed infrastructure layer.

**Two template variants.** The repository must contain two template directories: agent-loop (priority build) and agentic-workflow (designed now, built later).

## Should Have

**Optional MemoryHub integration.** Developers should be able to add MemoryHub support by running `memoryhub config init`, which wires up dual-path access (MCP for LLM tool calling, SDK for programmatic agent-code access) without modifying the agent subclass.

**Working examples in the scaffold.** The scaffolded project should include example prompts, tools, skills, rules, and eval cases that demonstrate the expected patterns and can be run immediately.

**Infrastructure setup documentation.** Clear documentation pointing to rh-ai-quickstart/ai-architecture-charts for deploying the infrastructure services that agents consume.

**Environment-portable configuration.** Configuration via YAML with `${VAR:-default}` environment variable substitution, so the same configuration file works locally and on OpenShift with different environment values.

**Red Hat UBI base image.** The Containerfile should use a Red Hat UBI base image for enterprise compatibility and FIPS readiness.

## Nice to Have

**Health check and readiness endpoints.** Pre-built HTTP endpoints for Kubernetes liveness and readiness probes.

**Example AGENTS.md.** An AGENTS.md populated with common patterns that demonstrates the format for new developers.

**Integration test scaffold.** A test structure that can run the agent against a local or remote LLM endpoint to verify end-to-end behavior.

**Context budget management.** Tracking of how much context window is consumed by skill loading, with warnings when the budget is exceeded.
