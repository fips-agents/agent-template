# Constraints

## Language and Runtime

Templates are Python-only, async throughout. Every I/O-bound operation in BaseAgent (model calls, tool invocations, MCP communication, memory access) is async. This is not a preference but a requirement: agents that block on I/O cannot efficiently handle concurrent tool calls or streaming responses.

## No Framework Dependencies

BaseAgent carries no agent framework dependencies. No LangChain, no LangGraph, no CrewAI, no AutoGen. The LLM client layer is litellm (for provider portability) and the MCP client is FastMCP v3. Everything else is either the Python standard library or lightweight parsing/validation libraries (pydantic, python-frontmatter, httpx). This constraint exists because frameworks impose opinions about control flow, state management, and composition that conflict with keeping BaseAgent simple and agent subclasses small.

## LLM Client

litellm is the sole LLM client library. It provides an OpenAI-compatible interface to 100+ providers. This is the one external dependency that enables provider portability, and it is non-negotiable -- without it, every provider requires its own client code.

## Container Images

All containers use Red Hat UBI base images. FIPS compliance may be required depending on the deployment environment; templates must not preclude FIPS mode (no hardcoded dependencies on non-FIPS cryptographic libraries).

## Deployment Target

Helm charts must work on OpenShift. The template assumes OpenShift-specific capabilities (Routes, SecurityContextConstraints) but should degrade gracefully to vanilla Kubernetes where possible.

## Template Distribution

The template is a public GitHub repository cloned by the `fips-agents` CLI. The `.claude/` directory drives the AI-assisted development experience. This distribution model (clone, not install) means the template must be self-contained -- no post-clone dependency resolution beyond `pip install` for Python packages.

## Content Formats

Prompts are Markdown with YAML frontmatter. Tools use the `@tool` decorator convention (matching FastMCP). Skills follow the agentskills.io specification exactly. Rules are plain Markdown files. These formats are fixed design decisions, not suggestions.

## MCP Protocol Version

FastMCP v3 for the MCP client. Not v2. This applies to both local tool server connections and remote MCP server integration.

## Image Immutability

Code, tools, prompts, skills, and rules are baked into the container image. Only environment-specific configuration (endpoint URLs, credentials, tuning parameters) is external. This is an enterprise traceability constraint: every deployed state must map to a single image tag and git commit.

## Build Order

The agent loop template ships first. The agentic workflow template is designed but deferred until the core BaseAgent and agent loop are proven in production.

## MemoryHub Integration

The MemoryHub Python SDK is the programmatic access path for agent code. MCP is the LLM access path. Both are optional -- an agent must function correctly without MemoryHub configured.
