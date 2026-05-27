# ${AGENT_NAME:-My Agent}

<!-- Capability manifest for this agent.
     Populated by /create-agent from AGENT_PLAN.md.
     Served at runtime via GET /.well-known/agents.md -->

## Description

<!-- Populated by /create-agent from the Purpose section of AGENT_PLAN.md -->

A single-loop AI agent built on the fipsagents BaseAgent framework.

## API

- **Endpoint**: `POST /v1/chat/completions` (OpenAI-compatible)
- **Streaming**: SSE via `stream: true`
- **Health**: `GET /healthz`
- **Info**: `GET /v1/agent-info` (JSON capability summary)

## Capabilities

<!-- Populated by /create-agent -->

## Tools

<!-- Populated by /create-agent. Format:

| Tool | Visibility | Description |
|------|------------|-------------|
| `tool_name` | `llm_only` | What it does |

-->

## Model

- **Provider**: `${MODEL_PROVIDER:-openai}`
- **Model**: `${MODEL_NAME:-}`
- **Endpoint**: OpenAI-compatible chat completions

## Constraints

- Maximum loop iterations per turn: `${MAX_ITERATIONS:-10}`
- Immutable container image -- prompts, tools, and rules are baked in
- Requires an OpenAI-compatible LLM endpoint

## Configuration

Runtime behavior is configured via `agent.yaml` with `${VAR:-default}`
environment variable substitution. See `agent.yaml` for the full schema.

## Version

0.1.0
