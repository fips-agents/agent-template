# Workflow Agent Project

A multi-node workflow agent built on the fipsagents workflow framework.
Composes BaseNode and AgentNode instances into a directed graph with
typed state.

## Build and Run

```bash
make install       # Create .venv, install dependencies
make run-local     # Run the workflow locally (port 8080)
make test          # Run pytest
make lint          # Lint with ruff
make build         # Build container (podman, linux/amd64)
make deploy PROJECT=<ns>   # Deploy to OpenShift via Helm
```

## Project Structure

```
src/agent.py       # Workflow definition — nodes, state, graph wiring
src/workflow/      # Framework — do not edit
tools/             # One @tool-decorated .py file per tool
prompts/           # Markdown with YAML frontmatter, one per prompt
skills/            # agentskills.io directories with SKILL.md
rules/             # Plain Markdown, one constraint per file
agent.yaml         # Config with ${VAR:-default} env var substitution
chart/             # Helm chart for OpenShift deployment
evals/             # Eval cases
```

## Conventions

- Use `BaseNode` for routing, transformation, and gating (no LLM).
  Use `AgentNode` for nodes that need LLM, tools, or MCP.
- The core method is `process(state) -> state`. Do not implement
  `step()` on AgentNodes.
- State is a Pydantic model with `extra="forbid"` — data only, no
  execution metadata.
- Tools use `@tool(description=..., visibility=...)` — every tool must
  declare its visibility plane.
- Do not edit `src/workflow/` — that is the framework.
- Do not import `openai` directly — use AgentNode's `call_model*` methods.
- Run `make test && make lint` before committing.

## Testing

```bash
make test          # Unit tests
make test-cov      # With coverage report
make eval          # Run eval cases from evals/evals.yaml
```

## Configuration

`agent.yaml` controls workflow behavior. All values support
`${VAR:-default}` environment variable substitution. Key env vars:

- `MODEL_ENDPOINT` — LLM API endpoint
- `MODEL_NAME` — Model identifier
- `MAX_ITERATIONS` — Loop cap for AgentNodes
- `LOG_LEVEL` — Python logging level
