# CLAUDE.md

This is a workflow project built on the workflow framework. The framework composes agents and lightweight nodes into directed graphs with typed state. Each node implements `process(state) -> state`, and the `WorkflowRunner` manages execution lifecycle.

## Development Workflow

```bash
make install       # Create .venv, install all dependencies
make run-local     # Run the workflow locally
make test          # Run pytest
make test-cov      # Run pytest with coverage report
make eval          # Run eval cases from evals/evals.yaml
make lint          # Lint with ruff
make build         # Build container (podman, linux/amd64)
make deploy PROJECT=<ns>   # Deploy to OpenShift
```

## Slash Command Workflow

### Scaffolding Pipeline

Run in order. Each step consumes the prior step's artifact.

**`/plan-agent`** -- Design the workflow: nodes, edges, state model, tools, prompts. Produces `AGENT_PLAN.md`. No code is written.

**`/create-agent`** -- Reads `AGENT_PLAN.md` and generates everything: `src/agent.py` (graph definition, node classes, state model), tools, prompts, skills, rules, updated `agent.yaml`. Runs `make test` and `make lint` to verify.

**`/exercise-agent`** -- Reads the implementation and designs test scenarios. Validates that nodes process state correctly, conditional edges route as expected, and tool calls work within AgentNodes. Writes eval cases to `evals/evals.yaml`.

**`/deploy-agent`** -- Pre-flight checks, container build, push, Helm deploy, pod verification.

### Extension Commands

**`/add-tool`** -- Add a new tool (used by AgentNodes). Generates the tool file in `tools/` with `@tool` decorator and visibility.

**`/add-skill`** -- Add a new skill directory following agentskills.io spec.

**`/add-memory`** -- Wire MemoryHub integration for persistent memory across workflow runs.

## Project Structure

```
src/agent.py           # YOUR workflow definition — nodes, state, graph wiring
src/workflow/          # Framework — do not edit
tools/                 # One @tool-decorated .py file per tool
prompts/system.md      # System prompt for AgentNodes. Add more as needed.
skills/<name>/SKILL.md # One directory per skill, agentskills.io spec
rules/                 # Plain Markdown, one constraint per file
agent.yaml             # Config with ${VAR:-default} env var substitution
chart/                 # Helm chart for OpenShift deployment
evals/                 # Eval cases and runner
```

## Writing Your Workflow

Your workflow lives in `src/agent.py`. It defines three things: a state model, node classes, and a graph.

### State Model

State is a Pydantic model that flows through the graph. It must subclass `WorkflowState`:

```python
from workflow import WorkflowState

class MyState(WorkflowState):
    query: str
    result: str = ""
    confidence: float = 0.0
```

Keep state minimal -- data only, no execution metadata. `WorkflowState` uses `extra="forbid"` to catch typos.

### Node Types

**`BaseNode`** -- Lightweight node for routing, transformation, gating. No LLM, no tools. Override `process(state) -> state`.

```python
from workflow import BaseNode, node

@node()
class FilterNode(BaseNode):
    async def process(self, state: MyState) -> MyState:
        # Pure logic, no LLM call
        return state.model_copy(update={"confidence": 0.8})
```

**`AgentNode`** -- Full agent node with LLM, tools, prompts, memory, MCP. Inherits all BaseAgent capabilities. Override `process(state) -> state`.

```python
from workflow.agent_node import AgentNode
from workflow import node

@node()
class AnalyzeNode(AgentNode):
    async def process(self, state: MyState) -> MyState:
        self.add_message("user", f"Analyze: {state.query}")
        response = await self.call_model()
        return state.model_copy(update={"result": response.content})
```

### Graph Definition

Wire nodes with edges (fixed routes), conditional edges (dynamic routes), and an entry point:

```python
from workflow import Graph, WorkflowRunner, END

def build_graph() -> Graph:
    graph = Graph(state_type=MyState)

    graph.add_node("filter", FilterNode())
    graph.add_node("analyze", AnalyzeNode())

    graph.set_entry_point("filter")
    graph.add_conditional_edge(
        "filter",
        lambda s: "analyze" if s.confidence > 0.5 else END,
    )
    graph.add_edge("analyze", END)

    return graph
```

### Running the Workflow

```python
runner = WorkflowRunner(build_graph(), max_steps=10)
result = await runner.start(MyState(query="What is X?"))
```

`WorkflowRunner` handles setup (AgentNode initialization), execution (stepping through the graph), and shutdown (AgentNode cleanup).

## Tool System (Two Planes)

Same as the agent-loop template. Every tool declares its visibility:

| Visibility | Who calls it | Use for |
|-----------|-------------|---------|
| `llm_only` | LLM decides via tool-calling | Search, retrieval, information gathering |
| `agent_only` | Agent code via `self.use_tool()` | Validation, formatting, internal logic |
| `both` | Either | Rare -- only when genuinely needed by both |

```python
from fipsagents.baseagent.tools import tool

@tool(description="Search the web", visibility="llm_only")
async def web_search(query: str) -> str:
    ...
```

Tools are available to all AgentNode instances in the workflow.

## Prompt Format

Markdown with YAML frontmatter in `prompts/`:

```markdown
---
name: system
description: Default system prompt
variables:
  - name: role
    default: "a helpful assistant"
---

You are {role}. Provide clear, accurate, and concise responses.
```

## Skills (agentskills.io)

Same convention as agent-loop. One directory per skill, `SKILL.md` with YAML frontmatter. Only frontmatter loads at startup.

## Rules

Plain Markdown in `rules/`. No frontmatter. One constraint per file. Injected into system prompt at startup.

## Configuration (`agent.yaml`)

Uses `${VAR:-default}` for env var substitution. Key env vars:

- `MODEL_ENDPOINT` -- LLM API endpoint
- `MODEL_NAME` -- Model identifier
- `MAX_ITERATIONS` -- Loop cap (for AgentNodes that use internal loops)
- `LOG_LEVEL` -- Python logging level

## Common Mistakes

- **Do not edit `src/workflow/`.** It is the framework. Your code goes in `src/agent.py`, `tools/`, `prompts/`, `skills/`, and `rules/`.
- **Do not implement `step()` on AgentNodes.** The core method is `process(state) -> state`. The workflow runner drives execution, not the agent loop.
- **Do not import `openai` directly.** Use AgentNode's `call_model*` methods.
- **Do not hardcode model names or endpoints.** Use `agent.yaml` with `${VAR:-default}`.
- **Do not skip `visibility` on tools.** Every tool must declare its plane.
- **Do not omit `tool_call_id` when appending tool results.** The API requires it.
- **Do not put execution metadata in state.** State is data only. The runner tracks execution.
- **Do not build on macOS without `--platform linux/amd64`.** Use `make build`.

## Deployment

1. `make test` -- tests must pass
2. `git status` -- no uncommitted changes
3. `make build IMAGE_NAME=<name> IMAGE_TAG=<tag>`
4. Push image to registry
5. Configure `chart/values.yaml`
6. `make deploy PROJECT=<namespace>`
7. Verify: `oc get pods -n <ns>`, `oc logs <pod> -n <ns>`

The image is immutable: code, tools, prompts, skills, rules, and `agent.yaml` defaults are all baked in.

## Dependencies

- **openai** -- LLM client (async SDK for OpenAI-compatible endpoints)
- **fastmcp** (v3) -- MCP client
- **pydantic** -- Config validation, state models, structured output
- **pyyaml** -- Config parsing
- **httpx** -- Async HTTP
- **python-frontmatter** -- Prompt/skill file parsing
- **memoryhub** (optional) -- MemoryHub SDK
