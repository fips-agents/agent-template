# Brownfield Integration Guide

Workflows don't have to be built from scratch. Existing deployed agents -- REST APIs, MCP servers, and A2A-speaking agents -- can participate as workflow nodes without being rewritten. This guide covers the integration patterns, when to use each, and how to extract a local node into a standalone remote service when it outgrows its container.

## When to Use RemoteNode vs a Custom Bridge

`RemoteNode` is the right choice when the remote agent already speaks the fipsagents HTTP contract: `POST /process` with state in, state out. This is the case for any agent built from this template and deployed as a standalone service.

For everything else -- legacy REST APIs, MCP servers, A2A agents -- write a thin `BaseNode` bridge that handles the impedance mismatch between your workflow state and the remote API. The bridge is owned by the workflow developer, not the remote service's team. This is intentional: the remote service stays unmodified, and all the adaptation logic lives in one place you control.

## The Remote Agent HTTP Contract

Any agent deployed from this template exposes a single endpoint:

```
POST {endpoint}/process
Content-Type: application/json

{"state": { ... }, "state_type": "my_workflow.state.MyState"}
```

On success, the agent returns `200` with the updated state:

```json
{"state": { ... updated state ... }}
```

The contract is deliberately simple. The remote agent receives state, processes it, and returns updated state. It has no knowledge of the workflow graph, edge routing, or other nodes.

## Using RemoteNode

### Via agent.yaml (auto-wrap)

The most common approach is to declare the remote node in `agent.yaml`. The `WorkflowRunner` auto-wraps it at runtime:

```yaml
nodes:
  research:
    type: remote
    endpoint: ${RESEARCH_AGENT_URL:-http://research-agent:8080}
    timeout: 60.0
    retries: 3
```

The graph definition doesn't change. The runner resolves the node type at execution time, constructing a `RemoteNode` per invocation based on the config.

### Via explicit construction

When you need finer-grained control over timeout or retry behavior per-node:

```python
from fipsagents.workflow import RemoteNode, Graph, END

graph = Graph(state_type=MyState)
graph.add_node("research", RemoteNode(
    name="research",
    endpoint="http://research-agent:8080",
    path="/process",
    timeout=60.0,
    retries=3,
))
graph.add_edge("research", END)
```

## REST Bridge Pattern

For existing REST APIs that don't speak the fipsagents contract, write a `BaseNode` bridge that maps between workflow state and the API's request/response format:

```python
from fipsagents.workflow import BaseNode, node
import httpx

@node()
class SearchBridge(BaseNode):
    """Bridge to an existing search API."""

    def __init__(self, api_url: str):
        super().__init__(name="search")
        self.api_url = api_url

    async def process(self, state: MyState) -> MyState:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.api_url}/search",
                params={"q": state.query, "limit": 10},
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        return state.model_copy(update={"search_results": results})
```

The bridge is thin by design. Complex response normalization, pagination, or retry logic belongs in a separate service layer -- the bridge itself should do nothing more than translate state fields into API calls and vice versa.

## MCP Bridge Pattern

When the remote agent exposes capabilities as MCP tools, use an `AgentNode` that connects via MCP. This gives the LLM access to the remote agent's tools for reasoning:

```python
from fipsagents.workflow import AgentNode, node

@node()
class McpResearchNode(AgentNode):
    """Workflow node that delegates to an MCP-backed research agent."""

    async def process(self, state: MyState) -> MyState:
        await self.connect_mcp("http://research-mcp:8080/mcp")
        self.add_message("user", f"Research this topic: {state.query}")
        response = await self.call_model()
        return state.model_copy(update={"research": response.content})
```

Use this pattern when the remote agent's value comes from tool-augmented reasoning rather than simple data transformation. Unlike a REST bridge, the LLM here decides which MCP tools to invoke and how to compose their results -- you're delegating reasoning, not just data access.

## A2A Bridge Pattern

For agents that speak the Agent-to-Agent protocol, the pattern is the same as a REST bridge: a `BaseNode` that handles protocol details. A2A is still evolving, so this is a recipe to adapt rather than a stable framework class:

```python
@node()
class A2ABridge(BaseNode):
    """Bridge to an A2A-speaking agent."""

    def __init__(self, a2a_url: str):
        super().__init__(name="a2a_agent")
        self.a2a_url = a2a_url

    async def process(self, state: MyState) -> MyState:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.a2a_url}/tasks/send",
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"text": state.query}],
                        }
                    },
                },
            )
            result = resp.json()
            output = result["result"]["artifacts"][0]["parts"][0]["text"]
        return state.model_copy(update={"result": output})
```

This is a simplified example. Production A2A integration requires handling task status polling, streaming responses, and error cases per the A2A specification.

## Extracting a Local Node to Remote

When a local node outgrows its container -- it needs a GPU, different scaling characteristics, or separate team ownership -- the extraction process follows a clear path.

Wrap the node's `process()` logic in a minimal FastAPI app that speaks the fipsagents contract:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import importlib

app = FastAPI()

class ProcessRequest(BaseModel):
    state: dict
    state_type: str

class ProcessResponse(BaseModel):
    state: dict

@app.post("/process")
async def process(req: ProcessRequest) -> ProcessResponse:
    module_path, class_name = req.state_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    state_cls = getattr(module, class_name)
    state = state_cls.model_validate(req.state)

    updated = await do_processing(state)

    return ProcessResponse(state=updated.model_dump())
```

Deploy it as a separate container, then update the workflow's `agent.yaml` to declare the node as `type: remote` with the new endpoint URL. The graph definition doesn't change -- the runner auto-wraps it. The local node class can be deleted once the remote service is verified.

## Choosing the Right Pattern

| Situation | Pattern |
|-----------|---------|
| Remote agent speaks fipsagents contract | `RemoteNode` (via config or explicit) |
| Existing REST API | Custom `BaseNode` bridge |
| Remote agent exposes MCP tools | `AgentNode` with `connect_mcp()` |
| Remote agent speaks A2A | Custom `BaseNode` bridge (recipe above) |
| Local node needs extraction | FastAPI wrapper + `RemoteNode` |
