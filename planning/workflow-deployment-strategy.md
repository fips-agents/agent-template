# Workflow Deployment Strategy

## Context

The workflow template (templates/workflow/) currently produces a single container where all nodes run in-process. This document captures the deployment topology options and brownfield integration patterns discussed during initial design, as a roadmap for future work.

## Deployment Topologies

### Single Container (v1 — current)

All nodes share a process. Communication is function calls. State is in-memory. The WorkflowRunner calls process() on each node directly.

When to use:
- All nodes owned by the same team
- Uniform resource requirements
- Shared credentials and trust boundary
- Starting a new workflow (start simple, extract later)

### Multi-Container (v2 — planned)

Each agent deploys independently with its own image, scaling, and resource limits. The workflow node is a thin client that calls the remote agent via HTTP.

When to use:
- Nodes have different resource profiles (e.g., one needs GPU)
- Different teams own different nodes
- Independent scaling requirements
- Reusing an already-deployed agent

### Hybrid (v2 — planned)

Mix of in-process and remote nodes. Lightweight routing/transformation nodes stay in-process; heavy or shared agents deploy separately. This is the expected real-world pattern as workflows grow.

Configuration would live in agent.yaml:

```yaml
nodes:
  classify:
    type: local
  research:
    type: remote
    endpoint: ${RESEARCH_AGENT_URL:-http://research-agent:8080}
  summarize:
    type: local
```

The runner checks node type: local calls process() directly, remote makes an HTTP POST. Same graph definition, same state flow, different execution strategy.

## Decision Axis

The topology choice depends on five factors:

| Factor | Single container | Separate containers |
|--------|-----------------|-------------------|
| Trust boundary | Same team, same secrets | Different teams/tenants |
| Resource profile | Uniform needs | One node needs 10x memory or GPU |
| Scaling | Uniform traffic | One node gets 100x traffic |
| Reuse | Building fresh | Existing deployed agent to integrate |
| Simplicity | No reason to split | Clear reason exists |

Default to single container. Extract nodes when there's a concrete reason.

## Brownfield Integration

When integrating an existing deployed agent (REST, A2A, MCP) into a workflow, the pattern is a thin bridge node:

```python
@node()
class ResearchBridge(BaseNode):
    async def process(self, state: MyState) -> MyState:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://research-agent:8080/query",
                json={"question": state.query},
            )
        result = resp.json()
        return state.model_copy(update={"research_doc": result["document"]})
```

The bridge node:
- Is owned by the workflow developer, not the remote agent's developer
- Handles impedance mismatch between workflow state and remote API
- Encapsulates the protocol choice (REST, A2A, MCP) — the graph doesn't care
- Is typically a BaseNode (no LLM needed), unless it uses MCP (then AgentNode with connect_mcp())

### Protocol options for bridges

- **REST/HTTP**: httpx call in a BaseNode. Simplest, works with any API.
- **A2A**: If the remote agent speaks Agent-to-Agent protocol. Same shape — BaseNode wrapping the protocol client.
- **MCP**: AgentNode that calls connect_mcp() to discover and invoke remote tools. Good when the remote agent exposes capabilities as MCP tools.
- **Message queue**: BaseNode that publishes a request and awaits a response. For async/event-driven patterns (deferred).

### Future CLI support

Potential commands to streamline brownfield integration:

- `fips-agents create bridge --from-openapi <url>` — Auto-generate a bridge node from an OpenAPI spec
- `fips-agents extract-node <name> --from <workflow>` — Extract an in-process node into a standalone agent with its own container, replacing it with a remote bridge in the workflow

## Deployment CLI Evolution

Current: `make deploy` deploys a single container.

Future options:
- `fips-agents deploy <workflow>` — deploy with topology from agent.yaml
- `fips-agents extract-node <name> --from <workflow>` — split a node into its own deployment

The key principle: developers start with a single container and extract when they have a reason. The design should not force topology decisions at creation time.
