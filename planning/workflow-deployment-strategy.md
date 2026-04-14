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

See Resolved Design Decisions below for the settled configuration approach.

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

## Resolved Design Decisions

### 1. Node type configuration location

Node type configuration lives in `AgentConfig` (fipsagents package) as a `NodeConfig` Pydantic model. This allows `WorkflowRunner` to consult node config at execution time and auto-wire remote nodes. The `nodes:` section in `agent.yaml` maps node names to their deployment topology.

### 2. State serialization format

Pydantic `model_dump_json()` / `model_validate_json()` is the serialization format for remote calls. Constraint: all state fields must be JSON-serializable (no file handles, generators, or arbitrary objects). This is enforced naturally since workflow state models use `extra="forbid"`.

### 3. RemoteNode is a framework class

`RemoteNode` ships as a first-class framework class in the fipsagents package alongside `BaseNode` and `AgentNode`. It handles serialization, HTTP transport, exponential backoff retries, and error mapping. Developers who need custom protocols can still write a plain `BaseNode` bridge instead.

### 4. A2A integration: documented pattern only

A2A integration is a documented bridge recipe in the brownfield integration guide, not a framework class. The protocol is still evolving. `RemoteNode` covers HTTP; A2A will be revisited when the protocol stabilizes.

### 5. Runner auto-wraps remote nodes

`WorkflowRunner` auto-wraps nodes when `agent.yaml` declares them as `type: remote`. The graph definition stays topology-agnostic — same graph works for local dev (all in-process) and production (some nodes remote). The runner checks `node_configs` before calling `process()` and substitutes a `RemoteNode` transparently.
