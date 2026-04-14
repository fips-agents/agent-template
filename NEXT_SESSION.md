# Next Session: Workflow Deployment Topology (#24)

## Goal

Enable workflows to mix in-process and remote nodes. Today all nodes run
in a single container. We need to support calling already-deployed agents
(REST, MCP, A2A) as workflow nodes, and extracting nodes into separate
deployments.

Tracking issue: #24

## Design context

The design doc at `planning/workflow-deployment-strategy.md` is settled. Key
decisions still open are listed under Priority 1 below. Read the doc before
starting — it covers the three topologies (single, multi, hybrid), brownfield
integration patterns, and the decision axis.

## Priority 1: Finalize the four open design questions

These are listed in issue #24. Resolve them and update
`planning/workflow-deployment-strategy.md` with the answers.

1. **Where does node type configuration live?**
   The doc proposes a `nodes:` section in `agent.yaml`. Decide: is this in
   AgentConfig (and therefore in the fipsagents package), or in the template's
   agent.yaml only? If it's in AgentConfig, it needs a Pydantic model.

2. **State serialization format for remote calls.**
   Pydantic `model_dump_json()` / `model_validate_json()` is the natural
   choice. Confirm this and document any constraints (e.g., all state fields
   must be JSON-serializable, no arbitrary objects).

3. **Should RemoteNode be a framework class or a documented pattern?**
   The doc shows a bridge node as a plain BaseNode with httpx. The question
   is whether the framework should provide `RemoteNode(endpoint, ...)` that
   handles serialization, HTTP, retries, and error mapping automatically.
   Recommendation: yes, ship it as a framework class. It's <100 lines and
   eliminates boilerplate for the most common case.

4. **A2A integration depth.**
   Options: (a) just a bridge recipe in docs, (b) an `A2ANode` framework
   class, (c) defer entirely. Recommendation: (a) for now — A2A is still
   early. Document the pattern, ship RemoteNode for HTTP, revisit A2A when
   the protocol stabilizes.

## Priority 2: Implement RemoteNode in the fipsagents package

### What to do

1. Create `packages/fipsagents/src/fipsagents/workflow/remote_node.py`
2. `RemoteNode(BaseNode)` with:
   - Constructor: `endpoint` (str), `path` (str, default "/process"),
     `timeout` (float, default 30.0), `retries` (int, default 2)
   - `process(state)`: serialize state → POST to endpoint → deserialize
     response back to state type
   - Exponential backoff on retries (reuse BackoffConfig pattern from
     agent loop)
   - Error handling: HTTP errors → raise with clear message (the runner's
     error edge system handles routing)
3. Export from `packages/fipsagents/src/fipsagents/workflow/__init__.py`
4. Tests in `packages/fipsagents/tests/test_remote_node.py` — mock httpx,
   test serialization round-trip, error handling, retries

### Remote agent HTTP contract

The remote agent exposes a single endpoint:

```
POST /process
Content-Type: application/json

{
  "state": { ... serialized workflow state ... },
  "state_type": "fully.qualified.ClassName"
}

Response 200:
{
  "state": { ... updated state ... }
}
```

This is deliberately simple. The remote agent doesn't need to know about
the workflow — it receives state, processes it, returns state.

### Key files

- `packages/fipsagents/src/fipsagents/workflow/remote_node.py` — new
- `packages/fipsagents/src/fipsagents/workflow/__init__.py` — add export
- `packages/fipsagents/src/fipsagents/__init__.py` — add export
- `packages/fipsagents/tests/test_remote_node.py` — new

## Priority 3: Add node topology configuration to agent.yaml

### What to do

1. Add `NodeConfig` Pydantic model to config.py (or a new workflow_config.py):
   ```yaml
   nodes:
     classify:
       type: local
     research:
       type: remote
       endpoint: ${RESEARCH_AGENT_URL:-http://research-agent:8080}
       path: /process
       timeout: 30.0
       retries: 2
   ```
2. Add `nodes: dict[str, NodeConfig]` to `AgentConfig` (default empty)
3. Update `WorkflowRunner` or `Graph` to consult node config at execution
   time — if a node name has a `remote` config, wrap it in a RemoteNode
   automatically (or require the graph to use RemoteNode explicitly — decide
   in Priority 1)
4. Update both template `agent.yaml` files with documented `nodes:` section
5. Tests for config parsing and remote node wiring

### Key files

- `packages/fipsagents/src/fipsagents/baseagent/config.py` — NodeConfig model
- `packages/fipsagents/src/fipsagents/workflow/runner.py` — topology-aware execution
- `templates/agent-loop/agent.yaml` — document nodes section
- `templates/workflow/agent.yaml` — document nodes section

## Priority 4: Brownfield integration guide

`docs/brownfield-integration.md` — how to integrate existing agents into
workflows. Covers:

- When to use RemoteNode vs a custom bridge BaseNode
- REST bridge pattern (with example)
- MCP bridge pattern (AgentNode + connect_mcp)
- A2A bridge pattern (recipe, not framework class)
- The remote agent HTTP contract
- Extracting a local node to remote (manual process for now)

## Architecture context

Read before starting:

- `planning/workflow-deployment-strategy.md` — the design doc (read above)
- `packages/fipsagents/src/fipsagents/workflow/runner.py` — WorkflowRunner
  execution loop, step limit, retry logic, error edges
- `packages/fipsagents/src/fipsagents/workflow/graph.py` — Graph definition:
  nodes, edges, conditional edges, error edges
- `packages/fipsagents/src/fipsagents/workflow/node.py` — BaseNode (L15-46)
- `packages/fipsagents/src/fipsagents/workflow/protocol.py` — WorkflowNode
  protocol: `name: str` + `async process(state) -> state`
- `packages/fipsagents/src/fipsagents/workflow/agent_node.py` — AgentNode
  (BaseAgent + process() method)
- `packages/fipsagents/src/fipsagents/baseagent/config.py` — AgentConfig,
  BackoffConfig (reuse for retry logic)

## Prior session context

- Pluggable memory backends shipped (v0.2.0): memoryhub, sqlite, pgvector,
  custom. Issues #27-31 all closed.
- Sandbox hardening v2 (#26) is still open, planned for the session after
  this one.
- Workflow framework was extracted to fipsagents package in commit df0d387.
  Full test suite (307 tests) in place.

## Cluster endpoints (if needed for testing remote nodes)

| Model | URL | API |
|-------|-----|-----|
| all-MiniLM-L6-v2 (embedding) | `https://all-minilm-l6-v2-embedding-model.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com` | TEI, `POST /embed` |
| Granite 3.3 8B Instruct | `https://granite-3-3-8b-instruct-granite-model.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com` | OpenAI-compat |
