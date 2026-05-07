# Subagent Tool Design (#165)

## Problem

A primary agent often needs to delegate part of a turn to a specialist
agent and incorporate the result without restructuring the entire
conversation as a workflow graph. The current options are unsatisfying:

- **Workflow graph.** Powerful but heavyweight. Topology is static at
  deploy time. Every delegation pair has to be a node-and-edge in the
  graph, which is the wrong shape for ad-hoc, model-driven decisions
  about *whether* to delegate.
- **Hand-rolled HTTP call.** A tool that does `httpx.post` to another
  agent's `/v1/chat/completions` works in the small, but skips the
  framework's logging, tracing, cost attribution, identity propagation,
  and permission enforcement. Every team that builds it ends up
  reimplementing the same five concerns badly.

The motivating use cases are deliberately non-coding:

- **Customer service.** A general-purpose support agent delegates an
  account lookup to a tenant-specific account agent that has the
  customer's records and tools wired in.
- **Operations.** A primary ops agent delegates runbook search to a
  runbook-specialist agent whose MCP servers and prompts are tuned for
  retrieval-and-summarise workflows.
- **Data analysis.** An analyst agent delegates SQL synthesis to a
  query-builder agent that has the schema baked into its prompt.
- **Document processing.** A coordinator agent fans out per-file
  analysis across N identical worker agents and aggregates results.

Every one of these is a tool call in spirit: the primary agent forms an
intent, names a target, passes a payload, gets back a structured
result, and continues. Subagent-as-tool makes that pattern first-class.

## Design principle

**Composable delegation as a tool, not a workflow primitive.** The
subagent tool sits at the tool plane — the same surface the LLM uses
for any other capability. Discovery is config-driven (`subagents:` in
`agent.yaml`), invocation is a normal tool call, and the result is a
typed payload the LLM can branch on.

This is intentionally orthogonal to the workflow graph. Workflows
remain the right shape when topology is known up front, when the
graph itself encodes business logic, or when nodes need to share
typed state. Subagent-as-tool is the right shape when the primary
agent decides *at runtime* whether to delegate, based on the user's
intent and the model's judgement.

The framework gives the tool everything it needs to be a good
citizen: conversation isolation, cost roll-up, trace propagation,
permission scoping, identity inheritance, depth bounds. The user
writes one config block and gets all of that for free.

## Subagent lifecycle

```
                Parent agent step
                       │
                       ▼
        ┌────────────────────────────┐
        │  LLM emits tool_call:      │
        │  delegate_to_agent(...)    │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Permission gate (#164)    │
        │  scope check + ask flow    │
        └────────────┬───────────────┘
                     │ allow
                     ▼
        ┌────────────────────────────┐
        │  Resolve subagent config   │
        │  by `agent_name`           │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Build subagent context:   │
        │  - fresh messages list     │
        │  - inherited identity      │
        │  - scoped permissions      │
        │  - injected trace headers  │
        │  - depth + 1               │
        └────────────┬───────────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
       remote transport   inprocess transport
       (HTTP, RemoteNode) (instantiate class)
            │                 │
            └────────┬────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Subagent runs its own     │
        │  step loop (isolated)      │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Build SubagentResult:     │
        │  - final content           │
        │  - tokens_used             │
        │  - tool_calls_made         │
        │  - cost_usd                │
        │  - span_id                 │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Roll up into parent:      │
        │  - BudgetEnforcer          │
        │  - close OTEL child span   │
        │  - structured log          │
        └────────────┬───────────────┘
                     │
                     ▼
            Parent agent continues
```

The subagent's intermediate messages are not visible to the parent.
The parent sees one tool call, one tool result. The subagent's full
trace is reconstructable from OTEL spans and structured logs but does
not pollute the parent's conversation.

## Architecture

### Where this lives

- **Stock tool.** A new file in the template's `tools/` directory:
  `tools/delegate_to_agent.py`. Decorated with `@tool(visibility="both")`.
  Auto-discovered by the existing tool registry on agent setup.
- **Config.** A new top-level `subagents:` block in `agent.yaml`,
  validated by a new `SubagentConfig` Pydantic model on `AgentConfig`.
- **Transport adapters.** Two implementations under
  `fipsagents.subagents`:
  - `RemoteSubagentTransport` — wraps an `httpx.AsyncClient` and posts
    to the target's `/v1/chat/completions`. Reuses
    `propagation.inject_trace_context()`.
  - `InProcessSubagentTransport` — instantiates a subagent class in
    the same process. Used in tests and for tightly-coupled
    compositions.
- **Streaming events.** New `StreamEvent` variants:
  - `SubagentInvoked(agent_name, task, span_id)` — emitted when the
    delegation starts.
  - `SubagentCompleted(agent_name, result, tokens_used)` — emitted
    when the subagent returns.
  - `SubagentFailed(agent_name, error)` — emitted on timeout or crash.
  - `SubagentDelta(agent_name, delta)` — v2, emitted while the
    subagent is streaming.

### Reuse, do not reinvent

- **Trace propagation.** `fipsagents.server.propagation` already
  implements W3C Trace Context. The remote transport calls
  `inject_trace_context()` before posting; the receiving agent
  already calls `extract_trace_context()` on inbound requests.
- **HTTP transport semantics.** `RemoteNode` in
  `fipsagents.workflow` already shells out to a remote agent over
  OpenAI-compatible HTTP. The subagent remote transport is a
  generalisation — same wire shape, different invocation surface
  (tool call vs. workflow node).
- **Cost accounting.** `BudgetEnforcer` already tracks tokens and
  cost. The subagent tool calls `BudgetEnforcer.record_subagent(
  result.tokens_used, result.cost_usd)` on completion.
- **Permission enforcement.** The permission policy (#164) runs in
  `BaseAgent.handle_tool_call` before dispatch. The subagent tool
  goes through the same gate as any other tool.

## Configuration shape

```yaml
subagents:
  - name: research_helper
    description: \"Searches internal knowledge base and returns a synthesised brief.\"
    when_to_use: \"Use when the user asks an open-ended policy question.\"
    transport:
      type: remote
      url: ${RESEARCH_HELPER_URL:-http://research-helper:8080/v1}
      timeout_seconds: 60
    permission_scope: research-readonly
    identity: inherit
    max_depth: 3

  - name: account_lookup
    description: \"Looks up an account by ID and returns relevant fields.\"
    when_to_use: \"Use when the user references an account by ID.\"
    transport:
      type: remote
      url: ${ACCOUNT_LOOKUP_URL}
      timeout_seconds: 15
    permission_scope: account-read
    identity:
      service_account: account-reader  # explicit override
    max_depth: 1
```

Field reference:

- `name` — the identifier the LLM uses in the `agent_name` parameter
  (or the suffix of the synthesised tool name if we go that route).
- `description` — surfaced in the tool schema so the LLM knows what
  this subagent does.
- `when_to_use` — a hint baked into the tool's docstring/schema so
  the LLM has guidance on selection.
- `transport.type` — `remote` or `inprocess`.
- `transport.url` — for remote, the OpenAI-compatible endpoint base.
  Standard `${VAR:-default}` env-var substitution applies.
- `transport.timeout_seconds` — per-call timeout; subagent failures
  do not block the parent indefinitely.
- `permission_scope` — references a named rule set in
  `PermissionConfig.scopes` (#164). The subagent runs under
  `min(parent_scope, this_scope)`.
- `identity` — `inherit` (default; subagent runs as the caller) or
  `service_account: <name>` (subagent runs as a fixed service
  identity, useful when the subagent has its own kagenti-issued
  credentials).
- `max_depth` — cap on delegation chains. The framework tracks depth
  in the trace context and rejects subagent calls that would exceed
  the cap.

## Permission propagation

Permissions are the security-load-bearing part of this design. The
rules:

1. **The parent's permission scope is the upper bound.** A subagent
   cannot invoke a tool the parent itself is not authorised to
   invoke. The framework enforces this by intersecting the parent's
   active rule set with the subagent's declared `permission_scope`.
2. **Subagents declare what they need.** `permission_scope` on the
   subagent config identifies a named rule set in the parent's
   `PermissionConfig.scopes`. The framework rejects the call at
   config-validation time if the named scope does not exist.
3. **Identity inheritance defaults to the caller.** A subagent
   invoked under `identity: inherit` carries the parent's
   kagenti-issued identity in its outbound HTTP headers. The
   subagent's own permission policy then layers on top — the
   subagent decides what *its* identity is allowed to do, the parent
   decides what the subagent invocation is allowed to do, and the
   intersection wins.
4. **Service-account override.** When a subagent has its own
   credentials (e.g. an account-lookup agent with read-only DB
   access), the parent specifies `identity: service_account: <name>`.
   The framework looks up the service account from kagenti and
   issues the call under that identity. This is the right shape for
   subagents that need stronger isolation than the caller's identity
   provides.
5. **Depth-bounded.** A delegation chain (A → B → C → D) is bounded
   by the `max_depth` of the *first* subagent in the chain. The
   framework increments a depth counter in the trace baggage; the
   receiving agent rejects calls that exceed the cap with a
   structured error.

There is no escalation path. A subagent cannot ask for capabilities
the parent does not have. If a workflow needs more privilege at a
deeper level, that is a deployment-topology decision (different
service account at the chain root), not a runtime policy decision.

## Cost & tracing

### Cost roll-up

Every `SubagentResult` includes `tokens_used` and `cost_usd`. The
parent's `BudgetEnforcer` records these against the parent's session.
The parent's per-step token cap, per-session token cap, and per-USD
cost cap all apply. A single subagent that exceeds the parent's
remaining budget is short-circuited; the tool call returns a
`BudgetExceededError` and the parent's loop continues with that
information.

Pricing rules (`PricingConfig`) are evaluated *at the subagent's
endpoint*, not at the parent's. A subagent talking to a more
expensive model costs more, and that cost flows correctly to the
parent's budget.

### OTEL spans

The remote transport creates a child span under the parent's current
span. Span attributes:

- `subagent.name` — the subagent identifier.
- `subagent.transport` — `remote` or `inprocess`.
- `subagent.depth` — current delegation depth.
- `subagent.permission_scope` — the named scope under which the
  subagent ran.
- `subagent.tokens_used` — tokens consumed.
- `subagent.cost_usd` — dollar cost.
- `error.type` — populated on failure (`Timeout`, `MaxDepth`,
  `RemoteCrash`, `BudgetExceeded`).

Inprocess transport spans are emitted similarly, with
`subagent.transport: inprocess`.

The W3C Trace Context propagation ensures that the receiving
subagent's OTEL spans are children of the parent's, so the entire
delegation tree is reconstructable in any OTLP-compatible viewer.

### Structured logs

Every delegation emits a structured log line at start and end:

```json
{
  \"event\": \"subagent.invoked\",
  \"parent_session_id\": \"sess_abc\",
  \"parent_message_index\": 14,
  \"agent_name\": \"research_helper\",
  \"depth\": 1,
  \"transport\": \"remote\"
}
```

`parent_message_index` is essential for replay: a debugger can
identify exactly which assistant turn invoked the subagent.

## Streaming model

### v1 (first PR): buffered

The subagent runs to completion. The tool returns a single
`SubagentResult`. The parent sees `SubagentInvoked` at the start and
`SubagentCompleted` at the end of the tool call.

This is the simplest shape and unblocks the headline use cases. It is
also adequate for ad-hoc delegations where the subagent's intermediate
output is not interesting to the operator.

### v2 (follow-on): nested deltas

The subagent's `StreamEvent`s are forwarded onto the parent's stream
as `SubagentDelta` events with the subagent's `agent_name` attached.
The parent's stream interleaves its own deltas with the subagent's,
and the gateway / UI can render both — likely as a nested panel or
threaded view (open question for the UI repo).

The framework requirements for v2:

- `SubagentDelta(agent_name, delta)` carries the original event
  unmodified except for the namespace.
- The parent's `astep_stream` consumer treats `SubagentDelta` as
  opaque — the gateway is responsible for routing deltas back to
  the right rendering surface.
- Backpressure: the remote transport reads the SSE stream as fast as
  the subagent emits. The parent's stream is bounded by the existing
  per-stream queue; if the consumer is slow, deltas drop with a
  `SubagentDeltaDropped` warning event. Retrofitting this onto v1's
  buffered model is non-breaking — v1 simply skips deltas.
- Buffered subagent results remain available even when streaming; the
  final `SubagentCompleted` event carries the same payload as the
  v1 tool result.

## Error handling

Failure modes and their semantics:

- **Timeout (`SubagentTimeoutError`).** The transport's
  `timeout_seconds` elapsed without the subagent returning. Tool call
  returns a structured error; the parent's LLM may retry or replan.
- **Remote crash (`SubagentRemoteError`).** The remote endpoint
  returned 5xx or the connection broke. Tool call returns a
  structured error with the upstream status. Retry policy is the
  parent's loop's responsibility (existing `loop.max_iterations`).
- **Max depth (`MaxDelegationDepthError`).** The framework detected
  that this call would exceed the chain's `max_depth`. Tool call
  returns a structured error; this is non-recoverable for the
  current delegation path.
- **Budget exceeded (`BudgetExceededError`).** The parent's budget
  was exhausted mid-subagent. The subagent is allowed to finish (we
  do not abort mid-call), but the result includes a budget warning,
  and the parent's next step terminates if the budget is still
  blown.
- **Permission denied (`PermissionDeniedError`).** The intersection
  of parent and subagent scopes denied the call. Same shape as a
  normal tool denial under #164.
- **Inprocess crash (`SubagentCrashedError`).** An exception escaped
  the inprocess subagent. The framework catches and converts to a
  structured error so the parent's loop continues cleanly.

Retries are the parent's LLM's job. The framework does not auto-retry
subagent calls — that conflates failure modes (timeout vs. crash vs.
denial) that the model should handle differently.

## Tradeoffs

### One tool with `agent_name` parameter, vs. N synthesised tools

**One tool (`delegate_to_agent(agent_name, task, context)`).**
- Pros: clean schema, constant token cost in the tool list, explicit
  argument that lets the LLM compose.
- Cons: discovery requires the LLM to read the `description` and
  `when_to_use` fields out of the tool schema's free-text section;
  picking the right subagent is a model-side reasoning step.

**N synthesised tools (one per registered subagent).**
- Pros: tool descriptions surface naturally in the tool list; the
  model picks via standard tool selection.
- Cons: tool list size grows linearly with subagents; on small models
  (Granite 8B, etc.) this can blow the context budget.

**Recommendation.** Ship one tool with `agent_name`, but synthesise
auto-generated `when_to_use` hints into the tool description. If real
deployments find selection unreliable, add a config knob
`subagent_tool_strategy: single | per_agent` later. The single-tool
shape is the smaller commitment.

### Static config vs. kagenti-discovered registry

**Static config in `agent.yaml`.**
- Pros: matches the immutable-image model. Subagents are deployment
  facts, baked in at scaffold time. Audit-friendly.
- Cons: rosters are per-agent. Cross-tenant or per-tenant rosters
  require redeploying.

**kagenti-discovered registry.**
- Pros: per-tenant rosters, dynamic capability discovery, central
  governance.
- Cons: runtime dependency on kagenti for every cold-start; adds a
  failure mode to agent setup.

**Recommendation.** Start static. Add a `discovery: kagenti` mode in
the `subagents:` block as a follow-on once the static path proves
itself. The discovery extension does not break existing configs —
agents that do not opt in keep the static behaviour.

## Open questions

1. **`agent_name` parameter vs. N synthesised tools.** Tradeoffs above.
   Defer to first PR; pick one and instrument.
2. **Subagent registry scoping.** Static-only at first; kagenti
   discovery as a follow-on.
3. **kagenti-discovered targets.** Out of scope for v1; design the
   `discovery: kagenti` extension separately.
4. **Streaming failure modes.** v2 design needs explicit answers on
   buffered windowing, backpressure, and partial-delta semantics.
   These are best decided once the UI's rendering shape is known
   (issue in `fips-agents/ui-template`).
5. **Composition with workflow graph.** An `AgentNode` may itself
   invoke a subagent. Likely fine — `AgentNode` runs a BaseAgent,
   BaseAgent has the tool. Document the pattern in the workflow
   docs; no code change needed.
6. **Inprocess identity scoping.** When transport is `inprocess`, the
   subagent shares the parent's process and therefore its kagenti
   identity (no HTTP boundary to override at). Should the framework
   forbid `identity: service_account: <name>` for inprocess
   transport, or transparently rebind the identity context for the
   duration of the call? Lean toward forbid + warn; revisit if real
   cases show up.
7. **Sub-subagent depth accounting.** `max_depth` is set on the first
   subagent in a chain. Is that the right shape, or should each
   subagent in the chain be able to set its own cap (with min
   wins)? Lean toward chain-root cap for simplicity.
8. **Resilience to subagent restarts.** If a remote subagent rolls
   over mid-call, retries that succeed land in a fresh process —
   acceptable, but worth documenting that subagents must be
   stateless or session-aware to be re-callable.

## Out of scope

- **A subagent marketplace or registry service.** The registry is
  per-agent config, baked in at deploy time. Cross-agent discovery
  is `fipsagents-platform`'s concern, not this issue's.
- **Cross-tenant subagent routing.** Identity boundary is kagenti's
  job; the framework does not auto-route across tenants.
- **v2 streaming.** Roadmap only; v1 ships buffered.
- **Subagent-driven workflow rewriting.** The graph stays static.
  Subagents do not rewrite the workflow's edges.
- **Multi-modal subagent payloads beyond what tool calls already
  support.** If a subagent takes a file or image argument, that goes
  through the existing `file_ids` / multimodal path, not a new
  subagent-specific channel.

## Related

- Issue #165 — the feature ticket this design backs.
- Issue #164 — per-tool permission policy; required for
  `permission_scope`.
- Issue #163 — Question tool; soft dependency for \"ask before
  delegating\" patterns.
- `RemoteNode` in `fipsagents.workflow` — transport prototype.
- `propagation.py` in `fipsagents.server` — W3C Trace Context.
- UI rendering for subagents (issue in `fips-agents/ui-template`) —
  open design questions on the rendering side; final shape is
  decided after v1 ships and we have real telemetry to ground the
  discussion.
