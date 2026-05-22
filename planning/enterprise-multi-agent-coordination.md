# Enterprise Multi-Agent Coordination

This document is a design reference for multi-agent coordination at
enterprise scale. It extends the WorkItemStore protocol (see
`work-item-coordination-design.md`) with the higher-level concepts
needed when agents are not just processing a queue but participating
in a talent marketplace: inspecting work, building reputations, earning
trust, and operating under governance.

Much of what is described here will eventually live in kagenti.
Implementation will start from fips-agents because fips-agents owns the
WorkItemStore primitives that the higher-level coordination depends on.
This document captures the target state so both projects build toward
compatible interfaces.


## 1. The Marketplace Metaphor

The coordination model is a talent marketplace, not a job queue. The
distinction matters because it determines what information flows where
and who makes assignment decisions.

### How a job queue works

A message enters the queue. A worker pops it. The worker processes it
or dead-letters it. The queue guarantees delivery; the worker
guarantees execution. Neither party inspects the other.

```
  producer ──► queue ──► worker ──► done
                │
                └── dead-letter on failure
```

This model works for homogeneous workers doing identical tasks. It
breaks down when workers have different capabilities, different trust
levels, and the work itself has variable sensitivity.

### How a marketplace works

A requester posts a job with a budget, required skills, acceptance
criteria, and priority. Workers browse available jobs filtered by
their capabilities. They evaluate fit before committing. The requester
(or a delegated reviewer) inspects the deliverable before accepting.
Workers accumulate reputation based on their track record.

```
  requester ──► post(budget, skills, criteria)
                     │
              marketplace pool
                     │
  workers ──► browse(my_skills) ──► checkout ──► deliver
                                                    │
  requester ──► inspect ──► accept/reject ──► trust += f(outcome)
```

Four properties separate this from a queue:

1. **Inspectability.** Work items are not opaque messages. They carry
   structured metadata that workers and reviewers can read before
   committing.
2. **Bidirectionality.** Workers choose work; work does not choose
   workers. (In the advanced auction model, both sides express
   preferences.)
3. **Contractual delivery.** Acceptance is contingent on criteria. The
   deliverable is not "processed" — it is accepted or rejected. Payment
   (trust credit) follows acceptance.
4. **Reputation.** Workers accumulate a track record. Reliable workers
   get access to more sensitive or complex work. Unreliable workers are
   progressively restricted.

### Why this matters for enterprise agents

Most published agent coordination research assumes a model that does
not hold in enterprise environments:

| Assumption | Enterprise reality |
|---|---|
| One human directing one agent (or a hierarchy) | Multiple agents and humans sharing a work pool |
| The human is always in the loop | Some decisions must be automated; some must not |
| The agent has full access to everything | Nobody has full access to everything |
| Agent quality is uniform | Some agents are more trusted than others |
| Work sensitivity is uniform | Some work is more sensitive than other work |
| Work arrives synchronously from one source | Work arrives from webhooks, cron, humans, and agents decomposing parent items |

The marketplace model addresses all of these. The WorkItemStore
protocol (`work-item-coordination-design.md`) provides the primitives.
This document provides the coordination patterns built on top.


## 2. Agent Capability Profiles

The WorkItemStore design uses boolean + ordinal capability matching
for checkout filtering. That covers the mechanical question ("can this
agent call the tools this work item needs?") but not the full
enterprise picture.

### Knowledge, Skills, and Abilities (KSA)

Borrowing from HR and military personnel management, agent capabilities
decompose into three orthogonal dimensions:

**Knowledge** — what the agent has been trained on or has access to.
Domain models, RAG corpora, memory stores, specialized prompts. An
agent with a medical knowledge base should not pick up financial
compliance work, even if it has the right tools connected. Knowledge
is the hardest dimension to verify mechanically; it is closest to
"domain expertise."

**Skills** — what tools and integrations the agent can use. MCP servers,
cluster access, web search, code execution. This is the most
machine-checkable dimension: if the MCP server connected at startup,
the skill exists. If it did not, it does not. The `mcp:{server_name}`
auto-discovery in WorkItemStore's capability matching handles this
directly.

**Abilities** — what the agent has demonstrated it can do reliably.
Trust level, past success rate on similar items, model tier. An agent
running on a 7B parameter model might have the right tools (skills)
and the right RAG corpus (knowledge) but consistently fail complex
reasoning tasks (ability). This is the least machine-checkable and
most interesting dimension.

```
                    ┌──────────────────┐
                    │   Knowledge      │
                    │   (domain data,  │
                    │    RAG, memory)  │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
  ┌───────▼───────┐  ┌──────▼───────┐  ┌───────▼───────┐
  │   Skills      │  │  Abilities   │  │  Trust        │
  │ (tools, MCP,  │  │ (track       │  │ (accumulated  │
  │  integrations)│  │  record,     │  │  reputation,  │
  │               │  │  model tier) │  │  permissions) │
  └───────────────┘  └──────────────┘  └───────────────┘
```

### Capability advertisement

Agents declare capabilities at registration time. In fips-agents, this
is `agent.yaml` under `server.work_items.capabilities`. In kagenti,
this is the agent manifest.

Some capabilities are static (declared at build time, baked into the
image):

```yaml
server:
  work_items:
    capabilities:
      - name: "skill:code_review"
      - name: "knowledge:compliance_v2"
      - name: "model:tool_calling"
```

Some capabilities are discovered at runtime:

- MCP server connections succeed or fail → `mcp:{server_name}`
  present or absent.
- Model capabilities (vision, tool calling) probed at startup via
  `probe_role_support()` or model metadata.
- Trust level changes over the agent's lifetime (see section 3).
- Cluster RBAC may change based on policy updates.

The registration protocol should accommodate both: static capabilities
from config, dynamic capabilities from runtime probes, and externally
managed capabilities from kagenti (trust scores, RBAC grants).

### Capability-requirement matching levels

Work items express requirements at three levels:

1. **Hard requirements.** Agent MUST have these. Missing = cannot
   checkout. Example: `cluster:admin` for infrastructure work,
   `trust: 4` for production deployments.

2. **Soft preferences.** Agent SHOULD have these. Missing = lower
   priority match but still eligible. Example: `model:vision` for
   UI testing (a text-only agent could still review the code, just
   not the screenshots).

3. **Exclusions.** Agent must NOT have these. Example: no agents with
   external network access for air-gapped work; no agents with
   production database credentials for dev-only testing.

v1 (fips-agents) implements hard requirements only. Soft preferences
are a ranking concern that the pull-model checkout handles implicitly
(agents self-select based on their own assessment). Exclusions follow
the Kubernetes taint/toleration model and are deferred to kagenti,
which owns the identity and RBAC layer.

```
  Matching level     Where enforced       Status
  ─────────────────────────────────────────────────
  Hard requirements  WorkItemStore        v1 (fips-agents)
  Soft preferences   Scheduler/ranking    v2 (kagenti)
  Exclusions         Identity/RBAC        v2 (kagenti)
```


## 3. Trust Accumulation

Trust is the enterprise answer to "how do we let agents do
increasingly important work without a human reviewing everything?"

### The problem with binary trust

Most agent frameworks treat trust as a switch: either the agent is
autonomous or it is not. Enterprise environments need a gradient:

- A new agent starts fully supervised. Every output is reviewed.
- After a track record of successful completions on low-stakes work,
  the agent earns limited autonomy on specific task types.
- After sustained reliability, the agent can handle progressively
  more sensitive work with less oversight.
- A security violation or repeated failure reverts trust for the
  affected scope.

This is how human employees are managed. New hires get code review on
every commit. After three months, they get review only on critical
paths. After a year, they approve others' work. There is no reason
agent trust should be less nuanced.

### Trust model (conceptual)

Trust is scoped and ordinal:

```
  Agent: processing-agent-7b
    trust:code_review    = 3   (can review non-critical code unsupervised)
    trust:deploy         = 0   (all deployments require human approval)
    trust:data_extract   = 2   (can extract data, review required for PII)
    trust:ops_remediate  = 1   (can diagnose, remediation requires approval)
```

Trust thresholds gate work item access through the existing capability
matching mechanism. A work item with
`required_capabilities: [{name: "trust:deploy", value: 3}]` excludes
any agent with `trust:deploy < 3`.

### Trust signals

Trust scores are computed from observable data:

| Signal | Measures | Source |
|---|---|---|
| Completion rate | items completed / items checked out | WorkItemStore |
| Acceptance rate | items accepted / items submitted for review | WorkItemStore |
| Budget efficiency | actual cost / budgeted cost | WorkItemStore + per-turn limits |
| Lease reliability | items completed within lease / total checkouts | WorkItemStore |
| Security record | tool inspection violations, guardrail triggers | TraceStore, SecurityConfig audit log |
| Error rate | items failed / items attempted | WorkItemStore |

These signals are raw data. The mapping from signals to a trust score
is a policy decision that kagenti owns. Different organizations will
weight these signals differently. A financial institution might weight
security record at 10x the weight of budget efficiency. A development
shop might weight completion rate highest.

### Trust lifecycle

```
  Agent deployed (trust = 0 for all scopes)
      │
      ▼
  Supervised operation (all outputs reviewed)
      │
      │  N successful completions, 0 security violations
      ▼
  Limited autonomy (low-stakes work auto-accepted)
      │
      │  Continued reliability on expanded scope
      ▼
  Standard autonomy (most work auto-accepted, high-stakes reviewed)
      │
      │  Security violation or sustained failure
      ▼
  Trust revoked for affected scope (reverts to supervised)
      │
      │  Administrative reinstatement + demonstrated recovery
      ▼
  Gradual re-trust (same progression, possibly with tighter thresholds)
```

Key properties:

- Trust accumulates slowly and decays sharply on violations. This is
  asymmetric by design — earning trust takes sustained performance,
  losing it takes a single incident.
- Trust can be revoked administratively (human override) at any time.
  This is a compliance requirement in regulated industries.
- Trust is per-agent-identity, not per-session. An agent that builds
  trust over 1000 sessions does not lose it when a session ends.
  This requires kagenti's persistent identity model.
- Trust history is auditable. Every trust-level change (earn, revoke,
  administrative override) is logged with timestamp, actor, and reason.

### Where trust lives

Trust computation and enforcement is a **kagenti concern**. fips-agents
provides the data:

- WorkItemStore records attempt history, outcomes, and cost.
- TraceStore records tool calls, guardrail triggers, and security
  events.
- fips-agents exposes trust as a capability dimension for matching.

kagenti aggregates these signals into a trust score and makes it
available as an agent attribute. fips-agents queries it at checkout
time instead of relying on the static `agent.yaml` value.

This requires no protocol changes to WorkItemStore. The `Capability`
model already supports ordinal matching. The only change is the data
source: static config in v1, kagenti API in v2.


## 4. Acceptance Patterns

The WorkItemStore design doc defines `accept()` and `reject()` as
primitives. This section explores the patterns agent developers build
on top.

### Pattern 1: Adversarial review agent

A dedicated agent whose only job is to validate completed work against
acceptance criteria. The worker calls `complete(review_required=True)`;
the reviewer subscribes to `review_pending` items, reads the acceptance
criteria, inspects the deliverable, and calls `accept()` or
`reject(reason)`.

The key insight: the evaluator does not need to be able to do the
work. It only needs to judge the output. The reviewer can use a
stronger model tier (Opus reviewing Haiku's work), apply different
tools (linters, test runners, schema validators), or be a completely
different agent type with prompts optimized for critique.

When acceptance criteria are measurable (tests pass, schema validates,
metric meets threshold), the reviewer can be fully automated. When
criteria are subjective, the reviewer should use the strongest
available model and may escalate to human review on low-confidence
judgments.

### Pattern 2: Requirements gate agent

A dedicated agent that validates work item postings before they enter
the pool. Every submitted item passes through the gate agent, which
checks: are acceptance criteria measurable? Is the budget reasonable?
Are required capabilities specific enough? Is the description
unambiguous? Items that pass are `create()`-ed into the store; items
that fail are returned to the source with feedback.

This prevents garbage-in-garbage-out and is particularly valuable
when items arrive from automated sources (webhooks, decomposition by
other agents) where no human reviews the posting before it enters
the pool.

### Pattern 3: Human-in-the-loop review

The worker calls `complete(review_required=True)`. A notification
(webhook, email, Slack) reaches a human reviewer. The human inspects
the result via a dashboard or UI and submits their decision. An
`HttpWebhookSource` (#188) receives the decision and the event handler
calls `accept()` or `reject()` on the store.

The work item sits in `review_pending` for as long as the human takes.
The lease has already been released (the worker completed its part).
No agent resources are consumed while waiting.

### Pattern 4: Graduated autonomy

Combine trust levels with review requirements to create a spectrum
from fully supervised to fully autonomous:

```
  Trust level     Review policy                           Governance
  ──────────────────────────────────────────────────────────────────
  0-1             All work requires human review          Full audit
  2               Work below cost threshold auto-         Spot-check
                  accepted; above requires human review   audit
  3               All work auto-accepted by review        Post-hoc
                  agent; human notified                   sampling
  4-5             Auto-accepted; post-hoc audit           Statistical
                  sampling replaces pre-acceptance        audit
                  review
```

This is how enterprises actually scale AI agent adoption. Not "fully
autonomous from day one" (unacceptable risk in regulated industries)
and not "human reviews everything forever" (does not scale past a
handful of agents).

The trust level determines the `review_required` flag behavior:

- At trust 0-1, the framework forces `review_required=True` on every
  `complete()` call, regardless of what the agent requests.
- At trust 2-3, the agent's choice is respected for work below a
  configurable cost threshold; above the threshold,
  `review_required=True` is forced.
- At trust 4-5, the agent's choice is always respected. A separate
  audit process samples completed items for post-hoc review.

The enforcement logic lives in the server layer (or kagenti), not in
the agent. The agent always calls `complete()` with its honest
assessment of whether review is needed. The platform overrides when
policy requires it.


## 5. Fleet Coordination

This section sketches what kagenti needs to provide so fips-agents can
design compatible interfaces. These are not fips-agents features; they
are the platform capabilities that fips-agents will integrate with.

### Agent registry

Agents register with kagenti on startup. Registration includes:

- **Identity**: Keycloak client credentials, SPIFFE SVID, or
  equivalent. See `docs/architecture.md`, "Agent Identity (Kagenti)".
- **Capabilities**: The KSA profile described in section 2, including
  both static (config-declared) and dynamic (runtime-discovered)
  capabilities.
- **Availability**: Whether the agent is ready to accept work. An
  agent that is mid-request, draining for shutdown, or in an error
  state should not receive new assignments.
- **Capacity**: How many concurrent work items the agent can handle.
  Most agents handle one at a time; agents backed by workflow graphs
  or subagent delegation may handle multiple.

kagenti maintains a live registry of agent instances. Agents heartbeat
to indicate liveness. Dead agents are deregistered; their leased work
items expire naturally via WorkItemStore's lease mechanism. Agents can
update their capabilities dynamically (e.g., when trust changes).

### Work distribution strategies

Three models, each appropriate for different deployment patterns:

**Pull model (fips-agents default).** Agents poll via
`list_available()` or get triggered by CronSource, then checkout
available items. No central scheduler needed. Simple, robust, works
with any number of replicas behind a shared store. Disadvantage:
suboptimal matching (agents take the first available item rather than
the best-fit).

**Push model (kagenti enhancement).** A scheduler with visibility into
both the work pool and the agent registry matches items to agents and
assigns directly. Optimal matching, no wasted polls, centralized
scheduling policies (fairness, affinity, cost minimization). Requires
kagenti infrastructure and introduces a scheduler as a single point
of failure.

**Auction model (future).** Agents bid on work items; the scheduler
selects the best bid based on capabilities, trust, cost estimate, and
availability. Most efficient for heterogeneous fleets with different
cost profiles. A `KagentiWorkItemStore` could implement `checkout()` as
"submit bid and await grant" without protocol changes.

### Cross-agent communication

Today, fips-agents supports parent-to-child delegation via
subagent-as-tool (#165). What is missing:

- **Peer-to-peer coordination.** Agent A needs to notify Agent B
  of something without going through a parent. Example: a monitoring
  agent detects an anomaly and needs to alert a remediation agent
  directly.

- **Broadcast.** An agent completes work that affects shared state
  and all interested agents need to know. Example: a schema migration
  agent completes a migration and all data-processing agents need to
  pick up the new schema.

Options, in increasing order of coupling:

| Mechanism | Coupling | Latency | Complexity |
|---|---|---|---|
| Work item handoff notes | Lowest | High (polling) | Low |
| Event bus (Kafka/Redis) | Medium | Low (pub/sub) | Medium |
| Direct HTTP between agents | Highest | Lowest | High |
| Shared memory store (MemoryHub) | Medium | Medium (polling) | Low |

v1 uses work item handoff notes and shared memory for all cross-agent
communication. This is sufficient for the common case (agent A
finishes, agent B picks up) and does not require new infrastructure.

Direct HTTP between agents is an open design question for kagenti.
It requires service discovery (which agents exist?), authentication
(is this caller allowed to contact me?), and protocol agreement (what
format does the message take?). These are all identity and routing
concerns that belong in kagenti, not in fips-agents.


## 6. Unexplored Problems

These are real enterprise problems with no published solutions in the
agent coordination space. This section documents them so they inform
future design work.

### Semantic conflict detection

Two agents working on different work items can produce outputs that are
individually correct but mutually contradictory:

- Agent A updates a compliance policy. Agent B generates a report
  using the old policy version.
- Agent A modifies a database schema. Agent B writes a migration
  script targeting the old schema.
- Agent A reclassifies training data. Agent B trains a model on the
  pre-reclassification labels.
- Agent A changes an API contract. Agent B writes integration tests
  against the old contract.

File-level merge conflicts (git) are the simplest case and git handles
them. But most semantic conflicts involve database state, API
contracts, policy documents, or data classifications where git has no
visibility.

Options:

| Approach | Coverage | Complexity | Feasibility |
|---|---|---|---|
| Optimistic concurrency (version stamps on shared resources) | Resource-level | Low | v1 hooks exist |
| Resource reservation (agent declares what shared state it will touch at checkout time) | Declared resources only | Medium | v2 |
| Domain-specific conflict detectors (custom tooling per use case) | Full for that domain | High per domain | Case-by-case |
| Post-hoc reconciliation (a dedicated agent reviews all completed items in a batch for consistency) | Broad | Medium | v2 |

No good general solution exists. v1 provides hooks: artifact references
in handoff notes, dependency tracking via `depends_on`, and version
stamps on progress updates. Conflict detection beyond these hooks is
a domain-specific concern.

### Cost governance

The WorkItemStore protocol handles per-item budgets. Three additional
cost dimensions need governance at the fleet level:

**Fleet-level cost control.** Total spend across all agents per
hour/day/month. An organization needs to cap total AI spend regardless
of how many agents are running or how many work items are in the pool.
This is a kagenti concern because it requires fleet visibility.

**Cost attribution.** Which team's budget does this agent's work
charge to? In a shared pool, Agent A (owned by Team X) might process
work items posted by Team Y. Whose budget gets charged? This requires
integration with enterprise cost management systems (chargeback,
showback) that are outside the fips-agents + kagenti stack.

**Runaway cost detection.** An agent is spending 10x the average on a
single work item. Is it stuck in a loop? Working on something
genuinely complex? The doom-loop guard (#167) catches repetitive tool
calls, but a non-repetitive agent steadily consuming budget on a dead
end is a different pattern. This requires baseline modeling: what is
the expected cost distribution for this type of work item? Statistical
anomaly detection on budget consumption is a future kagenti feature.

### Work item provenance and audit

In regulated industries (finance, healthcare, defense), every decision
needs a traceable chain of accountability:

- Who created this work item? (Human, agent, automated source)
- Who approved it for the pool? (Requirements gate agent, human
  manager)
- Who worked on it? (Full attempt history with handoff notes)
- Who reviewed the result? (Reviewer identity and rationale)
- Who accepted it? (Human or automated acceptance, with justification)

WorkItemStore's `attempt_history`, `created_by`, and `completed_by`
provide the raw data. TraceStore records the detailed execution trace
for each attempt. Audit aggregation — assembling these records into a
compliance-ready report — is a platform concern (kagenti or
fipsagents-platform).

The critical implementation requirement: provenance data must be
immutable once written. `attempt_history` is append-only by design.
Session revert (#168) does not erase trace data. This is intentional.

### Multi-tenancy

Multiple teams sharing a work pool with visibility boundaries:

- Agent A can see items tagged for Team X but not Team Y.
- Agent A can see items at classification level "internal" but not
  "restricted."
- Agents within a team can see each other's progress; agents in
  different teams cannot.

This is an identity and authorization concern. kagenti owns identity;
WorkItemStore needs to support filtering by tenant or scope, but the
authorization logic lives outside the store.

The store's interface accommodates this through `list_available()`
filtering. A kagenti-backed store implementation would inject tenant
filters based on the calling agent's identity. The SQLite and Postgres
backends in fips-agents do not enforce tenancy because they lack an
identity provider.


## 7. Relationship to Existing Research

Most published agent coordination research falls into categories that
do not address enterprise requirements. Understanding where they stop
is useful for positioning this work.

### Single-agent loop

Anthropic's "Effective Harnesses for Long-Running Agents," Claude Code,
Devin, and similar systems. One agent, many context windows, serialized
work. Coordination is handled by progress files and structured handoff
within a single agent's execution history.

**Where it stops.** No concurrent work, no shared pool, no trust,
no capability matching. Scaling requires multiple independent agents,
each with their own work, which is not coordination — it is isolation.

### Hierarchical multi-agent

CrewAI, AutoGen, LangGraph. A manager agent delegates to specialist
agents within a single session. The manager owns the plan and the
specialists execute steps.

**Where it stops.** The hierarchy is rigid: one manager, N workers,
defined at graph construction time. There is no dynamic discovery,
no capability matching at assignment time, no trust accumulation,
no persistence between sessions. If the manager crashes, the entire
hierarchy dies. If a specialist is not pre-registered in the graph,
it cannot be used.

fips-agents' subagent-as-tool (#165) provides this pattern. The
coordination problem starts when agents operate independently across
sessions, which hierarchical models do not address.

### Swarm / colony

Academic research on emergent behavior from simple rules (ant colony
optimization, particle swarm). Many agents with identical, simple
rules producing complex aggregate behavior.

**Where it stops.** Enterprise work requires predictable, auditable
outcomes. Emergent behavior is the opposite of that. When a
compliance officer asks "why did the agent make this decision?", the
answer cannot be "the swarm converged." Additionally, swarm models
assume homogeneous agents; enterprise fleets are heterogeneous by
design.

### What is missing in published work

| Capability | Published frameworks | fips-agents + kagenti |
|---|---|---|
| Persistent coordination across sessions | None | WorkItemStore + leases |
| Trust-gated work assignment | None | Capability matching + trust scores |
| Budget-aware handoff | None | Budget headroom enforcement |
| Acceptance review as a first-class primitive | None | `accept()` / `reject()` on store |
| Graduated autonomy from supervised to autonomous | None | Trust levels + review policy |
| Fleet cost governance | None (per-session only) | Per-item + fleet-level (kagenti) |
| Audit trail for regulated industries | Partial (logging only) | Structured provenance data |


## 8. Implementation Phasing

### Phase 1: WorkItemStore protocol (fips-agents)

Issue #214. The foundation layer.

Deliverables:
- `WorkItemStore` ABC in `fipsagents.server.work_items`
- `NullWorkItemStore` (default no-op)
- `SqliteWorkItemStore` (dev, testing, single-node)
- Five stock LLM tools (`check_available_work`, `checkout_work_item`,
  `complete_work_item`, `release_work_item`, `update_work_progress`)
- Budget headroom enforcement
- Boolean + ordinal capability matching
- Six `StreamEvent` variants
- Session continuity rule template
- `WorkItemConfig` on `ServerConfig`

Integration points with existing features:
- CronSource (#188) triggers periodic work polling
- Per-turn limits (#195) enforce per-item budgets
- Doom-loop guard (#167) catches stuck workers
- Subagent-as-tool (#165) enables in-process review patterns
- Session persistence (#182) preserves work context across restarts

### Phase 2: Enterprise backends (fips-agents)

Deliverables:
- `PostgresWorkItemStore` (asyncpg, advisory locks, shared
  `storage.database_url`)
- `GitHubWorkItemStore` (maps to Issues API for open-source agent
  fleets)
- Prometheus metrics for work item queue health
  (`agent_work_items_checked_out_total`, `_completed_total`,
  `_duration_seconds`, `_lease_expiries_total`)
- REST endpoints for external integration (`/v1/work-items/*`)

### Phase 3: Fleet coordination (kagenti)

Deliverables:
- Agent registry with capability advertisement and heartbeat
- Trust accumulation engine (signal aggregation, scoped scoring,
  decay and revocation)
- Push-model work distribution (scheduler with optimal matching)
- Fleet cost governance (per-hour/day/month caps, chargeback
  attribution)
- Audit trail aggregation (compliance-ready provenance reports)

Prerequisites: kagenti identity infrastructure (Keycloak operator,
SPIFFE SVIDs), WorkItemStore Phase 1 (for signal data).

### Phase 4: Advanced patterns (kagenti)

Deliverables:
- Auction-based work assignment (bid/grant protocol)
- Semantic conflict detection hooks (resource reservation, domain
  detector API)
- Multi-tenant work pools (tenant-scoped visibility, authorization
  filters)
- Cross-agent peer communication (service discovery, message routing)

Prerequisites: Phase 3 agent registry, kagenti RBAC.

### Phase boundary principle

Each phase is independently valuable. An organization can run a single
agent with SqliteWorkItemStore (Phase 1) and get structured work
management with handoff notes and budget control. Adding Postgres
(Phase 2) adds durability and concurrent access. Adding kagenti
(Phase 3) adds fleet management and trust. Phase 4 adds optimization.

No phase requires the next phase to deliver value. No phase breaks
backward compatibility with the previous phase.


## 9. Interface Contracts Between Layers

This section defines the integration surface between fips-agents and
kagenti so both projects can build toward compatible interfaces
without tight coupling.

### fips-agents provides to kagenti

| Data | Source | Format |
|---|---|---|
| Work item attempt history | `WorkItemStore.get().attempt_history` | `list[Attempt]` with actor_id, timestamps, outcome |
| Tool execution traces | `TraceStore.list_traces_for_session()` | Span trees with tool names, durations, errors |
| Security audit events | `fipsagents.security.audit.*` loggers | Structured JSON (tool inspection, guardrail, permission) |
| Budget utilization | Per-item cost tracking in WorkItemStore | Tokens consumed, cost_usd, budget remaining |
| Capability declarations | `agent.yaml` → `WorkItemConfig.capabilities` | `list[Capability]` (name, value) |
| Runtime capability probes | MCP connections, model metadata | Dynamic `mcp:{name}`, `model:{capability}` |

### kagenti provides to fips-agents

| Data | Access pattern | Format |
|---|---|---|
| Trust scores (per scope) | HTTP API, queried at checkout time | `dict[str, float]` mapping scope to score |
| Agent identity | K8s Secret (client credentials) | OAuth2 Client Credentials |
| Fleet cost policy | Config injection or API | Thresholds, caps, attribution rules |
| Tenant scope | Identity token claims | Tenant ID, classification level |
| Review policy overrides | Config injection or API | Trust-level → review_required mapping |

### Contract stability

The integration surface is deliberately narrow. fips-agents exposes
raw data (attempts, traces, costs); kagenti computes derived values
(trust scores, fleet metrics, compliance reports). Neither project
needs to understand the other's internals.

If kagenti is not deployed, fips-agents falls back to static trust
values from `agent.yaml`. The degraded mode is fully functional — it
just lacks dynamic trust accumulation and fleet coordination. This is
the same graceful-degradation pattern used throughout fips-agents
(NullSessionStore, NullTraceStore, NullWorkItemStore).


## Related

- `planning/work-item-coordination-design.md` — the WorkItemStore
  protocol this document builds on. Read that first for the ABC,
  models, and primitives.
- `docs/responsibilities.md` — layer boundaries. Fleet scheduling,
  trust management, and multi-tenant isolation are platform concerns
  (kagenti), not fips-agents concerns.
- `docs/architecture.md` — settled architecture decisions. Agent
  Identity (Kagenti) section describes the identity model that trust
  accumulation depends on.
- `planning/subagent-tool-design.md` — parent-to-child delegation
  (#165). The starting point for cross-agent communication.
- `planning/event-triggered-design.md` — EventSource/EventSink (#188).
  CronSource enables ambient agent workers; HttpWebhookSource enables
  human-in-the-loop review.
- Anthropic, "Effective Harnesses for Long-Running Agents" — the
  single-agent progress file pattern that motivates persistent
  multi-agent coordination.
