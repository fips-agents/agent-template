# Work Item Coordination Design

## Problem

fips-agents handles single-turn and multi-turn interactions well, and
the event-triggered mode (#188) extends the framework to reactive and
scheduled workloads. But a class of real-world agent deployments
remains unsupported: coordinated, multi-agent work on a shared pool
of tasks.

Consider these scenarios:

- **Fleet code review.** A cron-triggered agent wakes up, scans a
  backlog of pending PRs, claims one, reviews it, posts comments,
  and marks the item complete. A second instance of the same agent
  wakes on the same schedule, sees the first PR is already claimed,
  and picks the next one.
- **Multi-stage document processing.** A coordinator agent receives
  a batch of 500 contracts. It creates a work item per contract,
  each tagged with `needs: docling, trust >= 2`. Specialist agents
  with Docling MCP servers check out items matching their
  capabilities, extract terms, and post results. A summarizer agent
  picks up completed extractions and produces a digest.
- **Ops runbook execution.** An alert fires. The triage agent
  creates a work item describing the incident. A diagnostics agent
  checks it out, runs its toolset, writes findings into the handoff
  note, and releases the item. A remediation agent picks it up,
  reads the diagnosis, and executes the fix.

Anthropic's "Effective Harnesses for Long-Running Agents" describes a
progress file + feature list pattern for single-agent, multi-context-
window work. That pattern breaks down when multiple agents (and
humans) work concurrently on a shared pool. Concurrent access needs
leases, not files. Capability matching needs structure, not free text.
Handoff between agents of different types needs a protocol, not
conventions.

fips-agents already has the building blocks: session persistence
(#182), event-triggered mode (#188, CronSource, WebhookSource),
AgentState with reducers (#190), compaction (#166), subagent-as-tool
(#165), per-turn resource limits (#195), and doom-loop detection
(#167). What is missing is a coordination protocol for shared work.

## Design Principles

1. **WorkItemStore is server-layer, additive, and optional.**
   BaseAgent has no concept of work items. The server wraps the
   store, the stock LLM tools call it, and agents that do not
   configure it see zero behavior change.
2. **Lease, do not lock.** Hard locks require explicit unlock and
   break on crashes. Leases expire. An agent that dies mid-work
   loses its lease and the item returns to the pool.
3. **The framework provides primitives, not policy.** Review
   patterns, decomposition strategies, and trust models are
   agent-developer choices. The framework provides the store, the
   tools, and the events.
4. **Budget headroom is non-negotiable.** An agent must always
   reserve enough budget to write a handoff note, update the work
   item, and release its lease. The framework enforces this.
5. **Capability matching is declarative.** Agents declare what they
   can do; work items declare what they need. Matching happens at
   checkout time, not in the agent's prompt.

## Work Item Model

A work item is a unit of work that can be discovered, claimed,
executed, and handed off. The model is deliberately richer than a
task queue message because it must support multi-agent coordination
across context windows, agent types, and time.

```python
class WorkItem(BaseModel):
    # Identity
    id: str                                    # wi_{timestamp}_{random}
    title: str
    description: str
    
    # Status
    status: WorkItemStatus                     # available, checked_out,
                                               # completed, failed,
                                               # review_pending, blocked
    
    # Requirements
    required_capabilities: list[Capability]     # what the worker needs
    
    # Budget
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_duration_seconds: int | None = None
    
    # Priority
    priority: int = 0                          # higher = more urgent
    
    # Assignment
    assignee: str | None = None                # actor_id of current holder
    lease_expires_at: datetime | None = None
    
    # Hierarchy
    parent_id: str | None = None               # decomposition tree
    depends_on: list[str] = Field(default_factory=list)
    
    # Acceptance
    acceptance_criteria: list[str] = Field(default_factory=list)
    
    # Handoff
    handoff_note: HandoffNote | None = None     # structured progress context
    
    # Provenance
    created_by: str
    created_at: datetime
    completed_by: str | None = None
    completed_at: datetime | None = None
    attempt_history: list[Attempt] = Field(default_factory=list)

class Capability(BaseModel):
    name: str                                  # "mcp:web_search", "cluster:admin",
                                               # "trust:3", "skill:docling"
    value: str | int | float | bool = True     # for ordinal capabilities

class HandoffNote(BaseModel):
    accomplished: list[str]                    # what was done
    attempted: list[str] = Field(default_factory=list)   # what was tried and failed
    remaining: list[str] = Field(default_factory=list)   # what is left
    blockers: list[str] = Field(default_factory=list)    # why this stalled
    artifacts: dict[str, str] = Field(default_factory=dict)  # key=label, value=ref
    context: str = ""                          # free-form for the next actor

class Attempt(BaseModel):
    actor_id: str
    started_at: datetime
    ended_at: datetime
    outcome: str                               # "completed", "released", "expired",
                                               # "failed"
    handoff_note: HandoffNote | None = None
```

### Status transitions

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
             ┌─────────────┐    checkout()    ┌──────────────┐ │
  create ──► │  available   ├────────────────►│ checked_out  │ │
             └──────┬──────┘                  └──────┬───────┘ │
                    ▲                                │         │
                    │  release()                     │         │
                    │  or lease expiry               │         │
                    ├────────────────────────────────┘         │
                    │                                          │
                    │                    complete()            │
                    │              ┌──────────────────┐        │
                    │              │                  ▼        │
                    │         ┌────┴───────┐   ┌───────────┐  │
                    │         │ review_    │   │ completed  │  │
                    │         │ pending    │   └────────────┘  │
                    │         └────┬───────┘                   │
                    │              │ reject()                  │
                    │              └───────────────────────────┘
                    │                                
                    │                    fail()
                    │              ┌────────────┐
                    └──────────────┤  failed    │ (may retry → available)
                                   └────────────┘
```

`blocked` is an additional status for items whose `depends_on` list
contains incomplete items. The store automatically transitions
`blocked → available` when all dependencies complete. This is
evaluated lazily on `list_available()`, not eagerly on every
`complete()`.

## WorkItemStore ABC

Module: `fipsagents.server.work_items`

```python
class WorkItemStore(ABC):
    @abstractmethod
    async def create(self, item: WorkItem) -> WorkItem: ...

    @abstractmethod
    async def list_available(
        self,
        *,
        capabilities: list[Capability] | None = None,
        max_results: int = 10,
        parent_id: str | None = None,
    ) -> list[WorkItem]: ...

    @abstractmethod
    async def checkout(
        self,
        item_id: str,
        actor_id: str,
        *,
        lease_duration: timedelta | None = None,
    ) -> WorkItem: ...

    @abstractmethod
    async def renew_lease(
        self,
        item_id: str,
        actor_id: str,
        *,
        lease_duration: timedelta | None = None,
    ) -> WorkItem: ...

    @abstractmethod
    async def complete(
        self,
        item_id: str,
        *,
        result: dict[str, Any] | None = None,
        handoff_note: HandoffNote | None = None,
        review_required: bool = False,
    ) -> WorkItem: ...

    @abstractmethod
    async def release(
        self,
        item_id: str,
        *,
        handoff_note: HandoffNote | None = None,
    ) -> WorkItem: ...

    @abstractmethod
    async def fail(
        self,
        item_id: str,
        *,
        error: str,
        handoff_note: HandoffNote | None = None,
        retry: bool = False,
    ) -> WorkItem: ...

    @abstractmethod
    async def update_progress(
        self,
        item_id: str,
        *,
        progress: dict[str, Any],
    ) -> WorkItem: ...

    @abstractmethod
    async def get(self, item_id: str) -> WorkItem | None: ...

    @abstractmethod
    async def accept(self, item_id: str) -> WorkItem: ...

    @abstractmethod
    async def reject(
        self,
        item_id: str,
        *,
        reason: str,
    ) -> WorkItem: ...

    async def expire_leases(self) -> list[WorkItem]:
        """Release items whose leases have expired.

        Called periodically by the server. Default: no-op (backends
        that support TTL natively may not need this).
        """
        return []

    async def close(self) -> None: ...
```

`list_available()` returns items ordered by priority (descending),
then creation time (ascending). It filters by capability match and
excludes items that are checked out, completed, or blocked.
`capabilities` is the requesting agent's declared set -- items whose
`required_capabilities` are not a subset of the agent's capabilities
are excluded.

`checkout()` is atomic. If two agents race to checkout the same item,
exactly one succeeds and the other gets `WorkItemAlreadyCheckedOut`.
The winning agent's `actor_id` and `lease_expires_at` are set on the
item; the attempt is recorded in `attempt_history`.

`complete(review_required=True)` transitions to `review_pending`
instead of `completed`. The item stays in `review_pending` until
`accept()` or `reject()` is called.

`fail(retry=True)` transitions back to `available` with the attempt
recorded. `fail(retry=False)` transitions to terminal `failed`.

## Lease-Based Checkout

### Lease lifecycle

```
  checkout(item_id, actor_id, lease_duration=300s)
       │
       ▼
  ┌─────────────────────────────────────┐
  │ lease active                        │
  │ assignee = actor_id                 │
  │ lease_expires_at = now + 300s       │
  │                                     │
  │  renew_lease(item_id, actor_id)     │
  │  → lease_expires_at = now + 300s    │
  │                                     │
  │  update_progress(item_id, progress) │
  │  → implicit lease renewal           │
  └──────────────┬──────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   complete()        lease expires
   or release()      → status = available
   or fail()         → attempt recorded
                     → handoff_note preserved
```

### Lease expiry semantics

When a lease expires, the item transitions back to `available`. The
attempt is recorded with `outcome: "expired"`. If the agent wrote a
handoff note via `update_progress()` before expiring, that note is
preserved on the item as `handoff_note` so the next agent can read it.

Lease expiry is handled by `expire_leases()`, called periodically.
For SQLite, the server runs this on a configurable interval (default:
60 seconds) via an `asyncio.Task`. For Postgres, a `WHERE
lease_expires_at < now()` clause on `list_available()` handles it
implicitly, with `expire_leases()` as a cleanup pass for
`attempt_history` bookkeeping.

### Lease duration defaults

Work items can declare their own `max_duration_seconds`. If not set,
the global `server.work_items.lease_duration` applies (default: 300
seconds). `checkout()` accepts an explicit `lease_duration` that
overrides both.

The hierarchy: explicit `checkout()` arg > item-level
`max_duration_seconds` > global config.

## Budget Headroom

An agent must never exhaust its budget without writing a handoff note.
The framework enforces this through a headroom reservation.

### How it works

`server.work_items.budget_headroom_pct` (default: 10) reserves a
percentage of the agent's per-turn budget for handoff operations. When
the agent's remaining budget drops below the headroom threshold, the
framework:

1. Emits a `BudgetHeadroomWarning` stream event.
2. On the next tool call attempt, injects a system message: "You are
   approaching your budget limit. Write a handoff note and release or
   complete this work item."
3. If the agent ignores the warning and attempts another non-handoff
   tool call, the framework blocks it with a structured error and
   force-emits a `HandoffRequired` event.

The agent's LLM tools (`complete_work_item`, `release_work_item`)
are always permitted, even after the hard cutoff. The framework
guarantees enough budget for one final model call to produce a
handoff note.

### Handoff note structure

```python
class HandoffNote(BaseModel):
    accomplished: list[str]       # what was done
    attempted: list[str]          # what was tried and failed
    remaining: list[str]          # what is left
    blockers: list[str]           # why this stalled
    artifacts: dict[str, str]     # label → reference (commit SHA, file path, URL)
    context: str                  # free-form for the next actor
```

This structure is not free text. The next agent can programmatically
read `remaining` to understand its scope, check `blockers` to avoid
repeating failed approaches, and follow `artifacts` to pick up where
the previous agent left off.

### Relationship to per-turn limits (#195)

Per-turn limits (`max_tokens_per_turn`, `max_cost_per_turn_usd`) and
budget headroom serve different purposes. Per-turn limits cap
individual model calls. Budget headroom caps the total work budget and
reserves a tail for cleanup. Both can fire independently.

When a work item has `max_tokens` or `max_cost_usd`, the server
configures per-turn limits proportionally and sets up the headroom
reservation at checkout time. When neither is set, headroom tracking
is best-effort (based on global per-turn limits if configured, or
disabled).

## Agent Capability Matching

### Capability declaration

Agents declare capabilities in `agent.yaml`:

```yaml
server:
  work_items:
    enabled: true
    capabilities:
      - name: "mcp:web_search"
      - name: "mcp:docling"
      - name: "skill:code_review"
      - name: "cluster:read"
      - name: "trust"
        value: 3
      - name: "model:vision"
```

Capabilities are also auto-discovered from the agent's runtime state:
connected MCP servers contribute `mcp:{server_name}` capabilities,
loaded skills contribute `skill:{skill_name}`, and the model's
declared capabilities (vision, tool calling) contribute
`model:{capability}`.

### Matching semantics

A work item's `required_capabilities` is a conjunction: the agent
must satisfy every requirement. Matching rules:

- **Boolean capabilities.** `name: "mcp:web_search"` matches if the
  agent has a capability with `name="mcp:web_search"`. The value is
  ignored (presence is sufficient).
- **Ordinal capabilities.** `name: "trust", value: 3` matches if the
  agent has a capability with `name="trust"` and `value >= 3`.
  Comparison is numeric.
- **Negation is not supported in v1.** "Must NOT have cluster:write"
  is better handled by permission scoping, not capability matching.

This is deliberately simpler than Kubernetes scheduling. Taints,
tolerations, affinity, and anti-affinity are deferred. The matching
logic is a single function, easily extended.

### Relationship to PermissionSource (#164)

Capability matching and permission enforcement are orthogonal:

- **Capabilities** gate work assignment: "Can this agent do this
  type of work?"
- **Permissions** gate tool execution: "Is this agent allowed to
  call this tool right now?"

An agent can have the `mcp:kubectl` capability (it has kubectl
tools connected) but be denied the `kubectl_delete` tool by its
permission policy. Capabilities describe potential; permissions
describe policy.

## Acceptance and Review

### Review flow

Some work items need validation before they count as done. The
`complete(review_required=True)` call transitions the item to
`review_pending` instead of `completed`. The item stays there until
an authorized actor calls `accept()` or `reject()`.

```
  Worker agent                    Reviewer
       │                              │
       │  complete(review_required    │
       │           =True)             │
       │──────────────────────────────►
       │                              │
       │                   list_available(
       │                     status=review_pending)
       │                              │
       │                   get(item_id)
       │                   inspect result
       │                              │
       │              ┌───────────────┤
       │              ▼               ▼
       │         accept()        reject(reason)
       │              │               │
       │              ▼               ▼
       │         completed        available
       │                         (handoff_note includes
       │                          rejection reason)
```

### Framework primitives, not policy

The framework provides `accept()` and `reject()` on the store. It
does not enforce who can call them. Common patterns that agent
developers can build on top:

- **Adversarial review.** A second agent with review capabilities
  validates the work against acceptance criteria.
- **Human-in-the-loop.** A webhook delivers a human review decision
  to the agent via HttpWebhookSource; the event handler calls
  `accept()` or `reject()`.
- **Self-review with different model.** The same agent calls a
  higher-tier model to validate its own work. (Useful when the
  worker runs on a cost-efficient model.)
- **Requirements gate.** A dedicated agent validates that new work
  items have measurable acceptance criteria before they enter the
  pool. This is a pre-creation pattern, not a review pattern.

## Cron-Based Workers

Event-triggered mode (#188) already supports CronSource. The work
item coordination system composes naturally with it:

```yaml
server:
  event_sources:
    - type: cron
      schedule: "*/5 * * * *"        # every 5 minutes
      event_type: check-for-work

  work_items:
    enabled: true
    backend: sqlite
    capabilities:
      - name: "skill:code_review"
      - name: "mcp:github"
```

The agent's event handler (via `translate_event()` customisation or
prompt engineering) follows this pattern:

```
  CronSource fires
       │
       ▼
  list_available(capabilities=my_capabilities)
       │
       ├── no items → return, sleep until next cron tick
       │
       ├── items available
       │       │
       │       ▼
       │   checkout(item_id, actor_id=my_session_id)
       │       │
       │       ▼
       │   do the work (tool calls, model calls)
       │       │
       │       ├── success → complete(item_id, result, handoff_note)
       │       ├── partial → release(item_id, handoff_note)
       │       └── failure → fail(item_id, error, handoff_note)
       │
       ▼
  sleep until next cron tick
```

This is the "ambient agent" pattern. The agent is always deployed,
wakes periodically, processes available work, and sleeps. No external
orchestrator is needed. Multiple replicas of the same agent can run
concurrently -- the lease mechanism prevents double-processing.

## Stock LLM Tools

Five tools are auto-registered when `server.work_items.enabled` is
true. All are `visibility="llm_only"` -- the LLM decides when to
check for work, claim items, and report results.

```python
@tool(visibility="llm_only")
async def check_available_work(
    max_results: int = 5,
) -> list[dict]:
    """Check the work item pool for items matching your capabilities.

    Returns a list of available work items you are qualified to do,
    ordered by priority. Call checkout_work_item to claim one.
    """
    ...

@tool(visibility="llm_only")
async def checkout_work_item(
    item_id: str,
    lease_duration_seconds: int | None = None,
) -> dict:
    """Claim a work item. You now own it until the lease expires
    or you release/complete it. Read the handoff_note if present --
    a previous agent may have made progress.
    """
    ...

@tool(visibility="llm_only")
async def complete_work_item(
    item_id: str,
    result_summary: str,
    accomplished: list[str],
    review_required: bool = False,
) -> dict:
    """Mark a work item as complete. Provide a summary of what you
    accomplished. Set review_required=True if this needs validation.
    """
    ...

@tool(visibility="llm_only")
async def release_work_item(
    item_id: str,
    accomplished: list[str],
    remaining: list[str],
    blockers: list[str] | None = None,
    context: str = "",
) -> dict:
    """Release a work item back to the pool. Another agent will pick
    it up. Write a clear handoff note -- the next agent has no other
    context about your progress.
    """
    ...

@tool(visibility="llm_only")
async def update_work_progress(
    item_id: str,
    status_message: str,
    accomplished_so_far: list[str] | None = None,
) -> dict:
    """Update progress on a checked-out work item. Also renews your
    lease. Call this periodically on long-running work.
    """
    ...
```

The tool implementations call through to the `WorkItemStore` via a
reference on the server, following the same pattern as the question
tool (#163) and subagent tool (#165): `make_work_item_tools(server)`
factory function that closes over the server reference.

## Conflict Detection

Two agents working different items can produce conflicting changes
to shared state. This is fundamentally hard and v1 does not attempt
a general solution. It provides hooks.

### What v1 does

- **Lease exclusivity.** Only one agent holds a given work item at a
  time. This prevents conflict on the item itself.
- **Artifact references.** `HandoffNote.artifacts` carries references
  (commit SHAs, file paths, API versions) that downstream agents can
  inspect for staleness.
- **Version stamps on progress.** `update_progress()` includes a
  monotonic version counter. The store rejects stale updates
  (optimistic concurrency on the item row).

### What v1 does not do

- Detect that two agents editing different work items both modified
  the same file.
- Semantic conflict detection (e.g., two agents configuring
  contradictory firewall rules).
- Resource reservation (e.g., "this work item needs exclusive access
  to the staging database").

### Guidance for agent developers

For git-based workflows, agents should work on branches and let merge
conflict detection handle file-level conflicts naturally.

For database or API state, agents should use idempotent operations
and check preconditions before mutating. The handoff note is the
right place to document preconditions the next agent should verify.

Resource reservation (locking a database, an environment, a
namespace) is a v2 concern. The `Capability` model can express "needs
exclusive access to X" but the framework does not enforce exclusivity
beyond the work item lease.

## Trust Accumulation

Trust is ultimately a kagenti and platform concern, not a
fips-agents-core concern. But the WorkItemStore protocol needs to
participate in the trust model by threading trust levels through
capability matching.

### v1: trust as a capability

Trust level is declared as a `Capability` with an ordinal value:

```yaml
capabilities:
  - name: "trust"
    value: 3
```

Work items that require `trust: 3` exclude agents with `trust: 2`.
Trust values are static per deployment -- set in `agent.yaml`, baked
into the image.

### Future: dynamic trust

Dynamic trust accumulation (earn points through successful
completions, lose points through failures) belongs in kagenti, which
has agent identity, credential management, and fleet visibility. The
integration path:

1. kagenti tracks completion/failure rates per agent identity.
2. kagenti exposes a trust score via its agent metadata API.
3. fips-agents queries trust at checkout time instead of relying on
   the static `agent.yaml` value.

This requires no protocol changes to WorkItemStore -- the
`Capability` model already supports ordinal matching. The only change
is where the trust value comes from.

## Backends

### Phase 1

**NullWorkItemStore.** No-op, returns empty lists, raises
`NotImplementedError` on mutation. Default when
`server.work_items.enabled` is false.

**SqliteWorkItemStore.** Single-file, aiosqlite. Suitable for dev,
testing, and single-node deployments. Lease expiry via periodic
`expire_leases()` task. Atomic checkout via SQLite's
`BEGIN IMMEDIATE` transaction.

### Phase 2

**PostgresWorkItemStore.** asyncpg. Advisory locks for atomic
checkout (`pg_try_advisory_xact_lock`). Lease expiry via
`WHERE lease_expires_at < now()` on queries, with periodic cleanup
for history bookkeeping. Shares the `storage.database_url` from
`StorageConfig` (same as sessions and traces). Uses its own
connection pool (same pattern as `AgeGraphStore`).

### Phase 3 (community)

**GitHubWorkItemStore.** Maps to the GitHub Issues API. Work items
are issues with labels for status and capabilities. Checkout is
assignment + label transition. Handoff notes are issue comments.

This is a natural fit for open-source agent fleets where the work
backlog is already in GitHub Issues. The mapping is lossy (no lease
TTL enforcement by GitHub, no atomic checkout), but practical for
many use cases.

Other community backends (Jira, Linear, Redis) follow the same ABC.

### Backend factory

```python
def create_work_item_store(config: WorkItemConfig) -> WorkItemStore:
    if not config.enabled:
        return NullWorkItemStore()
    if config.backend == "sqlite":
        return SqliteWorkItemStore(config.database_url or "work_items.db")
    if config.backend == "postgres":
        return PostgresWorkItemStore(config.database_url)
    raise ValueError(f"Unknown work item backend: {config.backend}")
```

Follows the same factory pattern as `create_compactor()`,
`create_permission_source()`, and `create_event_source()`.

## Configuration

```yaml
server:
  work_items:
    enabled: false                              # default: off
    backend: null                               # null | sqlite | postgres
    database_url: ${WORK_ITEMS_DB_URL:-}        # falls back to storage.database_url
    lease_duration: 300                          # default lease in seconds
    budget_headroom_pct: 10                     # % of budget reserved for handoff
    expire_check_interval: 60                   # seconds between expiry sweeps

    capabilities:                               # this agent's capabilities
      - name: "mcp:web_search"
      - name: "skill:code_review"
      - name: "trust"
        value: 3
```

`WorkItemConfig` is a new field on `ServerConfig`, following the
same pattern as `SessionsConfig`, `TracesConfig`, and `GraphConfig`.

## Stream Events

Six new `StreamEvent` variants:

| Event | Fields |
|-------|--------|
| `WorkItemCheckedOut` | item_id, title, priority, lease_expires_at |
| `WorkItemCompleted` | item_id, title, outcome |
| `WorkItemReleased` | item_id, title, handoff_note_summary |
| `WorkItemFailed` | item_id, title, error |
| `BudgetHeadroomWarning` | item_id, remaining_pct, headroom_pct |
| `HandoffRequired` | item_id, remaining_pct |

These are emitted by the stock LLM tools and budget headroom logic.
They flow through the existing observer chain (MetricsCollector,
TraceCollector) and appear in traces and SSE streams.

## Prometheus Metrics

Four new metrics (requires `[metrics]` extra):

| Metric | Type | Labels |
|--------|------|--------|
| `agent_work_items_checked_out_total` | counter | status, capability |
| `agent_work_items_completed_total` | counter | status, outcome |
| `agent_work_item_duration_seconds` | histogram | outcome |
| `agent_work_item_lease_expiries_total` | counter | - |

## Server Wiring

### Lifecycle

`WorkItemStore` is created in `_lifespan()` after `agent.setup()`
and before event source startup:

```python
self._work_item_store = create_work_item_store(config.work_items)
```

A periodic `asyncio.Task` runs `expire_leases()` when the backend
is SQLite (Postgres handles this in queries). The task is cancelled
during shutdown.

Stock LLM tools are registered via `make_work_item_tools(server)`
during `_lifespan()`, after the store is created. They are only
registered when `work_items.enabled` is true.

### REST endpoints (optional, Phase 2)

```
POST   /v1/work-items                    # create
GET    /v1/work-items                    # list (filterable)
GET    /v1/work-items/{id}               # get
POST   /v1/work-items/{id}/checkout      # checkout
POST   /v1/work-items/{id}/complete      # complete
POST   /v1/work-items/{id}/release       # release
POST   /v1/work-items/{id}/accept        # accept review
POST   /v1/work-items/{id}/reject        # reject review
DELETE /v1/work-items/{id}               # cancel
```

These are for external integrations (dashboards, human review UIs,
CI/CD pipelines). They are not required for agent-to-agent
coordination, which happens through the LLM tools.

## Relationship to Platform Layers

### kagenti

kagenti owns fleet scheduling, agent identity, and trust management.
WorkItemStore provides the work pool; kagenti provides the worker
pool. The integration surface:

- kagenti's agent metadata includes capabilities that flow into the
  work item matching logic.
- kagenti's trust scores replace static `trust` capabilities in
  the dynamic trust model (future).
- kagenti's fleet scheduler could populate WorkItemStore on behalf
  of a higher-level orchestrator.

### OGX / LlamaStack

OGX could orchestrate work distribution at the platform level, with
WorkItemStore as the persistence backend. This is not v1 scope.

### fips-agents

fips-agents owns the WorkItemStore protocol, basic backends (null,
SQLite, Postgres), stock LLM tools, and the budget headroom
enforcement. It does not own fleet scheduling, trust management, or
cross-tenant coordination.

## Module Layout

```
fipsagents/server/
  work_items.py              # ABCs (WorkItemStore), models
                             # (WorkItem, Capability, HandoffNote, Attempt),
                             # stream events, create_work_item_store(),
                             # make_work_item_tools()
  work_item_stores/
    __init__.py
    null.py                  # NullWorkItemStore
    sqlite.py                # SqliteWorkItemStore (Phase 1)
    postgres.py              # PostgresWorkItemStore (Phase 2)
```

ABCs, models, and factory in `work_items.py`. Backend
implementations in `work_item_stores/` subdirectory to keep file
sizes small, following the `sources/` and `sinks/` pattern from
event-triggered mode.

## Open Questions

1. **Should work item decomposition be a framework feature or left
   to agent prompting?** A parent agent can create child work items
   with `parent_id` set, and the store tracks the hierarchy. But
   should the framework provide a `decompose_work_item` tool that
   calls the LLM to break a work item into sub-items? Lean toward
   leaving this to agent prompting -- decomposition strategy is
   domain-specific. The framework provides `parent_id` and hierarchy
   queries; the agent decides how to use them.

2. **How does cross-session state transfer work when an item is
   released and picked up by a different agent type?** The handoff
   note is the transfer mechanism. It is deliberately untyped beyond
   the `HandoffNote` structure -- `context` is free-form, `artifacts`
   is a string-to-string map. This is intentional: different agent
   types have different AgentState schemas, and the handoff note is
   the common ground. If stronger typing is needed, `artifacts` can
   reference files or database records with typed content.

3. **Should acceptance review be synchronous or async?** The store
   supports both. `complete(review_required=True)` is inherently
   async -- the item sits in `review_pending` until reviewed. But
   an agent could also call a subagent (#165) for synchronous
   in-process review before calling `complete()`. Lean toward
   defaulting to async (set `review_required=True`) and letting
   developers choose.

4. **What is the right default lease duration?** 300 seconds (5
   minutes) is a reasonable default for tool-calling agents that
   process structured data. Agents doing complex multi-step work
   should override per-item or per-checkout. The config knob exists
   at three levels (global, item, checkout call) specifically because
   there is no one right answer.

5. **How do we handle work item DAG ordering?** `depends_on` on the
   work item model enables DAG relationships. `list_available()`
   filters out items with incomplete dependencies (status = `blocked`
   evaluated lazily). This covers linear dependencies and fan-in.
   Fan-out (one item produces N children) is handled by the parent
   agent creating children with `parent_id`. Cycle detection is the
   creator's responsibility (the store does not enforce it in v1).

6. **Should agents bid on work items (auction model) vs.
   first-come-first-served checkout?** v1 ships first-come-first-
   served. Auction models (agents submit bids, a scheduler picks the
   best match) are a kagenti fleet scheduling concern. The
   WorkItemStore protocol does not preclude auctions -- a
   `KagentiWorkItemStore` could implement `checkout()` as "submit
   bid and wait for grant." But the SQLite and Postgres backends
   are FCFS.

## Out of Scope

- **Fleet scheduling.** Which agent instance gets which work item
  across a cluster is kagenti's concern. WorkItemStore handles the
  work pool; agent scheduling is separate.
- **Trust management lifecycle.** Earning, revoking, and auditing
  trust is kagenti's domain. fips-agents threads trust values
  through capability matching but does not compute them.
- **Cross-tenant work pools.** Multi-tenancy is an identity and
  authorization concern, not a work item store concern.
- **Semantic conflict resolution.** Detecting that two completed
  work items produced contradictory results is a domain-specific
  problem. The framework provides handoff notes and artifact
  references as inputs to conflict detection, but does not implement
  detection logic.
- **Work item templates / blueprints.** Pre-defined work item
  shapes (e.g., "code review" with standard acceptance criteria)
  are useful but belong in agent-specific tooling, not the
  framework.
- **UI for work item management.** Dashboards and human review
  interfaces are `fips-agents/ui-template` concerns. The REST
  endpoints provide the API surface.

## Dependency Graph

```
#182 (session state)    ─┐
#188 (event-triggered)  ─┼─► work item coordination
#165 (subagent-as-tool) ─┤      │
#195 (per-turn limits)  ─┤      ├──► acceptance agent pattern (agent-level)
#167 (doom-loop)        ─┘      ├──► kagenti fleet scheduling (platform-level)
                                └──► GitHubWorkItemStore (Phase 3)
```

All prerequisites are shipped. Work item coordination is additive
and backward-compatible -- agents that do not enable it see no
behavior change.

## Related

- Anthropic, "Effective Harnesses for Long-Running Agents" -- the
  progress file + feature list pattern that motivates multi-agent
  coordination.
- Issue #188 -- event-triggered mode; CronSource enables ambient
  agent workers.
- Issue #165 -- subagent-as-tool; enables in-process review and
  delegation patterns.
- Issue #164 -- permission policy; orthogonal to capability matching.
- Issue #190 -- reducer-based state recovery; handoff artifacts can
  be surfaced through AgentState.
- `docs/responsibilities.md` -- layer boundaries; fleet scheduling
  and trust management are platform concerns.
