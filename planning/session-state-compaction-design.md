# Session State, Compaction, and Pluggable Permissions Design

Foundation design for issues #163 (Question tool), #164 (per-tool permission policy), #166 (auto-compaction), and #168 (session fork). These four issues share a single contract surface: the shape of the session row, the lifecycle of pending state, and the layering of compaction and permission resolution. This document pins the contract once so each downstream issue can reference it instead of re-spec'ing locally.

## Status

Authoritative foundation design. Per-feature design (the LLM-facing tool surface, the permission rule grammar, the compaction summarization prompt) lives in each issue's body, not here. This doc covers only what is shared.

## Problem

We are about to ship four features that all touch session state. None of them currently spec the schema. If we land them in any order without a shared foundation, we will:

- Re-do the session schema two or three times as each feature discovers what it needs.
- Build incompatible pause/resume semantics — the Question tool's "operator answers later" path and the Permission policy's "ask before tool runs" path are structurally identical and must share plumbing.
- Lose tool-call/tool-result pairing on compaction (a documented failure mode in LangChain and n8n) because compaction has no way to know which tool calls are still in flight.
- Get session fork wrong because the originals would already be summarized away, with no path back to the unsummarized history.

## Constraints inherited from existing architecture

- **BaseAgent stays unaware of persistence.** Sessions, traces, metrics, and now compaction and permission resolution are all server-layer concerns. BaseAgent works on `self.messages` and emits `StreamEvent`s; the server wraps that with persistence and observation. This layering is not negotiable.
- **Stores are pluggable ABCs with `null` defaults.** `SessionStore`, `TraceStore`, `MetricsCollector`, and `ChunkStore` all follow this pattern. Compaction and permission sourcing follow it too.
- **The framework targets multiple deployment surfaces.** "Regular" Red Hat OpenShift AI (vanilla vLLM, no agent-aware identity service), Kagenti-managed deployments (per-tenant identity and policy), and OGX/LlamaStack-fronted deployments (server-side shields and tool orchestration). Where permissions come from differs across these; the framework cannot assume one source.
- **Compaction is client-side.** Anthropic's server-side `compact_2026_01_12` extension exists, but vLLM, llama.cpp, Bedrock, Vertex, and OpenAI-compatible endpoints generally do not. The framework owns compaction.

## Architecture overview

```
┌──────────────────────────────────────────────────────────────┐
│ OpenAIChatServer                                             │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ SessionStore │   │  Compactor   │   │ PermissionSource │  │
│  │  (existing)  │   │     (new)    │   │      (new)       │  │
│  └──────┬───────┘   └──────┬───────┘   └─────────┬────────┘  │
│         │                  │                     │           │
│         └──── reads/writes ─session row ─────────┘           │
│                            │                                 │
│                  ┌─────────▼─────────┐                       │
│                  │     TraceStore    │ ◄── audit log,         │
│                  │     (existing)    │     pre-compaction     │
│                  └───────────────────┘     history            │
└──────────────────────────────────────────────────────────────┘
                          │
                  loads/saves around
                          │
┌──────────────────────────▼──────────────────────────────────┐
│ BaseAgent                                                   │
│  - self.messages (with stable IDs, see §4.1)                │
│  - StreamEvents: existing + Question* + Compaction* + ...   │
│  - No knowledge of compaction, sessions, or permissions     │
└─────────────────────────────────────────────────────────────┘
```

Three new ABCs sit alongside the existing ones. The session row grows new columns. BaseAgent's only required change is stable message IDs; everything else is server-layer.

## Schema commitments

These five items are the foundation. They are cheap to add now and expensive to retrofit. Land them before any of #163, #164, #166, or #168.

### 4.1 Stable message IDs

Every entry in `self.messages` carries a stable `id` (ULID). The ID is assigned at message construction time and never changes. Required for:

- Session fork lineage (`forked_at_message_id`).
- Compaction's `replaces_message_ids` list inside the compaction marker.
- The Question tool's `answers_to_question_id` resume parameter.

ULIDs sort lexicographically by creation time, which makes "most recent N messages" trivial without an extra timestamp column.

### 4.2 Session row schema additions

The current `SessionRecord` (in `fipsagents.server.sessions`) gains:

| Column | Type | Purpose |
|---|---|---|
| `parent_session_id` | `SessionId \| None` | Fork lineage (#168). Null for non-forked sessions. |
| `forked_at_message_id` | `MessageId \| None` | Where in the parent the fork happened. Null for non-forked. |
| `pending_question` | `QuestionState \| None` | Set when the agent has emitted a Question and is waiting (#163). |
| `open_tool_calls` | `list[ToolUseId]` | Tool calls dispatched but not yet resolved. Default `[]`. |
| `pending_subagent_calls` | `list[SubagentCallId]` | Subagent invocations in flight. Default `[]`. |
| `permission_scope_active` | `str \| None` | Named permission scope this session runs under (#164). |
| `compaction_state` | `CompactionState` | Tracks whether compaction has run and its marker IDs. |

`messages` continues to hold the active (post-compaction) view that the LLM sees on resume. The trace store holds the pre-compaction history.

### 4.3 Pending-state semantics

`pending_question`, `open_tool_calls`, and `pending_subagent_calls` together define whether a session is in a "safe to operate on" state. The contract:

- A session with any non-empty pending state **cannot be compacted** until it is resolved or explicitly carried forward. Compactor implementations check this and emit `CompactionSkipped(reason="pending_state")` if violated.
- A session with `pending_question` set **cannot accept a new chat-completion request** that does not include `answers_to_question_id`. The server returns an HTTP 409 with the question state.
- A session with non-empty `open_tool_calls` after a turn ends is a bug — surface it as a structured error, do not silently drop the calls.
- Forking a session with pending state copies the pending state to the fork by default. Operators can opt to clear it via a fork parameter.

### 4.4 Fork lineage

Forks are a row insert, not a copy. The fork's `messages` is initialized to the parent's `messages` truncated at `forked_at_message_id`. Re-expansion to the unsummarized pre-compaction history walks the trace store from the parent up to `forked_at_message_id`.

### 4.5 Storage backends

The new columns apply to both `SqliteSessionStore` and `PostgresSessionStore`. Migrations are additive — existing sessions get null/empty defaults and continue to work. `NullSessionStore` ignores the new columns (it is ephemeral by design).

## Compactor ABC

```python
class Compactor(ABC):
    @abstractmethod
    async def should_compact(self, session: SessionRecord) -> bool: ...

    @abstractmethod
    async def compact(self, session: SessionRecord) -> CompactionPlan:
        """Return a plan; the server applies it under a session-row lock."""
```

Two stock implementations:

- `NullCompactor` — `should_compact` always returns False. Default. No behavior change vs. current main.
- `LLMSummarizer` — token-threshold trigger (configurable, default 80% of model's effective context window). Calls a configured LLM (defaults to the agent's primary model, override-able to a smaller cheap model) with a system prompt that produces the marker payload.

### 5.1 Compaction marker

The compactor produces a synthetic message inserted at the start of the active message list:

```python
class CompactionMarker(BaseModel):
    summary: str
    original_goal: str  # extracted from the first user message
    preserved_tool_call_ids: list[str]  # tool_use_ids re-emitted post-marker
    permission_scope_at_compaction: str | None
    compacted_at_token_count: int
    replaces_message_ids: list[MessageId]
    created_at: datetime
```

The marker is a `system` or `developer` role message (configurable, parallel to memory's `prefix_role`). Messages whose IDs appear in `replaces_message_ids` are dropped from `messages_active` but remain in the trace store.

### 5.2 Tool-call pairing

Compaction must not produce orphaned tool-use/tool-result pairs. The rule: if a `tool_use` block falls inside the compacted range but its corresponding `tool_result` falls outside (or vice versa), both are kept. The marker's `preserved_tool_call_ids` records which tool calls were preserved across the boundary.

### 5.3 Stream events

```python
class CompactionStarted(StreamEvent):
    trigger_token_count: int
    threshold: int

class CompactionCompleted(StreamEvent):
    summary_tokens: int
    messages_compacted: int
    messages_preserved: int

class CompactionSkipped(StreamEvent):
    reason: Literal["pending_state", "below_threshold", "disabled"]
    detail: str | None
```

`TraceCollector` observes these events as it does any other; no special handling.

### 5.4 Re-entrancy

Compaction holds a row-level session lock. A second compaction attempt on the same session blocks until the first finishes. Compaction triggered by a turn-in-progress is deferred to end-of-turn so it does not race with model output.

## PermissionSource ABC

```python
class PermissionSource(ABC):
    @abstractmethod
    async def resolve(
        self,
        scope: str,
        identity: Identity,
        tool: ToolDescriptor,
        args: dict,
    ) -> PermissionDecision: ...

@dataclass
class PermissionDecision:
    action: Literal["allow", "deny", "ask"]
    rule_id: str | None
    audit_metadata: dict
```

Three stock implementations, selected by `permissions.source` in `agent.yaml`:

- `StaticPermissionSource` — rules baked into `agent.yaml`. The default. Matches "regular RHOAI" deployments where there is no agent-aware identity/policy service.
- `KagentiPermissionSource` — resolves rules from Kagenti's policy service at session-start time, scoped to the authenticated identity and tenant.
- `OGXPermissionSource` — defers to LlamaStack shields. The agent-side decision is `allow` with a hint that the shield will enforce server-side; the framework does not duplicate the policy check.

The `permission_scope_active` column on the session row records the scope name. The active source resolves rules under that scope at decision time. Rule caching is per-session (resolved once at session start, refreshable on demand) to keep tool-call latency bounded.

`agent.yaml`:

```yaml
permissions:
  mode: enforce            # enforce | observe
  source: static           # static | kagenti | ogx
  default_scope: research-readonly
  rules:                   # only consumed when source: static
    - tool: "kubectl_*"
      args_match: { verb: ["delete", "drain", "cordon"] }
      action: ask
    - tool: "*"
      action: ask
```

When `source: kagenti`, the `rules:` block is ignored and a warning is logged at startup. When `source: ogx`, the framework still consults the source for `deny` rules at the agent boundary (defense in depth) but defers everything else to the shield.

The `ask` action's operator-prompt path goes through the Question tool's plumbing (#163). This is why #163 is a hard dependency on `ask` and why the foundation must land before #164.

## Cross-cutting: stream events

New `StreamEvent` variants introduced by the foundation:

- `QuestionAsked(question_id, prompt, options, ...)` — see #163 for shape.
- `QuestionAnswered(question_id, selected, custom_text)` — see #163.
- `CompactionStarted` / `CompactionCompleted` / `CompactionSkipped` — see §5.3.
- `PermissionDecisionMade(tool, action, rule_id, scope)` — observability, not a control event. Logged to `fipsagents.security.audit.permissions`.

Existing `StreamEvent` consumers continue to work; new events are additive.

## How each downstream issue consumes the foundation

### #163 — Question tool

- Sets `pending_question` on the session row when `ask_user` is invoked.
- Server emits `QuestionAsked` over SSE and ends the stream.
- Resume request includes `answers_to_question_id`; server clears `pending_question` and resumes the agent loop with a synthetic tool-result message.
- The question's `id` is the message ID of the `ask_user` invocation.
- Per-feature design lives in #163.

### #164 — Permission policy

- `PermissionSource` resolves the action for every tool call before dispatch.
- `allow` and `deny` are synchronous.
- `ask` invokes the Question tool's plumbing — sets `pending_question` with a tool-confirmation shape, ends the stream, resume on answer.
- `permission_scope_active` is recorded at session creation (or on first tool call if not specified).
- Per-feature design lives in #164.

### #166 — Auto-compaction

- `LLMSummarizer` is the stock `Compactor` implementation.
- Trigger logic is part of the Compactor, not BaseAgent.
- Refuses to run while pending state is set.
- Emits compaction stream events.
- Per-feature design lives in #166.

### #168 — Session fork

- Fork creates a new session row with `parent_session_id` and `forked_at_message_id` set.
- Active messages are copied; pending state copies by default.
- Re-expansion to pre-compaction history reads from the trace store using the parent's session ID.
- Per-feature design lives in #168.

## Phased rollout

- **Phase 0 (this design + tracker issue).** Stable message IDs, session schema additions, `Compactor` ABC + `NullCompactor`, `PermissionSource` ABC + `StaticPermissionSource` skeleton, compaction stream events. Backward-compatible — all new columns default to safe values.
- **Phase 1 — #163.** Question tool consumes `pending_question` lifecycle.
- **Phase 2 — #164.** Permission policy consumes `PermissionSource` and Question plumbing for `ask`. Unblocks `permission_scope` enforcement in the subagent path (the v1 scope cut from #173).
- **Phase 3 — #166.** `LLMSummarizer` Compactor implementation.
- **Phase 4 — #168.** Session fork using `parent_session_id` + `forked_at_message_id`.

Phases 1 and 2 can land in parallel after Phase 0; Phase 3 depends on Phase 0 only; Phase 4 is independent of Phases 1–3 except for needing stable message IDs.

## Open questions

1. **Compactor model selection.** Default to the agent's primary model, or always use a smaller summarizer (Haiku-class)? Larger model = better summary fidelity but higher latency on every overflow. Smaller model = predictable cost but lossier. Likely configurable per agent; default to a smaller model.
2. **PermissionSource caching window.** Session-start resolution is the proposal. Should there be a max staleness (re-resolve on long sessions)? Probably yes, configurable, default 30 minutes for `KagentiPermissionSource`.
3. **`pending_question` parallelism.** Default proposal: one open question per session. Worth confirming under realistic UI flows (e.g. a parallel tool call also triggering `ask`).
4. **Trace store retention vs. fork re-expansion.** Re-expanding a fork requires the trace store to still hold the pre-compaction history. Trace stores have a configurable retention policy (`max_age_hours`). If a fork is created from a session whose traces have been pruned, re-expansion fails — what's the failure shape? Options: best-effort partial re-expansion, hard error, or pin the trace lifetime to any session that forks from it.

## Out of scope for this design

- The summarization prompt itself for `LLMSummarizer`. That's per-feature design in #166.
- The full permission rule grammar (wildcards, `args_match` patterns). That's per-feature design in #164.
- The Question tool's argument schema and UI rendering. The framework spec lives in #163; rendering is the UI-template repo's problem.
- Cross-tenant routing and identity itself. Identity remains Kagenti's domain; this design only consumes whatever identity the runtime provides.
- A subagent registry resolver. Kagenti-discovered subagents are tracked in #180.

## Sources

- [Anthropic Compaction (API docs)](https://platform.claude.com/docs/en/build-with-claude/compaction) — `compact_2026_01_12`, `pause_after_compaction`, `trigger.input_tokens` mechanism.
- [Anthropic Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) — `clear_tool_uses_20250919`, tool-use/result pairing rules.
- [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — original-goal preservation, compaction loss profiles.
- [LangGraph add-memory](https://docs.langchain.com/oss/python/langgraph/add-memory) — `SummarizationNode`, `RunningSummary`, checkpointer-as-fork-lineage pattern.
- [LangChain `trim_messages`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html) — `start_on`/`end_on`, ToolMessage pairing constraint.
- [OpenAI Runs API — `truncation_strategy`](https://platform.openai.com/docs/api-reference/runs/createRun) — auto vs last_messages truncation, drop-middle behavior.
- `planning/subagent-tool-design.md` — preceding design doc this one mirrors in shape.
