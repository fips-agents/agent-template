# Session State & Compaction Foundation Design (#182)

## Problem

Issues #163 (Question tool), #164 (per-tool permission policy), #166
(auto-compaction), and #168 (session fork) each touch session state.
Without a shared foundation, each would define its own columns,
pending-state semantics, and event types — producing incompatible
schemas and duplicated compaction-safety logic.

## Design principle

Land one additive, backward-compatible schema migration and a minimal
set of ABCs. Downstream issues consume these contracts without
re-specifying. Existing deployments see zero behavior change.

## Message IDs

Every entry in `BaseAgent.messages` carries a stable `id` field:

    msg_{unix_ms_hex:012}_{random_hex:12}

- Sortable by creation time.
- No external dependency (no python-ulid).
- Set at construction via `_stamp_message_id()`; never changes.
- Messages loaded from pre-#182 sessions are backfilled on load.
- The `id` is an extension field — OpenAI-compatible clients ignore it.

## Session schema

Seven new columns on the sessions table, all with null/empty defaults:

| Column | Type | Default | Consumer |
|--------|------|---------|----------|
| `parent_session_id` | TEXT | NULL | #168 (fork lineage) |
| `forked_at_message_id` | TEXT | NULL | #168 |
| `pending_question` | TEXT/JSON | NULL | #163 (Question tool) |
| `open_tool_calls` | JSON array | `[]` | #166 (compaction guard) |
| `pending_subagent_calls` | JSON array | `[]` | #166 |
| `permission_scope_active` | TEXT | NULL | #164 (permission scope) |
| `compaction_state` | JSON object | `{}` | #166 (compaction cursor) |

### Access methods

Two new methods on `SessionStore` (default no-op for backward compat):

- `update_state(session_id, **fields)` — partial update of state columns.
  Allowed fields: the seven above. Unknown fields silently ignored.
- `get_state(session_id)` — returns all state columns as a dict.

`create()` extended with optional `parent_session_id`,
`forked_at_message_id`, `permission_scope_active` keyword args.

### Migration strategy

Additive `ALTER TABLE ADD COLUMN` statements, idempotent (SQLite checks
`PRAGMA table_info`, Postgres uses `IF NOT EXISTS`). Existing sessions
load with null/empty defaults. `NullSessionStore` inherits the default
no-op implementations.

## Compactor ABC

Module: `fipsagents.server.compactor`

```python
class Compactor(ABC):
    async def should_compact(messages, *, state=None) -> bool
    async def compact(messages, *, state=None) -> CompactionResult
    async def close() -> None  # default no-op
```

`should_compact` is separate from `compact` so the server can skip
compaction when pending state exists without invoking the (potentially
expensive) compactor.

`CompactionResult` carries: `messages`, `original_count`,
`compacted_count`, `skipped`, `skip_reason`.

`CompactionState` tracks: `last_compacted_at`,
`last_compacted_message_id`, `compaction_count`.

`NullCompactor` (default): always returns False / passthrough.
`create_compactor(backend)` factory.

The LLM-driven compactor implementation comes in #166.

## PermissionSource ABC

Module: `fipsagents.server.permissions`

```python
class PermissionSource(ABC):
    async def resolve(tool_name, *, scope=None, context=None) -> PermissionDecision
    async def close() -> None  # default no-op
```

`PermissionDecision` carries: `action` (allow/deny/ask), `tool`,
`rule_id`, `scope`, `reason`.

`PermissionRule` carries: `id`, `tool` (name or `*`), `action`, `scope`.

Implementations:
- `NullPermissionSource` — allow all (default, backward-compatible).
- `StaticPermissionSource` — first-match-wins from config rules,
  wildcard support (`*`), scope filtering, configurable default action.
- `KagentiPermissionSource`, `OGXPermissionSource` — deferred.

`create_permission_source(backend, *, rules, default_action)` factory.

## Stream events

Six new `@dataclass` event types added to `StreamEvent` union:

| Event | Fields | Consumer |
|-------|--------|----------|
| `CompactionStarted` | session_id, message_count | #166 |
| `CompactionCompleted` | session_id, original_count, compacted_count | #166 |
| `CompactionSkipped` | reason, session_id | #166 |
| `PermissionDecisionMade` | tool, action, rule_id, scope | #164 |
| `QuestionAsked` | question_id, question_text, session_id | #163 |
| `QuestionAnswered` | question_id, answer_text, session_id | #163 |

SSE serialization under namespaced delta keys:
`compaction.{type}`, `permission.{tool,action,...}`,
`question.{type}`.

## Server wiring

### Pending-question guard

If a session has a non-null `pending_question` and the incoming request
does not include `answers_to_question_id`, the server returns HTTP 409
with the question state. This prevents concurrent requests from
clobbering the question flow.

`answers_to_question_id` field added to `ChatCompletionRequest`.

### Compaction pending-state guard

Before invoking compaction, the server checks `open_tool_calls`,
`pending_subagent_calls`, and `pending_question`. If any are non-empty,
compaction is skipped with `CompactionSkipped(reason="pending_state")`.

### Permission scope on session creation

`permission_scope_active` is set from `ServerConfig.permissions.source`
when a session is created. Downstream #164 uses this to resolve
permissions per-session.

## Configuration

```yaml
server:
  compaction:
    enabled: false
    backend: null  # null | llm
    threshold_messages: 50
  permissions:
    source: null   # null | static
    default_action: allow
    rules: []
```

## Downstream dependency graph

```
#182 (this issue — Phase 0)
  │
  ├──► #163 (Question tool — Phase 1)
  ├──► #164 (Permission policy — Phase 2, parallel with #163)
  ├──► #166 (Auto-compaction — Phase 3)
  └──► #168 (Session fork — Phase 4)
```

## Out of scope

- Question tool implementation (#163)
- Permission rule grammar (wildcards, args_match, named scopes) (#164)
- Compaction summarization prompt and LLMSummarizer (#166)
- Session fork API endpoints and re-expansion logic (#168)
- KagentiPermissionSource, OGXPermissionSource
