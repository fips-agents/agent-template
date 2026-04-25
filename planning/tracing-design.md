# Tracing and Observability Design (#81)

## Problem

Agents produce no durable trace of what happened during a request —
model calls, tool executions, decisions, timings, token costs. The
StreamMetrics data is computed but only surfaces in the HTTP response
and is never persisted. When something goes wrong, the only record is
whatever the log level happened to capture.

## Design principle

Same as session persistence: when an observability stack is available
(OTEL, LlamaStack telemetry), emit standard traces. When it's not,
provide a lightweight internal trace store (SQLite) that gives
post-hoc debugging without any infrastructure. Zero-dependency
internal mode — no OTEL SDK required for the fallback.

## Trace model

A trace represents one request through the agent. It contains spans,
which represent operations within that request.

```
Trace (one per request)
├── request span       (HTTP layer, server module)
│   ├── step span      (one per agent step/iteration)
│   │   ├── model_call span   (LLM client call)
│   │   ├── tool span         (tool execution, if any)
│   │   ├── tool span         (parallel tool, if any)
│   │   └── memory span       (deferred memory injection, if first step)
│   ├── step span
│   │   └── model_call span
│   └── ...
└── session_load / session_save spans (if session persistence enabled)
```

### Span fields

```python
@dataclass
class Span:
    trace_id: str           # shared across all spans in the request
    span_id: str            # unique to this span
    parent_span_id: str | None
    name: str               # e.g. "model_call", "tool:search", "step:1"
    start_time: float       # time.monotonic()
    end_time: float | None
    status: str             # "ok" | "error"
    attributes: dict        # operation-specific data
    events: list[dict]      # timestamped events within the span
```

### Span attributes by type

**request**: method, path, session_id, model, stream (bool)

**step**: iteration number, finish_reason

**model_call**: model name, prompt_tokens, completion_tokens,
  time_to_first_token, temperature, tool_count

**tool**: tool_name, call_id, visibility, duration_ms, is_error,
  inspection_findings (from ToolInspector)

**memory**: backend, result_count, query (truncated)

## Architecture

```
 BaseAgent                           Server
 ─────────                           ──────
 astep_stream() ─── emits ──►  StreamEvent
       │                            │
       │                     TraceCollector
       │                       (listens)
       │                            │
       ▼                            ▼
 StreamMetrics              Span accumulation
 (already computed)         (wraps around events)
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                    NullStore   SqliteStore   OTELExporter
                   (logging)   (edge/dev)    (enterprise)
```

### Where tracing hooks go

**BaseAgent (framework)**:
- Does NOT import or know about tracing directly
- Already emits StreamEvents with timing data
- Already computes StreamMetrics
- Already logs tool executions with call_id
- No changes needed to BaseAgent for tracing — the data is all there

**Server module (HTTP layer)**:
- The TraceCollector wraps around `astep_stream()` and observes the
  events as they flow through
- Creates the request span, step spans, and child spans by watching
  for ContentDelta (model responding), ToolCallDelta (tool starting),
  ToolResultEvent (tool finished), StreamComplete (step done)
- This is the same observer pattern the SSE serializer uses — it
  consumes StreamEvents without modifying them

**TraceCollector** (new module):
- Stateful observer that builds spans from StreamEvents
- Plugs in between `astep_stream()` and the SSE serializer
- Async — doesn't block the response stream

## TraceCollector interface

```python
class TraceCollector:
    """Observes StreamEvents and builds trace spans."""

    def __init__(self, store: TraceStore, trace_id: str):
        self.store = store
        self.trace_id = trace_id

    def begin_request(self, attributes: dict) -> None:
        """Open the root request span."""

    async def observe(
        self, events: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[StreamEvent]:
        """Wrap an event stream, building spans as events flow.

        Yields events unchanged — this is a pure observer.
        """

    async def end_request(self) -> None:
        """Close the root span and persist the trace."""
```

Usage in the server:

```python
collector = TraceCollector(store, trace_id=request_id)
collector.begin_request({"model": model_name, "stream": True})
events = agent.astep_stream(**overrides)
observed = collector.observe(events)  # wraps, does not consume
async for sse_chunk in stream_events_as_sse(observed, model_name):
    yield sse_chunk
await collector.end_request()
```

## Storage interface

```python
class TraceStore(ABC):
    """Pluggable trace persistence backend."""

    @abstractmethod
    async def save_trace(self, trace: Trace) -> None:
        """Persist a completed trace."""

    @abstractmethod
    async def get_trace(self, trace_id: str) -> Trace | None:
        """Retrieve a trace by ID."""

    @abstractmethod
    async def list_traces(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[TraceSummary]:
        """List recent traces (summary only, not full spans)."""

    @abstractmethod
    async def delete_traces_before(self, cutoff: datetime) -> int:
        """Housekeeping: remove old traces. Return count deleted."""
```

## Backends

### NullTraceStore (default)

No persistence. Traces are logged to `fipsagents.tracing` at DEBUG
level as structured JSON, then discarded. This gives structured
log output that a log collector (fluentd, vector) can pick up
without any storage overhead.

### SqliteTraceStore

SQLite-backed trace storage for edge/dev. Schema:

```sql
CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    model       TEXT,
    session_id  TEXT,
    status      TEXT NOT NULL DEFAULT 'ok',
    spans       TEXT NOT NULL,  -- JSON array of Span objects
    summary     TEXT            -- JSON: token counts, duration, tool/step counts
);
CREATE INDEX idx_traces_started ON traces (started_at);
```

If session persistence (#78) is also using SQLite, share the same
database file via a config option. Different tables, same connection.

### OTELExporter

Translates our Span model to OpenTelemetry spans and exports via
the standard OTLP exporter. Only instantiated when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set (or `mode: external`).

The `opentelemetry-sdk` and `opentelemetry-exporter-otlp` packages
are optional dependencies — never imported unless this backend is
selected.

Mapping: our Span → OTEL Span is straightforward since our model
is intentionally OTEL-shaped (trace_id, span_id, parent, attributes,
events).

## Query endpoints

When trace storage is enabled (not null), the server exposes:

```
GET /v1/traces                          → list recent traces
GET /v1/traces/{trace_id}               → full trace with all spans
GET /v1/traces/{trace_id}/spans         → spans only (lighter)
```

These are debug/admin endpoints, not part of the OpenAI-compat API.

## Configuration

```yaml
# agent.yaml
observability:
  mode: auto            # auto | internal | external | off
  trace_storage: null   # null | sqlite | postgres
  sqlite_path: "./traces.db"
  max_age_hours: 168    # auto-delete traces older than 7 days
  sampling_rate: 1.0    # 1.0 = trace every request (reduce for high-traffic)
```

Auto-detection logic:
- If `OTEL_EXPORTER_OTLP_ENDPOINT` is set → `external` (OTEL export)
- Else if `trace_storage` is configured → `internal` (local store)
- Else → `off` (NullTraceStore, structured logging only)

## Relationship to existing instrumentation

### StreamMetrics

StreamMetrics is already computed in `astep_stream()`. The
TraceCollector should extract these and attach them to the
model_call span's attributes rather than recomputing. This avoids
duplicate timing logic.

### Audit logger

The `fipsagents.security.audit` logger captures tool inspection
findings. The TraceCollector should add these as events on the
tool span, linking security findings to the specific tool call
in the trace. The audit logger continues to emit independently —
tracing supplements it, doesn't replace it.

### Call IDs

Tool calls already have `call_id` fields. The TraceCollector uses
these as span IDs (or links to span IDs) so that tool spans
correlate with the tool_calls in the message history and the
audit log entries.

## What this does NOT cover

- **Prometheus metrics endpoint**: Separate concern. A `/metrics`
  endpoint with request counts, latencies, error rates is useful
  but orthogonal to tracing. Could share the same config section.
- **Distributed tracing across agents**: When agent A calls agent B
  via RemoteNode, propagating trace context (W3C Trace Context
  headers) is important but is a follow-on to single-agent tracing.
- **Dashboard or UI**: Traces are queryable via the REST endpoints.
  Visualization is left to Jaeger/Grafana (external) or a future
  debug UI.

## Implementation order

1. Span/Trace data model (dataclasses)
2. TraceCollector (observer wrapping astep_stream)
3. NullTraceStore (structured logging, default)
4. Server integration (wrap astep_stream, add /v1/traces endpoints)
5. SqliteTraceStore
6. OTELExporter (optional dependency)
7. Housekeeping (max_age_hours cleanup)

## Shared storage with session persistence (#78)

If both features use SQLite, they should share the database file:

```yaml
server:
  storage:
    backend: sqlite           # shared backend
    sqlite_path: "./agent.db" # single file
  sessions:
    enabled: true
  traces:
    enabled: true
```

This collapses the per-feature backend config into a shared
storage layer. The `SessionStore` and `TraceStore` implementations
receive the same connection/path and use different tables.

This also suggests a refinement to the session persistence design:
instead of `sessions.backend`, use a shared `server.storage.backend`
with feature-level enable/disable.
