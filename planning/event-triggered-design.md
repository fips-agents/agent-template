# Event-Triggered Mode Design (#188)

## Problem

The fipsagents framework only supports chat-triggered agents. Every
interaction enters through `POST /v1/chat/completions`. This breaks
down for reactive automation (S3 file arrival, GitHub PR, PagerDuty
alert), pipeline integration (Kafka topic, Redis Stream), and
scheduled work (daily compliance check, weekly report). In each case
the deployer reimplements session management, cost tracking,
compaction, and observability outside the framework. Event-triggered
mode brings those concerns inside.

## Design Principles

1. **EventSource is purely additive.** Server-layer only. Sources
   produce synthetic messages that enter via `astep_stream()`.
   BaseAgent contract unchanged.
2. **One session per source instance.** `source_id` becomes the
   session key. Sources can override per-event via `session_key`.
3. **Dual-mode.** Event sources run alongside the chat endpoint,
   serialised through `_agent_lock`.
4. **Source-native backpressure.** No framework queuing. Kafka
   consumer lag, Redis `XPENDING`, webhook HTTP 429 are the signals.
5. **Safety features are non-optional.** Compaction, cost ceiling,
   and doom-loop checks apply to every event processing run.

## EventSource ABC

Module: `fipsagents.server.events`

```python
class EventSource(ABC):
    source_id: str

    async def setup(self) -> None: ...

    @abstractmethod
    async def consume(self) -> AsyncIterator[InboundEvent]:
        """Yield events as they arrive.

        Implementers must use ``async def`` with ``yield``.
        Must be cancellation-safe -- the server cancels the
        consuming task on shutdown.
        """
        ...

    async def acknowledge(self, event_id: str) -> None: ...  # default no-op

    async def close(self) -> None: ...
```

`consume()` is an async generator -- subclasses implement it with
`yield`. The server cancels it on shutdown, so implementations must
be cancellation-safe. `acknowledge()` commits offsets (Kafka) or
XACKs (Redis); fire-and-forget sources leave the default no-op.

`NullEventSource` (never yields, returns immediately) is provided
for testing and as a documentation example.

## InboundEvent Envelope

```python
class InboundEvent(BaseModel):
    event_id: str
    event_type: str                          # "github.pull_request.opened"
    payload: dict[str, Any]
    source: str                              # matches EventSource.source_id
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_key: str | None = None           # override source_id session
```

## EventSink ABC

```python
class OutboundEvent(BaseModel):
    correlation_id: str                      # event_id of the InboundEvent that triggered this
    event_type: str                          # "response" | "processing_failed"
    payload: dict[str, Any]
    source: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

class EventSink(ABC):
    async def setup(self) -> None: ...
    @abstractmethod
    async def emit(self, event: OutboundEvent) -> None: ...
    async def close(self) -> None: ...
```

Implementations: `NullSink` (default, discard), `LogSink` (structured
JSON), `HttpCallbackSink` (POST to URL), `KafkaSink` (1b),
`RedisStreamSink` (1b).

## Transport Implementations

### Phase 1a (no new dependencies)

**HttpWebhookSource.** Registers a `POST {path}` route on the
existing FastAPI app during `setup()`. Incoming requests are
HMAC-verified (if `secret` configured), wrapped in `InboundEvent`,
and placed on an `asyncio.Queue`. `consume()` reads the queue.
The HTTP handler returns 202 immediately.

Signature header name is configurable (default:
`X-Hub-Signature-256`). `hmac.compare_digest()` prevents timing
attacks. Missing/invalid signatures receive 401.

**CronSource.** `consume()` computes the next fire time from a
5-field cron expression, sleeps via `asyncio.sleep()`, and yields
an `InboundEvent`. Minimal cron parser, stdlib only.

### Phase 1b (optional extras, implemented)

**KafkaSource** -- `fipsagents[kafka]` extra (`aiokafka`). Wraps
`AIOKafkaConsumer` with consumer group semantics. Auto-commit is
disabled; `acknowledge()` commits offsets after successful event
processing. Consumer group membership handles load balancing across
replicas. SASL/SSL configuration via `security_protocol`, `sasl_mechanism`,
`sasl_username`, `sasl_password`.

**RedisStreamSource** -- `fipsagents[redis]` extra (`redis[hiredis]`).
Wraps `XREADGROUP` for Redis Streams. Consumer groups are auto-created
with `MKSTREAM` on setup. `acknowledge()` calls `XACK` to remove
messages from the pending-entries list. Configurable block time
controls polling latency. Pending entries via `XPENDING` provide
retry visibility.

## Server Integration

The server starts one `asyncio.Task` per configured source during
lifespan startup. Each task runs a processing loop with retry:

```python
async def _event_loop(self, source: EventSource, sink: EventSink) -> None:
    async for event in source.consume():
        now = lambda: datetime.now(tz=UTC)
        retry = source.config.retry  # RetryConfig
        for attempt in range(retry.max_attempts):
            try:
                response = await self._process_event(event)
                await sink.emit(OutboundEvent(
                    correlation_id=event.event_id, event_type="response",
                    payload={"content": response}, source=event.source,
                    timestamp=now(),
                ))
                await source.acknowledge(event.event_id)
                break
            except tuple(retry.retriable_errors):
                if attempt + 1 < retry.max_attempts:
                    delay = min(retry.backoff_base ** attempt, retry.backoff_max)
                    await asyncio.sleep(delay)
                    continue
                await sink.emit(OutboundEvent(
                    correlation_id=event.event_id,
                    event_type="processing_failed",
                    payload={"error": traceback.format_exc()},
                    source=event.source, timestamp=now(),
                ))
                await source.acknowledge(event.event_id)
            except Exception:
                await sink.emit(OutboundEvent(
                    correlation_id=event.event_id,
                    event_type="processing_failed",
                    payload={"error": traceback.format_exc()},
                    source=event.source, timestamp=now(),
                ))
                await source.acknowledge(event.event_id)
                break
```

`acknowledge()` is only called after successful processing **or**
after all retries are exhausted / a non-retriable error occurs.

### How events enter `astep_stream()`

`_process_event` follows the same pattern as `_run_agent_sync`:

1. Call `translate_event(event)` to build messages (see below).
2. Resolve session: key is `event.session_key or source.source_id`.
   Load or create via session store (upsert semantics).
3. Run `_maybe_compact()` if threshold is reached.
4. Acquire `_agent_lock` (shared with chat completions).
5. Load session messages into `agent.messages`, append translated
   event messages, run `astep_stream()`.
6. Drain events, accumulate response from `ContentDelta`.
7. Save session, emit to sink, acknowledge event.

### Lock serialisation and starvation

Event processing and chat completions share `_agent_lock`. This
means a long-running event (multi-step tool use) blocks chat
requests, and vice versa. This is intentional for v1 — the agent
has one conversation history and interleaving would corrupt state.

**Starvation risk:** A cron source firing every minute or a webhook
burst can starve the chat endpoint. Mitigations, in order of
recommendation:

1. **Deploy event-only agents.** The recommended pattern for
   production: event-triggered agents run in separate pods without
   the chat endpoint exposed. This eliminates contention entirely.
2. **Priority queue (v2).** A future enhancement could use a
   priority asyncio queue where chat requests preempt pending
   events. Not in Phase 1a scope.
3. **Source-level rate limiting.** `max_events_per_second` (see
   Security section) bounds throughput per source. Default: 10/s
   for webhooks, 1/s for cron.

The design does not attempt concurrent event processing in v1.
Concurrent access to `agent.messages` would require a fundamentally
different session model.

### Lifecycle wiring

Event sources and sinks are set up after `agent.setup()` in the
lifespan, and closed (with task cancellation) before
`agent.shutdown()`. New instance attributes on `OpenAIChatServer`:

```python
self._event_sources: list[EventSource] = []
self._event_sink: EventSink | None = None
self._event_tasks: list[asyncio.Task] = []
```

## Session Model

| Source type | Default session key |
|-------------|-------------------|
| `webhook` | `event:{path}` |
| `cron` | `event:cron:{event_type}` |
| `kafka` | `event:kafka:{topic}:{consumer_group}` |
| `redis` | `event:redis:{stream}:{consumer_group}` |

The `event:` prefix distinguishes event sessions from chat sessions.
Session expiry is source-configurable via `session_ttl_hours`
(default: 168). Long-lived event sessions make compaction (#166)
practically required. Explicit `source_id` on any source config
overrides the default derivation.

## Interaction with Safety Features

**Compaction (#166).** Long-lived event sessions hit thresholds
naturally. `_maybe_compact()` handles this. The pending-state
guard (#182) prevents compaction during active tool calls. No
compactor changes needed.

**Cost ceiling (#195).** `max_tokens_per_turn` and
`max_iterations_per_turn` prevent any single event from burning
unbounded tokens. Per-session budget accumulates across events.

**Doom-loop detection (#167).** Breaks stuck tool loops from
malformed events. `LoopBreakEvent` is emitted, response packaged
as `processing_failed`, event acknowledged to prevent redelivery.

## Event Translation

`translate_event()` lives in `fipsagents.server.events`, **not** on
BaseAgent. BaseAgent must not import server-layer types (the
dependency flows server → baseagent, never the reverse).

```python
def default_translate_event(event: InboundEvent) -> list[dict[str, str]]:
    """Convert an inbound event into conversation messages.

    Default: system message with event context + user message
    with JSON payload.
    """
    return [
        {"role": "system", "content": (
            f"You received a {event.event_type!r} event from "
            f"{event.source!r} at {event.timestamp.isoformat()}. "
            f"Process this event and take appropriate action."
        )},
        {"role": "user", "content": json.dumps(
            event.payload, indent=2, default=str
        )},
    ]
```

Customisation: pass a `translate_fn` parameter to
`OpenAIChatServer` (or override `_translate_event()` on a server
subclass). Agent subclasses that need custom translation should
register a translate function during server construction, not
override a BaseAgent method.

## Configuration

```yaml
server:
  event_sources:
    - type: webhook
      source_id: github-prs          # optional; default: "event:{path}"
      path: /events/github
      secret: ${GITHUB_WEBHOOK_SECRET}
      event_type_header: X-GitHub-Event
      session_ttl_hours: 720
      max_events_per_second: 10       # token-bucket rate limit (default: 10)

    - type: kafka
      source_id: doc-ingest           # optional; default: "event:kafka:{topic}:{consumer_group}"
      bootstrap_servers: ${KAFKA_BOOTSTRAP}
      topic: document-ingestion
      consumer_group: my-agent
      security_protocol: SASL_SSL
      sasl_mechanism: PLAIN
      sasl_username: ${KAFKA_USER}
      sasl_password: ${KAFKA_PASSWORD}

    - type: redis
      source_id: task-queue           # optional; default: "event:redis:{stream}:{consumer_group}"
      redis_url: ${REDIS_URL}         # supports rediss:// for TLS
      stream: task-stream
      consumer_group: my-agent
      block_ms: 1000                  # polling timeout in milliseconds

    - type: cron
      schedule: "0 9 * * 1-5"         # 5-field POSIX cron (no @macros)
      event_type: daily-check
      max_events_per_second: 1        # default for cron: 1

  event_sink:
    type: kafka
    bootstrap_servers: ${KAFKA_BOOTSTRAP}
    topic: agent-responses
    # OR:
    # type: redis
    # redis_url: ${REDIS_URL}
    # stream: agent-responses
    # maxlen: 10000                     # optional, approximate trimming
    # OR:
    # type: http_callback
    # url: ${CALLBACK_URL}
```

**`source_id` derivation:** If `source_id` is omitted, it is derived
from the source type and identifying fields (see session model table
above). Explicit `source_id` overrides the default.

**`session_ttl_hours`** overrides the global `server.sessions.max_age_hours`
for sessions created by this source. If both are set, the source-level
value wins. If neither is set, sessions have no expiry.

**Cron expression subset:** 5-field POSIX cron (`minute hour day-of-month
month day-of-week`). Supports ranges (`1-5`), lists (`1,3,5`), steps
(`*/15`), and wildcards (`*`). No `@yearly`/`@reboot` macros, no
seconds field, no `L`/`W`/`#` extensions.

Parsed into `EventSourceConfig` (discriminated union on `type`)
and `EventSinkConfig` in `fipsagents.server.models`. Factory
functions `create_event_source(config)` and `create_event_sink(config)`
in `fipsagents.server.events` follow the same pattern as
`create_compactor()` and `create_permission_source()`.

## Error Handling

Configurable per-source retry policy:

```yaml
retry:
  max_attempts: 3
  backoff_base: 2.0
  backoff_max: 60.0
  retriable_errors: [TimeoutError, SubagentTimeoutError]
```

Non-retriable errors (validation, missing tools, budget exceeded)
skip retry and emit `processing_failed` to the sink immediately.

Dead-letter is the source transport's responsibility: Kafka DLQ
topics, Redis `XCLAIM`, webhook sender retries. Failed events
always produce a `processing_failed` `OutboundEvent` for audit.

## Security Considerations

- **Webhook HMAC-SHA256** per source, `hmac.compare_digest()`,
  configurable signature header.
- **Rate limiting** per source via `max_events_per_second` with
  token-bucket limiter. Defaults: 10/s for webhooks, 1/s for cron.
  Excess events buffered, not dropped. Disabled when set to `0`.
- **Payload validation** via optional `payload_schema` (JSON Schema).
  Invalid payloads: 400 for webhooks, `validation_failed` event
  for async sources.
- **Transport security**: Kafka `SASL_SSL`/mTLS, Redis TLS via
  `rediss://`, webhooks inherit server TLS (OpenShift Route).
- **No code execution from payloads.** Payloads serialised to JSON
  text, never `eval()`ed.

## Observability

Each event processing run produces its own trace via `TraceCollector`.
Trace structure mirrors chat completion requests with additional
`event_processing`, `translate`, and `sink_emit` spans.

Three new `StreamEvent` variants:

| Event | Fields |
|-------|--------|
| `EventReceived` | event_id, event_type, source |
| `EventProcessed` | event_id, source, duration_ms |
| `EventFailed` | event_id, source, error, retriable |

Three new Prometheus metrics (requires `[metrics]` extra):

| Metric | Type | Labels |
|--------|------|--------|
| `agent_events_received_total` | counter | source, event_type |
| `agent_events_processed_total` | counter | source, event_type, status |
| `agent_event_processing_duration_seconds` | histogram | source, event_type |

## Out of Scope

- Event-to-event composition between agents (use workflow + RemoteNode)
- Custom event infrastructure (implement EventSource ABC directly)
- Concurrent event processing (sequential via `_agent_lock` in v1)
- Cross-device session sync (session-store concern, not event concern)
- Event filtering/routing DSL (use Knative Eventing, EventBridge)

## Dependency Graph

```
#195 (cost ceiling)   ─┐
#167 (doom-loop)      ─┼─► #188 (event-triggered mode) ─► #189 (OTEL event log) ─► #190 (reducer recovery)
#166 (compaction)     ─┘
```

All prerequisites are shipped. Phase 1a (#188 base) and Phase 1b (Kafka, Redis) are both implemented.

## Module Layout

```
fipsagents/server/
  events.py             # ABCs (EventSource, EventSink), models
                        # (InboundEvent, OutboundEvent), stream events,
                        # default_translate_event(),
                        # create_event_source(), create_event_sink()
  sources/
    __init__.py
    null.py             # NullEventSource (never yields, for testing)
    webhook.py          # HttpWebhookSource (Phase 1a)
    cron.py             # CronSource + minimal 5-field POSIX parser (Phase 1a)
    kafka.py            # KafkaSource (Phase 1b)
    redis.py            # RedisStreamSource (Phase 1b)
  sinks/
    __init__.py
    null.py             # NullSink (Phase 1a)
    log.py              # LogSink (Phase 1a)
    http_callback.py    # HttpCallbackSink (Phase 1a)
    kafka.py            # KafkaSink (Phase 1b)
    redis.py            # RedisStreamSink (Phase 1b)
```

ABCs, models, and factories in `events.py`. Transports split into
subdirectories to keep files small and avoid pulling optional deps
at import time. Phase 1b (Kafka, Redis) imports only when their
respective extras are installed.
