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
    async def consume(self) -> AsyncIterator[InboundEvent]: ...

    async def acknowledge(self, event_id: str) -> None: ...  # default no-op

    async def teardown(self) -> None: ...
```

`consume()` must be cancellation-safe -- the server cancels it on
shutdown. `acknowledge()` commits offsets (Kafka) or XACKs (Redis);
fire-and-forget sources leave the default no-op.

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
    event_id: str                            # correlation from InboundEvent
    event_type: str                          # "response" | "processing_failed"
    payload: dict[str, Any]
    source: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

class EventSink(ABC):
    async def setup(self) -> None: ...
    @abstractmethod
    async def emit(self, event: OutboundEvent) -> None: ...
    async def teardown(self) -> None: ...
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

### Phase 1b (optional extras)

**KafkaSource** -- `fipsagents[kafka]` extra (`aiokafka`). Wraps
`AIOKafkaConsumer`. `acknowledge()` commits offsets. Consumer
group membership handles load balancing across replicas.

**RedisStreamSource** -- `fipsagents[redis]` extra (`redis[hiredis]`).
Wraps `XREADGROUP`. `acknowledge()` calls `XACK`. Pending entries
via `XPENDING` provide retry visibility.

## Server Integration

The server starts one `asyncio.Task` per configured source during
lifespan startup. Each task runs a processing loop:

```python
async def _event_loop(self, source: EventSource, sink: EventSink) -> None:
    async for event in source.consume():
        try:
            response = await self._process_event(event)
            await sink.emit(OutboundEvent(
                event_id=event.event_id, event_type="response",
                payload={"content": response}, source=event.source,
                timestamp=datetime.utcnow(),
            ))
            await source.acknowledge(event.event_id)
        except Exception:
            await sink.emit(OutboundEvent(
                event_id=event.event_id, event_type="processing_failed",
                payload={"error": traceback.format_exc()},
                source=event.source, timestamp=datetime.utcnow(),
            ))
            await source.acknowledge(event.event_id)
```

### How events enter `astep_stream()`

`_process_event` follows the same pattern as `_run_agent_sync`:

1. Call `agent.translate_event(event)` to build messages.
2. Resolve session: key is `event.session_key or source.source_id`.
   Load or create via session store (upsert semantics).
3. Run `_maybe_compact()` if threshold is reached.
4. Acquire `_agent_lock` (shared with chat completions).
5. Load session messages into `agent.messages`, append translated
   event messages, run `astep_stream()`.
6. Drain events, accumulate response from `ContentDelta`.
7. Save session, emit to sink, acknowledge event.

### Lifecycle wiring

Event sources and sinks are set up after `agent.setup()` in the
lifespan, and torn down (with task cancellation) before
`agent.shutdown()`. New instance attributes on `OpenAIChatServer`:

```python
self._event_sources: list[EventSource] = []
self._event_sink: EventSink | None = None
self._event_tasks: list[asyncio.Task] = []
```

## Session Model

| Source type | Default session key |
|------------|-------------------|
| `webhook` | `event:{path}` |
| `cron` | `event:cron:{event_type}` |
| `kafka` | `event:kafka:{topic}:{group}` |
| `redis` | `event:redis:{stream}:{group}` |

The `event:` prefix distinguishes event sessions from chat sessions.
Session expiry is source-configurable via `session_ttl_hours`
(default: 168). Long-lived event sessions make compaction (#166)
practically required.

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

## BaseAgent Change

One optional method -- the only BaseAgent change:

```python
def translate_event(self, event: InboundEvent) -> list[dict[str, str]]:
    """Convert an inbound event into conversation messages.

    Default: system message with event context + user message
    with JSON payload. Override to specialise.
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

## Configuration

```yaml
server:
  event_sources:
    - type: webhook
      path: /events/github
      secret: ${GITHUB_WEBHOOK_SECRET}
      event_type_header: X-GitHub-Event
      session_ttl_hours: 720

    - type: kafka
      bootstrap_servers: ${KAFKA_BOOTSTRAP}
      topic: document-ingestion
      consumer_group: my-agent

    - type: cron
      schedule: "0 9 * * 1-5"
      event_type: daily-check

  event_sink:
    type: http_callback
    url: ${CALLBACK_URL}
```

Parsed into `EventSourceConfig` (discriminated union on `type`)
and `EventSinkConfig` in `fipsagents.server.models`.

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
  token-bucket limiter. Excess events buffered, not dropped.
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

All three prerequisites are shipped. #188 is unblocked.

## Module Layout

```
fipsagents/server/
  events.py             # ABCs, models, stream events, factories
  sources/
    __init__.py
    webhook.py          # HttpWebhookSource
    cron.py             # CronSource
    kafka.py            # KafkaSource (1b)
    redis.py            # RedisStreamSource (1b)
  sinks/
    __init__.py
    null.py             # NullSink
    log.py              # LogSink
    http_callback.py    # HttpCallbackSink
    kafka.py            # KafkaSink (1b)
    redis.py            # RedisStreamSink (1b)
```

ABCs and models in `events.py`. Transports split into subdirectories
to keep files small and avoid pulling optional deps at import time.
