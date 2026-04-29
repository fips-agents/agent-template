# Changelog

All notable changes to the `fipsagents` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.16.0] - 2026-04-28

File Upload track — server-side document ingest with pluggable storage, pluggable inline parsers, content-based MIME sniffing, and a pluggable virus-scanner contract. Closes the bulk of [#100](https://github.com/fips-agents/agent-template/issues/100).

### Added

- **`FileStore` ABC + Null/Sqlite backends + `FilesConfig`**. New top-level `files` block on `AgentConfig` with size cap, allowed MIME list, scanner config, and a `bytes_dir` for local-fs storage. `NullFileStore` (default, ephemeral) and `SqliteFileStore` (single-replica edge/dev) ship; metadata in SQLite, bytes on local FS.
- **`POST /v1/files` upload endpoint and `file_ids` on chat completions**. Multipart upload returns a `FileRecord`. Chat completion requests can carry `file_ids: [...]` so the server attaches uploaded files to the conversation context.
- **`DELETE` and `LIST` endpoints for `/v1/files`** ([#100](https://github.com/fips-agents/agent-template/issues/100)). Pagination + filter by `session_id`. Hard-delete removes both metadata and bytes.
- **Inline file parsing via `PlaintextParser` + `DoclingParser`**. Pluggable `FileParser` ABC with per-MIME dispatch. `PlaintextParser` covers text/markdown/JSON; `DoclingParser` (opt-in via `[files]` extra) covers PDF, DOCX, PPTX, XLSX, HTML, CSV, images. Parsed text is injected into the prompt via the `_attached_file` LLM-tool pattern.
- **Content-based MIME sniffing for `/v1/files` uploads**. Uses libmagic (via `python-magic`) to determine MIME type from bytes rather than trusting client-supplied `Content-Type`. Mismatches surface as 415. Optional dependency — falls back to client-supplied type when unavailable.
- **Pluggable `VirusScanner` with HTTP sidecar contract**. `ScannerConfig` lets operators wire any HTTP-speaking scanner (default contract: `POST /scan` returns `{infected: bool, viruses: [...]}`). `fail_mode: open | closed` for production tuning. Helm chart ships a ClamAV sidecar example.
- **`PostgresFileStore`** ([#127](https://github.com/fips-agents/agent-template/pull/127)). Mirrors `PostgresSessionStore` line-for-line: lazy asyncpg pool, `IF NOT EXISTS` schema, BIGINT size, TIMESTAMPTZ timestamps. Closes the Postgres half of `FileStore` for enterprise multi-replica deployments.

### Notes

- **`HttpFileStore` deferred** — the platform-routed backend still raises `NotImplementedError`; will land alongside `fipsagents-platform`'s `/v1/files` surface.
- **S3-compatible bytes backend deferred** — Module 9 of the examples site documents MinIO as the future target. Tracked on [#100](https://github.com/fips-agents/agent-template/issues/100); ADR pending.
- **Optional extras**: `pip install fipsagents[files]` pulls in Docling (~500 MB with torch). Container builds remain opt-in to keep image size down for agents that don't need parsing.
- **Scaffolded `agent.yaml` config + ClamAV sidecar** ships in the agent-loop template via [#126](https://github.com/fips-agents/agent-template/pull/126). New projects pick up the upload track without manual wiring; the chart toggles the sidecar via `files.virusScanner.enabled`.

## [0.15.0] - 2026-04-28

Cost Tracking v2 — pricing, budget enforcement, observability. Layers dollar amounts and configurable USD limits on top of the raw token accumulator that shipped in 0.14.x. Closes the bulk of [#104](https://github.com/fips-agents/agent-template/issues/104).

### Added

- **`PricingConfig` + `GET /v1/sessions/{id}/usage`** ([#121](https://github.com/fips-agents/agent-template/pull/121)). New top-level `pricing` field on `AgentConfig` with a per-model `PricingRate` table (USD per 1k tokens for input / output / cached, plus an optional `per_request` flat fee). `compute_cost()` pure helper in `fipsagents.server.pricing` follows OpenAI cached-token semantics. `GET /v1/sessions/{id}/usage` layers the configured rates over the cumulative `cost_data` accumulator and returns a single computed-cost view that BudgetEnforcer, the gateway, and the UI can consume without each re-implementing rate lookup.
- **`tenant_id` + `session_id` labels on `agent_tokens_total`** ([#122](https://github.com/fips-agents/agent-template/pull/122)). Opt-in via the new `metrics.token_label_mode: "model" | "tenant" | "session"` setting; default (`"model"`) preserves the existing label space exactly. The server extracts `X-Tenant` from the request headers (gateway-stamped, falls back to `"default"`) and `session_id` from the `ChatCompletionRequest`. The `"session"` mode is documented as high-cardinality and opt-in for deployments with external aggregation (federation, Mimir); `GET /v1/sessions/{id}/usage` remains the preferred per-session view.
- **`BudgetEnforcer` with per-session + per-tenant USD limits** ([#123](https://github.com/fips-agents/agent-template/pull/123)). Configurable via the new `BudgetConfig` (per-session / per-tenant `warn_usd` + `limit_usd`, `mode: enforce | observe`). Follows the `MetricsCollector` / `TraceCollector` observer pattern: `check_before_request()` reads cumulative session cost from the session store (works across restarts/replicas) and per-tenant cost from an in-process accumulator, raising `BudgetExceededError` when a hard limit would be crossed; `record_after_request()` refreshes the in-process counter and logs a single soft-warning per scope per identifier. `BudgetExceededError` maps to **HTTP 402 Payment Required** with a structured detail body (`error`, `scope`, `identifier`, `current_usd`, `limit_usd`) so callers can distinguish budget rejection from rate-limit (`429`) and auth (`401/403`). `observe` mode downgrades raising to log-only. Per-tenant scope is "this agent process's view" — accurate for single-replica; multi-replica tenant aggregation is documented as out of scope here.
- **OTEL GenAI semantic conventions on trace spans** ([#124](https://github.com/fips-agents/agent-template/pull/124)). `TraceCollector` stamps the standard GenAI attribute names alongside the legacy ones so OTEL backends (Tempo, Honeycomb, Grafana Cloud, etc.) get the keys they expect: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.system` on the request span; `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.system` on the model_call span. Legacy `prompt_tokens` / `completion_tokens` / `total_tokens` / `total_time` are preserved verbatim — no breakage for existing trace consumers (`TraceSummary` aggregation, `/v1/traces/{id}` viewers, custom dashboards). `OTELTraceStore` carries the new keys through automatically via its generic span-attribute copy loop.

### Notes

- **Backward-compatible defaults.** All four features are opt-in. `BudgetConfig.is_active()` returns `False` when no limits are set, so existing deployments without budget config see zero behavior change. `metrics.token_label_mode` defaults to `"model"`. `pricing` defaults to all-zero rates so `/usage` returns `cost_usd: 0.0` until rates are configured. OTEL GenAI attributes are additive on existing spans.
- **Cluster-smoked end-to-end** against `fipsagents-platform 0.2.1` and `RedHatAI/gpt-oss-20b` on RHPDS. Cumulative `cost_data` GET, `/v1/sessions/{id}/usage`, `agent_tokens_total{tenant_id,session_id}` labels, and BudgetEnforcer soft-warnings will be verified post-publish during the 0.15.0 re-smoke.

## [0.14.2] - 2026-04-28

### Added

- **`HttpSessionStore.get_cost_data` reads from the platform.** Replaces the `NotImplementedError` placeholder with a real `GET /v1/sessions/{id}/cost_data`, closing the cumulative-cost gap noted in 0.14.0's release notes. HTTP-backed deployments now get the same cumulative shallow-merge semantics that SQLite/Postgres provide natively — the per-turn accumulator on `OpenAIChatServer` reads existing totals before computing the merge, so multi-turn sessions converge on cumulative numbers instead of last-write-wins. 5 new tests across the unit and e2e suites; total `fipsagents` suite 765 → 770.

### Notes

- Requires `fipsagents-platform>=0.2.1`. Older platforms 404 cleanly on the read path and the agent degrades to the previous last-write-wins behavior — same operational shape as 0.14.0/0.14.1, no breakage.

## [0.14.1] - 2026-04-27

### Fixed

- **Cost tracking now records token usage in production** — `LLMClient.call_model_stream_raw` now sets `stream_options={"include_usage": True}` by default when streaming, so vLLM and other OpenAI-compat servers emit the terminal usage chunk that `OpenAIChatServer._persist_cost_data` relies on. Without this, `StreamMetrics.prompt_tokens` / `completion_tokens` stayed `None` and `cost_data` accumulators on the session never advanced past `{}`. Surfaced during the cluster smoke for [#116](https://github.com/fips-agents/agent-template/issues/116); fixes [#118](https://github.com/fips-agents/agent-template/issues/118). Callers can opt out by passing `stream_options={"include_usage": False}` (or supplying a different value) — the default uses `setdefault` semantics.

## [0.14.0] - 2026-04-27

### Added

- **`SessionStore.update()`** — partial-update method on the ABC for recording per-session accumulator state without rewriting message history. Signature: `update(session_id, *, cost_data: dict | None = None) -> bool`. Implementations on `Null` (no-op `False`), `Sqlite` (Python-side shallow merge), `Postgres` (native `||` JSONB merge), and `Http` (maps to `PATCH /v1/sessions/{id}`). First slice of [#104](https://github.com/fips-agents/agent-template/issues/104) (Cost Tracking).
- **`SessionStore.get_cost_data()`** — symmetric reader so the server-side accumulator can read existing totals before writing cumulative ones back. Implemented on `Null` / `Sqlite` / `Postgres`; `Http` raises `NotImplementedError` until the platform exposes a GET endpoint (tracked at [fipsagents-platform#4](https://github.com/fips-agents/fipsagents-platform/issues/4)).
- **Per-turn token-usage persistence** — `OpenAIChatServer` extracts `prompt_tokens` / `completion_tokens` from each turn's terminal `StreamComplete` event (sync and streaming paths) and accumulates `input_tokens`, `output_tokens`, `cached_tokens`, `model`, and `turn_count` onto the session's `cost_data` via `SessionStore.update()`. Persistence failures are caught and logged so cost-tracking issues never break the chat response.
- **`cost_data` column** on the `sessions` table — `TEXT NOT NULL DEFAULT '{}'` on SQLite, `JSONB NOT NULL DEFAULT '{}'::jsonb` on Postgres. Existing databases pick up the column on first connect via idempotent `ALTER TABLE ADD COLUMN` migrations; no operator action required.

### Changed

- `SqliteSessionStore.save()` switches from `INSERT OR REPLACE` to `ON CONFLICT(session_id) DO UPDATE SET messages, updated_at` so `cost_data` survives saves of new messages. Postgres's `save()` already had the right shape.

### Notes

- HTTP-backed deployments currently fall back to per-turn-delta writes (last-write-wins) for `cost_data` because the platform doesn't yet expose a read endpoint — see [fipsagents-platform#4](https://github.com/fips-agents/fipsagents-platform/issues/4). SQLite/Postgres backends get cumulative semantics for free.
- Cost data shape (`input_tokens`, `output_tokens`, `cached_tokens`, `model`, `turn_count`) is owned by the server layer; pricing, budget enforcement, and aggregation endpoints are deferred follow-ups on [#104](https://github.com/fips-agents/agent-template/issues/104).

## [0.13.0] - 2026-04-27

### Added

- **HTTP-backed store implementations** — `HttpSessionStore`, `HttpTraceStore`, `HttpFeedbackStore` in `fipsagents.server.http`. Drop-in replacements for the existing SQLite/Postgres backends that delegate persistence to a sibling [`fipsagents-platform`](https://github.com/fips-agents/fipsagents-platform) service over its REST surface. Closes the agent-side half of the Cross-Agent Platform Service work tracked in [#114](https://github.com/fips-agents/agent-template/issues/114) (architecture decision in [#112](https://github.com/fips-agents/agent-template/issues/112)).
- **Per-store backend override** — `SessionsConfig`, `TracesConfig` and `FeedbackConfig` each gain an optional `backend: sqlite | postgres | http` field. When unset, the store inherits `storage.backend`. Lets an operator route, eg, `feedback.backend: http` while keeping sessions/traces on local SQLite.
- **Platform routing config** — `StorageConfig` gains `platform_url` and `platform_token`. The static token is used for service-to-service flows; per-request `Authorization` headers from inbound chat requests take precedence and are forwarded to the platform via a contextvar populated by a new `_HttpStoreContextMiddleware`. W3C `traceparent` is forwarded the same way so platform writes participate in the same distributed trace as the chat completion that generated them.
- **`platform-client` extra** — explicit opt-in marker for HTTP-backed deployments (httpx itself is already a core dependency).

### Changed

- `OpenAIChatServer._lifespan` now resolves each store's backend independently and only acquires a SQLite connection when at least one enabled store needs it. The housekeeping task is skipped entirely when every active store is HTTP-backed (the platform owns its own housekeeping cycle).

### Notes

- `delete_before()` on every `Http*Store` is a logged no-op — the platform service is responsible for housekeeping cross-tenant data.
- `HttpFeedbackStore.add()` returns the platform-generated `feedback_id`; the agent's pre-generated id and `created_at` on the inbound `FeedbackRecord` are intentionally discarded (matches the platform's `POST /v1/feedback` contract).

## [0.12.0] - 2026-04-27

### Added

- **User feedback collection** — `POST/GET /v1/feedback`, `GET /v1/feedback/stats` with pluggable `FeedbackStore` backends (null, sqlite, postgres). Records ratings (thumbs-up/-down), comments, corrections, and aggregated stats.
- **In-place feedback updates** — `PATCH /v1/feedback/{feedback_id}` mutates an existing record (rating change, comment edit) rather than accumulating duplicates. Backed by a new `update()` method on the `FeedbackStore` ABC; partial payloads (None means "leave unchanged"). Returns 404 if the id is unknown, 200 with the updated record otherwise.
- **Trace ID surfacing** — every chat completion response now carries an `X-Trace-Id` header (sync and streaming) and the final SSE usage chunk includes a top-level `trace_id` field. Lets clients correlate completions with traces and submit feedback against a known trace.
- **Identity attribution on feedback** — `FeedbackRecord` gains a `user_id` field (default `"anonymous"`) populated from the gateway-issued `X-Auth-Subject` header (gateway-template#21 v1). Both SQLite and Postgres carry idempotent `ADD COLUMN` migrations so pre-cutover databases survive without downtime; legacy rows surface as `"anonymous"`. New `user_id` query filter on `GET /v1/feedback`.
- **Scaffolded feedback config** — `fips-agents create agent` now writes a `feedback:` block in `agent.yaml` and a `[feedback]` extra hint, so new projects pick up the feature without manual wiring.
- **Local smoke test** — `scripts/smoke-feedback.sh` exercises the full ui → gateway → agent stack offline (no LLM required, ~30 checks). Asserts the gateway strips spoofed `X-Auth-Subject` headers in anonymous mode.

### Changed

- `CreateFeedbackRequest.trace_id` is now optional. When omitted the server synthesises a stand-alone identifier, so feedback works even if tracing is disabled or sampled out (orphan records are still stored).

### Architecture

- **Cross-Agent Platform Service decision** — `docs/architecture.md` gains a new section recording the Option-4 decision from [#112](https://github.com/fips-agents/agent-template/issues/112): a sibling [`fips-agents/fipsagents-platform`](https://github.com/fips-agents/fipsagents-platform) repo will expose the `FeedbackStore` / `SessionStore` / `TraceStore` ABCs over REST, and `HttpFeedbackStore` / `HttpSessionStore` / `HttpTraceStore` (tracked in [#114](https://github.com/fips-agents/agent-template/issues/114)) will let agents route persistence to it. The 0.12.0 release of `fipsagents` is what unblocks the platform repo's first release.

## [0.11.0] - 2026-04-25

### Added

- **LLM adapter sidecar** (`packages/llm-adapter/`) with 8 providers: Anthropic, Bedrock (Claude), Bedrock Converse (Llama/Mistral/DeepSeek/Qwen/Nova), Azure OpenAI, OpenAI-compatible (vLLM/TGI/Together/Groq/etc.), Ollama, llama.cpp, Vertex AI/Gemini. New `provider` field in `LLMConfig` with automatic endpoint rewriting.
- **Session persistence** — `SessionStore` ABC with Null, SQLite, Postgres backends. REST endpoints (`POST/GET/DELETE /v1/sessions`), auto-create-on-first-use, session ID validation. Requires `[server]` extra for SQLite.
- **Tracing** — `TraceCollector` observer builds span trees from `StreamEvent`s. `TraceStore` ABC with Null (structured JSON logging), SQLite, Postgres backends. Query via `GET /v1/traces`.
- **Shared storage layer** — unified `StorageConfig` (`null`/`sqlite`/`postgres`) with per-feature enable flags and background housekeeping for expired data.
- **Prometheus metrics** — request/tool/token counters and duration histograms at `GET /metrics`. Requires `[metrics]` extra.
- **OTEL trace export** — `OTELTraceStore` wraps any `TraceStore`, translates to OpenTelemetry spans, exports via OTLP. Requires `[otel]` extra.
- **W3C distributed trace propagation** — `traceparent` header extraction/injection for `RemoteNode` HTTP calls in multi-agent workflows.

### Changed

- Server module refactored from single `__init__.py` into a proper package (`app.py`, `models.py`, `sessions.py`, `tracing.py`, `collector.py`, `metrics.py`, `otel.py`, `propagation.py`, `sqlite.py`).
- `SqliteConnectionManager` deduplicates connections by resolved path when both session and trace stores use SQLite. `PostgresTraceStore` fully implemented (was previously a stub falling back to `NullTraceStore`).
- Helm charts support conditional LLM adapter sidecar injection with `ADAPTER_PROVIDER` env var and health probes.
- Extracted shared OpenAI SDK helpers into `providers/_openai_helpers.py` for cross-provider reuse.

### Fixed

- Updated MCP integration tests for MemoryHub's refactored unified `memory(action=...)` tool API.
- Added missing `ServerConfig` to test stub agent.

## [0.9.0] - 2026-04-24

Initial stable release.
