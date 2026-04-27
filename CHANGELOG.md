# Changelog

All notable changes to the `fipsagents` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
