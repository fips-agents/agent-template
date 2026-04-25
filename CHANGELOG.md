# Changelog

All notable changes to the `fipsagents` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
