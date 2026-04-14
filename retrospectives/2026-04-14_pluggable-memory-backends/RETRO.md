# Retrospective: Pluggable Memory Backends

**Date:** 2026-04-14
**Effort:** Make agent memory work without MemoryHub. Add configurable backend dispatch, SQLite and PGVector backends, developer guide.
**Issues:** #27 (tracking), #28, #29, #30, #31
**Commits:** b16ad47, 03055b9, 5981d24

## What We Set Out To Do

NEXT_SESSION.md laid out four priorities in dependency order:

1. Make the MemoryClientBase interface public, add `backend` field to MemoryConfig, refactor the factory to dispatch on it
2. SQLite backend with FTS5 keyword search (zero-dependency local dev)
3. PGVector backend with semantic vector search (production without MemoryHub)
4. Developer guide: how to implement and register a custom backend

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Added empty-string validator to MemoryConfig.backend | Good pivot | `${MEMORY_BACKEND:-}` env var substitution produces empty string, not None — would fail Pydantic Literal validation |
| SQLite LIKE fallback rewritten to per-word OR | Good pivot | Review agent found that FTS5 syntax chars in the query (e.g., unclosed quotes) made the raw LIKE pattern match nothing |
| Three unused imports removed from test_tools.py | Incidental fix | Pre-existing lint failures from prior commit; review agent cleaned them while running ruff |
| No scope deferrals or missed requirements | — | NEXT_SESSION.md plan was precise enough that all four priorities shipped as specified |

## What Went Well

- **NEXT_SESSION.md as a session contract.** Every priority had specific files, line numbers, schemas, and key decisions pre-made. Zero time spent on discovery or design debate. This is the third session in a row where detailed NEXT_SESSION.md planning paid off — it's now a validated pattern.
- **Parallel sub-agent execution.** Each priority decomposed into implement + test + review. Independent tasks (config changes, template YAML updates) ran in parallel with core implementation. The review agent caught two real bugs (empty-string validator, LIKE fallback).
- **Test coverage.** 307 tests, all passing. Every dispatch branch covered. SQLite tests are real integration tests (actual DB). PGVector tests are properly mocked (no PostgreSQL dependency in CI). Contract test class documented for custom backends.
- **Clean commit history.** Three commits with clear separation: feature, docs, housekeeping. No fixup commits, no reverts.
- **Session close audit caught stale docs.** architecture.md still referenced `self.memory is None` and MemoryHub-only integration. Caught and fixed before the session ended. This prompted creating `/session-close` as a reusable skill.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| agent.py still passes only config_path to create_memory_client(), not the full MemoryConfig | Follow-up | The factory works via backward compat, but passing config would enable backend dispatch without auto-detect. Low priority — works correctly as-is. |
| PGVector factory doesn't close httpx.AsyncClient or asyncpg.Pool on agent shutdown | Accept | Both are safe to leave open in the current architecture. A future lifecycle hook could wire cleanup, but no agent currently needs it. |
| No end-to-end integration test for PGVector (requires real PostgreSQL) | Accept | Unit tests with mocks cover the code paths. Real integration testing happens at deploy time against the cluster's PGVector. |
| Issues #28-31 still open on GitHub | Fix now | Need to close them — work is done and committed. |

## Action Items

- [ ] Close issues #28, #29, #30, #31 on GitHub (work is committed)
- [ ] Update NEXT_SESSION.md for the next session's agenda
- [ ] Consider wiring MemoryConfig into agent.py's create_memory_client() call (low priority)

## Patterns

**Start:**
- Using `/session-close` at the end of every session. This session's manual audit caught stale docs and a missing version bump — the new skill codifies those checks.

**Stop:**
- Nothing new to stop. Previous patterns (run full tests, don't mock to hide errors, review agent after implementation) were all followed.

**Continue:**
- Detailed NEXT_SESSION.md with file paths, line numbers, and pre-made design decisions. Three sessions running, zero wasted discovery time.
- The implement → review cycle with sub-agents. Review found 2 bugs this session (empty-string coercion, LIKE fallback), 5 bugs last session. It consistently catches real issues.
- Pre-commit secret scanning before every commit.
- Clean commit separation (feature / docs / housekeeping).
