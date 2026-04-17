# Retrospective: Memory Prefix Slot (#47)

**Date:** 2026-04-16
**Effort:** Add stable memory-prefix slot to BaseAgent for cache-friendly injection
**Issues:** #47, #49 (follow-up: model probe), #50-#53 (follow-up: example cleanup)
**Commits:** `90d5695`, `b326104`

## What We Set Out To Do

Implement the `build_memory_prefix()` hook on BaseAgent so memory is injected once at session start as a stable prefix (index 1 in `self.messages`), keeping inference-server KV caches warm across turns. Add `prefix_role` config to support the OpenAI harmony `developer` role alongside the universal `system` role. Cap prefix size with `max_prefix_chars`.

Scope from NEXT_SESSION.md: ~60 lines in BaseAgent, ~20 in DemoChatAgent, small docs tweak, 5-8 tests.

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Added `prefix_role` config knob (not in original plan) | Good pivot | Cluster poke showed gpt-oss-20b handles `developer` role natively via harmony format; granite handles it via template fallback. Config knob lets agents opt in. |
| DemoChatAgent update deferred | Scope deferral | User directed: "examples can be reworked later." Filed as #50. |
| Integration tests exposed factory bug — `setup()` never passed `config=` to `create_memory_client` | Missed (pre-existing) | `MemoryConfig.backend` was a dead field through `setup()`. Non-memoryhub backends couldn't be selected. Fixed in `b326104`. |
| Factory path resolution bug — used raw `config.config_path` instead of resolved positional arg | Missed (pre-existing) | Sentinel default now disambiguates explicit path from omitted arg. |

## What Went Well

- **Plan-first workflow** caught the `developer` role question before any code was written — the cluster poke (4 curl commands) gave concrete evidence instead of speculation.
- **Review sub-agent** found the whitespace-only content gap — a 2-character fix that would have been a real bug on backends with padded content.
- **Integration tests justified themselves immediately** — found two real bugs in the factory that 14 unit tests with `FakeMemoryClient` couldn't catch. The factory path resolution issue would have silently broken every non-memoryhub backend deployed through `setup()`.
- **Session close checklist** caught the uncommitted state and stale docs before the session ended.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| #47 still open on GitHub | Fix now | Close after retro |
| Example agents have dead system-prompt guards | Follow-up | #50-#53 filed |
| No integration test for sqlite/pgvector backends | Accept | Markdown covers the factory dispatch path; other backends need external deps |
| `_parse_sections` uses `## ` not `# ` — initially wrote test fixtures with wrong heading level | One-off | Fixed immediately; markdown backend docs are clear, test author just didn't check |

## Action Items

- [ ] Close #47 on GitHub
- [ ] #50 — demo-chat-agent cleanup (highest value: adopts prefix, removes per-turn recall)
- [ ] #53 — agent-loop template cleanup (highest urgency: new scaffolded agents get dead code)

## Patterns

**Start:** Write at least one integration test that exercises `setup()` end-to-end with real files for any feature that touches the setup lifecycle. Unit tests with fakes proved the logic but missed two factory bugs that integration tests found in minutes.

**Continue:** Plan-first workflow with cluster validation before implementation. The `developer` role research saved a bad assumption from becoming a bad default.

**Continue:** Sub-agent review catching edge cases (whitespace filtering) that the implementation agent missed.
