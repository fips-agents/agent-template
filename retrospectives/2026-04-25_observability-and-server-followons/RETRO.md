# Retrospective: Observability Stack & Server Follow-Ons

**Date:** 2026-04-25
**Effort:** Server module hardening + full observability stack + v0.11.0 release
**Issues:** #95 (OTEL), #96 (Prometheus), #97 (trace propagation), #79 (RAG, descoped)
**Commits:** ee27c22, 6fb0a65, 9b2c709, d9582f5, b40ddce

## What We Set Out To Do

NEXT_SESSION.md had three priorities:
1. Vector storage design (#79) — design phase
2. Server follow-on improvements from the sessions/tracing review — 4 concrete items
3. Observability issues (#95, #96, #97) — 3 features with a dependency chain

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| RAG (#79) descoped from agent-template | Good pivot | RAG belongs as a separate ecosystem service (`fips-agents create rag-store`), not baked into the agent template. Same separation as UI and gateway. |
| 3 planned server follow-on commits → 1 | Scope adjustment | Files were interleaved across improvements; forcing 3 clean commits would have created commits that don't build independently. |
| 4 planned observability commits → 2 (feature + lint fix) | Scope adjustment | Same interleaving. Pragmatic collapse with no downside. |
| ID hash functions moved from otel.py to propagation.py | Good pivot | Review caught that propagation.py imported from otel.py, making trace context extraction fail without `[otel]` extra. The hash functions are pure hashlib. |

## What Went Well

- **Review agents caught 5 real bugs** across 3 review cycles: root span trace ID not propagated to OTEL (broken trace trees), `str(stream)` producing `"True"` instead of `"true"` (Prometheus convention), streaming error path recording `status="ok"`, propagation.py depending on otel.py (transitive ImportError), agent shutdown ordering (DB close before agent shutdown).
- **Clean dependency chain execution**: Prometheus (independent) → OTEL (depends on tracing) → propagation (depends on OTEL IDs) built in order with no backtracking.
- **45 new tests, zero regressions** at any point. 684 → 729.
- **Release pipeline worked first try**: tag → build → publish, both jobs green.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| `app.py` at 531 lines, over 512 soft limit | Accept | No clean split point; the file is the server entry point. Would need a major refactor to extract further. |
| OTEL `force_flush()` blocks event loop on shutdown | Accept | Only fires during server shutdown. `asyncio.to_thread` would be correct but low-value. |
| W3C traceparent regex doesn't reject `ff` version or all-zeros IDs | Follow-up | Spec edge case; functional for all real-world traces. |
| `RemoteNode.set_trace_context()` is caller-opt-in | Follow-up | `WorkflowRunner` should auto-inject when tracing is active. Not wired yet. |
| v0.10.0 was never released to PyPI | Accept | Version was bumped in pyproject.toml but never tagged. Jumped to v0.11.0. No external consumers were affected. |

## Action Items

- [x] All immediate fixes applied during session
- [ ] WorkflowRunner auto-injection of trace context into RemoteNodes (future enhancement)

## Patterns

**Start:** Nothing new. Process was smooth this session.

**Stop:** Nothing new.

**Continue:**
- Implement → review → fix cycle. 11th consecutive session with real bugs caught (5 this session). Non-negotiable.
- NEXT_SESSION.md with priorities and context. 11th session, zero wasted discovery.
- `/session-close` checklist before wrapping up.
- Collapsing commits when files are interleaved rather than forcing artificial separation.
- Descoping aggressively when a concern belongs in a different repo/service.
