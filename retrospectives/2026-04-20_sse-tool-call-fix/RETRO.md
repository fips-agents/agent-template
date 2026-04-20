# Retrospective: SSE Tool Call Fix (#72)

**Date:** 2026-04-20
**Effort:** Fix tool call id/name missing in SSE stream for subsequent model iterations, deploy to FIPS cluster
**Issues:** #72
**Commits:** 5e37a1b, 7afef39 (main); 48cb23f (ui-template fix/tool-call-id-sse-72)
**Release:** fipsagents v0.6.1

## What We Set Out To Do

Fix #72: when `astep_stream` loops across model iterations (tool A → result → tool B), only the first tool call's SSE deltas included `id` and `name`. The UI showed one tool pill instead of two. NEXT_SESSION.md identified three files to investigate and a reproduction scenario.

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| `agent.py` ruled out as root cause | Good discovery | NEXT_SESSION.md flagged it, but `tool_buf` reset and `first` flag were already correct |
| UI fix in both ui-template and data-analyst-demo | Scope expansion | Same index-keyed bug existed in the client; fixing server alone would still show one pill |
| fipsagents v0.6.1 release | Necessary | PyPI dependency chain — cluster BuildConfigs install from PyPI |

## What Went Well

- **NEXT_SESSION.md payoff**: three file paths, reproduction command, and test approach were all correct. Zero discovery overhead — went straight to the fix.
- **Review agent caught a real bug**: the inserted test clobbered the `def` line of the next test function. Reviewer spotted it before commit. Direct improvement from last retro's "no review agent used" gap.
- **Parallel builds**: all 4 OpenShift BuildConfig builds ran concurrently (~2.5 min for Go, ~3 min for Python). Total wall time for full stack redeploy was under 5 minutes.
- **Tight scope**: 3-line server fix, 3-line UI fix, one test. No scope creep.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| ui-template#14 not yet merged | Follow-up | PR open, needs merge + rebase of feature branch |
| data-analyst-demo app.js fix not committed to git | Follow-up | Applied locally for BuildConfig, needs committing |
| #72 not yet verified on cluster | Follow-up | User handling in separate session |
| calculus-demo still on fipsagents 0.6.0 | Accept | Not affected by #72 (single tool call agent) |

## Action Items

- [ ] Merge ui-template#14
- [ ] Commit app.js fix in data-analyst-demo
- [ ] Verify #72 fix on FIPS cluster (user handling separately)

## Patterns

**Start:** Nothing new.

**Stop:** Nothing new.

**Continue:**
- NEXT_SESSION.md with file paths and reproduction steps (ten sessions, still paying off every time)
- Review agent on every implementation (caught a real bug this session, validating last retro's gap identification)
- BuildConfig over remote builds (fast, no SSH dependency, native x86_64)
