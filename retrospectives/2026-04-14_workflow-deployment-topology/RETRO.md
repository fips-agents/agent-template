# Retrospective: Workflow Deployment Topology

**Date:** 2026-04-14
**Effort:** Enable workflows to mix in-process and remote nodes (#24)
**Issues:** #24 (closed), #32 (follow-up)
**Commits:** e3e39d3, bdbc5de
**Version:** 0.2.0 -> 0.3.0

## What We Set Out To Do

Issue #24 had four open design questions and a clear implementation plan
laid out in NEXT_SESSION.md across four priorities:

1. Resolve where node config lives, state serialization format, whether
   RemoteNode is a framework class, and A2A integration depth
2. Implement RemoteNode in the fipsagents package
3. Add NodeConfig to AgentConfig and auto-wrap in WorkflowRunner
4. Write a brownfield integration guide

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Added "runner auto-wrap" as 5th design decision | Good pivot | Natural follow-on from NodeConfig — keeps graphs topology-agnostic |
| All 4 original design questions confirmed as-recommended | As planned | NEXT_SESSION.md pre-made decisions held up without debate |

No scope deferrals. No missed requirements.

## What Went Well

- **Pre-made decisions eliminated design overhead.** NEXT_SESSION.md had
  recommendations for all four questions with rationale. Confirmation took
  under 2 minutes. This is the strongest validation yet of the "decide in
  the planning doc, confirm at session start" pattern.

- **Sub-agent parallelism.** 3 batches of parallel workers (3 + 2 + 2) plus
  a dedicated reviewer. Wall time was dominated by waiting on the longest
  worker per batch, not serial execution. The reviewer found 4 real issues
  (missing `__future__` import, relative imports, doc inaccuracy, missing
  type annotation) that were all fixed before commit.

- **Test coverage.** 27 new tests across 3 files. 333 total passing. The
  test worker ran the full suite as part of its task, and the coordinator
  re-ran independently to verify.

- **Clean commit separation.** Feature commit (code + tests + design doc),
  then docs/version commit (brownfield guide, CLAUDE.md, NEXT_SESSION.md,
  version bump). Matches the established pattern.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| Non-JSON responses and missing `state` key propagate as raw exceptions, not RemoteNodeError | Follow-up | Filed #32 |
| RemoteNode creates new httpx.AsyncClient per request | Accept | Negligible overhead vs network call |
| Double-retry amplification (RemoteNode retries x runner node_retries) only documented in YAML comments | Accept | RemoteNode docstring already explains the intentional layering |

## Action Items

- [x] Close #24
- [x] File #32 for RemoteNodeError response parsing wrapping
- [x] Bump version to 0.3.0
- [x] Update CLAUDE.md with RemoteNode/NodeConfig
- [x] Write NEXT_SESSION.md for sandbox hardening v2 (#26)

## Patterns

**Start:** Nothing new needed. Process worked well this session.

**Stop:** Nothing new.

**Continue:**
- Detailed NEXT_SESSION.md with file paths, pre-made decisions, and contracts
- Implement -> review -> fix cycle with sub-agents
- `/session-close` checklist (caught lint error, doc staleness, version bump)
- Clean commit separation (feature + docs)
- Pre-commit secret scanning via gitleaks
