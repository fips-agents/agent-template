# Retrospective: OpenShift Deployment & Pipeline Design

**Date:** 2026-04-14
**Effort:** Deploy code-sandbox-agent to OpenShift, design sandbox profiles and code refactoring pipeline
**Issues:** #33 (FIPS, not started)
**Commits:** a638363..e1237d4

## What We Set Out To Do

From NEXT_SESSION.md (3 priorities):
1. Deploy code-sandbox-agent to OpenShift and validate end-to-end
2. FIPS cluster testing (#33)
3. Harden shared-memory example on-cluster

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| ec2-dev-2 unreachable, switched to BuildConfig | Good pivot | BuildConfig builds natively on x86_64 in-cluster, faster than QEMU emulation on Mac. May be the better default going forward. |
| FIPS testing deferred | Scope deferral | Cluster not provisioned yet. Tracked in #33. |
| Shared-memory deployment deferred | Scope deferral | Time spent on pipeline design discussion instead. Higher value. |
| Added sandbox profiles + code refactoring pipeline design | Good pivot | Emerged from post-deployment discussion. Captured in planning/code-execution-pipeline.md. |
| Fixed 6 pre-existing test_memory.py failures | Unplanned fix | MagicMock auto-attribute creation broke SDK method fallback tests. Had been accumulating as noise in every session-close. |

## What Went Well

- Review agent caught 2 real bugs before build: readyz 503 tuple return (FastAPI doesn't handle tuple status codes like Flask) and relative `Path(".")` base_dir that would break if uvicorn's CWD changed.
- BuildConfig pivot was seamless. Zero wasted time — just switched build strategy when SSH failed.
- Full on-cluster test matrix passed first try (after the API key fix). Landlock confirmed active (ABI v5), read-only root confirmed, resources well within limits.
- The design discussion produced a concrete, implementable plan (planning/code-execution-pipeline.md) covering sandbox profiles, pipeline tiers, and MCP server topology for code refactoring agents.
- Fixing the test_memory.py failures eliminates noise from every future session-close.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| No tests for server.py | Follow-up | Need unit tests for /chat, /healthz, /readyz endpoints |
| litellm OPENAI_API_KEY not documented at template level | Fix now | Fixed in e1237d4 — added to architecture.md and agent.yaml |
| FIPS cluster testing still pending | Follow-up | Tracked in #33, waiting on cluster provisioning |
| Shared-memory example not deployed | Follow-up | Deferred to next session |
| BuildConfig created a Dockerfile symlink we had to clean up | Nit | Renamed Containerfile to Dockerfile. Consider standardizing on Dockerfile for examples since BuildConfig requires it by default. |

## Action Items

- [ ] Write tests for examples/code-sandbox-agent/server.py
- [ ] FIPS cluster testing (#33) when cluster is available
- [ ] Deploy shared-memory example to OpenShift
- [ ] Implement sandbox profiles (planning/code-execution-pipeline.md Section 1)

## Patterns

**Start:**
- Prefer BuildConfig over remote builds when the cluster is available. Native x86_64, no SSH dependency, images land directly in the internal registry. Reserve ec2-dev-2 for builds that can't use BuildConfig (multi-arch, builds needing tools not in the base image).
- Document operational gotchas (like the OPENAI_API_KEY requirement) in both the architecture doc and the template's config comments. Two readers, two locations.

**Stop:**
- Letting test failures accumulate across sessions. The 6 memory test failures existed through 4 retros before being fixed. Should have been fixed when first noticed.

**Continue:**
- Implement -> review -> fix cycle. The review agent continues to find real bugs (2 this session, 2-5 in previous sessions).
- Detailed NEXT_SESSION.md. Five sessions running, zero wasted discovery time.
- `/session-close` checklist before wrapping up.
- Design discussions captured as planning docs, not just conversation.
