# Retrospective: Long-Running Agent Epic (#215) + Tool Approval (#105)

**Date:** 2026-05-22
**Effort:** Shipped prompt assembly, WorkItemStore follow-ups, human-in-the-loop tool approval, self-healing, and trust infrastructure in a single session.
**Issues:** #215 (phases 2-4), #105, #214
**Commits:** c083303..24acfb7 (11 commits on feat/215-prompt-assembly)
**PR:** #217

## What We Set Out To Do

Work through the priority queue from NEXT_SESSION.md in order:

1. Merge PR #216 (WorkItemStore Phase 1) + tag v0.29.0
2. Prompt Assembly — named layers with identity/personality
3. WorkItemStore follow-ups — Postgres backend, REST API, budget headroom, capability auto-discovery
4. Human-in-the-loop tool approval (#105)
5. Self-Healing — learned skills, trust-scoped writes, skill versioning
6. Trust + Scoreboard — trust accumulation/decay, scoreboard REST endpoints

All six delivered. DSPy integration explicitly descoped from Phase 4 after discussion.

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Phase 4 scoped to trust+scoreboard only (no DSPy, no maturation lifecycle) | Good pivot | DSPy is speculative; trust model needs real-world validation first |
| Tool blocking for budget headroom replaced with system-message steering | Good pivot | Intercepting tool dispatch is complex and fragile; system message + per-turn limits provides a sufficient safety net |
| Capability auto-discovery excludes tool names | Good pivot | Tool names change based on MCP exposure, creating noise; MCP servers and skills are stable identifiers |

## What Went Well

- Massive throughput: 6 features, 11 commits, 6270 lines, 160+ new tests, zero regressions at every checkpoint
- Consistent delegation pattern: implement sub-agent + lint/test verification, parallel where independent
- Every feature followed established patterns (stock tools, server routes, config models) — no architectural novelty required
- Test count climbed steadily (2129 -> 2325) with full-suite verification after each commit
- Plan mode caught the right scope questions before implementation (Phase 4 DSPy deferral, Phase 1 follow-up scoping)

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| PR #217 is too large (11 commits, 6270 lines, 5 features) | Moderate | Accept — features are logically separable by commit; splitting retroactively would create merge conflicts. Future sessions should open separate PRs per feature. |
| No end-to-end integration test for the full requires_approval flow | Moderate | Follow-up — needs a test with real session store + question answer injection |
| Budget headroom not integration-tested with actual astep_stream model calls | Moderate | Follow-up — unit tests verify math; integration requires LLM mock with token counts |
| Learned skills not tested together with prompt assembly layers | Low | Follow-up — each tested independently; cross-feature integration is a gap |
| postgres.py exceeds 512-line target (727 lines) | Low | Accept — proportional to sqlite.py (763 lines); ABC surface area drives the size |
| `_agent` reference in trust_routes fragile between requests | Low | Accept — routes return 404 gracefully; proper fix requires persisting trust state on the server instance |

## Action Items

- [ ] Split future feature work into separate PRs (1 feature = 1 PR)
- [ ] Add integration test for requires_approval with session store resume
- [ ] Add integration test for budget headroom in astep_stream with mocked model responses
- [ ] Test learned skills loading through prompt assembly's capabilities layer

## Patterns

**Start:**
- One PR per feature, even when features are queued sequentially. A 6270-line PR is hard to review and hard to bisect if something breaks.
- For features that reuse existing infrastructure (like requires_approval reusing the permission "ask" flow), verify the integration path end-to-end, not just the new entry point.

**Stop:**
- Nothing new to stop. Previous patterns (run full tests between commits, don't mock errors, plan before implementing) were all followed well.

**Continue:**
- Plan mode before each feature — caught the DSPy scoping issue and the Phase 1 follow-up prioritization question early
- Sub-agent delegation with parallel independent work — kept throughput high without sacrificing quality
- Pre-commit secret scanning before every commit
- Session close checklist — caught the doc staleness gap that would have shipped otherwise
