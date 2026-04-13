# Retrospective: Template Validation Sprint

**Date:** 2026-04-13
**Effort:** Validate the agent-loop template end-to-end: eval schema, extension commands, deploy pipeline, and close out open issues.
**Issues:** #20 (deferred), #21 (closed), #22 (closed), #23 (deferred)
**Commits:** 469d78c, f07b507

## What We Set Out To Do

Four priorities from the previous session's NEXT_SESSION.md:
1. Fix eval schema mismatch between /exercise-agent and run_evals.py
2. Genericize shipped evals.yaml
3. Audit and fix /add-tool, /add-skill, /add-memory extension commands
4. Deploy smoke test on OpenShift

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Added #21 (AGENTS.md) and #22 (test audit) to session scope | Good pivot | Both were quick wins that completed the decoupling sweep started last session |
| Debugged .claude/settings.local.json permissions override | Unplanned fix | Blocking issue — every tool call prompted for approval despite --dangerously-skip-permissions |
| Smoke test expanded from "pod starts" to "full LLM call" | Good pivot | User pushed for complete validation; the partial test was leaving value on the table |

## What Went Well

- All four original priorities completed plus two bonus issues closed
- The eval schema mismatch was exactly where the previous session predicted it would be — good issue-filing discipline paid off
- Deploy smoke test validated the entire pipeline: Containerfile, Helm chart, litellm routing, both tool planes, structured output, graceful shutdown
- Extension command audit caught a real bug (chmod truncation in /add-memory) that would have broken containers silently

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| .claude/settings.local.json accumulated narrow permission rules across sessions, overriding --dangerously-skip-permissions | Fix now | Deleted the file. See "Patterns" for the process fix. |
| First smoke test pass stopped at "crashes at expected location" instead of completing with a real LLM | Process gap | User had to push for the full test. See "Patterns." |

## Action Items

- [x] Delete .claude/settings.local.json (done in session)
- [x] Run smoke test with live LLM (done in session)

## Patterns

**Start:**
- When a test can be run fully, run it fully. "It fails at the expected point" is not a passing test — it's a partial test. If resources are available (API keys, cluster access), use them. If they're not, say so explicitly and discuss with the user whether to defer or find an alternative.
- Periodically audit .claude/ project settings for accumulated state. Permission allowlists grow silently across sessions and can degrade the experience without an obvious cause.

**Stop:**
- Declaring smoke tests complete when they only validate initialization. A smoke test means the thing actually runs its core function, not just that it starts up.

**Continue:**
- Writing NEXT_SESSION.md with specific, actionable priorities and predicted bug locations — today's session confirmed two bugs exactly where the previous session said to look.
- Using sub-agents for parallel audit + implementation, with a review pass before committing.
- The pattern of filing issues for deferred work during a session, then closing them in the next session.
