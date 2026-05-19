# Retrospective: Subagent-as-tool v1 shakedown

**Date:** 2026-05-08
**Effort:** Triage of issues opened with subagent-as-tool, design synthesis for the foundation underneath #163/#164/#166/#168, end-to-end demo build (calculus-coordinator → calculus-agent specialist via subagent-as-tool), and three-model scorecard to pick a champion coordinator.
**Issues:** closed #165 (v1 retroactively); opened #179, #180, #181 (v2 follow-ups), #182 (Phase 0 foundation tracker), #184 (ad-hoc spawn_agent companion), #185, #186 (framework bugs found during shakedown)
**PRs:** agent-template #183, ui-template #31, examples #31
**Commits (this work):** d2dea07, 639578b (agent-template); e5a6d79, df8d17c (ui-template); ee53ba8, 32fa895 (examples)

## What We Set Out To Do

The session opened as exploratory triage — "let's discuss our open issues, particularly the ones we opened today." It widened twice on user direction:

1. After triaging #165, into design work on the #163/#164 cluster (Question tool + per-tool permission policy).
2. After "deploy to cluster, do all and only stop if you run into a problem," into build-and-ship: scaffold a coordinator, deploy the full stack, evaluate three models, scale down to a single L40S champion.

The arc was triage → design → ship. Unusual to span all three in one session.

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Closed #165 entirely (vs commenting) | Good pivot | v1 was already shipped via PR #173. The PR body's "Closes the bulk of #165" is not a recognized closing keyword, so the issue stayed open. Carved cleanly into v2 issues #179/#180/#181. |
| Foundation tracker #182 + design doc | Good pivot (scope addition) | Noticed mid-discussion that #163/#164/#166/#168 share schema concerns. Pinning the contract once via design doc avoids re-spec'ing 4×. Stable message IDs, pending-state columns, fork lineage all defined once. |
| Compaction research via subagent | Good pivot | Findings shaped the design doc concretely (marker-based rolling summary, tool-call/result pairing as a documented LLM failure class, client-side enforcement). Research → architecture, not research → Slack message. |
| gpt-oss-20b → Ministral coordinator swap | Forced pivot | gpt-oss's RLHF prior overrode strict prompts on trivial calculus. Nemotron's vLLM tool-call parser is named `_no_streaming.py` and doesn't surface tool calls in streaming mode. Ministral was the third option. |
| Add Granite 4.0 H-Small + 4.1 8B to evaluation | Scope addition | External recommendation forwarded mid-session. Useful detour — produced reproducible scorecard methodology. |
| Deploy `calculus-helper` MCP into `calculus-mcp` namespace | Missed requirement caught late | Assumed the existing course-v11 calculus-agent was a working specialist. Its image expected an MCP namespace that didn't exist on the cluster. The specialist had been hallucinating. Discovery cost an iteration. |
| Coordinator nodeSelector to `g6e.4xlarge` | Good pivot | First reschedule landed Ministral on the OTHER team's eval-gpu node. Adding the nodeSelector ensured it could only land on gpu-cluster nodes. |

## What Went Well

- **Foundation caught before piecemeal implementation.** Design doc + #182 lock the schema contract once; #163/#164/#166/#168 reference it instead of re-spec'ing.
- **Research → design was tight.** Compaction research's findings ended up as actual contract decisions in the design doc with primary-source citations.
- **Scorecard methodology was reproducible.** `/tmp/scorecard.py` exists; raw timings, token counts via `stream_options.include_usage`, six prompts in fixed order. Champion picked on numbers, not vibes. The TTFC vs TTFT distinction (with Ministral conceptual at 42 ms) directly addressed the rounding-to-zero risk.
- **Multi-repo coordination clean.** Three PRs, each scoped to its own concern, no leaked changes between them. CLAUDE.md updates eventually landed on each repo's branch.
- **Two real bugs filed with reproductions.** #185 (observers drop subagent events), #186 (URL doubling). Concrete repro steps, proposed fixes.
- **Surgical cluster cleanup.** `cluster-api-delete-machine` annotation pattern + nodeSelector pin kept the eval-gpu scaledown from disturbing the parallel test on the same machineset.
- **Diagnostic flow once provoked.** Specialist hallucination was caught only after user question, but the path from "did it really work?" → confirmed hallucination → traced to dead namespace → deployed missing MCP was tight (~10 min).

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| MCP backend assumed working without sanity check; specialist was hallucinating until user provoked the question | Process problem | Mitigation: hit each load-bearing endpoint with a curl before declaring end-to-end works. |
| Visual UI verification skipped after final Ministral relocation — curl-only confirmation | Fix now | User did the visual verification post-retro and reported "still some bugs" → tracked for next session, not blocking. |
| vLLM Granite 4.0 H-Small hermes parser double-encoding bug — diagnosed cleanly but no upstream issue filed | Owner: user | User will file separately. |
| fipsagents non-streaming `usage: null` bug — diagnosed but no fipsagents issue filed | Owner: user | User will file separately. |
| Three CLAUDE.md updates batched at session-close instead of landing with the work | Process slop | Fixed as part of session-close skill; pattern for next session: doc updates land in the same commit as behavior change. |
| Six-revision prompt-engineering thrash on gpt-oss-20b before testing model behavior with a minimal harness | Process problem | Mitigation: when a model resists a strict prompt, run a minimal curl-with-tool-only harness first to bound model-can't vs prompt-can't. |
| Two off-by-one errors in `oc patch` JSON-Patch index paths against args arrays | One-off | Default to replacing the whole list when mutating positional args. |
| Demo MCP grounding fragile — depends on `calculus-mcp` namespace existing, no alerting | Documentation gap | Resolved in `examples/calculus-coordinator/CLAUDE.md` "Demo dependencies" section (commit 32fa895). |
| Final system prompt lacked rationale for its strict shape | Documentation gap | Resolved by frontmatter comment in `prompts/system.md` (commit 32fa895). |
| Demo URL + helm `--set` reference values existed only in conversation | Discoverability | Resolved in `examples/calculus-coordinator/CLAUDE.md` "Demo dependencies" section (commit 32fa895). |
| "Diagnose-but-don't-file" pattern: #165's verbal closing keyword + two un-filed framework bugs treated as filed | Recurring pattern | Mitigation: open the issue in the same step as the diagnosis. Verbally agreeing to file is not a filed issue. |

## Action Items

- [x] Document demo dependencies in `calculus-coordinator/CLAUDE.md` (commit 32fa895, examples PR #31)
- [x] Comment system-prompt rationale in `prompts/system.md` (commit 32fa895, examples PR #31)
- [ ] Visual verification reported "still some bugs" — debug in next session.
- [ ] User to file vLLM Granite 4.0 H-Small parser bug separately.
- [ ] User to file fipsagents non-streaming `usage: null` bug separately.

## Patterns

**Start:**

- Sanity-check load-bearing dependencies (specialist endpoints, MCP backends, model availability) with one curl before declaring an end-to-end demo working. Cheaper than discovering a hallucinating specialist after a scorecard run.
- File issues at the moment of diagnosis. "Verbal commitment to file" is the failure mode that produced #165 staying open + two un-filed framework bugs in this session.
- Visual UI verification on every UI-touching demo, even when curl-equivalent SSE confirms the wire is correct.
- When a model resists a strict prompt, isolate "model can't" from "prompt can't" with a 30-second curl harness against the model directly.

**Stop:**

- Batching CLAUDE.md updates at session-close. Doc landings happen with the work.
- Patching positional-arg arrays via JSON-Patch index paths. Replace the whole list when mutating it.

**Continue:**

- Foundation-design-before-piecemeal-implementation when a cluster of issues shares a contract surface. Pays off more than the design-doc effort.
- Reproducible scorecard methodology (raw timings, captured tokens via `stream_options.include_usage`, fixed prompt set). Generalizable to future model evaluations.
- Surgical cluster operations (annotation-driven machine deletion, node-affinity pinning) instead of "scale and hope" patterns.
- Cross-linking issues + PRs with prose explaining *why*, not just `Closes #N` keywords (which several models in this evaluation also fumble — turns out humans can too).
