# Retrospective: Code Sandbox + Shared Memory Demo

**Date:** 2026-04-14
**Effort:** Build the code execution sandbox sidecar (#25) and a two-agent shared-memory example using MemoryHub.
**Issues:** #25 (closed), #26 (open — v2 hardening)
**Commits:** 6562879, 8b68e83, 7ed1d81, 9e8c77c

## What We Set Out To Do

From NEXT_SESSION.md: build a UBI-based sidecar container that executes LLM-generated Python safely, plus a `code_executor` tool, Helm chart updates, and an example workflow. Extended mid-session to include a two-agent demo showing MemoryHub shared memory.

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Example workflow node deferred | Scope deferral | No architectural impact; sandbox + tool is the useful deliverable |
| NetworkPolicy deferred to #26 | Scope deferral | Sidecar shares pod network; AST guardrails block network imports as v1 compensation |
| Added `getattr`/`setattr`/`delattr`/`breakpoint`/`input` to blocked calls | Good pivot | Review agent identified a real getattr+string-concatenation bypass |
| Fixed Containerfile `pip install .` ordering | Good pivot | Review caught that source files weren't copied before install step |
| memory.py rewritten for SDK v0.5.0 | Unplanned fix | SDK renamed all methods, changed return types, changed auth lifecycle. Discovered live during demo. |
| `demos/` renamed to `examples/` | Good pivot | User suggestion — matches open-source convention |
| Agent 2 uses agent-loop instead of workflow framework | Good pivot | Workflow classes (Graph, WorkflowRunner) aren't exported from fipsagents package — only in template's src/workflow/ |
| Cross-agent memory sharing uses same identity | Workaround | MemoryHub project enrollment not set up; user-scope memories aren't visible cross-user |

## What Went Well

- Sub-agent parallelization was effective: guardrails + executor built simultaneously, then tool + helm + docs in a second parallel batch. Review agent caught 5 real issues.
- 85 tests, 98% coverage on the sandbox. Integration tests exercise realistic LLM code patterns (stats, JSON, regex, datetime, combinatorics, Decimal precision).
- End-to-end demo worked: three services (sandbox :8000, problem-solver :8001, code-writer :8002), Granite LLM, MemoryHub, and the Code Writer visibly changed its output (float → Decimal) based on memories from the Problem Solver.
- Kept credentials out of git: .memoryhub.yaml uses env var placeholders, MEMORY_HUB_SETUP.md is gitignored, API keys never committed.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| fipsagents package has no test suite — memory.py fix is untested except via live demo | Follow-up | Create `packages/fipsagents/tests/` with at least memory.py coverage |
| Sandbox Containerfile not build-tested | Follow-up | Needs remote x86_64 build on ec2-dev |
| Granite 8b doesn't reliably use tool calling (generates JSON as text) | Accept | Use GPT-OSS-20B or larger models for agents requiring tool use. Document in template. |
| Workflow framework not in fipsagents package | Follow-up | WorkflowRunner, Graph, BaseNode, AgentNode only exist in template's src/workflow/. Consider extracting to package. |
| MemoryHub cross-user sharing requires project enrollment | Accept for now | Need admin API or manual ConfigMap edit to set up projects. Document the pattern. |

## Action Items

- [ ] Create fipsagents test suite (at minimum: memory.py, config.py, tools.py)
- [ ] Build-test sandbox Containerfile on ec2-dev
- [ ] Document Granite 8b tool-calling limitation in template CLAUDE.md
- [ ] Consider extracting workflow framework to fipsagents package
- [ ] Set up MemoryHub project enrollment for demo agents

## Patterns

**Start:**
- When integrating with an external SDK, check method signatures and return types first (`inspect.signature`). The memory.py ↔ SDK v0.5.0 mismatch wasted 3 iterations because we discovered it live instead of probing upfront.
- For demos that span multiple services, write a single `setup.sh` and document the startup sequence. The 3-terminal pattern worked but is fragile.

**Stop:**
- Nothing new to stop. Previous retro's "stop declaring smoke tests complete when they only validate initialization" was followed well this session — the demo ran real LLM calls, real MemoryHub writes, real sandbox execution.

**Continue:**
- The review agent pattern: implement → review → fix cycle. The review found the getattr bypass, Containerfile bug, and resp.json crash — all would have shipped without it.
- Writing NEXT_SESSION.md with cluster endpoints and open items. It worked perfectly to start this session.
- Parallel sub-agents for independent implementation tasks, sequential for dependent ones.
- Pre-commit secret scanning before every commit.
