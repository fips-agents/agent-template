# Retrospective: FIPS Sandbox + Gemma 4 + Multi-Agent Delegation

**Date:** 2026-04-19
**Effort:** Validate Gemma 4 tool calling, end-to-end FIPS code sandbox, multi-agent delegation, UI upstream
**Issues:** #33 (closed), #72 (opened)
**Commits:** bd9b6f9..66dd7b0 (8 commits)
**Release:** fipsagents v0.6.0

## What We Set Out To Do

1. Validate Gemma 4 native tool calling on the FIPS cluster via vLLM's `gemma4` parser
2. Re-validate the calculus demo on mcp-rhoai with updated BaseAgent
3. End-to-end test of code sandbox on FIPS cluster (issue #33)

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Two BaseAgent bugs found (MCP result serialization, content null) | Good pivot | Latent bugs surfaced by first real MCP + Gemma 4 usage |
| Simple Gemma test → fipsagents v0.6.0 release | Scope expansion | Bug fixes + 22 unreleased commits warranted a release |
| "Test sandbox on FIPS" → full 4-component demo deployment | Scope expansion | User wanted real-world scenarios, not manual exec tests |
| UI features upstream (7 features from calculus-ui) | Unplanned | User noticed template UI was bare; natural time to fix it |
| `/v1/agent-info` endpoint in server + gateway | Unplanned | Required by the upstreamed UI settings panel |
| Copy button on raw response modal | Unplanned | User request during testing |
| Public repo (data-analyst-demo) | Unplanned | User wanted the example shareable |

## What Went Well

- Two real bugs found by testing on infrastructure, not by reading code. MCP result serialization and `content: null` were latent for months.
- Cross-cluster MCP (FIPS → mcp-rhoai) worked first try. No TLS issues.
- Multi-agent delegation chain (3 hops, 40s) worked after adding route timeout annotation.
- `fips-agents create agent/gateway/ui` scaffolded deployable projects with minimal customization.
- UI feature upstream was efficient — copy static files, de-brand, done.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| #72: Tool call SSE serialization drops id/name for 2nd+ model iteration | Follow-up | Filed #72, documented in NEXT_SESSION.md |
| GPT-OSS-20B ignores code_executor for simple math | Accept | Model capability limitation (20B params), not framework bug |
| `build_system_prompt()` fix not yet released to PyPI | Accept | Ships with next release (#72 fix) |
| No review agent used this session | Process gap | Interactive testing replaced formal review; two bugs slipped through (agent-info reading messages, gateway missing agent-info proxy) |
| data-analyst-demo has ephemeral cluster URLs | Accept | Uses `${VAR:-default}` pattern, documented in README |

## Action Items

- [x] Filed #72 for tool call SSE serialization
- [x] Created NEXT_SESSION.md with #72 priority and debugging guidance
- [x] Public repo at redhat-ai-americas/data-analyst-demo
- [x] Closed #33

## Patterns

**Start:** When adding an endpoint to the server package, also add the pass-through to the gateway template in the same session. This session required two extra rebuild cycles because agent-info was added to the server but not the gateway.

**Stop:** Nothing new.

**Continue:**
- NEXT_SESSION.md with cluster state and where-to-look guidance (nine sessions, zero wasted discovery)
- `/session-close` checklist (caught architecture.md staleness again)
- Finding bugs by deploying to real infrastructure
- Seizing upstream opportunities when they arise naturally during demo work
