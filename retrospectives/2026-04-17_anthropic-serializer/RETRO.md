# Retrospective: Anthropic Messages Wire Format Serializer

**Date:** 2026-04-17
**Effort:** Add Anthropic Messages streaming serializer (#41)
**Issues:** #41
**Commits:** b95da71, c32b090

## What We Set Out To Do

Add `stream_events_as_anthropic_messages` — a second wire format serializer translating BaseAgent's `StreamEvent` stream to Anthropic's named-event SSE format. The explicit success criterion from the issue: "A second serializer proves the convention. If adding Anthropic Messages requires introducing any registry, factory, or strategy pattern, stop and reassess."

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| (none) | — | Plan matched implementation exactly. No pivots or scope changes. |

## What Went Well

- **"Do we need this?" check before implementation.** Asked whether LiteLLM could handle this instead. Answer: no — LiteLLM operates on the model side, the serializer operates on the consumer side. Right question, prevented building something unnecessary or duplicative.
- **Convention validated.** Zero registries, factories, or strategy patterns needed. The type signature `(events, model_name, ...) -> AsyncIterator[str]` is the contract. Copy the module, change the mapping, done.
- **Review agent found a real bug.** `ToolCallDelta` with `call_id=None` on first delta produced `"id": ""` in the tool_use block. Fixed to skip (matching the OpenAI serializer's guard) before commit.
- **Edge case tests from review.** Interleaved tool call deltas, content-after-tools, empty arguments_delta — all added based on review findings.
- **Cleanest session in the project.** Two commits, one issue, no rework. Plan discussion → implement → review → fix → commit → push → close.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| MemoryHub integration tests still failing (3 tests) | Pending | User shipping MemoryHub MCP update — retest shortly |

## Action Items

- [ ] Retest MemoryHub integration tests after MCP server update lands

## Patterns

**Continue:**
- Implement -> review -> fix cycle. Review found 1 bug this session, consistently 1-5 per session across all 8 retros. The cost is ~2 minutes; the value is real bugs caught before commit.
- NEXT_SESSION.md with cluster state, issue status, and design decisions. Zero wasted discovery time at session start. Eight sessions running.
- `/session-close` checklist. Caught stale architecture.md reference this session.
- Pre-commit secret scanning via gitleaks.
- Clean commit separation (feature / docs).
- Asking "should we build this?" before building. This session's LiteLLM question; previous session's similar check on MCP data models.
