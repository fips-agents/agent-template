# Next Session — multi-provider (follow-on)

## Status: Epic complete

The multi-provider LLM client and fallback chain are shipped and released (fipsagents 0.32.0, CLI 0.17.1). This file tracks follow-on items that came out of the work but aren't urgent.

## Follow-on items (not blocking, pick up when relevant)

### 1. Live fallback integration test
The FallbackProvider has 19 unit tests but no live test against a real endpoint. A test that points the primary at a non-existent endpoint and falls back to a real one would close the gap. Low effort, high confidence gain.

### 2. Bump fipsagents to 0.33.0 for fallback chain
The fallback chain shipped after the 0.32.0 tag. Users installing 0.32.0 from PyPI don't get it. Cut a 0.33.0 release when convenient.

### 3. CLI 0.17.0 GitHub Release cleanup
The 0.17.0 tag exists on GitHub but its PyPI publish failed (black drift). 0.17.1 is the actual release. The 0.17.0 GitHub Release object could confuse users — consider deleting it or marking it as a pre-release.

### 4. Deeper #233 fix — vendor from PyPI sdist
The current fix reads vendored deps from UPSTREAM.toml, which is better than the hardcoded list. The deeper fix (sourcing vendored code from the PyPI sdist instead of cloning agent-template) would eliminate the version-staleness problem entirely. Medium effort.

### 5. #231 — OpenAIChatServer bypasses step() and platform.enabled
Skipped this session. Still open. The server's HTTP handler calls `astep_stream()` directly instead of `step()`, so `platform.enabled` has no effect on the serving path. Needs a design decision about how the Responses API path integrates with the server.

## What landed (pointer)

See `session-summaries/2026-08-21-multi-provider-llm.md` for the full ledger.
