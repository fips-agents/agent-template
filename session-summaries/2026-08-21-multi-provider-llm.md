# Session Summary — 2026-08-21 · multi-provider-llm · Add LiteLLM, native OpenAI, native Anthropic to BaseAgent

**Plan:** GitHub issues #233, #234, #235, #194 + new multi-provider feature   **Commits:** none yet (prepare-and-ask)
**Deployed:** none   **Model:** Claude Opus 4.6 (1M context)

## Plan vs. actual
Planned: multi-provider LLM client (LiteLLM default, native OpenAI, native Anthropic) + fix scaffolding bugs #233/#234/#235. Shipped: all planned work complete. Slipped: none. #194 (fallback chain) is a design hook only as planned — the provider abstraction enables it as follow-on.
Scope: stayed in scope; added CLI `--provider` flag as a natural extension.

## Shipped
- Provider ABC (`baseagent/providers/_base.py`) + factory (`__init__.py`) with three implementations
- `OpenAIProvider` extracted from `LLMClient` — direct `AsyncOpenAI`
- `LiteLLMProvider` — wraps `litellm.acompletion()`, OpenAI-compatible output
- `AnthropicProvider` — full translation: messages, tools, streaming chunk normalization, extended thinking via `reasoning_content`, structured output via tool-use pattern
- `LLMClient` refactored to delegate to provider; `astep_stream()` unchanged
- `LLMConfig.provider` expanded: `litellm` (default) | `openai` | `anthropic` | `bedrock`/`azure` (deprecated with warnings)
- New optional extras: `fipsagents[litellm]`, `fipsagents[anthropic]`
- Test updates: mock targets updated for provider delegation (75 LLM tests, 44 spawn tests, 2 config tests)
- **CLI repo**: `--provider` flag on `fips-agents create agent`, `customize_provider()` function
- **CLI repo**: #235 fix (`repo.index.add(".")` for dotfiles), #234 fix (vendored deps from UPSTREAM.toml), #233 (version notice)
- Documentation: template CLAUDE.md + repo CLAUDE.md updated

## Verification & confidence
- fipsagents test suite: 2453 passed, 27 skipped, 0 failed
- CLI test suite: 433 passed, 0 failed
- Lint: ruff clean
- Provider instantiation verified for all three backends + backward compat
- Confidence: **high** for the provider abstraction and OpenAI/LiteLLM paths. **Medium** for the Anthropic provider — translation logic is well-referenced from the adapter sidecar, but it hasn't been exercised against a real Anthropic API endpoint in this session (mock-only). Extended thinking and structured output paths in particular need live testing.

## Judgment calls & deviations
- Default provider changed from `openai` to `litellm` per user direction. This is a backward-incompatible config change for anyone using `LLMConfig()` without specifying a provider.
- Anthropic structured output uses tool-use pattern (force a tool call with the schema) rather than `response_format`, since Anthropic doesn't support OpenAI's `response_format` shape.
- Kept the adapter sidecar path working for `bedrock`/`azure` with deprecation warnings rather than removing it.
- `astep_stream()` was not modified — all providers normalize to OpenAI-compatible chunk shapes. This was a deliberate design choice to minimize blast radius.

## Backlog delta
Filed: none new. Closed: none (no commits yet). #194 (fallback chain) is enabled by the provider abstraction but not implemented. #231 (platform mode bypass) skipped per user direction.

## Drift & forward-collisions
- Backward: #194 (model fallback chain) is now straightforward to implement — the provider factory could return a chain. Still valid, scope unchanged.
- Forward: none identified.

## For the reviewer
- Sanity-check: the Anthropic provider's streaming chunk normalization (`_stream_as_openai_chunks`) synthesizes `SimpleNamespace` objects to duck-type as OpenAI chunks. Verify the attribute access patterns in `astep_stream()` are fully covered (content, reasoning_content, tool_calls, finish_reason, usage).
- Thin verification: the Anthropic provider has not been tested against a real API. The LiteLLM provider was verified to instantiate but not to make real calls.
- Wants guidance: should the default provider change from `litellm` to `openai` for backward compat, with LiteLLM set via the CLI `--provider` flag? Current: `litellm` is the default everywhere.

## Risks / watch-fors
- The `litellm` default means existing projects that upgrade fipsagents and don't specify `provider: openai` in their `agent.yaml` will get litellm, which requires `pip install fipsagents[litellm]`. The env-var fallback `${MODEL_PROVIDER:-openai}` in agent.yaml templates mitigates this for scaffolded projects.
- CLI changes span two repos (agent-template + fips-agents-cli) — they should be committed and released together to stay consistent.
