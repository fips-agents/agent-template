"""Tests for BaseAgent._inject_deferred_memory() (issue #85)
and injection_mode: user_turn + server tag stripping (issue #86).

Covers:
- Deferred patterns (lazy, jit, lazy_with_rebias) inject memory as a prefix
  message before the user turn.
- injection_mode="user_turn" appends memory inside XML tags on the user message.
- Guard against double injection (user_turn mode).
- No-ops for eager pattern, missing project_config, no user message, or
  empty results.
- Truncation via max_prefix_chars.
- Exception safety: search() failure is swallowed and messages are unchanged.
- Server tag stripping regex (the part that strips echoed tags from LLM output).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.config import AgentConfig, MemoryConfig
from fipsagents.baseagent.memory import MemoryClientBase, NullMemoryClient


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeMemoryClient(MemoryClientBase):
    """Returns a fixed list of results for any search query."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._results

    async def write(self, content: str, **kwargs: Any) -> dict[str, Any] | None:
        return None

    async def update(
        self, memory_id: str, content: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None

    async def report_contradiction(
        self, memory_id: str, description: str
    ) -> None:
        return None


class _FakeMemoryLoading:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern


class _FakeProjectConfig:
    def __init__(
        self, *, pattern: str = "eager", project_id: str | None = None
    ) -> None:
        self.project_id = project_id
        self.memory_loading = _FakeMemoryLoading(pattern)


class FakeMemoryHubClient(FakeMemoryClient):
    """Extends FakeMemoryClient with a project_config and search-call tracking."""

    def __init__(
        self,
        results: list[dict[str, Any]],
        *,
        project_config: _FakeProjectConfig | None = None,
    ) -> None:
        super().__init__(results)
        self._project_config = project_config
        self.search_calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def project_config(self) -> _FakeProjectConfig | None:
        return self._project_config

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append((query, kwargs))
        return await super().search(query, **kwargs)


class BrokenMemoryClient(MemoryClientBase):
    """Always raises on search() to test exception safety."""

    def __init__(self, *, project_config: _FakeProjectConfig | None = None) -> None:
        self._project_config_val = project_config

    @property
    def project_config(self) -> _FakeProjectConfig | None:
        return self._project_config_val

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise ConnectionError("MemoryHub unreachable")

    async def write(self, content: str, **kwargs: Any) -> dict[str, Any] | None:
        return None

    async def update(
        self, memory_id: str, content: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None

    async def report_contradiction(
        self, memory_id: str, description: str
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Subsystem stubs (same as test_memory_prefix.py)
# ---------------------------------------------------------------------------


class _RenderedPrompt:
    def render(self) -> str:
        return "You are a test agent."


class _PromptStub:
    def get(self, name: str) -> _RenderedPrompt:
        return _RenderedPrompt()


class _RulesStub:
    def get_combined_content(self) -> str:
        return ""


class _SkillsStub:
    def get_manifest(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    memory: MemoryClientBase | None = None,
    prefix_role: str = "system",
    max_prefix_chars: int = 8000,
    injection_mode: str = "prefix",
    injection_tag: str = "user_memories",
    max_results: int = 50,
    min_weight: float = 0.0,
) -> BaseAgent:
    """Build a minimal BaseAgent for deferred-memory tests."""
    config = AgentConfig(
        model={"name": "test-model", "endpoint": "http://localhost:1234/v1"},
        memory=MemoryConfig(
            backend="null",
            prefix_role=prefix_role,
            max_prefix_chars=max_prefix_chars,
            injection_mode=injection_mode,
            injection_tag=injection_tag,
            max_results=max_results,
            min_weight=min_weight,
        ),
    )

    class _TestAgent(BaseAgent):
        def __init__(self) -> None:
            self.config = config
            self.memory: MemoryClientBase = memory or NullMemoryClient()
            self.messages: list[dict[str, Any]] = []
            self.prompts = _PromptStub()
            self.rules = _RulesStub()
            self.skills = _SkillsStub()

    return _TestAgent()


def _seeded_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "What is the weather?"},
    ]


# ---------------------------------------------------------------------------
# Deferred injection tests (#85)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_lazy_injects_as_prefix() -> None:
    """lazy pattern + prefix mode inserts memory message at index 1."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Memory A"}, {"id": "b", "content": "Memory B"}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    assert len(agent.messages) == 3, (
        f"Expected 3 messages after injection, got {len(agent.messages)}: "
        f"{agent.messages}"
    )
    assert agent.messages[1]["role"] == "system"
    assert agent.messages[1]["content"] == "Memory A\n\n---\n\nMemory B"
    assert agent.messages[2]["role"] == "user"


@pytest.mark.asyncio
async def test_deferred_jit_injects_as_prefix() -> None:
    """jit pattern + prefix mode inserts memory message before the user turn."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "JIT memory"}],
        project_config=_FakeProjectConfig(pattern="jit"),
    )
    agent = _make_agent(memory=mem)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    assert len(agent.messages) == 3
    assert agent.messages[1]["content"] == "JIT memory"
    assert agent.messages[2]["role"] == "user"


@pytest.mark.asyncio
async def test_deferred_lazy_with_rebias_injects_as_prefix() -> None:
    """lazy_with_rebias pattern also injects memory before the user turn."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Rebias memory"}],
        project_config=_FakeProjectConfig(pattern="lazy_with_rebias"),
    )
    agent = _make_agent(memory=mem)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    assert len(agent.messages) == 3
    assert agent.messages[1]["content"] == "Rebias memory"
    assert agent.messages[2]["role"] == "user"


@pytest.mark.asyncio
async def test_deferred_eager_does_nothing() -> None:
    """eager pattern: _inject_deferred_memory() is a no-op."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Should not appear"}],
        project_config=_FakeProjectConfig(pattern="eager"),
    )
    agent = _make_agent(memory=mem)
    original = _seeded_messages()
    agent.messages = [dict(m) for m in original]

    await agent._inject_deferred_memory()

    assert agent.messages == original, (
        "eager pattern must not modify messages"
    )
    assert len(mem.search_calls) == 0, (
        "eager pattern must not call search()"
    )


@pytest.mark.asyncio
async def test_deferred_no_project_config_does_nothing() -> None:
    """Non-MemoryHub backend (project_config=None) → no injection."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Should not appear"}],
        project_config=None,
    )
    agent = _make_agent(memory=mem)
    original = _seeded_messages()
    agent.messages = [dict(m) for m in original]

    await agent._inject_deferred_memory()

    assert agent.messages == original
    assert len(mem.search_calls) == 0


@pytest.mark.asyncio
async def test_deferred_no_user_message_does_nothing() -> None:
    """No user message in history → no injection, no error."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Should not appear"}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem)
    agent.messages = [{"role": "system", "content": "You are a test agent."}]

    await agent._inject_deferred_memory()

    assert len(agent.messages) == 1
    assert len(mem.search_calls) == 0


@pytest.mark.asyncio
async def test_deferred_empty_results_does_nothing() -> None:
    """Empty search results → messages unchanged."""
    mem = FakeMemoryHubClient(
        [],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem)
    original = _seeded_messages()
    agent.messages = [dict(m) for m in original]

    await agent._inject_deferred_memory()

    assert len(agent.messages) == 2
    assert agent.messages == original


@pytest.mark.asyncio
async def test_deferred_uses_user_message_as_query() -> None:
    """search() is called with the user message content as the query."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Some memory"}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    assert len(mem.search_calls) == 1
    query, _ = mem.search_calls[0]
    assert query == "What is the weather?"


@pytest.mark.asyncio
async def test_deferred_passes_project_id() -> None:
    """project_id from project_config is forwarded to search()."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Project memory"}],
        project_config=_FakeProjectConfig(pattern="lazy", project_id="my-project"),
    )
    agent = _make_agent(memory=mem)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    assert len(mem.search_calls) == 1
    _, kwargs = mem.search_calls[0]
    assert kwargs.get("project_id") == "my-project"


@pytest.mark.asyncio
async def test_deferred_respects_max_prefix_chars() -> None:
    """Content exceeding max_prefix_chars is truncated with the standard suffix."""
    long_content = "x" * 200
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": long_content}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem, max_prefix_chars=50)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    # A prefix message should have been inserted at index 1.
    injected = agent.messages[1]["content"]
    assert injected == "x" * 50 + "\n\n… [truncated]", (
        f"Expected truncated content, got: {injected!r}"
    )


# ---------------------------------------------------------------------------
# User-turn injection mode tests (#86)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_turn_appends_to_user_message() -> None:
    """injection_mode=user_turn appends memories inside <user_memories> tags."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "User turn memory"}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem, injection_mode="user_turn")
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    # No new message should be inserted.
    assert len(agent.messages) == 2, (
        f"user_turn mode must not insert a new message, got {len(agent.messages)}"
    )
    user_content = agent.messages[1]["content"]
    assert "<user_memories>" in user_content
    assert "User turn memory" in user_content
    assert "</user_memories>" in user_content
    assert user_content.startswith("What is the weather?")


@pytest.mark.asyncio
async def test_user_turn_custom_tag() -> None:
    """Custom injection_tag is used when injection_mode=user_turn."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Context memory"}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(
        memory=mem, injection_mode="user_turn", injection_tag="agent_context"
    )
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    user_content = agent.messages[1]["content"]
    assert "<agent_context>" in user_content, (
        f"Expected custom tag <agent_context> in content: {user_content!r}"
    )
    assert "</agent_context>" in user_content
    assert "<user_memories>" not in user_content


@pytest.mark.asyncio
async def test_user_turn_double_injection_guard() -> None:
    """If user message already contains the tag, skip injection entirely."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Should not be added again"}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem, injection_mode="user_turn")
    already_injected_content = (
        "What is the weather?\n\n<user_memories>\nold memory\n</user_memories>"
    )
    agent.messages = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": already_injected_content},
    ]

    await agent._inject_deferred_memory()

    # search() must not be called and content must be unchanged.
    assert len(mem.search_calls) == 0, (
        "Double injection guard: search() must not be called when tag already present"
    )
    assert agent.messages[1]["content"] == already_injected_content, (
        "Content must not be modified when tag already present"
    )


@pytest.mark.asyncio
async def test_prefix_mode_inserts_separate_message() -> None:
    """injection_mode=prefix (explicit) inserts a new message before user turn."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Prefix memory"}],
        project_config=_FakeProjectConfig(pattern="jit"),
    )
    agent = _make_agent(memory=mem, injection_mode="prefix", prefix_role="system")
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    assert len(agent.messages) == 3, (
        f"Expected 3 messages (system + prefix + user), got {len(agent.messages)}"
    )
    injected = agent.messages[1]
    assert injected["role"] == "system"
    assert injected["content"] == "Prefix memory"
    # Original user message must follow unchanged.
    assert agent.messages[2]["role"] == "user"
    assert agent.messages[2]["content"] == "What is the weather?"


@pytest.mark.asyncio
async def test_deferred_injection_never_crashes() -> None:
    """search() raising an exception must not crash the agent; messages unchanged."""
    mem = BrokenMemoryClient(
        project_config=_FakeProjectConfig(pattern="lazy")
    )
    agent = _make_agent(memory=mem)
    original = _seeded_messages()
    agent.messages = [dict(m) for m in original]

    # Must not raise.
    await agent._inject_deferred_memory()

    assert agent.messages == original, (
        "Messages must be unchanged when memory search raises"
    )


# ---------------------------------------------------------------------------
# Server tag stripping (#86)
# ---------------------------------------------------------------------------


def test_collect_sync_strips_echoed_tags() -> None:
    """Regex stripping removes echoed <user_memories>…</user_memories> blocks."""
    tag = re.escape("user_memories")
    content = (
        "Here is my response.\n"
        "<user_memories>\n"
        "some memory\n"
        "</user_memories>\n"
        "More text."
    )
    result = re.sub(
        rf"<{tag}>.*?</{tag}>", "", content, flags=re.DOTALL
    ).strip()
    assert result == "Here is my response.\n\nMore text.", (
        f"Unexpected stripped content: {result!r}"
    )


# ---------------------------------------------------------------------------
# Budget preset tests (#87)
# ---------------------------------------------------------------------------


def test_budget_small_sets_defaults() -> None:
    """MemoryConfig(budget='small') applies the small-tier preset values."""
    cfg = MemoryConfig(budget="small")
    assert cfg.max_prefix_chars == 500, (
        f"Expected max_prefix_chars=500, got {cfg.max_prefix_chars}"
    )
    assert cfg.max_results == 5, (
        f"Expected max_results=5, got {cfg.max_results}"
    )
    assert cfg.min_weight == 0.7, (
        f"Expected min_weight=0.7, got {cfg.min_weight}"
    )


def test_budget_medium_sets_defaults() -> None:
    """MemoryConfig(budget='medium') applies the medium-tier preset values."""
    cfg = MemoryConfig(budget="medium")
    assert cfg.max_prefix_chars == 4000, (
        f"Expected max_prefix_chars=4000, got {cfg.max_prefix_chars}"
    )
    assert cfg.max_results == 20, (
        f"Expected max_results=20, got {cfg.max_results}"
    )
    assert cfg.min_weight == 0.5, (
        f"Expected min_weight=0.5, got {cfg.min_weight}"
    )


def test_budget_large_sets_defaults() -> None:
    """MemoryConfig(budget='large') applies the large-tier preset values."""
    cfg = MemoryConfig(budget="large")
    assert cfg.max_prefix_chars == 8000, (
        f"Expected max_prefix_chars=8000, got {cfg.max_prefix_chars}"
    )
    assert cfg.max_results == 50, (
        f"Expected max_results=50, got {cfg.max_results}"
    )
    assert cfg.min_weight == 0.3, (
        f"Expected min_weight=0.3, got {cfg.min_weight}"
    )


def test_budget_none_uses_field_defaults() -> None:
    """MemoryConfig() with no budget uses the field-level defaults."""
    cfg = MemoryConfig()
    assert cfg.max_prefix_chars == 8000, (
        f"Expected default max_prefix_chars=8000, got {cfg.max_prefix_chars}"
    )
    assert cfg.max_results == 50, (
        f"Expected default max_results=50, got {cfg.max_results}"
    )
    assert cfg.min_weight == 0.0, (
        f"Expected default min_weight=0.0, got {cfg.min_weight}"
    )


def test_budget_explicit_override() -> None:
    """An explicit field value wins over the budget preset for that field only."""
    cfg = MemoryConfig(budget="small", max_results=15)
    # Explicit value overrides the preset's max_results=5.
    assert cfg.max_results == 15, (
        f"Expected explicit max_results=15 to override preset, got {cfg.max_results}"
    )
    # Non-overridden fields still come from the preset.
    assert cfg.max_prefix_chars == 500, (
        f"Expected preset max_prefix_chars=500, got {cfg.max_prefix_chars}"
    )
    assert cfg.min_weight == 0.7, (
        f"Expected preset min_weight=0.7, got {cfg.min_weight}"
    )


def test_budget_custom_uses_field_defaults() -> None:
    """budget='custom' is not a named preset and falls through to field defaults."""
    cfg = MemoryConfig(budget="custom")
    assert cfg.max_prefix_chars == 8000, (
        f"Expected field default max_prefix_chars=8000, got {cfg.max_prefix_chars}"
    )
    assert cfg.max_results == 50, (
        f"Expected field default max_results=50, got {cfg.max_results}"
    )
    assert cfg.min_weight == 0.0, (
        f"Expected field default min_weight=0.0, got {cfg.min_weight}"
    )


@pytest.mark.asyncio
async def test_deferred_max_results_passed_to_search() -> None:
    """max_results from MemoryConfig is forwarded as a kwarg to search()."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "Memory A"}],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem, max_results=3)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    assert len(mem.search_calls) == 1, (
        f"Expected exactly one search call, got {len(mem.search_calls)}"
    )
    _, kwargs = mem.search_calls[0]
    assert kwargs.get("max_results") == 3, (
        f"Expected max_results=3 in search kwargs, got {kwargs!r}"
    )


@pytest.mark.asyncio
async def test_deferred_min_weight_filters_results() -> None:
    """Results below min_weight are filtered out before injection."""
    mem = FakeMemoryHubClient(
        [
            {"id": "a", "content": "High weight", "weight": 0.9},
            {"id": "b", "content": "Low weight", "weight": 0.3},
            {"id": "c", "content": "Medium weight", "weight": 0.7},
        ],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem, min_weight=0.6)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    injected = agent.messages[1]["content"]
    assert "High weight" in injected, (
        f"Expected 'High weight' (0.9 >= 0.6) to be present: {injected!r}"
    )
    assert "Medium weight" in injected, (
        f"Expected 'Medium weight' (0.7 >= 0.6) to be present: {injected!r}"
    )
    assert "Low weight" not in injected, (
        f"Expected 'Low weight' (0.3 < 0.6) to be filtered out: {injected!r}"
    )


@pytest.mark.asyncio
async def test_deferred_min_weight_zero_no_filter() -> None:
    """min_weight=0.0 (default) passes all results through regardless of weight."""
    mem = FakeMemoryHubClient(
        [
            {"id": "a", "content": "Very low", "weight": 0.01},
            {"id": "b", "content": "Zero weight", "weight": 0.0},
            {"id": "c", "content": "Normal", "weight": 0.8},
        ],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    agent = _make_agent(memory=mem, min_weight=0.0)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    # All three should appear since min_weight=0.0 disables filtering.
    injected = agent.messages[1]["content"]
    assert "Very low" in injected, (
        f"Expected 'Very low' to pass min_weight=0.0 filter: {injected!r}"
    )
    assert "Zero weight" in injected, (
        f"Expected 'Zero weight' to pass min_weight=0.0 filter: {injected!r}"
    )
    assert "Normal" in injected, (
        f"Expected 'Normal' to pass min_weight=0.0 filter: {injected!r}"
    )


@pytest.mark.asyncio
async def test_min_weight_default_preserves_no_weight_field() -> None:
    """Results missing a 'weight' key default to 1.0 and are never filtered."""
    mem = FakeMemoryHubClient(
        [
            {"id": "a", "content": "No weight field"},
            {"id": "b", "content": "Has weight field", "weight": 0.5},
        ],
        project_config=_FakeProjectConfig(pattern="lazy"),
    )
    # Use a strict min_weight — the weightless result must still survive.
    agent = _make_agent(memory=mem, min_weight=0.9)
    agent.messages = _seeded_messages()

    await agent._inject_deferred_memory()

    injected = agent.messages[1]["content"]
    assert "No weight field" in injected, (
        f"Expected result without 'weight' key to default to 1.0 and pass "
        f"min_weight=0.9 filter: {injected!r}"
    )
    assert "Has weight field" not in injected, (
        f"Expected result with weight=0.5 to be filtered by min_weight=0.9: "
        f"{injected!r}"
    )
