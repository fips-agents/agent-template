"""Tests for BaseAgent.build_memory_prefix() and setup() prefix injection.

Covers the memory-prefix slot added by #47: default behavior across backends,
subclass override, truncation, role configuration, and positional stability.
"""

from __future__ import annotations

from typing import Any

import pytest

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.config import AgentConfig, MemoryConfig
from fipsagents.baseagent.memory import MemoryClientBase, NullMemoryClient


# ---------------------------------------------------------------------------
# Test double — in-memory backend returning canned search results
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


# ---------------------------------------------------------------------------
# Minimal agent factory — bypasses full setup()
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    memory: MemoryClientBase | None = None,
    prefix_role: str = "system",
    max_prefix_chars: int = 8000,
) -> BaseAgent:
    """Build a BaseAgent with enough wiring for prefix tests.

    Bypasses full setup() — sets config, memory, and stub subsystems
    directly so build_memory_prefix() and build_system_prompt() work
    without a filesystem, LLM, or network connection.
    """
    config = AgentConfig(
        model={"name": "test-model", "endpoint": "http://localhost:1234/v1"},
        memory=MemoryConfig(
            backend="null",
            prefix_role=prefix_role,
            max_prefix_chars=max_prefix_chars,
        ),
    )

    class _PrefixTestAgent(BaseAgent):
        def __init__(self) -> None:
            # Minimal init — skip config_path resolution
            self.config = config
            self.memory: MemoryClientBase = memory or NullMemoryClient()
            self.messages: list[dict[str, Any]] = []
            # Lightweight stubs for subsystems used by build_system_prompt()
            self.prompts = _stub_prompts()
            self.rules = _stub_rules()
            self.skills = _stub_skills()

    return _PrefixTestAgent()


# -- Subsystem stubs ----------------------------------------------------------


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


def _stub_prompts() -> _PromptStub:
    return _PromptStub()


def _stub_rules() -> _RulesStub:
    return _RulesStub()


def _stub_skills() -> _SkillsStub:
    return _SkillsStub()


# ---------------------------------------------------------------------------
# Tests — build_memory_prefix() core behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_backend_returns_none() -> None:
    """NullMemoryClient produces no results, so prefix must be None."""
    agent = _make_agent()
    result = await agent.build_memory_prefix()
    assert result is None


@pytest.mark.asyncio
async def test_default_prefix_joins_content() -> None:
    """Multiple results are joined with the standard separator."""
    agent = _make_agent(
        memory=FakeMemoryClient(
            [
                {"id": "a", "content": "Memory one"},
                {"id": "b", "content": "Memory two"},
                {"id": "c", "content": "Memory three"},
            ]
        )
    )
    result = await agent.build_memory_prefix()
    assert result == "Memory one\n\n---\n\nMemory two\n\n---\n\nMemory three"


@pytest.mark.asyncio
async def test_blank_content_entries_skipped() -> None:
    """Entries with empty content strings are dropped from the prefix."""
    agent = _make_agent(
        memory=FakeMemoryClient(
            [
                {"id": "a", "content": "Useful"},
                {"id": "b", "content": ""},
                {"id": "c", "content": "Also useful"},
            ]
        )
    )
    result = await agent.build_memory_prefix()
    assert result == "Useful\n\n---\n\nAlso useful"


@pytest.mark.asyncio
async def test_results_without_content_key_skipped() -> None:
    """Results missing the 'content' key entirely are silently ignored."""
    agent = _make_agent(
        memory=FakeMemoryClient(
            [
                {"id": "a", "content": "Has content"},
                {"id": "b", "stub": "Only a stub"},
            ]
        )
    )
    result = await agent.build_memory_prefix()
    assert result == "Has content"


@pytest.mark.asyncio
async def test_whitespace_only_content_skipped() -> None:
    """Entries with whitespace-only content are treated as blank."""
    agent = _make_agent(
        memory=FakeMemoryClient(
            [
                {"id": "a", "content": "Good"},
                {"id": "b", "content": "   "},
                {"id": "c", "content": "\n\t"},
            ]
        )
    )
    result = await agent.build_memory_prefix()
    assert result == "Good"


@pytest.mark.asyncio
async def test_all_empty_content_returns_none() -> None:
    """When every result has empty content the prefix must be None, not an empty string."""
    agent = _make_agent(
        memory=FakeMemoryClient(
            [{"id": "a", "content": ""}, {"id": "b", "content": ""}]
        )
    )
    result = await agent.build_memory_prefix()
    assert result is None


# ---------------------------------------------------------------------------
# Tests — truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_prefix_chars_truncation() -> None:
    """Content exceeding max_prefix_chars is cut and tagged as truncated."""
    agent = _make_agent(
        memory=FakeMemoryClient([{"id": "a", "content": "x" * 100}]),
        max_prefix_chars=50,
    )
    result = await agent.build_memory_prefix()
    assert result is not None
    assert result == "x" * 50 + "\n\n… [truncated]"


@pytest.mark.asyncio
async def test_max_prefix_chars_zero_disables_limit() -> None:
    """Setting max_prefix_chars=0 disables truncation entirely."""
    content = "y" * 20_000
    agent = _make_agent(
        memory=FakeMemoryClient([{"id": "a", "content": content}]),
        max_prefix_chars=0,
    )
    result = await agent.build_memory_prefix()
    assert result is not None
    assert len(result) == 20_000
    assert result == content


# ---------------------------------------------------------------------------
# Tests — subclass override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subclass_override_returns_custom_string() -> None:
    """Subclasses can return any string from build_memory_prefix."""
    base = _make_agent(
        memory=FakeMemoryClient([{"id": "a", "content": "default would be this"}])
    )

    class _CustomPrefixAgent(base.__class__):
        async def build_memory_prefix(self) -> str | None:
            return "custom prefix"

    agent = _CustomPrefixAgent()
    result = await agent.build_memory_prefix()
    assert result == "custom prefix"


@pytest.mark.asyncio
async def test_subclass_override_returns_none() -> None:
    """Subclasses can suppress the prefix entirely by returning None."""
    base = _make_agent(
        memory=FakeMemoryClient([{"id": "a", "content": "default would be this"}])
    )

    class _NoPrefixAgent(base.__class__):
        async def build_memory_prefix(self) -> str | None:
            return None

    agent = _NoPrefixAgent()
    result = await agent.build_memory_prefix()
    assert result is None


# ---------------------------------------------------------------------------
# Tests — setup() wiring (step-10 simulation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_injects_system_and_prefix() -> None:
    """When memories are available, setup injects system prompt then prefix."""
    agent = _make_agent(
        memory=FakeMemoryClient([{"id": "a", "content": "mem"}])
    )

    # Simulate setup step 10: system prompt + optional prefix.
    agent.messages.append({"role": "system", "content": agent.build_system_prompt()})
    prefix = await agent.build_memory_prefix()
    if prefix:
        agent.messages.append(
            {"role": agent.config.memory.prefix_role, "content": prefix}
        )

    assert len(agent.messages) == 2, (
        f"Expected 2 messages (system + prefix), got {len(agent.messages)}: "
        f"{agent.messages}"
    )
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[1]["role"] == "system"
    assert agent.messages[1]["content"] == "mem"


@pytest.mark.asyncio
async def test_setup_skips_prefix_when_no_memories() -> None:
    """With NullMemoryClient, setup produces only the system prompt message."""
    agent = _make_agent()

    agent.messages.append({"role": "system", "content": agent.build_system_prompt()})
    prefix = await agent.build_memory_prefix()
    assert prefix is None

    assert len(agent.messages) == 1, (
        f"Expected 1 message (system only), got {len(agent.messages)}: "
        f"{agent.messages}"
    )


@pytest.mark.asyncio
async def test_developer_role_config() -> None:
    """prefix_role='developer' causes the prefix message to use that role."""
    agent = _make_agent(
        memory=FakeMemoryClient([{"id": "a", "content": "mem"}]),
        prefix_role="developer",
    )

    agent.messages.append({"role": "system", "content": agent.build_system_prompt()})
    prefix = await agent.build_memory_prefix()
    if prefix:
        agent.messages.append(
            {"role": agent.config.memory.prefix_role, "content": prefix}
        )

    assert len(agent.messages) == 2, (
        f"Expected 2 messages, got {len(agent.messages)}: {agent.messages}"
    )
    assert agent.messages[1]["role"] == "developer"


# ---------------------------------------------------------------------------
# Tests — positional stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefix_stability_across_turns() -> None:
    """The prefix at index 1 must remain untouched as the conversation grows."""
    agent = _make_agent(
        memory=FakeMemoryClient([{"id": "a", "content": "stable mem"}])
    )

    agent.messages.append({"role": "system", "content": agent.build_system_prompt()})
    prefix = await agent.build_memory_prefix()
    assert prefix == "stable mem"
    agent.messages.append(
        {"role": agent.config.memory.prefix_role, "content": prefix}
    )

    # Simulate 3 conversation turns.
    for i in range(3):
        agent.messages.append({"role": "user", "content": f"Turn {i}"})
        agent.messages.append({"role": "assistant", "content": f"Reply {i}"})

    assert agent.messages[1]["content"] == "stable mem", (
        "Memory prefix at index 1 was mutated or displaced after conversation turns"
    )
    assert len(agent.messages) == 8, (
        f"Expected 8 messages (system + prefix + 3*(user+assistant)), "
        f"got {len(agent.messages)}"
    )


# ---------------------------------------------------------------------------
# Tests — pattern-aware build_memory_prefix() behavior
# ---------------------------------------------------------------------------


class _FakeMemoryLoading:
    """Minimal stand-in for the SDK's MemoryLoading config."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern


class _FakeProjectConfig:
    """Minimal stand-in for the MemoryHub SDK's ProjectConfig."""

    def __init__(self, *, pattern: str = "eager", project_id: str | None = None) -> None:
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


@pytest.mark.asyncio
async def test_eager_pattern_calls_search_with_project_id() -> None:
    """Eager pattern with a project_id passes query='', mode='index',
    max_results=50, and project_id to search."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "mem one"}, {"id": "b", "content": "mem two"}],
        project_config=_FakeProjectConfig(pattern="eager", project_id="test-project"),
    )
    agent = _make_agent(memory=mem)
    result = await agent.build_memory_prefix()

    assert result == "mem one\n\n---\n\nmem two"
    assert len(mem.search_calls) == 1
    query, kwargs = mem.search_calls[0]
    assert query == ""
    assert kwargs["mode"] == "index"
    assert kwargs["max_results"] == 50
    assert kwargs["project_id"] == "test-project"


@pytest.mark.asyncio
async def test_eager_pattern_without_project_id() -> None:
    """Eager pattern with project_id=None omits project_id from search kwargs."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "data"}],
        project_config=_FakeProjectConfig(pattern="eager", project_id=None),
    )
    agent = _make_agent(memory=mem)
    result = await agent.build_memory_prefix()

    assert result == "data"
    assert len(mem.search_calls) == 1
    _, kwargs = mem.search_calls[0]
    assert "project_id" not in kwargs
    assert kwargs["mode"] == "index"
    assert kwargs["max_results"] == 50


@pytest.mark.parametrize(
    "pattern",
    ["lazy", "lazy_with_rebias", "jit"],
    ids=["lazy", "lazy_with_rebias", "jit"],
)
@pytest.mark.asyncio
async def test_deferred_pattern_returns_none(pattern: str) -> None:
    """Non-eager patterns (lazy, lazy_with_rebias, jit) return None
    without calling search — memory loading is deferred to post-turn."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "should not appear"}],
        project_config=_FakeProjectConfig(pattern=pattern, project_id="proj"),
    )
    agent = _make_agent(memory=mem)
    result = await agent.build_memory_prefix()

    assert result is None
    assert len(mem.search_calls) == 0, (
        f"search() should not be called for pattern={pattern!r}"
    )


@pytest.mark.asyncio
async def test_fallback_when_no_project_config() -> None:
    """When project_config is None (non-MemoryHub backend), search is
    called with query='' — the legacy fallback path (empty query returns all)."""
    mem = FakeMemoryHubClient(
        [{"id": "a", "content": "fallback data"}],
        project_config=None,
    )
    agent = _make_agent(memory=mem)
    result = await agent.build_memory_prefix()

    assert result == "fallback data"
    assert len(mem.search_calls) == 1
    query, kwargs = mem.search_calls[0]
    assert query == ""
    assert "mode" not in kwargs
    assert "project_id" not in kwargs
