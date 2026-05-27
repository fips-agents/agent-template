"""Tests for the ``spawn_agent`` tool factory.

Verifies :func:`make_spawn_tool` and the internal ``_spawn`` / ``_build_child_registry``:
validation, depth cap, tool subset, event lifecycle, token rollup, tool metadata,
and defensive behaviour when parent attributes are absent.
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from fipsagents.baseagent.config import AgentConfig, SpawnConfig
from fipsagents.baseagent.events import (
    SpawnAgentCompleted,
    SpawnAgentFailed,
    SpawnAgentInvoked,
)
from fipsagents.baseagent.tools import _TOOL_MARKER, tool
from fipsagents.baseagent.tools._registry import ToolRegistry
from fipsagents.baseagent.tools.spawn import (
    _ROLE_RE,
    _build_child_registry,
    _spawn,
    make_spawn_tool,
)
from fipsagents.subagents.types import MaxDelegationDepthError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parent(
    spawn_config: SpawnConfig | None = None,
    tools: ToolRegistry | None = None,
    delegation_depth: int = 0,
) -> types.SimpleNamespace:
    config = AgentConfig(spawn=spawn_config or SpawnConfig())
    return types.SimpleNamespace(
        config=config, tools=tools or ToolRegistry(),
        _subagent_events=[], _subagent_token_usage=[],
        _delegation_depth=delegation_depth,
    )


def _registry_with_tools(*names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for name in names:
        @tool(description=f"Dummy {name}", visibility="llm_only", name=name)
        async def _fn(**kw: object) -> str:
            return "ok"
        reg.register(_fn)
    return reg


def _chunk(content: str | None, finish: str | None = None,
           prompt_tok: int = 15, compl_tok: int = 8):
    delta = types.SimpleNamespace(
        content=content, reasoning_content=None, tool_calls=None,
    )
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish)
    usage = None
    if finish:
        usage = types.SimpleNamespace(
            prompt_tokens=prompt_tok, completion_tokens=compl_tok,
            total_tokens=prompt_tok + compl_tok,
        )
    return types.SimpleNamespace(
        choices=[choice], usage=usage, id="chatcmpl-test", model="test-model",
    )


@pytest.fixture()
def mock_openai(monkeypatch):
    """Patch AsyncOpenAI so LLMClient works without a real endpoint."""
    async def fake_create(**kwargs):
        class Stream:
            def __init__(self):
                self._chunks = [
                    _chunk("Hello from child"),
                    _chunk(None, finish="stop"),
                ]
                self._i = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._i >= len(self._chunks):
                    raise StopAsyncIteration
                c = self._chunks[self._i]
                self._i += 1
                return c
        return Stream()

    mock_cls = MagicMock()
    inst = MagicMock()
    inst.chat.completions.create = AsyncMock(side_effect=fake_create)
    mock_cls.return_value = inst
    monkeypatch.setattr("fipsagents.baseagent.llm.AsyncOpenAI", mock_cls)
    return inst


# Shorthand — most tests share the same _spawn signature tail.
_S = dict(system_prompt="s", task="t", tools=None, model=None, max_iterations=None)


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.asyncio
    async def test_invalid_role_name_raises(self):
        with pytest.raises(ValueError, match="Invalid role name"):
            await _spawn(_make_parent(), role="bad name!", **_S)

    @pytest.mark.asyncio
    async def test_role_with_dashes_raises(self):
        with pytest.raises(ValueError, match="Invalid role name"):
            await _spawn(_make_parent(), role="role-with-dashes", **_S)

    @pytest.mark.asyncio
    async def test_role_must_start_with_letter(self):
        with pytest.raises(ValueError, match="Invalid role name"):
            await _spawn(_make_parent(), role="2helper", **_S)

    @pytest.mark.asyncio
    async def test_role_name_with_numbers_ok(self, mock_openai):
        r = json.loads(await _spawn(_make_parent(), role="helper2", **{**_S, "max_iterations": 1}))
        assert r["agent_name"] == "helper2"

    @pytest.mark.asyncio
    async def test_underscore_role_name_ok(self, mock_openai):
        r = json.loads(await _spawn(_make_parent(), role="my_helper", **{**_S, "max_iterations": 1}))
        assert r["agent_name"] == "my_helper"

    @pytest.mark.asyncio
    async def test_spawn_disabled_raises(self):
        agent = _make_parent(spawn_config=SpawnConfig(enabled=False))
        with pytest.raises(ValueError, match="disabled"):
            await _spawn(agent, role="helper", **_S)

    @pytest.mark.asyncio
    async def test_model_not_in_allowed_list_raises(self):
        agent = _make_parent(spawn_config=SpawnConfig(allowed_models=["gpt-4o"]))
        with pytest.raises(ValueError, match="not in allowed_models"):
            await _spawn(agent, role="helper", system_prompt="s", task="t",
                         tools=None, model="llama-70b", max_iterations=None)

    @pytest.mark.asyncio
    async def test_model_allowed_when_none(self, mock_openai):
        r = json.loads(await _spawn(
            _make_parent(), role="helper", system_prompt="s", task="t",
            tools=None, model="anything", max_iterations=1))
        assert r["agent_name"] == "helper"

    @pytest.mark.asyncio
    async def test_model_allowed_when_in_list(self, mock_openai):
        agent = _make_parent(spawn_config=SpawnConfig(allowed_models=["gpt-4o", "llama"]))
        r = json.loads(await _spawn(
            agent, role="helper", system_prompt="s", task="t",
            tools=None, model="gpt-4o", max_iterations=1))
        assert r["agent_name"] == "helper"


# ---------------------------------------------------------------------------
# Tests: Depth cap
# ---------------------------------------------------------------------------

class TestDepthCap:
    @pytest.mark.asyncio
    async def test_depth_exceeded_raises(self):
        agent = _make_parent(spawn_config=SpawnConfig(max_depth=3), delegation_depth=3)
        with pytest.raises(MaxDelegationDepthError):
            await _spawn(agent, role="helper", **_S)

    @pytest.mark.asyncio
    async def test_depth_at_limit_succeeds(self, mock_openai):
        agent = _make_parent(spawn_config=SpawnConfig(max_depth=3), delegation_depth=2)
        r = json.loads(await _spawn(agent, role="helper", **{**_S, "max_iterations": 1}))
        assert r["agent_name"] == "helper"

    @pytest.mark.asyncio
    async def test_depth_failed_event_emitted(self):
        agent = _make_parent(spawn_config=SpawnConfig(max_depth=2), delegation_depth=2)
        with pytest.raises(MaxDelegationDepthError):
            await _spawn(agent, role="researcher", **_S)
        ev = agent._subagent_events
        assert len(ev) == 1
        assert isinstance(ev[0], SpawnAgentFailed)
        assert ev[0].error_type == "MaxDelegationDepthError"
        assert ev[0].role == "researcher"
        assert ev[0].span_id.startswith("spawn-")


# ---------------------------------------------------------------------------
# Tests: _build_child_registry
# ---------------------------------------------------------------------------

class TestBuildChildRegistry:
    def test_empty_names_gives_empty_registry(self):
        assert _build_child_registry(_registry_with_tools("a", "b"), []).get_all() == []

    def test_copies_named_tools(self):
        child = _build_child_registry(_registry_with_tools("a", "b", "c"), ["a", "c"])
        assert {t.name for t in child.get_all()} == {"a", "c"}

    def test_raises_on_unknown_tool(self):
        with pytest.raises(ValueError, match="nonexistent"):
            _build_child_registry(_registry_with_tools("a"), ["nonexistent"])

    def test_error_lists_available_tools(self):
        with pytest.raises(ValueError, match="alpha") as exc_info:
            _build_child_registry(_registry_with_tools("alpha", "beta"), ["missing"])
        assert "beta" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: Tool subset via _spawn
# ---------------------------------------------------------------------------

class TestToolSubset:
    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="nonexistent"):
            await _spawn(_make_parent(), role="helper", system_prompt="s",
                         task="t", tools=["nonexistent"], model=None, max_iterations=None)

    @pytest.mark.asyncio
    async def test_valid_tool_subset_accepted(self, mock_openai):
        agent = _make_parent(tools=_registry_with_tools("calc"))
        r = json.loads(await _spawn(agent, role="helper", system_prompt="s",
                                    task="t", tools=["calc"], model=None, max_iterations=1))
        assert r["agent_name"] == "helper"

    @pytest.mark.asyncio
    async def test_empty_tools_list(self, mock_openai):
        agent = _make_parent(tools=_registry_with_tools("calc"))
        r = json.loads(await _spawn(agent, role="helper", system_prompt="s",
                                    task="t", tools=[], model=None, max_iterations=1))
        assert r["agent_name"] == "helper"

    @pytest.mark.asyncio
    async def test_none_tools_gives_no_tools(self, mock_openai):
        agent = _make_parent(tools=_registry_with_tools("calc"))
        r = json.loads(await _spawn(agent, role="helper", **{**_S, "max_iterations": 1}))
        assert r["agent_name"] == "helper"


# ---------------------------------------------------------------------------
# Tests: Event sequence
# ---------------------------------------------------------------------------

class TestEventSequence:
    @pytest.mark.asyncio
    async def test_invoked_event_fields(self, mock_openai):
        agent = _make_parent(delegation_depth=1)
        await _spawn(agent, role="researcher", system_prompt="Be helpful",
                     task="Do research", tools=None, model="custom-model", max_iterations=1)
        inv = agent._subagent_events[0]
        assert isinstance(inv, SpawnAgentInvoked)
        assert (inv.role, inv.task, inv.tools, inv.model, inv.depth) == (
            "researcher", "Do research", [], "custom-model", 2)

    @pytest.mark.asyncio
    async def test_completed_event_on_success(self, mock_openai):
        agent = _make_parent()
        await _spawn(agent, role="helper", **{**_S, "max_iterations": 1})
        ev = agent._subagent_events
        assert len(ev) == 2
        assert isinstance(ev[0], SpawnAgentInvoked)
        assert isinstance(ev[1], SpawnAgentCompleted)
        assert ev[1].role == "helper"
        assert ev[1].content == "Hello from child"

    @pytest.mark.asyncio
    async def test_failed_event_on_error(self):
        agent = _make_parent(spawn_config=SpawnConfig(max_depth=1), delegation_depth=1)
        with pytest.raises(MaxDelegationDepthError):
            await _spawn(agent, role="helper", **_S)
        assert len(agent._subagent_events) == 1
        assert isinstance(agent._subagent_events[0], SpawnAgentFailed)

    @pytest.mark.asyncio
    async def test_span_id_consistent(self, mock_openai):
        agent = _make_parent()
        raw = await _spawn(agent, role="helper", **{**_S, "max_iterations": 1})
        inv, comp = agent._subagent_events
        assert inv.span_id.startswith("spawn-")
        assert inv.span_id == comp.span_id
        assert json.loads(raw)["span_id"] == inv.span_id


# ---------------------------------------------------------------------------
# Tests: Token rollup
# ---------------------------------------------------------------------------

class TestTokenRollup:
    @pytest.mark.asyncio
    async def test_token_usage_appended_on_success(self, mock_openai):
        agent = _make_parent()
        await _spawn(agent, role="helper", **{**_S, "max_iterations": 1})
        assert len(agent._subagent_token_usage) == 1
        tok = agent._subagent_token_usage[0]
        assert (tok["input"], tok["output"], tok["cached"]) == (15, 8, 0)


# ---------------------------------------------------------------------------
# Tests: Tool metadata
# ---------------------------------------------------------------------------

class TestToolMetadata:
    def _meta(self):
        return getattr(make_spawn_tool(_make_parent()), _TOOL_MARKER)

    def test_tool_name(self):
        assert self._meta().name == "spawn_agent"

    def test_tool_visibility(self):
        assert self._meta().visibility == "llm_only"

    def test_tool_description(self):
        assert "ephemeral" in self._meta().description.lower()

    def test_tool_parameters(self):
        props = set(self._meta().parameters.get("properties", {}).keys())
        for p in ("role", "task", "system_prompt", "tools", "model", "max_iterations"):
            assert p in props


# ---------------------------------------------------------------------------
# Tests: Defensive — missing parent attributes
# ---------------------------------------------------------------------------

class TestDefensive:
    @pytest.mark.asyncio
    async def test_no_crash_when_events_missing(self, mock_openai):
        agent = types.SimpleNamespace(
            config=AgentConfig(), tools=ToolRegistry(),
            _subagent_token_usage=[], _delegation_depth=0,
        )
        r = json.loads(await _spawn(agent, role="helper", **{**_S, "max_iterations": 1}))
        assert r["agent_name"] == "helper"

    @pytest.mark.asyncio
    async def test_no_crash_when_token_usage_missing(self, mock_openai):
        agent = types.SimpleNamespace(
            config=AgentConfig(), tools=ToolRegistry(),
            _subagent_events=[], _delegation_depth=0,
        )
        r = json.loads(await _spawn(agent, role="helper", **{**_S, "max_iterations": 1}))
        assert r["agent_name"] == "helper"

    @pytest.mark.asyncio
    async def test_raises_when_config_missing(self):
        agent = types.SimpleNamespace(
            tools=ToolRegistry(), _subagent_events=[],
            _subagent_token_usage=[], _delegation_depth=0,
        )
        with pytest.raises(ValueError, match="no config"):
            await _spawn(agent, role="helper", **_S)


# ---------------------------------------------------------------------------
# Tests: Role name regex (data-driven)
# ---------------------------------------------------------------------------

class TestRoleRegex:
    @pytest.mark.parametrize("name", ["helper", "Helper", "myAgent2", "agent_v3", "A"])
    def test_valid(self, name: str):
        assert _ROLE_RE.match(name) is not None

    @pytest.mark.parametrize("name", ["2x", "bad name", "a-b", "a.b", "", " ", "x!"])
    def test_invalid(self, name: str):
        assert _ROLE_RE.match(name) is None
