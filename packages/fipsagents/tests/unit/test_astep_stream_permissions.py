"""Tests for per-tool permission policy in astep_stream."""

from types import SimpleNamespace

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.events import (
    PermissionDecisionMade,
    QuestionAsked,
    StreamComplete,
    ToolResultEvent,
)
from fipsagents.baseagent.tools import ToolRegistry, tool
from fipsagents.server.permissions import (
    PermissionRule,
    StaticPermissionSource,
)


@tool(description="Echo input", visibility="both")
async def echo(msg: str) -> str:
    return f"echo:{msg}"


@tool(description="Dangerous tool", visibility="both")
async def dangerous(cmd: str) -> str:
    return f"ran:{cmd}"


def _tc_delta(index, *, call_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments or "")
    return SimpleNamespace(index=index, id=call_id, function=fn)


def _chunk(*, tool_calls=None, content=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


class _StubLLM:
    def __init__(self, turns):
        self._turns = list(turns)
        self._idx = 0

    async def call_model_stream_raw(self, messages, tools=None, **kw):
        turn = self._turns[self._idx]
        self._idx += 1
        for c in turn:
            yield c


class _StubAgent(BaseAgent):
    def __init__(self, *, llm, permission_source=None, permission_mode="enforce",
                 extra_tools=None):
        self.llm = llm
        self.config = None
        self.messages = []
        self.tools = ToolRegistry()
        self._question_pending = None
        self._question_events = []
        self._subagent_events = []
        self._subagent_token_usage = []
        self._delegation_depth = 0
        self._inbound_auth_header = None
        self._reasoning_parser = None
        self._permission_source = permission_source
        self._permission_mode = permission_mode
        self._permission_preapproved = set()

        if extra_tools:
            for t in extra_tools:
                self.tools.register(t)

    async def _inject_deferred_memory(self):
        return None


@pytest.mark.asyncio
async def test_allow_executes_normally():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="echo", action="allow")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is False
    assert "echo:hello" in tool_results[0].content


@pytest.mark.asyncio
async def test_deny_returns_error():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Understood"),
            _chunk(finish_reason="stop"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="echo", action="deny")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is True
    assert "DENIED" in tool_results[0].content
    assert "echo:hello" not in tool_results[0].content


@pytest.mark.asyncio
async def test_deny_emits_permission_event():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="OK"),
            _chunk(finish_reason="stop"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="echo", action="deny")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    perm_events = [e for e in events if isinstance(e, PermissionDecisionMade)]
    assert len(perm_events) == 1
    assert perm_events[0].action == "deny"


@pytest.mark.asyncio
async def test_allow_emits_permission_event():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="echo", action="allow")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    perm_events = [e for e in events if isinstance(e, PermissionDecisionMade)]
    assert len(perm_events) == 1
    assert perm_events[0].action == "allow"


@pytest.mark.asyncio
async def test_ask_pauses_with_question():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="echo", action="ask")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    question_events = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(question_events) == 1
    assert "requires approval" in question_events[0].question_text

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "question"

    assert agent._question_pending is not None
    assert agent._question_pending.get("permission_ask") is True
    assert agent._question_pending["tool_name"] == "echo"
    assert agent._question_pending["tool_args"] == {"msg": "hello"}


@pytest.mark.asyncio
async def test_no_source_allows_all():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent = _StubAgent(llm=llm, permission_source=None, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is False

    perm_events = [e for e in events if isinstance(e, PermissionDecisionMade)]
    assert len(perm_events) == 0


@pytest.mark.asyncio
async def test_observe_mode_logs_no_block():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="echo", action="deny")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, permission_mode="observe",
                       extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is False
    assert "echo:hello" in tool_results[0].content

    perm_events = [e for e in events if isinstance(e, PermissionDecisionMade)]
    assert len(perm_events) == 1
    assert perm_events[0].action == "deny"


@pytest.mark.asyncio
async def test_multiple_tools_deny_one():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_echo", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(tool_calls=[_tc_delta(1, call_id="call_danger", name="dangerous")]),
            _chunk(tool_calls=[_tc_delta(1, arguments='{"cmd": "rm -rf"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="dangerous", action="deny")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, extra_tools=[echo, dangerous])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2

    echo_result = [e for e in tool_results if e.name == "echo"][0]
    assert echo_result.is_error is False
    assert "echo:hello" in echo_result.content

    danger_result = [e for e in tool_results if e.name == "dangerous"][0]
    assert danger_result.is_error is True
    assert "DENIED" in danger_result.content


@pytest.mark.asyncio
async def test_preapproved_skips_check():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="*", action="deny")]
    )
    agent = _StubAgent(llm=llm, permission_source=src, extra_tools=[echo])
    agent._permission_preapproved.add("call_123")

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is False
    assert "echo:hello" in tool_results[0].content

    perm_events = [e for e in events if isinstance(e, PermissionDecisionMade)]
    assert len(perm_events) == 0
