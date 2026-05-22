"""Integration tests for ask_user tool with astep_stream."""

from types import SimpleNamespace

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.events import QuestionAsked, ToolResultEvent, StreamComplete
from fipsagents.baseagent.tools.question import make_question_tool
from fipsagents.baseagent.tools import ToolRegistry, tool


@tool(description="Echo input", visibility="both")
async def echo(msg: str) -> str:
    return f"echo:{msg}"


def _tc_delta(index, *, call_id=None, name=None, arguments=None):
    """Build a SimpleNamespace tool-call delta."""
    fn = SimpleNamespace(
        name=name,
        arguments=arguments or "",
    )
    return SimpleNamespace(index=index, id=call_id, function=fn)


def _chunk(*, tool_calls=None, content=None, finish_reason=None):
    """Build a SimpleNamespace streaming chunk."""
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
    def __init__(self, *, llm, extra_tools=None):
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

        self.tools.register(make_question_tool(self))
        if extra_tools:
            for t in extra_tools:
                self.tools.register(t)

    async def _inject_deferred_memory(self):
        return None


ASK_ARGS = '{"prompt": "Which?", "options": [{"label": "A"}, {"label": "B"}]}'


@pytest.mark.asyncio
async def test_astep_stream_question_pauses():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_123", name="ask_user")]),
            _chunk(tool_calls=[_tc_delta(0, arguments=ASK_ARGS)]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    question_events = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(question_events) == 1
    q = question_events[0]
    assert q.question_text == "Which?"
    assert len(q.options) == 2
    assert q.options[0]["label"] == "A"

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].name == "ask_user"

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "question"

    assert agent._question_pending is not None
    assert agent._question_pending["tool_call_id"] == "call_123"

    assistant_msgs = [m for m in agent.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert "tool_calls" in assistant_msgs[0]

    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_123"


@pytest.mark.asyncio
async def test_astep_stream_question_stamps_tool_call_id():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_xyz_999", name="ask_user")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"prompt": "Confirm?", "options": [{"label": "Yes"}, {"label": "No"}]}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    assert agent._question_pending is not None
    assert agent._question_pending["tool_call_id"] == "call_xyz_999"
    assert [e for e in events if isinstance(e, StreamComplete)][0].finish_reason == "question"


@pytest.mark.asyncio
async def test_astep_stream_multi_tool_with_question():
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="call_echo", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hello"}')]),
            _chunk(tool_calls=[_tc_delta(1, call_id="call_q", name="ask_user")]),
            _chunk(tool_calls=[_tc_delta(1, arguments='{"prompt": "Choose", "options": [{"label": "X"}, {"label": "Y"}]}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2

    echo_result = [e for e in tool_results if e.name == "echo"][0]
    assert "echo:hello" in echo_result.content

    question_result = [e for e in tool_results if e.name == "ask_user"][0]
    assert question_result.call_id == "call_q"

    assert [e for e in events if isinstance(e, StreamComplete)][0].finish_reason == "question"

    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert {m["tool_call_id"] for m in tool_msgs} == {"call_echo", "call_q"}

    assert agent._question_pending is not None
    assert agent._question_pending["tool_call_id"] == "call_q"
