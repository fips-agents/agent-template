"""Tests for doom-loop detection in astep_stream."""

from types import SimpleNamespace

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.config import LoopConfig, LoopGuardConfig
from fipsagents.baseagent.events import (
    LoopBreakEvent,
    StreamComplete,
    ToolResultEvent,
)
from fipsagents.baseagent.tools import ToolRegistry, tool


@tool(description="Fetch data", visibility="both")
async def fetch(url: str) -> str:
    return f"data:{url}"


@tool(description="Store data", visibility="both")
async def store(key: str, value: str) -> str:
    return f"stored:{key}={value}"


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
    """Yields a different set of chunks on each successive model call."""

    def __init__(self, turns):
        self._turns = list(turns)
        self._idx = 0

    async def call_model_stream_raw(self, messages, tools=None, **kw):
        turn = self._turns[self._idx % len(self._turns)]
        self._idx += 1
        for c in turn:
            yield c


class _StubAgent(BaseAgent):
    def __init__(self, *, llm, guard_cfg=None, extra_tools=None):
        self.llm = llm
        self.messages = []
        self.tools = ToolRegistry()
        self._question_pending = None
        self._question_events = []
        self._subagent_events = []
        self._subagent_token_usage = []
        self._delegation_depth = 0
        self._inbound_auth_header = None
        self._reasoning_parser = None
        self._permission_source = None
        self._permission_mode = "enforce"
        self._permission_preapproved = set()

        loop_cfg = LoopConfig(guard=guard_cfg or LoopGuardConfig())
        self.config = SimpleNamespace(
            loop=loop_cfg,
            tools=SimpleNamespace(enabled=True),
            memory=SimpleNamespace(
                loading_pattern="eager",
                injection_mode="prefix",
                injection_tag="user_memories",
                max_prefix_chars=8000,
                max_results=50,
                min_weight=0.0,
                prefix_role="system",
            ),
        )

        if extra_tools:
            for t in extra_tools:
                self.tools.register(t)

    async def _inject_deferred_memory(self):
        return None


def _make_tool_turn(call_id, name, args_json):
    """Build a single-tool-call turn (open + args + finish)."""
    return [
        _chunk(tool_calls=[_tc_delta(0, call_id=call_id, name=name)]),
        _chunk(tool_calls=[_tc_delta(0, arguments=args_json)]),
        _chunk(finish_reason="tool_calls"),
    ]


def _make_content_turn(text):
    return [
        _chunk(content=text),
        _chunk(finish_reason="stop"),
    ]


@pytest.mark.asyncio
async def test_repeat_detection_fires():
    """Same tool+args called 3 times (default threshold) triggers LoopBreakEvent."""
    same_turn = _make_tool_turn("call_1", "fetch", '{"url": "http://x.com"}')
    llm = _StubLLM([same_turn])
    agent = _StubAgent(llm=llm, extra_tools=[fetch])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    loop_breaks = [e for e in events if isinstance(e, LoopBreakEvent)]
    assert len(loop_breaks) == 1
    assert loop_breaks[0].tool_name == "fetch"
    assert loop_breaks[0].repeat_count == 3
    assert loop_breaks[0].last_args == {"url": "http://x.com"}
    assert loop_breaks[0].last_error is None

    completes = [e for e in events if isinstance(e, StreamComplete)]
    assert len(completes) == 1
    assert completes[0].finish_reason == "loop_break"


@pytest.mark.asyncio
async def test_below_threshold_no_break():
    """Same tool called only 2 times (under default threshold of 3) does not trigger."""
    same_turn = _make_tool_turn("call_1", "fetch", '{"url": "http://x.com"}')
    content_turn = _make_content_turn("Done")
    llm = _StubLLM([same_turn, same_turn, content_turn])
    # Override so it returns turns in sequence, not cycling.
    llm._turns = [same_turn, same_turn, content_turn]

    class _SeqLLM:
        def __init__(self, turns):
            self._turns = list(turns)
            self._idx = 0
        async def call_model_stream_raw(self, messages, tools=None, **kw):
            turn = self._turns[self._idx]
            self._idx += 1
            for c in turn:
                yield c

    llm = _SeqLLM([same_turn, same_turn, content_turn])
    agent = _StubAgent(llm=llm, extra_tools=[fetch])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    loop_breaks = [e for e in events if isinstance(e, LoopBreakEvent)]
    assert len(loop_breaks) == 0

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2

    completes = [e for e in events if isinstance(e, StreamComplete)]
    assert completes[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_window_sizing():
    """With window=5: calls A, A, B, B, B -> B triggers at 3, A does not (only 2 in window)."""

    class _SeqLLM:
        def __init__(self, turns):
            self._turns = list(turns)
            self._idx = 0
        async def call_model_stream_raw(self, messages, tools=None, **kw):
            turn = self._turns[self._idx]
            self._idx += 1
            for c in turn:
                yield c

    turn_a = _make_tool_turn("call_a", "fetch", '{"url": "http://a.com"}')
    turn_b = _make_tool_turn("call_b", "store", '{"key": "k", "value": "v"}')

    llm = _SeqLLM([turn_a, turn_a, turn_b, turn_b, turn_b])
    guard_cfg = LoopGuardConfig(repeat_threshold=3, pattern_window=5)
    agent = _StubAgent(llm=llm, guard_cfg=guard_cfg, extra_tools=[fetch, store])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    loop_breaks = [e for e in events if isinstance(e, LoopBreakEvent)]
    assert len(loop_breaks) == 1
    assert loop_breaks[0].tool_name == "store"
    assert loop_breaks[0].repeat_count == 3


@pytest.mark.asyncio
async def test_guard_disabled():
    """When guard is disabled, no LoopBreakEvent even with many repeats."""
    same_turn = _make_tool_turn("call_1", "fetch", '{"url": "http://x.com"}')
    content_turn = _make_content_turn("Done")

    class _SeqLLM:
        def __init__(self, turns):
            self._turns = list(turns)
            self._idx = 0
        async def call_model_stream_raw(self, messages, tools=None, **kw):
            turn = self._turns[self._idx]
            self._idx += 1
            for c in turn:
                yield c

    llm = _SeqLLM([same_turn] * 5 + [content_turn])
    guard_cfg = LoopGuardConfig(enabled=False)
    agent = _StubAgent(llm=llm, guard_cfg=guard_cfg, extra_tools=[fetch])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    loop_breaks = [e for e in events if isinstance(e, LoopBreakEvent)]
    assert len(loop_breaks) == 0

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 5

    completes = [e for e in events if isinstance(e, StreamComplete)]
    assert completes[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_different_args_different_hash():
    """Same tool with different args does not trigger repeat detection."""

    class _SeqLLM:
        def __init__(self, turns):
            self._turns = list(turns)
            self._idx = 0
        async def call_model_stream_raw(self, messages, tools=None, **kw):
            turn = self._turns[self._idx]
            self._idx += 1
            for c in turn:
                yield c

    turns = [
        _make_tool_turn("c1", "fetch", '{"url": "http://a.com"}'),
        _make_tool_turn("c2", "fetch", '{"url": "http://b.com"}'),
        _make_tool_turn("c3", "fetch", '{"url": "http://c.com"}'),
        _make_content_turn("Done"),
    ]

    llm = _SeqLLM(turns)
    agent = _StubAgent(llm=llm, extra_tools=[fetch])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    loop_breaks = [e for e in events if isinstance(e, LoopBreakEvent)]
    assert len(loop_breaks) == 0

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 3

    completes = [e for e in events if isinstance(e, StreamComplete)]
    assert completes[0].finish_reason == "stop"
