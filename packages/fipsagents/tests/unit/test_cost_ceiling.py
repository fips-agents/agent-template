"""Tests for per-turn resource limits (cost ceiling) in astep_stream."""

from types import SimpleNamespace

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.config import LimitsConfig, PricingConfig, PricingRate
from fipsagents.baseagent.events import (
    LimitExceeded,
    StreamComplete,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.baseagent.tools import ToolRegistry, tool


@tool(description="Echo input", visibility="both")
async def echo(msg: str) -> str:
    return f"echo:{msg}"


# ---------------------------------------------------------------------------
# Stub helpers (mirrors test_astep_stream_permissions.py pattern)
# ---------------------------------------------------------------------------


def _tc_delta(index, *, call_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments or "")
    return SimpleNamespace(index=index, id=call_id, function=fn)


def _chunk(*, tool_calls=None, content=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _usage(prompt=0, completion=0, total=None):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total if total is not None else prompt + completion,
    )


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
    def __init__(self, *, llm, limits=None, extra_tools=None, pricing=None):
        self.llm = llm
        if limits is not None or pricing is not None:
            self.config = SimpleNamespace(
                model=SimpleNamespace(
                    limits=limits,
                    model="test-model",
                ),
                tools=SimpleNamespace(enabled=True),
                memory=SimpleNamespace(
                    loading_pattern="eager",
                    prefix_role="system",
                    max_prefix_chars=8000,
                    injection_mode="prefix",
                    injection_tag="user_memories",
                    max_results=50,
                    min_weight=0.0,
                ),
                pricing=pricing if pricing is not None else PricingConfig(),
            )
        else:
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
        self._permission_source = None
        self._permission_mode = "enforce"
        self._permission_preapproved = set()
        self.memory = SimpleNamespace(project_config=None)

        if extra_tools:
            for t in extra_tools:
                self.tools.register(t)

    async def _inject_deferred_memory(self):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_limit_triggers():
    """LLM reports usage exceeding max_tokens_per_turn -> LimitExceeded emitted."""
    limits = LimitsConfig(max_tokens_per_turn=100)
    llm = _StubLLM([
        [
            _chunk(content="Hello world"),
            _chunk(finish_reason="stop", usage=_usage(prompt=80, completion=50)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 1
    assert limit_events[0].limit_type == "tokens"
    assert limit_events[0].threshold == 100.0
    assert limit_events[0].actual == 130.0  # 80 + 50

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "limit"


@pytest.mark.asyncio
async def test_iteration_limit_triggers():
    """LLM makes tool calls repeatedly, exceeds max_iterations_per_turn."""
    limits = LimitsConfig(max_iterations_per_turn=2)
    llm = _StubLLM([
        # Turn 1: tool call
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "a"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        # Turn 2: another tool call — hits iteration limit (model_calls == 2)
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c2", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "b"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 1
    assert limit_events[0].limit_type == "iterations"
    assert limit_events[0].threshold == 2.0
    assert limit_events[0].actual == 2.0

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "limit"


@pytest.mark.asyncio
async def test_no_limit_backward_compat():
    """All limits None -> normal behavior, no LimitExceeded."""
    limits = LimitsConfig()  # all None
    llm = _StubLLM([
        [
            _chunk(content="All good"),
            _chunk(finish_reason="stop", usage=_usage(prompt=500, completion=200)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 0

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_no_config_backward_compat():
    """Agent with config=None (test stubs that bypass setup) -> no limits, no crash."""
    llm = _StubLLM([
        [
            _chunk(content="Fine"),
            _chunk(finish_reason="stop", usage=_usage(prompt=100, completion=50)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=None)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 0

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_limit_prevents_tool_dispatch():
    """When token limit is exceeded after a model call that also has tool_calls,
    the tools should NOT be dispatched."""
    limits = LimitsConfig(max_tokens_per_turn=50)
    llm = _StubLLM([
        # Model returns a tool call AND usage that exceeds the limit.
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "blocked"}')]),
            _chunk(finish_reason="tool_calls", usage=_usage(prompt=40, completion=30)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    # Limit should fire.
    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 1
    assert limit_events[0].limit_type == "tokens"

    # Tool should NOT have been dispatched.
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 0

    # The tool call deltas are still emitted (they happen during streaming
    # before usage is known), but the actual execution is prevented.
    tc_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert len(tc_deltas) > 0

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "limit"


@pytest.mark.asyncio
async def test_token_limit_under_threshold_no_trigger():
    """Usage exactly at or below the threshold does not trigger the limit."""
    limits = LimitsConfig(max_tokens_per_turn=100)
    llm = _StubLLM([
        [
            _chunk(content="Under budget"),
            _chunk(finish_reason="stop", usage=_usage(prompt=50, completion=50)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 0

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_cost_limit_triggers():
    """LLM reports usage exceeding max_cost_per_turn_usd -> LimitExceeded emitted."""
    # Set pricing: $0.10 per 1K input tokens, $0.20 per 1K output tokens
    # Turn uses 500 input + 500 output = $0.05 + $0.10 = $0.15 total
    pricing = PricingConfig(
        default=PricingRate(
            input_per_1k=0.10,
            output_per_1k=0.20,
        )
    )
    limits = LimitsConfig(max_cost_per_turn_usd=0.10)  # Set limit below the actual cost
    llm = _StubLLM([
        [
            _chunk(content="Hello world"),
            _chunk(finish_reason="stop", usage=_usage(prompt=500, completion=500)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits, pricing=pricing)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 1
    assert limit_events[0].limit_type == "cost"
    assert limit_events[0].threshold == 0.10
    assert limit_events[0].actual == 0.15  # (500/1000)*0.10 + (500/1000)*0.20

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "limit"


@pytest.mark.asyncio
async def test_cost_limit_under_threshold_no_trigger():
    """Cost exactly at or below the threshold does not trigger the limit."""
    pricing = PricingConfig(
        default=PricingRate(
            input_per_1k=0.10,
            output_per_1k=0.20,
        )
    )
    limits = LimitsConfig(max_cost_per_turn_usd=0.20)  # Set limit above the actual cost
    llm = _StubLLM([
        [
            _chunk(content="Under budget"),
            _chunk(finish_reason="stop", usage=_usage(prompt=500, completion=500)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits, pricing=pricing)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 0

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_cost_limit_prevents_tool_dispatch():
    """When cost limit is exceeded after a model call that also has tool_calls,
    the tools should NOT be dispatched."""
    pricing = PricingConfig(
        default=PricingRate(
            input_per_1k=0.10,
            output_per_1k=0.20,
        )
    )
    limits = LimitsConfig(max_cost_per_turn_usd=0.05)  # Very low limit
    llm = _StubLLM([
        # Model returns a tool call AND usage that exceeds the cost limit.
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "blocked"}')]),
            _chunk(finish_reason="tool_calls", usage=_usage(prompt=500, completion=500)),
        ],
    ])
    agent = _StubAgent(llm=llm, limits=limits, pricing=pricing, extra_tools=[echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=10):
        events.append(ev)

    # Limit should fire.
    limit_events = [e for e in events if isinstance(e, LimitExceeded)]
    assert len(limit_events) == 1
    assert limit_events[0].limit_type == "cost"

    # Tool should NOT have been dispatched.
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 0

    # The tool call deltas are still emitted (they happen during streaming
    # before usage is known), but the actual execution is prevented.
    tc_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert len(tc_deltas) > 0

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "limit"
