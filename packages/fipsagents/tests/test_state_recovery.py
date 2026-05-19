"""Tests for reducer-based state recovery (#190).

Covers AgentState, state_schema_key, StateReducerObserver,
reconstruct_events, recover_state, and BaseAgent reduce/after_event defaults.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepResult
from fipsagents.baseagent.config import AgentConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamEvent,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.baseagent.state import (
    AgentState,
    StateCheckpoint,
    StateReducerObserver,
    state_schema_key,
)
from fipsagents.server.recovery import reconstruct_events, recover_state
from fipsagents.server.tracing import Span, Trace


# ---------------------------------------------------------------------------
# Test state subclass + helpers
# ---------------------------------------------------------------------------


class _TestState(AgentState):
    items: list[str] = []
    counter: int = 0


class _OtherState(AgentState):
    name: str = ""
    value: float = 0.0


class _MockAgent:
    """Minimal agent double for StateReducerObserver tests."""

    def __init__(self) -> None:
        self._agent_state = _TestState()
        self._reduce_calls: list[StreamEvent] = []
        self._after_event_calls: list[StreamEvent] = []

    def reduce(self, state: _TestState, event: StreamEvent) -> _TestState:
        self._reduce_calls.append(event)
        if isinstance(event, ToolResultEvent):
            return state.model_copy(update={"items": [*state.items, event.content]})
        return state

    async def after_event(self, state: _TestState, event: StreamEvent) -> None:
        self._after_event_calls.append(event)


class _MockSessionStore:
    def __init__(self, state_data: dict[str, Any] | None = None) -> None:
        self._state = state_data or {}

    async def get_state(self, session_id: str) -> dict[str, Any]:
        return self._state

    async def update_state(self, session_id: str, **fields: Any) -> bool:
        self._state.update(fields)
        return True


class _MockTraceStore:
    def __init__(self, traces: list[Trace] | None = None) -> None:
        self._traces = traces or []

    async def list_traces_for_session(
        self,
        session_id: str,
        *,
        after_trace_id: str | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        return self._traces


async def _aiter(*events: StreamEvent) -> AsyncIterator[StreamEvent]:
    for e in events:
        yield e


# ---------------------------------------------------------------------------
# 1. AgentState basics
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_extra_forbid_rejects_unknown_fields(self):
        with pytest.raises(Exception):
            _TestState(items=[], counter=0, bogus="nope")

    def test_subclass_typed_fields(self):
        s = _TestState(items=["a", "b"], counter=42)
        assert s.items == ["a", "b"]
        assert s.counter == 42

    def test_model_dump_validate_roundtrip(self):
        original = _TestState(items=["x"], counter=7)
        dumped = original.model_dump()
        restored = _TestState.model_validate(dumped)
        assert restored == original

    def test_default_values(self):
        s = _TestState()
        assert s.items == []
        assert s.counter == 0


# ---------------------------------------------------------------------------
# 2. StateCheckpoint
# ---------------------------------------------------------------------------


class TestStateCheckpoint:
    def test_construction(self):
        cp = StateCheckpoint(
            state={"items": ["a"], "counter": 1},
            last_trace_id="t1",
            last_span_id="s1",
            checkpoint_at="2026-01-01T00:00:00Z",
            schema_version="TestState:abc123",
        )
        assert cp.last_trace_id == "t1"
        assert cp.state["counter"] == 1


# ---------------------------------------------------------------------------
# 3. state_schema_key
# ---------------------------------------------------------------------------


class TestStateSchemaKey:
    def test_same_class_same_key(self):
        assert state_schema_key(_TestState) == state_schema_key(_TestState)

    def test_different_fields_different_key(self):
        assert state_schema_key(_TestState) != state_schema_key(_OtherState)

    def test_non_pydantic_returns_qualname(self):

        class Plain:
            pass

        assert state_schema_key(Plain) == Plain.__qualname__

    def test_key_contains_classname(self):
        key = state_schema_key(_TestState)
        assert "_TestState:" in key

    def test_key_format(self):
        key = state_schema_key(_TestState)
        parts = key.split(":")
        assert len(parts) == 2
        assert len(parts[1]) == 16  # sha256 hex prefix


# ---------------------------------------------------------------------------
# 4. StateReducerObserver
# ---------------------------------------------------------------------------


class TestStateReducerObserver:
    @pytest.mark.asyncio
    async def test_events_pass_through(self):
        agent = _MockAgent()
        observer = StateReducerObserver(agent)
        events_in = [
            ContentDelta(content="hello"),
            ToolResultEvent(call_id="c1", name="search", content="found"),
        ]
        collected = [e async for e in observer.observe(_aiter(*events_in))]
        assert collected == events_in

    @pytest.mark.asyncio
    async def test_reduce_called_for_each_event(self):
        agent = _MockAgent()
        observer = StateReducerObserver(agent)
        events_in = [
            ContentDelta(content="a"),
            ToolResultEvent(call_id="c1", name="t", content="b"),
            ContentDelta(content="c"),
        ]
        _ = [e async for e in observer.observe(_aiter(*events_in))]
        assert len(agent._reduce_calls) == 3

    @pytest.mark.asyncio
    async def test_state_updated_by_reduce(self):
        agent = _MockAgent()
        observer = StateReducerObserver(agent)
        events_in = [
            ToolResultEvent(call_id="c1", name="t", content="r1"),
            ToolResultEvent(call_id="c2", name="t", content="r2"),
        ]
        _ = [e async for e in observer.observe(_aiter(*events_in))]
        assert agent._agent_state.items == ["r1", "r2"]

    @pytest.mark.asyncio
    async def test_after_event_called_without_replay(self):
        agent = _MockAgent()
        observer = StateReducerObserver(agent, replay=False)
        events_in = [ContentDelta(content="x")]
        _ = [e async for e in observer.observe(_aiter(*events_in))]
        assert len(agent._after_event_calls) == 1

    @pytest.mark.asyncio
    async def test_replay_skips_after_event(self):
        agent = _MockAgent()
        observer = StateReducerObserver(agent, replay=True)
        events_in = [
            ContentDelta(content="x"),
            ToolResultEvent(call_id="c", name="t", content="y"),
        ]
        _ = [e async for e in observer.observe(_aiter(*events_in))]
        assert len(agent._reduce_calls) == 2
        assert len(agent._after_event_calls) == 0

    @pytest.mark.asyncio
    async def test_none_state_passes_through(self):
        """Agent with _agent_state=None should pass events unchanged."""
        agent = _MockAgent()
        agent._agent_state = None
        observer = StateReducerObserver(agent)
        events_in = [ContentDelta(content="hi")]
        collected = [e async for e in observer.observe(_aiter(*events_in))]
        assert collected == events_in
        assert len(agent._reduce_calls) == 0


# ---------------------------------------------------------------------------
# 5. reconstruct_events
# ---------------------------------------------------------------------------


def _make_trace(*spans: Span) -> Trace:
    return Trace(
        trace_id="t1",
        started_at="2026-01-01T00:00:00Z",
        spans=list(spans),
    )


class TestReconstructEvents:
    def test_tool_result_extracted(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="tool:search",
                events=[{
                    "name": "tool_result",
                    "timestamp": 1.0,
                    "body": {
                        "call_id": "c1", "name": "search",
                        "content": "result1", "is_error": False,
                    },
                }],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert events[0].content == "result1"
        assert events[0].call_id == "c1"

    def test_content_delta_extracted(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="request",
                events=[{
                    "name": "content_delta",
                    "timestamp": 2.0,
                    "body": {"content": "hello world"},
                }],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 1
        assert isinstance(events[0], ContentDelta)
        assert events[0].content == "hello world"

    def test_reasoning_delta_extracted(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="step",
                events=[{
                    "name": "reasoning_delta",
                    "timestamp": 0.5,
                    "body": {"content": "thinking..."},
                }],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 1
        assert isinstance(events[0], ReasoningDelta)

    def test_tool_call_delta_extracted(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="step",
                events=[{
                    "name": "tool_call_delta",
                    "timestamp": 1.5,
                    "body": {
                        "index": 0, "call_id": "c1",
                        "name": "search", "arguments_delta": '{"q":',
                    },
                }],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 1
        assert isinstance(events[0], ToolCallDelta)
        assert events[0].index == 0
        assert events[0].arguments_delta == '{"q":'

    def test_sorted_by_timestamp(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="step",
                events=[
                    {"name": "content_delta", "timestamp": 3.0, "body": {"content": "second"}},
                    {"name": "content_delta", "timestamp": 1.0, "body": {"content": "first"}},
                ],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 2
        assert events[0].content == "first"
        assert events[1].content == "second"

    def test_unknown_events_skipped(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="step",
                events=[
                    {"name": "some_future_event", "timestamp": 1.0, "body": {}},
                    {"name": "content_delta", "timestamp": 2.0, "body": {"content": "ok"}},
                ],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 1
        assert isinstance(events[0], ContentDelta)

    def test_messages_snapshot_skipped(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="step",
                events=[
                    {"name": "messages_snapshot", "timestamp": 1.0, "body": {"messages": []}},
                    {"name": "tool_result", "timestamp": 2.0, "body": {
                        "call_id": "c1", "name": "t", "content": "ok", "is_error": False,
                    }},
                ],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)

    def test_multi_span_aggregation(self):
        trace = _make_trace(
            Span(
                trace_id="t1", span_id="s1", name="tool:a",
                events=[{"name": "tool_result", "timestamp": 1.0, "body": {
                    "call_id": "c1", "name": "a", "content": "r1", "is_error": False,
                }}],
            ),
            Span(
                trace_id="t1", span_id="s2", name="tool:b",
                events=[{"name": "tool_result", "timestamp": 2.0, "body": {
                    "call_id": "c2", "name": "b", "content": "r2", "is_error": False,
                }}],
            ),
        )
        events = reconstruct_events(trace)
        assert len(events) == 2
        assert events[0].name == "a"
        assert events[1].name == "b"

    def test_empty_trace(self):
        trace = _make_trace()
        assert reconstruct_events(trace) == []


# ---------------------------------------------------------------------------
# 6. recover_state
# ---------------------------------------------------------------------------


def _checkpoint_json(
    state: dict[str, Any],
    schema_version: str,
    last_trace_id: str = "t0",
) -> str:
    return json.dumps({
        "state": state,
        "schema_version": schema_version,
        "last_trace_id": last_trace_id,
        "last_span_id": "s0",
        "checkpoint_at": "2026-01-01T00:00:00Z",
    })


class _RecoveryAgent:
    """Minimal agent double for recover_state tests."""
    state_type = _TestState

    def reduce(self, state: _TestState, event: StreamEvent) -> _TestState:
        if isinstance(event, ToolResultEvent):
            return state.model_copy(update={"items": [*state.items, event.content]})
        return state


class TestRecoverState:
    @pytest.mark.asyncio
    async def test_no_checkpoint_returns_none(self):
        agent = _RecoveryAgent()
        session = _MockSessionStore({})
        traces = _MockTraceStore()
        result = await recover_state(agent, "sess1", session, traces)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_checkpoint_no_traces(self):
        schema = state_schema_key(_TestState)
        cp = _checkpoint_json({"items": ["a"], "counter": 5}, schema)
        agent = _RecoveryAgent()
        session = _MockSessionStore({"checkpoint_state": cp})
        traces = _MockTraceStore([])
        result = await recover_state(agent, "sess1", session, traces)
        assert isinstance(result, _TestState)
        assert result.items == ["a"]
        assert result.counter == 5

    @pytest.mark.asyncio
    async def test_checkpoint_plus_replay(self):
        schema = state_schema_key(_TestState)
        cp = _checkpoint_json({"items": ["a"], "counter": 0}, schema)
        replay_trace = _make_trace(
            Span(
                trace_id="t2", span_id="s1", name="tool:search",
                events=[{
                    "name": "tool_result", "timestamp": 1.0,
                    "body": {"call_id": "c1", "name": "search",
                             "content": "replayed", "is_error": False},
                }],
            ),
        )
        agent = _RecoveryAgent()
        session = _MockSessionStore({"checkpoint_state": cp})
        traces = _MockTraceStore([replay_trace])
        result = await recover_state(agent, "sess1", session, traces)
        assert result.items == ["a", "replayed"]

    @pytest.mark.asyncio
    async def test_schema_mismatch_returns_none(self):
        cp = _checkpoint_json({"items": [], "counter": 0}, "WrongState:badhash")
        agent = _RecoveryAgent()
        session = _MockSessionStore({"checkpoint_state": cp})
        traces = _MockTraceStore()
        result = await recover_state(agent, "sess1", session, traces)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        agent = _RecoveryAgent()
        session = _MockSessionStore({"checkpoint_state": "not-valid-json{{"})
        traces = _MockTraceStore()
        result = await recover_state(agent, "sess1", session, traces)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_state_data_returns_none(self):
        """Valid JSON but data that fails Pydantic validation."""
        schema = state_schema_key(_TestState)
        cp = _checkpoint_json({"items": "not-a-list", "counter": "nan"}, schema)
        agent = _RecoveryAgent()
        session = _MockSessionStore({"checkpoint_state": cp})
        traces = _MockTraceStore()
        result = await recover_state(agent, "sess1", session, traces)
        assert result is None


# ---------------------------------------------------------------------------
# 7. BaseAgent reduce/after_event defaults
# ---------------------------------------------------------------------------


class _MinimalAgent(BaseAgent):
    state_type = _TestState

    async def step(self) -> StepResult:
        return StepResult.done()


class TestBaseAgentReducerDefaults:
    def test_default_reduce_returns_state_unchanged(self):
        config = AgentConfig(
            model=LLMConfig(endpoint="http://test:8321/v1", name="test"),
            loop=LoopConfig(max_iterations=1),
        )
        agent = _MinimalAgent(config=config)
        state = _TestState(items=["x"], counter=3)
        event = ContentDelta(content="hi")
        result = agent.reduce(state, event)
        assert result is state

    @pytest.mark.asyncio
    async def test_default_after_event_is_noop(self):
        config = AgentConfig(
            model=LLMConfig(endpoint="http://test:8321/v1", name="test"),
            loop=LoopConfig(max_iterations=1),
        )
        agent = _MinimalAgent(config=config)
        state = _TestState()
        # Should complete without raising.
        await agent.after_event(state, ContentDelta(content="x"))

    def test_agent_state_attr_exists(self):
        config = AgentConfig(
            model=LLMConfig(endpoint="http://test:8321/v1", name="test"),
            loop=LoopConfig(max_iterations=1),
        )
        agent = _MinimalAgent(config=config)
        assert hasattr(agent, "_agent_state")
        assert agent._agent_state is None


# ---------------------------------------------------------------------------
# 8. Backward compatibility — no state_type
# ---------------------------------------------------------------------------


class _StatelessAgent(BaseAgent):
    async def step(self) -> StepResult:
        return StepResult.done()


class TestBackwardCompatibility:
    def test_no_state_type_has_none_agent_state(self):
        config = AgentConfig(
            model=LLMConfig(endpoint="http://test:8321/v1", name="test"),
            loop=LoopConfig(max_iterations=1),
        )
        agent = _StatelessAgent(config=config)
        assert agent.state_type is None
        assert agent._agent_state is None

    @pytest.mark.asyncio
    async def test_observer_noop_with_none_state(self):
        agent = _MockAgent()
        agent._agent_state = None
        observer = StateReducerObserver(agent)
        events_in = [
            ContentDelta(content="a"),
            ToolResultEvent(call_id="c1", name="t", content="b"),
        ]
        collected = [e async for e in observer.observe(_aiter(*events_in))]
        assert collected == events_in
        assert len(agent._reduce_calls) == 0
        assert len(agent._after_event_calls) == 0
