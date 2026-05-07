"""End-to-end tests for the in-process subagent delegation path.

Exercises BaseAgent.setup() registering delegate_to_agent, direct tool
invocation, and the full astep_stream drain path with an LLM stub.

No real LLM or network calls are made.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, AsyncIterator

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepResult
from fipsagents.baseagent.config import (
    AgentConfig,
    InProcessTransportConfig,
    SubagentConfig,
)
from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    StreamMetrics,
    SubagentCompleted,
    SubagentInvoked,
    ToolResultEvent,
)


# ---------------------------------------------------------------------------
# Child stub — registered into sys.modules so InProcessSubagentTransport can
# import it via importlib.  Yields a fixed event sequence, echoing the task.
# ---------------------------------------------------------------------------


class _ChildAgent:
    """Minimal agent stub consumed by InProcessSubagentTransport."""

    def __init__(self, config_path: str = "agent.yaml") -> None:
        self.messages: list[dict] = []

    async def setup(self) -> None:
        pass

    async def astep_stream(self, **_kwargs: Any) -> AsyncIterator:
        # Extract the task from the last user message so we can echo it.
        task_text = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                task_text = msg.get("content", "")
                break

        yield ContentDelta(content="child says: ")
        yield ContentDelta(content=task_text)
        yield StreamComplete(
            finish_reason="stop",
            metrics=StreamMetrics(prompt_tokens=15, completion_tokens=20),
        )


_CHILD_MODULE = "fipsagents_test_stubs.ChildAgent"
_CHILD_MODULE_PATH = "fipsagents_test_stubs"
_CHILD_CLASS_NAME = "ChildAgent"


def _register_child_stub() -> None:
    mod = types.ModuleType(_CHILD_MODULE_PATH)
    setattr(mod, _CHILD_CLASS_NAME, _ChildAgent)
    sys.modules[_CHILD_MODULE_PATH] = mod


def _unregister_child_stub() -> None:
    sys.modules.pop(_CHILD_MODULE_PATH, None)


# ---------------------------------------------------------------------------
# Parent agent — a real BaseAgent subclass using provided config
# ---------------------------------------------------------------------------


class _ParentAgent(BaseAgent):
    """Concrete parent agent that delegates to _ChildAgent."""

    async def step(self) -> StepResult:
        return StepResult.done()


# ---------------------------------------------------------------------------
# Minimal AgentConfig factory
# ---------------------------------------------------------------------------


def _make_config_with_subagent() -> AgentConfig:
    return AgentConfig(
        subagents=[
            SubagentConfig(
                name="child",
                description="A child agent that echoes tasks.",
                when_to_use="Use to echo tasks.",
                transport=InProcessTransportConfig(
                    type="inprocess",
                    class_path=_CHILD_MODULE,
                ),
                max_depth=3,
                identity="inherit",
                permission_scope=None,
            )
        ]
    )


# ---------------------------------------------------------------------------
# Test: Setup registers delegate_to_agent
# ---------------------------------------------------------------------------


class TestSetupRegistersDelegate:
    @pytest.mark.asyncio
    async def test_subagents_populated_after_setup(self, tmp_path) -> None:
        _register_child_stub()
        try:
            config = _make_config_with_subagent()
            agent = _ParentAgent(
                tmp_path / "agent.yaml",
                config=config,
                base_dir=tmp_path,
            )
            await agent.setup()

            assert "child" in agent.subagents
            assert agent.subagents["child"].name == "child"
        finally:
            await agent.shutdown()
            _unregister_child_stub()

    @pytest.mark.asyncio
    async def test_delegate_tool_registered_after_setup(self, tmp_path) -> None:
        _register_child_stub()
        try:
            config = _make_config_with_subagent()
            agent = _ParentAgent(
                tmp_path / "agent.yaml",
                config=config,
                base_dir=tmp_path,
            )
            await agent.setup()

            tool_fn = agent.tools.get("delegate_to_agent")
            assert tool_fn is not None, (
                "delegate_to_agent tool not found in registry after setup(); "
                f"registered tools: {list(agent.tools._registry.keys())}"
            )
        finally:
            await agent.shutdown()
            _unregister_child_stub()

    @pytest.mark.asyncio
    async def test_no_delegate_tool_when_no_subagents(self, tmp_path) -> None:
        config = AgentConfig()  # no subagents
        agent = _ParentAgent(
            tmp_path / "agent.yaml",
            config=config,
            base_dir=tmp_path,
        )
        await agent.setup()
        tool_fn = agent.tools.get("delegate_to_agent")
        assert tool_fn is None, (
            "delegate_to_agent should not be registered when subagents is empty"
        )
        await agent.shutdown()


# ---------------------------------------------------------------------------
# Test: Direct tool invocation (no astep_stream)
# ---------------------------------------------------------------------------


class TestDirectToolInvocation:
    @pytest.mark.asyncio
    async def test_direct_invoke_returns_parseable_json(self, tmp_path) -> None:
        _register_child_stub()
        try:
            config = _make_config_with_subagent()
            agent = _ParentAgent(
                tmp_path / "agent.yaml",
                config=config,
                base_dir=tmp_path,
            )
            await agent.setup()

            result = await agent.tools.execute(
                "delegate_to_agent", agent_name="child", task="hello"
            )

            assert not result.is_error, f"Expected success, got error: {result.error}"
            parsed = json.loads(result.result)
            assert parsed["agent_name"] == "child"
            assert parsed["content"] == "child says: hello"
        finally:
            await agent.shutdown()
            _unregister_child_stub()

    @pytest.mark.asyncio
    async def test_direct_invoke_token_usage_shape(self, tmp_path) -> None:
        _register_child_stub()
        try:
            config = _make_config_with_subagent()
            agent = _ParentAgent(
                tmp_path / "agent.yaml",
                config=config,
                base_dir=tmp_path,
            )
            await agent.setup()

            result = await agent.tools.execute(
                "delegate_to_agent", agent_name="child", task="hello"
            )
            parsed = json.loads(result.result)

            assert parsed["tokens_used"] == {"input": 15, "output": 20, "cached": 0}, (
                f"Unexpected tokens_used: {parsed['tokens_used']}"
            )
        finally:
            await agent.shutdown()
            _unregister_child_stub()

    @pytest.mark.asyncio
    async def test_direct_invoke_appends_token_usage_to_buffer(self, tmp_path) -> None:
        _register_child_stub()
        try:
            config = _make_config_with_subagent()
            agent = _ParentAgent(
                tmp_path / "agent.yaml",
                config=config,
                base_dir=tmp_path,
            )
            await agent.setup()

            await agent.tools.execute(
                "delegate_to_agent", agent_name="child", task="hello"
            )

            assert len(agent._subagent_token_usage) == 1, (
                f"Expected 1 token usage entry, got {agent._subagent_token_usage}"
            )
            usage = agent._subagent_token_usage[0]
            assert usage["input"] == 15
            assert usage["output"] == 20
        finally:
            await agent.shutdown()
            _unregister_child_stub()

    @pytest.mark.asyncio
    async def test_direct_invoke_events_accumulate_pre_drain(self, tmp_path) -> None:
        """Events buffer accumulates during direct tool call — not drained until astep_stream."""
        _register_child_stub()
        try:
            config = _make_config_with_subagent()
            agent = _ParentAgent(
                tmp_path / "agent.yaml",
                config=config,
                base_dir=tmp_path,
            )
            await agent.setup()

            # Verify buffer is empty before the call.
            assert agent._subagent_events == []

            await agent.tools.execute(
                "delegate_to_agent", agent_name="child", task="hello"
            )

            # After direct execute(), events are in the buffer (not drained
            # because astep_stream hasn't run yet).
            assert len(agent._subagent_events) == 2, (
                f"Expected 2 events (Invoked + Completed), got: {agent._subagent_events}"
            )
            assert isinstance(agent._subagent_events[0], SubagentInvoked)
            assert isinstance(agent._subagent_events[1], SubagentCompleted)
        finally:
            await agent.shutdown()
            _unregister_child_stub()


# ---------------------------------------------------------------------------
# Stub LLM that emits one tool_call then a final stop response
# ---------------------------------------------------------------------------


class _StubLLMToolThenStop:
    """Yields a single tool_call turn for delegate_to_agent, then a stop turn.

    First call → tool_call for delegate_to_agent(agent_name="child", task="hi")
    Second call → final content "done" with finish_reason "stop"
    """

    def __init__(self) -> None:
        self._call_count = 0

    async def call_model_stream_raw(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self._call_count += 1
        if self._call_count == 1:
            # First call: emit a tool_call for delegate_to_agent
            import uuid

            call_id = f"call-{uuid.uuid4().hex[:8]}"
            args_json = json.dumps({"agent_name": "child", "task": "hi"})
            choice = types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            index=0,
                            id=call_id,
                            function=types.SimpleNamespace(
                                name="delegate_to_agent",
                                arguments=args_json,
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
            yield types.SimpleNamespace(choices=[choice], usage=None)
        else:
            # Second call: final content
            choice = types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    content="done",
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
            yield types.SimpleNamespace(
                choices=[choice],
                usage=types.SimpleNamespace(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )


# ---------------------------------------------------------------------------
# Test: astep_stream drains events in correct order
# ---------------------------------------------------------------------------


class TestAStepStreamDrain:
    @pytest.mark.asyncio
    async def test_event_order_invoked_completed_before_tool_result(
        self, tmp_path
    ) -> None:
        _register_child_stub()
        try:
            config = _make_config_with_subagent()
            agent = _ParentAgent(
                tmp_path / "agent.yaml",
                config=config,
                base_dir=tmp_path,
            )
            await agent.setup()

            # Replace the LLM with our stub.
            agent.llm = _StubLLMToolThenStop()
            agent._reasoning_parser = None

            # Seed a user message.
            agent.messages.append({"role": "user", "content": "delegate something"})

            collected: list = []
            async for event in agent.astep_stream():
                collected.append(event)

            # Find the SubagentInvoked, SubagentCompleted, and ToolResultEvent
            invoked = [e for e in collected if isinstance(e, SubagentInvoked)]
            completed = [e for e in collected if isinstance(e, SubagentCompleted)]
            tool_results = [e for e in collected if isinstance(e, ToolResultEvent)]

            assert len(invoked) == 1, f"Expected 1 SubagentInvoked, got {invoked}"
            assert len(completed) == 1, f"Expected 1 SubagentCompleted, got {completed}"
            assert len(tool_results) == 1, f"Expected 1 ToolResultEvent, got {tool_results}"

            # The invoked + completed events must appear before the tool result.
            idx_invoked = collected.index(invoked[0])
            idx_completed = collected.index(completed[0])
            idx_tool_result = collected.index(tool_results[0])

            assert idx_invoked < idx_tool_result, (
                f"SubagentInvoked (idx {idx_invoked}) must precede "
                f"ToolResultEvent (idx {idx_tool_result})"
            )
            assert idx_completed < idx_tool_result, (
                f"SubagentCompleted (idx {idx_completed}) must precede "
                f"ToolResultEvent (idx {idx_tool_result})"
            )
            assert idx_invoked < idx_completed, (
                f"SubagentInvoked (idx {idx_invoked}) must precede "
                f"SubagentCompleted (idx {idx_completed})"
            )

            # After astep_stream, the buffer should be empty (drained).
            assert agent._subagent_events == [], (
                f"_subagent_events should be empty after astep_stream drain; "
                f"got: {agent._subagent_events}"
            )
        finally:
            await agent.shutdown()
            _unregister_child_stub()
