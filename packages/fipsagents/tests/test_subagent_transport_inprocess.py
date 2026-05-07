"""Tests for InProcessSubagentTransport.

Uses a tiny test-only BaseAgent stub to drive the transport without spinning
up a real LLM or making network calls.
"""

from __future__ import annotations

import asyncio
import sys
from typing import AsyncIterator

import pytest

from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    StreamMetrics,
    ToolResultEvent,
)
from fipsagents.baseagent.config import InProcessTransportConfig
from fipsagents.subagents.transport import InProcessSubagentTransport
from fipsagents.subagents.types import (
    SubagentCrashedError,
    SubagentTimeoutError,
)


# ---------------------------------------------------------------------------
# Test-only BaseAgent stub
# ---------------------------------------------------------------------------


class _StubAgent:
    """Minimal BaseAgent surface for transport tests.

    Configurable via class attributes set by test helpers:
    - ``_events``: sequence of StreamEvents to yield from astep_stream.
    - ``_raise``: if not None, the exception to raise inside astep_stream.
    - ``_sleep``: if > 0, asyncio.sleep(n) before yielding any events.
    - ``setup_count``: incremented each time setup() is called.
    """

    _events: list = []
    _raise: BaseException | None = None
    _sleep: float = 0.0
    setup_count: int = 0

    def __init__(self, config_path: str = "agent.yaml") -> None:
        self.messages: list[dict] = []
        self.__class__.setup_count = 0  # reset per instance

    async def setup(self) -> None:
        self.__class__.setup_count += 1

    async def astep_stream(self, **_kwargs) -> AsyncIterator:
        if self._sleep > 0:
            await asyncio.sleep(self._sleep)
        if self._raise is not None:
            raise self._raise
        for event in self._events:
            yield event


def _register_stub(module_name: str, class_name: str, stub_class: type) -> None:
    """Insert a stub class into sys.modules so importlib can find it."""
    # Create a minimal module-like object.
    import types

    mod = types.ModuleType(module_name)
    setattr(mod, class_name, stub_class)
    sys.modules[module_name] = mod


def _remove_stub(module_name: str) -> None:
    sys.modules.pop(module_name, None)


def _make_config(
    class_path: str = "test_stubs.StubAgent",
    config_path: str | None = None,
) -> InProcessTransportConfig:
    return InProcessTransportConfig(
        type="inprocess",
        class_path=class_path,
        config_path=config_path,
    )


def _make_transport(stub_class: type, class_path: str = "test_stubs_ip.StubAgent") -> InProcessSubagentTransport:
    """Register stub in sys.modules and return a configured transport."""
    module_name, class_name = class_path.rsplit(".", 1)
    _register_stub(module_name, class_name, stub_class)
    config = _make_config(class_path)
    return InProcessSubagentTransport("helper", config)


def _make_stub_class(
    events: list,
    *,
    raise_exc: BaseException | None = None,
    sleep: float = 0.0,
) -> type:
    """Build a fresh _StubAgent subclass with the given configuration.

    Each call produces a distinct class so tests do not share state.
    """

    class FreshStub(_StubAgent):
        _events = list(events)
        _raise = raise_exc
        _sleep = sleep
        setup_count = 0

    return FreshStub


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestInProcessHappyPath:
    async def test_content_concatenated(self) -> None:
        events = [
            ContentDelta(content="hello "),
            ContentDelta(content="world"),
            StreamComplete(
                finish_reason="stop",
                metrics=StreamMetrics(prompt_tokens=10, completion_tokens=5),
            ),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub)

        result = await transport.invoke(task="say hello")

        assert result.content == "hello world"
        _remove_stub("test_stubs_ip")

    async def test_finish_reason_stop(self) -> None:
        events = [
            ContentDelta(content="done"),
            StreamComplete(
                finish_reason="stop",
                metrics=StreamMetrics(),
            ),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub)

        result = await transport.invoke(task="x")

        assert result.finish_reason == "stop"
        _remove_stub("test_stubs_ip")

    async def test_finish_reason_length(self) -> None:
        events = [
            ContentDelta(content="truncated"),
            StreamComplete(
                finish_reason="length",
                metrics=StreamMetrics(),
            ),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub)

        result = await transport.invoke(task="x")

        assert result.finish_reason == "length"
        _remove_stub("test_stubs_ip")

    async def test_agent_name_on_result(self) -> None:
        events = [
            ContentDelta(content="ok"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        stub = _make_stub_class(events)
        module_name = "test_stubs_ip_name"
        _register_stub(module_name, "StubAgent", stub)
        config = _make_config(f"{module_name}.StubAgent")
        transport = InProcessSubagentTransport("my_subagent", config)

        result = await transport.invoke(task="x")

        assert result.agent_name == "my_subagent"
        _remove_stub(module_name)

    async def test_cost_usd_is_zero(self) -> None:
        events = [StreamComplete(finish_reason="stop", metrics=StreamMetrics())]
        stub = _make_stub_class(events)
        transport = _make_transport(stub)

        result = await transport.invoke(task="x")

        assert result.cost_usd == 0.0
        _remove_stub("test_stubs_ip")

    async def test_span_id_is_none(self) -> None:
        events = [StreamComplete(finish_reason="stop", metrics=StreamMetrics())]
        stub = _make_stub_class(events)
        transport = _make_transport(stub)

        result = await transport.invoke(task="x")

        assert result.span_id is None
        _remove_stub("test_stubs_ip")


# ---------------------------------------------------------------------------
# Token capture
# ---------------------------------------------------------------------------


class TestTokenCapture:
    async def test_tokens_from_stream_complete(self) -> None:
        metrics = StreamMetrics(prompt_tokens=10, completion_tokens=5)
        events = [
            ContentDelta(content="ok"),
            StreamComplete(finish_reason="stop", metrics=metrics),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub, class_path="test_stubs_tokens.StubAgent")

        result = await transport.invoke(task="x")

        assert result.tokens_used == {"input": 10, "output": 5, "cached": 0}
        _remove_stub("test_stubs_tokens")

    async def test_missing_token_counts_default_to_zero(self) -> None:
        events = [
            ContentDelta(content="ok"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub, class_path="test_stubs_tokens2.StubAgent")

        result = await transport.invoke(task="x")

        assert result.tokens_used == {"input": 0, "output": 0, "cached": 0}
        _remove_stub("test_stubs_tokens2")

    async def test_no_stream_complete_gives_zeros(self) -> None:
        """If StreamComplete is never emitted, tokens default to zero."""
        events = [ContentDelta(content="ok")]
        stub = _make_stub_class(events)
        transport = _make_transport(stub, class_path="test_stubs_tokens3.StubAgent")

        result = await transport.invoke(task="x")

        assert result.tokens_used == {"input": 0, "output": 0, "cached": 0}
        _remove_stub("test_stubs_tokens3")


# ---------------------------------------------------------------------------
# Tool call counting
# ---------------------------------------------------------------------------


class TestToolCallCounting:
    async def test_two_tool_result_events_gives_two(self) -> None:
        events = [
            ToolResultEvent(call_id="c1", name="fn1", content="result1"),
            ToolResultEvent(call_id="c2", name="fn2", content="result2"),
            ContentDelta(content="done"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub, class_path="test_stubs_tc.StubAgent")

        result = await transport.invoke(task="x")

        assert result.tool_calls_made == 2
        _remove_stub("test_stubs_tc")

    async def test_no_tool_result_events_gives_zero(self) -> None:
        events = [
            ContentDelta(content="answer"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub, class_path="test_stubs_tc2.StubAgent")

        result = await transport.invoke(task="x")

        assert result.tool_calls_made == 0
        _remove_stub("test_stubs_tc2")

    async def test_error_tool_result_still_counted(self) -> None:
        events = [
            ToolResultEvent(call_id="c1", name="fn1", content="ERROR: oops", is_error=True),
            ContentDelta(content="fallback"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub, class_path="test_stubs_tc3.StubAgent")

        result = await transport.invoke(task="x")

        assert result.tool_calls_made == 1
        _remove_stub("test_stubs_tc3")


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestInProcessTimeout:
    async def test_timeout_raises_subagent_timeout_error(self) -> None:
        stub = _make_stub_class([], sleep=10.0)
        transport = _make_transport(stub, class_path="test_stubs_to.StubAgent")

        with pytest.raises(SubagentTimeoutError) as exc_info:
            await transport.invoke(task="x", timeout_seconds=0.05)

        err = exc_info.value
        assert err.agent_name == "helper"
        assert err.timeout_seconds == pytest.approx(0.05)
        _remove_stub("test_stubs_to")

    async def test_timeout_stores_timeout_seconds(self) -> None:
        stub = _make_stub_class([], sleep=10.0)
        transport = _make_transport(stub, class_path="test_stubs_to2.StubAgent")

        with pytest.raises(SubagentTimeoutError) as exc_info:
            await transport.invoke(task="x", timeout_seconds=0.03)

        assert exc_info.value.timeout_seconds == pytest.approx(0.03)
        _remove_stub("test_stubs_to2")


# ---------------------------------------------------------------------------
# Crash handling
# ---------------------------------------------------------------------------


class TestCrashHandling:
    async def test_runtime_error_becomes_crashed_error(self) -> None:
        stub = _make_stub_class([], raise_exc=RuntimeError("boom"))
        transport = _make_transport(stub, class_path="test_stubs_crash.StubAgent")

        with pytest.raises(SubagentCrashedError) as exc_info:
            await transport.invoke(task="x")

        err = exc_info.value
        assert err.agent_name == "helper"
        assert isinstance(err.original, RuntimeError)
        assert "boom" in str(err.original)
        _remove_stub("test_stubs_crash")

    async def test_value_error_becomes_crashed_error(self) -> None:
        stub = _make_stub_class([], raise_exc=ValueError("bad value"))
        transport = _make_transport(stub, class_path="test_stubs_crash2.StubAgent")

        with pytest.raises(SubagentCrashedError) as exc_info:
            await transport.invoke(task="x")

        assert isinstance(exc_info.value.original, ValueError)
        _remove_stub("test_stubs_crash2")

    async def test_crashed_error_is_catchable_as_subagent_error(self) -> None:
        from fipsagents.subagents.types import SubagentError

        stub = _make_stub_class([], raise_exc=RuntimeError("crash"))
        transport = _make_transport(stub, class_path="test_stubs_crash3.StubAgent")

        with pytest.raises(SubagentCrashedError) as exc_info:
            await transport.invoke(task="x")

        assert isinstance(exc_info.value, SubagentError)
        _remove_stub("test_stubs_crash3")


# ---------------------------------------------------------------------------
# Bad class path
# ---------------------------------------------------------------------------


class TestClassPathResolution:
    async def test_nonexistent_module_raises_crashed_error(self) -> None:
        config = _make_config("doesnotexist.module.NotAClass")
        transport = InProcessSubagentTransport("helper", config)

        with pytest.raises(SubagentCrashedError) as exc_info:
            await transport.invoke(task="x")

        err = exc_info.value
        assert err.agent_name == "helper"
        assert err.original is not None

    async def test_nonexistent_class_in_valid_module_raises_crashed_error(self) -> None:
        # Register a real module but reference a missing class.
        import types

        mod = types.ModuleType("test_stubs_badclass_mod")
        sys.modules["test_stubs_badclass_mod"] = mod
        config = _make_config("test_stubs_badclass_mod.DoesNotExist")
        transport = InProcessSubagentTransport("helper", config)

        with pytest.raises(SubagentCrashedError) as exc_info:
            await transport.invoke(task="x")

        assert isinstance(exc_info.value, SubagentCrashedError)
        sys.modules.pop("test_stubs_badclass_mod", None)


# ---------------------------------------------------------------------------
# Caching — setup() called only once
# ---------------------------------------------------------------------------


class TestAgentCaching:
    async def test_setup_called_only_once_across_invocations(self) -> None:
        def make_events():
            return [
                ContentDelta(content="ok"),
                StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
            ]

        class CountingStub(_StubAgent):
            setup_count = 0

            async def setup(self) -> None:
                CountingStub.setup_count += 1

            async def astep_stream(self, **_kwargs) -> AsyncIterator:
                for event in make_events():
                    yield event

        module_name = "test_stubs_cache"
        _register_stub(module_name, "CountingStub", CountingStub)
        config = _make_config(f"{module_name}.CountingStub")
        transport = InProcessSubagentTransport("helper", config)

        await transport.invoke(task="first")
        await transport.invoke(task="second")
        await transport.invoke(task="third")

        assert CountingStub.setup_count == 1, (
            f"Expected setup() to be called once, got {CountingStub.setup_count}"
        )
        _remove_stub(module_name)

    async def test_same_agent_instance_reused(self) -> None:
        """Verify the transport returns results from the same cached agent."""
        collected_instance_ids: list[int] = []

        class TrackingStub(_StubAgent):
            setup_count = 0

            async def setup(self) -> None:
                TrackingStub.setup_count += 1

            async def astep_stream(self, **_kwargs) -> AsyncIterator:
                collected_instance_ids.append(id(self))
                yield ContentDelta(content="ok")
                yield StreamComplete(finish_reason="stop", metrics=StreamMetrics())

        module_name = "test_stubs_cache2"
        _register_stub(module_name, "TrackingStub", TrackingStub)
        config = _make_config(f"{module_name}.TrackingStub")
        transport = InProcessSubagentTransport("helper", config)

        await transport.invoke(task="first")
        await transport.invoke(task="second")

        assert len(collected_instance_ids) == 2
        assert collected_instance_ids[0] == collected_instance_ids[1], (
            "Expected the same agent instance to be reused"
        )
        _remove_stub(module_name)


# ---------------------------------------------------------------------------
# Context prepending
# ---------------------------------------------------------------------------


class TestInProcessContextPrepending:
    async def test_context_prepended_before_task(self) -> None:
        received_messages: list[dict] = []

        class InspectingStub(_StubAgent):
            async def astep_stream(self, **_kwargs) -> AsyncIterator:
                # Capture the last user message appended before astep_stream.
                for msg in self.messages:
                    if msg.get("role") == "user":
                        received_messages.append(msg)
                yield ContentDelta(content="ok")
                yield StreamComplete(finish_reason="stop", metrics=StreamMetrics())

        module_name = "test_stubs_ctx"
        _register_stub(module_name, "InspectingStub", InspectingStub)
        config = _make_config(f"{module_name}.InspectingStub")
        transport = InProcessSubagentTransport("helper", config)

        await transport.invoke(task="What is the policy?", context="You are a compliance agent.")

        assert received_messages, "No user messages captured"
        content = received_messages[-1]["content"]
        assert "You are a compliance agent." in content
        assert "What is the policy?" in content
        context_pos = content.index("You are a compliance agent.")
        task_pos = content.index("What is the policy?")
        assert context_pos < task_pos
        _remove_stub(module_name)

    async def test_no_context_sends_task_only(self) -> None:
        received_messages: list[dict] = []

        class InspectingStub2(_StubAgent):
            async def astep_stream(self, **_kwargs) -> AsyncIterator:
                for msg in self.messages:
                    if msg.get("role") == "user":
                        received_messages.append(msg)
                yield ContentDelta(content="ok")
                yield StreamComplete(finish_reason="stop", metrics=StreamMetrics())

        module_name = "test_stubs_ctx2"
        _register_stub(module_name, "InspectingStub2", InspectingStub2)
        config = _make_config(f"{module_name}.InspectingStub2")
        transport = InProcessSubagentTransport("helper", config)

        await transport.invoke(task="just the task")

        assert received_messages
        assert received_messages[-1]["content"] == "just the task"
        _remove_stub(module_name)


# ---------------------------------------------------------------------------
# Headers ignored for inprocess transport
# ---------------------------------------------------------------------------


class TestHeadersIgnored:
    async def test_headers_do_not_cause_errors(self) -> None:
        events = [
            ContentDelta(content="ok"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        stub = _make_stub_class(events)
        transport = _make_transport(stub, class_path="test_stubs_hdrs.StubAgent")

        # Providing headers must not raise even though they are ignored.
        result = await transport.invoke(
            task="x",
            headers={"traceparent": "00-abc-def-01"},
        )

        assert result.content == "ok"
        _remove_stub("test_stubs_hdrs")
