"""Shared fixtures and helpers for MCP integration tests.

Provides:
- Custom pytest markers for different transport/dispatch types
- Agent config factory
- Trivial test tools (@tool decorated)
- Mock LLM response and streaming chunk builders
- A ``harness_agent`` fixture with manual wiring (no filesystem/MCP/memory)
- Common assertion helpers
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.events import (
    StreamComplete,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.baseagent.llm import LLMClient
from fipsagents.baseagent.tools import tool


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so pytest doesn't warn about unknown marks."""
    config.addinivalue_line(
        "markers", "local_tool: Tests using local @tool dispatch"
    )
    config.addinivalue_line(
        "markers", "mcp_http: Tests using MCP streamable-http transport"
    )
    config.addinivalue_line(
        "markers", "mcp_stdio: Tests using MCP stdio transport"
    )
    config.addinivalue_line(
        "markers", "llamastack: Tests using LlamaStack proxy dispatch"
    )
    config.addinivalue_line(
        "markers", "kagenti: Tests using Kagenti Gateway dispatch"
    )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> AgentConfig:
    """Build a minimal AgentConfig for tests."""
    defaults: dict[str, Any] = {
        "model": LLMConfig(
            endpoint="http://test:8321/v1",
            name="test-model",
            temperature=0.0,
            max_tokens=256,
        ),
        "loop": LoopConfig(
            max_iterations=5,
            backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
        ),
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ---------------------------------------------------------------------------
# Trivial test tools
# ---------------------------------------------------------------------------


@tool(description="Add two numbers", visibility="llm_only")
async def add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


@tool(description="Multiply two numbers", visibility="llm_only")
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a * b)


@tool(description="A tool that always fails", visibility="llm_only")
async def failing_tool(message: str) -> str:
    """Always raises an error.

    Args:
        message: Error message to raise.
    """
    raise ValueError(message)


# ---------------------------------------------------------------------------
# Mock LLM response helpers (non-streaming)
# ---------------------------------------------------------------------------


def _mock_litellm_response(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> SimpleNamespace:
    """Build a fake litellm non-streaming response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    """Build a fake tool_call object for non-streaming responses."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


# ---------------------------------------------------------------------------
# Mock streaming chunk helpers
# ---------------------------------------------------------------------------


def _make_stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    """Build a fake litellm streaming chunk."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _make_tc_delta(
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments_delta: str | None = None,
) -> SimpleNamespace:
    """Build a fake tool_call delta for streaming."""
    fn = SimpleNamespace(name=name, arguments=arguments_delta)
    return SimpleNamespace(index=index, id=call_id, function=fn)


async def _async_iter(chunks: list[Any]) -> AsyncIterator[Any]:
    """Async generator that yields each chunk in sequence.

    Use as the body of a ``call_model_stream_raw`` mock::

        async def mock_stream(messages, *, tools=None):
            async for chunk in _async_iter(chunks):
                yield chunk
    """
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# Streaming turn builders
# ---------------------------------------------------------------------------


def _tool_call_turn(
    call_id: str,
    name: str,
    args_json: str,
) -> list[object]:
    """Build streaming chunks for a single tool call turn."""
    return [
        _make_stream_chunk(
            tool_calls=[_make_tc_delta(0, call_id=call_id, name=name, arguments_delta=args_json)]
        ),
        _make_stream_chunk(finish_reason="tool_calls"),
    ]


def _content_turn(content: str) -> list[object]:
    """Build streaming chunks for a plain content turn."""
    return [
        _make_stream_chunk(content=content),
        _make_stream_chunk(finish_reason="stop"),
    ]


def _multi_tool_turn(
    calls: list[tuple[str, str, str]],
) -> list[object]:
    """Build a turn with multiple concurrent tool calls.

    Each entry in *calls* is ``(call_id, name, args_json)``.
    """
    chunks: list[object] = []
    for idx, (call_id, name, args_json) in enumerate(calls):
        chunks.append(
            _make_stream_chunk(
                tool_calls=[_make_tc_delta(idx, call_id=call_id, name=name, arguments_delta=args_json)]
            )
        )
    chunks.append(_make_stream_chunk(finish_reason="tool_calls"))
    return chunks


def _make_mock_stream(turn_chunks: list[list[object]]):
    """Return a ``call_model_stream_raw`` mock that cycles through turns.

    Each call to the returned async generator advances to the next turn's
    chunk list.  Excess calls yield an empty stop turn.
    """
    call_count = 0

    async def _mock(messages, *, tools=None):
        nonlocal call_count
        turn = call_count
        call_count += 1
        chunks = turn_chunks[turn] if turn < len(turn_chunks) else _content_turn("")
        for chunk in chunks:
            yield chunk

    return _mock


# ---------------------------------------------------------------------------
# Agent factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def harness_agent() -> AsyncIterator[BaseAgent]:
    """Create a BaseAgent with test tools registered and mocked LLM.

    Performs manual wiring — no filesystem, no MCP, no memory dependencies.
    Yields the configured agent and cleans up after the test.
    """
    config = _make_config()
    agent = BaseAgent(config=config)

    agent.config = config
    agent.llm = MagicMock(spec=LLMClient)
    agent.tools.register(add)
    agent.tools.register(multiply)
    agent.tools.register(failing_tool)
    agent.messages = [{"role": "system", "content": "You are a calculator."}]
    agent._reasoning_parser = None
    agent._setup_done = True

    yield agent

    await agent.shutdown()


# ---------------------------------------------------------------------------
# Common assertion helpers
# ---------------------------------------------------------------------------


def assert_tool_call_result_ordering(events: list[Any]) -> None:
    """Assert that ToolCallDelta events precede their matching ToolResultEvent."""
    tc_indices: dict[str, int] = {}
    tr_indices: dict[str, int] = {}

    for i, event in enumerate(events):
        if isinstance(event, ToolCallDelta) and event.call_id:
            tc_indices[event.call_id] = i
        if isinstance(event, ToolResultEvent):
            tr_indices[event.call_id] = i

    for call_id, result_idx in tr_indices.items():
        assert call_id in tc_indices, (
            f"ToolResultEvent for {call_id!r} has no prior ToolCallDelta "
            f"(events: {[type(e).__name__ for e in events]})"
        )
        tc_idx = tc_indices[call_id]
        assert tc_idx < result_idx, (
            f"ToolCallDelta for {call_id!r} (at {tc_idx}) must precede "
            f"ToolResultEvent (at {result_idx})"
        )


def assert_stream_completes(events: list[Any]) -> None:
    """Assert the event stream ends with StreamComplete."""
    assert events, "Event stream is empty"
    assert isinstance(events[-1], StreamComplete), (
        f"Expected StreamComplete as last event, got "
        f"{type(events[-1]).__name__} "
        f"(events: {[type(e).__name__ for e in events]})"
    )
