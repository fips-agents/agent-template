"""Integration tests for in-process MCP transport (FastMCP object).

Validates tool discovery, direct tool execution, and full agent dispatch
against a FastMCP server running in the same process — no subprocess, no
network.  The LLM is mocked.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.events import ToolResultEvent
from fipsagents.baseagent.llm import LLMClient

from .calculator_server import mcp as calculator_mcp
from .conftest import (
    _content_turn,
    _make_mock_stream,
    _tool_call_turn,
    assert_stream_completes,
    assert_tool_call_result_ordering,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> AgentConfig:
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


@pytest.fixture
async def mcp_inprocess_agent() -> AsyncIterator[BaseAgent]:
    """Agent wired to the calculator FastMCP server in-process."""
    config = _make_config()
    agent = BaseAgent(config=config)

    agent.config = config
    agent.llm = MagicMock(spec=LLMClient)
    agent.messages = [{"role": "system", "content": "You are a calculator."}]
    agent._reasoning_parser = None
    agent._setup_done = True

    # Pass the FastMCP server object directly — in-process transport.
    await agent.connect_mcp(calculator_mcp)

    if not agent.tools.get_all():
        await agent.shutdown()
        pytest.skip("No tools discovered from in-process calculator server")

    yield agent

    await agent.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.local_tool  # No external deps — runs like a local tool test.
class TestMcpInprocessDiscovery:
    """Verify in-process MCP tool discovery."""

    async def test_discovers_add_and_multiply(
        self, mcp_inprocess_agent: BaseAgent,
    ) -> None:
        names = {t.name for t in mcp_inprocess_agent.tools.get_all()}
        assert "add" in names, f"'add' not in {sorted(names)}"
        assert "multiply" in names, f"'multiply' not in {sorted(names)}"

    async def test_tools_registered_as_llm_only(
        self, mcp_inprocess_agent: BaseAgent,
    ) -> None:
        for meta in mcp_inprocess_agent.tools.get_all():
            assert meta.visibility == "llm_only"


@pytest.mark.local_tool
class TestMcpInprocessToolExecution:
    """Execute in-process MCP tools directly through the registry."""

    async def test_add(self, mcp_inprocess_agent: BaseAgent) -> None:
        result = await mcp_inprocess_agent.tools.execute("add", a=3, b=5)
        assert not result.is_error, f"Tool error: {result.error}"
        assert "8" in result.result

    async def test_multiply(self, mcp_inprocess_agent: BaseAgent) -> None:
        result = await mcp_inprocess_agent.tools.execute("multiply", a=4, b=7)
        assert not result.is_error, f"Tool error: {result.error}"
        assert "28" in result.result


@pytest.mark.local_tool
class TestMcpInprocessSyncDispatch:
    """Full sync step() with mocked LLM calling in-process MCP tools."""

    async def test_sync_round_trip(self, mcp_inprocess_agent: BaseAgent) -> None:
        agent = mcp_inprocess_agent
        agent.add_message("user", "What is 10 + 20?")

        turn1 = _tool_call_turn("call_add", "add", '{"a": 10, "b": 20}')
        turn2 = _content_turn("The answer is 30.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE
        assert "30" in result.result

        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs
        assert "30" in tool_msgs[0]["content"]


@pytest.mark.local_tool
class TestMcpInprocessStreamingDispatch:
    """Streaming astep_stream() with in-process MCP tools."""

    async def test_event_ordering(self, mcp_inprocess_agent: BaseAgent) -> None:
        agent = mcp_inprocess_agent
        agent.add_message("user", "Multiply 6 by 7.")

        turn1 = _tool_call_turn("call_mul", "multiply", '{"a": 6, "b": 7}')
        turn2 = _content_turn("42.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_tool_call_result_ordering(events)
        assert_stream_completes(events)

        tc_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tc_results) == 1
        assert tc_results[0].name == "multiply"
        assert "42" in tc_results[0].content
