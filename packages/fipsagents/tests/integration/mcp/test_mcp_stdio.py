"""Integration tests for MCP stdio transport.

Validates tool discovery, direct tool execution, and full agent dispatch
(sync + streaming) against a local MCP server running as a subprocess.
The LLM is mocked — these tests exercise the stdio transport path.

The test server is ``calculator_server.py`` in this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.config import (
    AgentConfig,
    BackoffConfig,
    LLMConfig,
    LoopConfig,
    McpServerConfig,
)
from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    ToolResultEvent,
)
from fipsagents.baseagent.llm import LLMClient

from .conftest import (
    _content_turn,
    _make_mock_stream,
    _multi_tool_turn,
    _tool_call_turn,
    assert_stream_completes,
    assert_tool_call_result_ordering,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SERVER_SCRIPT = Path(__file__).resolve().parent / "calculator_server.py"

# Sanity check at import time — if the script is missing, every test
# in this file would fail with a confusing error.
assert _SERVER_SCRIPT.is_file(), (
    f"calculator_server.py not found at {_SERVER_SCRIPT}"
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
async def mcp_stdio_agent() -> AsyncIterator[BaseAgent]:
    """Agent wired to the calculator MCP server over stdio.

    Starts the server as a subprocess, discovers tools, and cleans up
    (including killing the subprocess) after the test.
    """
    config = _make_config()
    agent = BaseAgent(config=config)

    # Manual wiring — bypass setup() to avoid filesystem/memory deps.
    agent.config = config
    agent.llm = MagicMock(spec=LLMClient)
    agent.messages = [{"role": "system", "content": "You are a calculator."}]
    agent._reasoning_parser = None
    agent._setup_done = True

    # Connect via stdio using McpServerConfig.
    stdio_cfg = McpServerConfig(
        command=sys.executable,
        args=[str(_SERVER_SCRIPT)],
    )
    await agent.connect_mcp(stdio_cfg)

    if not agent.tools.get_all():
        await agent.shutdown()
        pytest.skip("No tools discovered from stdio calculator server")

    yield agent

    await agent.shutdown()


# ---------------------------------------------------------------------------
# TestMcpStdioDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.mcp_stdio
class TestMcpStdioDiscovery:
    """Verify MCP tool discovery over stdio transport."""

    async def test_discovers_add_and_multiply(
        self, mcp_stdio_agent: BaseAgent,
    ) -> None:
        names = {t.name for t in mcp_stdio_agent.tools.get_all()}
        assert "add" in names, f"'add' not in {sorted(names)}"
        assert "multiply" in names, f"'multiply' not in {sorted(names)}"

    async def test_tools_registered_as_llm_only(
        self, mcp_stdio_agent: BaseAgent,
    ) -> None:
        for meta in mcp_stdio_agent.tools.get_all():
            assert meta.visibility == "llm_only", (
                f"Tool {meta.name!r} has visibility {meta.visibility!r}"
            )

    async def test_tool_schemas_generated(
        self, mcp_stdio_agent: BaseAgent,
    ) -> None:
        schemas = mcp_stdio_agent.tools.generate_schemas()
        assert len(schemas) >= 2, f"Expected >= 2 schemas, got {len(schemas)}"
        schema_names = {s["function"]["name"] for s in schemas}
        assert {"add", "multiply"}.issubset(schema_names), (
            f"Missing expected tools in schemas: {schema_names}"
        )


# ---------------------------------------------------------------------------
# TestMcpStdioToolExecution
# ---------------------------------------------------------------------------


@pytest.mark.mcp_stdio
class TestMcpStdioToolExecution:
    """Execute MCP tools directly through the registry (no LLM)."""

    async def test_add(self, mcp_stdio_agent: BaseAgent) -> None:
        result = await mcp_stdio_agent.tools.execute("add", a=3, b=5)
        assert not result.is_error, f"Tool error: {result.error}"
        assert "8" in result.result, (
            f"Expected '8' in add result: {result.result!r}"
        )

    async def test_multiply(self, mcp_stdio_agent: BaseAgent) -> None:
        result = await mcp_stdio_agent.tools.execute("multiply", a=4, b=7)
        assert not result.is_error, f"Tool error: {result.error}"
        assert "28" in result.result, (
            f"Expected '28' in multiply result: {result.result!r}"
        )

    async def test_float_precision(self, mcp_stdio_agent: BaseAgent) -> None:
        """Tool handles floats correctly."""
        result = await mcp_stdio_agent.tools.execute("add", a=1.5, b=2.5)
        assert not result.is_error, f"Tool error: {result.error}"
        assert "4" in result.result, (
            f"Expected '4' in float add result: {result.result!r}"
        )


# ---------------------------------------------------------------------------
# TestMcpStdioSyncDispatch
# ---------------------------------------------------------------------------


@pytest.mark.mcp_stdio
class TestMcpStdioSyncDispatch:
    """Full sync step() with mocked LLM calling stdio MCP tools."""

    async def test_sync_round_trip(self, mcp_stdio_agent: BaseAgent) -> None:
        """LLM calls add(10, 20); stdio MCP result flows back."""
        agent = mcp_stdio_agent
        agent.add_message("user", "What is 10 + 20?")

        turn1 = _tool_call_turn("call_add", "add", '{"a": 10, "b": 20}')
        turn2 = _content_turn("The answer is 30.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE
        assert "30" in result.result, f"Expected '30' in result: {result.result!r}"

        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, "No tool result in message history"
        assert "30" in tool_msgs[0]["content"]

    async def test_multi_tool_same_turn(self, mcp_stdio_agent: BaseAgent) -> None:
        """LLM calls both add and multiply in one turn."""
        agent = mcp_stdio_agent
        agent.add_message("user", "Add 2+3 and multiply 4*5.")

        turn1 = _multi_tool_turn([
            ("call_a", "add", '{"a": 2, "b": 3}'),
            ("call_m", "multiply", '{"a": 4, "b": 5}'),
        ])
        turn2 = _content_turn("Results: 5 and 20.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE

        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2, (
            f"Expected 2 tool messages, got {len(tool_msgs)}"
        )
        tool_contents = [m["content"] for m in tool_msgs]
        assert any("5" in c for c in tool_contents), (
            f"add result missing: {tool_contents}"
        )
        assert any("20" in c for c in tool_contents), (
            f"multiply result missing: {tool_contents}"
        )


# ---------------------------------------------------------------------------
# TestMcpStdioStreamingDispatch
# ---------------------------------------------------------------------------


@pytest.mark.mcp_stdio
class TestMcpStdioStreamingDispatch:
    """Streaming astep_stream() with mocked LLM calling stdio MCP tools."""

    async def test_event_ordering(self, mcp_stdio_agent: BaseAgent) -> None:
        """ToolCallDelta → ToolResultEvent → ContentDelta ordering."""
        agent = mcp_stdio_agent
        agent.add_message("user", "Multiply 6 by 7.")

        turn1 = _tool_call_turn("call_mul", "multiply", '{"a": 6, "b": 7}')
        turn2 = _content_turn("6 times 7 is 42.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_tool_call_result_ordering(events)
        assert_stream_completes(events)

        tc_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tc_results) == 1
        assert tc_results[0].name == "multiply"
        assert "42" in tc_results[0].content

        content_deltas = [e for e in events if isinstance(e, ContentDelta)]
        assert content_deltas, "Expected ContentDelta after tool result"

    async def test_metrics(self, mcp_stdio_agent: BaseAgent) -> None:
        """StreamComplete metrics count stdio tool calls correctly."""
        agent = mcp_stdio_agent
        agent.add_message("user", "Add 100 and 200.")

        turn1 = _tool_call_turn("call_sum", "add", '{"a": 100, "b": 200}')
        turn2 = _content_turn("That's 300.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_stream_completes(events)
        complete = events[-1]
        assert isinstance(complete, StreamComplete)
        assert complete.metrics.tool_calls == 1
        assert complete.metrics.model_calls >= 2


# ---------------------------------------------------------------------------
# TestMcpStdioLifecycle
# ---------------------------------------------------------------------------


@pytest.mark.mcp_stdio
class TestMcpStdioLifecycle:
    """Verify subprocess lifecycle management."""

    async def test_shutdown_cleans_up(self) -> None:
        """Agent shutdown closes the stdio MCP client without errors."""
        config = _make_config()
        agent = BaseAgent(config=config)
        agent.config = config
        agent.llm = MagicMock(spec=LLMClient)
        agent.messages = [{"role": "system", "content": "Test."}]
        agent._reasoning_parser = None
        agent._setup_done = True

        stdio_cfg = McpServerConfig(
            command=sys.executable,
            args=[str(_SERVER_SCRIPT)],
        )
        await agent.connect_mcp(stdio_cfg)

        # Should have at least one MCP client registered.
        assert len(agent._mcp_clients) == 1, (
            f"Expected 1 MCP client, got {len(agent._mcp_clients)}"
        )

        # Shutdown should complete without exceptions.
        await agent.shutdown()

        assert len(agent._mcp_clients) == 0, (
            "MCP clients not cleared after shutdown"
        )

    async def test_connect_via_config_object(self) -> None:
        """McpServerConfig with command field connects correctly."""
        config = _make_config()
        agent = BaseAgent(config=config)
        agent.config = config
        agent.llm = MagicMock(spec=LLMClient)
        agent.messages = []
        agent._reasoning_parser = None
        agent._setup_done = True

        stdio_cfg = McpServerConfig(
            command=sys.executable,
            args=[str(_SERVER_SCRIPT)],
        )
        await agent.connect_mcp(stdio_cfg)

        names = {t.name for t in agent.tools.get_all()}
        assert "add" in names
        assert "multiply" in names

        await agent.shutdown()
