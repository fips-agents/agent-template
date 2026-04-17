"""Integration tests for MCP streamable-http transport (unauthenticated).

Validates tool discovery, direct tool execution, and full agent dispatch
(sync + streaming) against a live MCP server over HTTP.  The LLM is mocked
— these tests exercise the MCP transport path, not model inference.

Target: calculus-helper-mcp on cluster n7pd5.  Override with the
``CALCULUS_HELPER_MCP_URL`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    ToolResultEvent,
)
from fipsagents.baseagent.llm import LLMClient

from .conftest import (
    _content_turn,
    _make_mock_stream,
    _tool_call_turn,
    assert_stream_completes,
    assert_tool_call_result_ordering,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_URL = (
    "https://mcp-server-calculus-helper-mcp.apps.cluster-n7pd5"
    ".n7pd5.sandbox5167.opentlc.com/mcp/"
)
MCP_URL = os.environ.get("CALCULUS_HELPER_MCP_URL", _DEFAULT_URL)

# Tools we expect the calculus-helper to expose (subset — enough for tests).
_EXPECTED_TOOLS = {"differentiate", "integrate", "evaluate_numeric", "simplify_expression"}


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
async def mcp_http_agent() -> AsyncIterator[BaseAgent]:
    """Agent wired to the live calculus-helper MCP server.

    Skips the test if the server is unreachable or returns no tools.
    The LLM is mocked — only the MCP transport path is real.
    """
    config = _make_config()
    agent = BaseAgent(config=config)

    # Manual wiring — bypass setup() to avoid filesystem/memory deps.
    agent.config = config
    agent.llm = MagicMock(spec=LLMClient)
    agent.messages = [{"role": "system", "content": "You are a calculus assistant."}]
    agent._reasoning_parser = None
    agent._setup_done = True

    # Real MCP connection.  connect_mcp swallows exceptions, so check
    # whether any tools were actually registered.
    await agent.connect_mcp(MCP_URL)

    if not agent.tools.get_all():
        await agent.shutdown()
        pytest.skip(
            f"No tools discovered from MCP at {MCP_URL} "
            "(server may be unreachable)"
        )

    yield agent

    await agent.shutdown()


# ---------------------------------------------------------------------------
# TestMcpHttpDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMcpHttpDiscovery:
    """Verify MCP tool discovery over streamable-http."""

    async def test_discovers_expected_tools(self, mcp_http_agent: BaseAgent) -> None:
        names = {t.name for t in mcp_http_agent.tools.get_all()}
        missing = _EXPECTED_TOOLS - names
        assert not missing, (
            f"Expected tools not discovered: {missing} "
            f"(got: {sorted(names)})"
        )

    async def test_tools_registered_as_llm_only(self, mcp_http_agent: BaseAgent) -> None:
        for meta in mcp_http_agent.tools.get_all():
            assert meta.visibility == "llm_only", (
                f"Tool {meta.name!r} has visibility {meta.visibility!r}, "
                f"expected 'llm_only'"
            )

    async def test_tool_schemas_generated(self, mcp_http_agent: BaseAgent) -> None:
        schemas = mcp_http_agent.tools.generate_schemas()
        assert len(schemas) >= len(_EXPECTED_TOOLS), (
            f"Expected >= {len(_EXPECTED_TOOLS)} schemas, got {len(schemas)}"
        )
        for schema in schemas:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn


# ---------------------------------------------------------------------------
# TestMcpHttpToolExecution
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMcpHttpToolExecution:
    """Execute MCP tools directly through the registry (no LLM involvement)."""

    async def test_evaluate_numeric(self, mcp_http_agent: BaseAgent) -> None:
        """evaluate_numeric computes 2 + 3 = 5."""
        result = await mcp_http_agent.tools.execute(
            "evaluate_numeric", expression="2 + 3",
        )
        assert not result.is_error, f"Tool error: {result.error}"
        assert "5" in result.result, (
            f"Expected '5' in result: {result.result!r}"
        )

    async def test_differentiate(self, mcp_http_agent: BaseAgent) -> None:
        """d/dx(x**2) = 2*x."""
        result = await mcp_http_agent.tools.execute(
            "differentiate", expression="x**2", variables=["x"],
        )
        assert not result.is_error, f"Tool error: {result.error}"
        assert "2*x" in result.result or "2x" in result.result, (
            f"Expected '2*x' in result: {result.result!r}"
        )

    async def test_simplify_trig_identity(self, mcp_http_agent: BaseAgent) -> None:
        """sin(x)**2 + cos(x)**2 simplifies to 1."""
        result = await mcp_http_agent.tools.execute(
            "simplify_expression",
            expression="sin(x)**2 + cos(x)**2",
            form="simplify",
        )
        assert not result.is_error, f"Tool error: {result.error}"
        assert "1" in result.result, (
            f"Expected '1' in simplified result: {result.result!r}"
        )

    async def test_integrate_basic(self, mcp_http_agent: BaseAgent) -> None:
        """Indefinite integral of 2*x is x**2."""
        result = await mcp_http_agent.tools.execute(
            "integrate", expression="2*x", variable="x",
        )
        assert not result.is_error, f"Tool error: {result.error}"
        assert "x**2" in result.result, (
            f"Expected 'x**2' in integral result: {result.result!r}"
        )

    async def test_tool_error_on_invalid_input(self, mcp_http_agent: BaseAgent) -> None:
        """MCP server returns an error for nonsense input."""
        result = await mcp_http_agent.tools.execute(
            "evaluate_numeric", expression="not_math ][][",
        )
        # The MCP server may return an error via is_error or as error text
        # in the result content — either is acceptable.
        has_error_flag = result.is_error
        has_error_text = (
            "error" in result.result.lower()
            or "Error" in result.result
            if result.result else False
        )
        assert has_error_flag or has_error_text, (
            f"Expected an error for invalid input, got: "
            f"is_error={result.is_error}, result={result.result!r}"
        )


# ---------------------------------------------------------------------------
# TestMcpHttpSyncDispatch
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMcpHttpSyncDispatch:
    """Full sync step() with mocked LLM generating tool calls to live MCP tools."""

    async def test_sync_round_trip(self, mcp_http_agent: BaseAgent) -> None:
        """LLM calls evaluate_numeric('2+3'); MCP result flows back correctly."""
        agent = mcp_http_agent
        agent.add_message("user", "What is 2 + 3?")

        turn1 = _tool_call_turn(
            "call_eval", "evaluate_numeric", '{"expression": "2 + 3"}',
        )
        turn2 = _content_turn("The answer is 5.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE
        assert "5" in result.result, f"Expected '5' in result: {result.result!r}"

        # Verify tool result appears in conversation history.
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, "No tool result in message history"
        assert "5" in tool_msgs[0]["content"], (
            f"Expected '5' in tool message: {tool_msgs[0]['content']!r}"
        )

    async def test_sync_differentiate_round_trip(self, mcp_http_agent: BaseAgent) -> None:
        """LLM calls differentiate(x**2, [x]); MCP result contains 2*x."""
        agent = mcp_http_agent
        agent.add_message("user", "Differentiate x squared.")

        turn1 = _tool_call_turn(
            "call_diff", "differentiate",
            '{"expression": "x**2", "variables": ["x"]}',
        )
        turn2 = _content_turn("The derivative of x^2 is 2x.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE

        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, "No tool result in message history"
        tool_content = tool_msgs[0]["content"]
        assert "2*x" in tool_content or "2x" in tool_content, (
            f"Expected derivative in tool result: {tool_content!r}"
        )


# ---------------------------------------------------------------------------
# TestMcpHttpStreamingDispatch
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMcpHttpStreamingDispatch:
    """Streaming astep_stream() with mocked LLM calling live MCP tools."""

    async def test_streaming_mcp_tool_event_ordering(
        self, mcp_http_agent: BaseAgent,
    ) -> None:
        """ToolCallDelta → ToolResultEvent → ContentDelta for MCP tools."""
        agent = mcp_http_agent
        agent.add_message("user", "Simplify sin(x)^2 + cos(x)^2.")

        turn1 = _tool_call_turn(
            "call_simp", "simplify_expression",
            '{"expression": "sin(x)**2 + cos(x)**2", "form": "simplify"}',
        )
        turn2 = _content_turn("It simplifies to 1.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_tool_call_result_ordering(events)
        assert_stream_completes(events)

        tc_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tc_results) == 1, (
            f"Expected 1 ToolResultEvent, got {len(tc_results)}"
        )
        assert tc_results[0].name == "simplify_expression"
        assert "1" in tc_results[0].content, (
            f"Expected '1' in simplification result: {tc_results[0].content!r}"
        )

        content_deltas = [e for e in events if isinstance(e, ContentDelta)]
        assert content_deltas, "Expected ContentDelta after tool result"

    async def test_streaming_metrics_include_mcp_tool(
        self, mcp_http_agent: BaseAgent,
    ) -> None:
        """StreamComplete metrics count MCP tool calls correctly."""
        agent = mcp_http_agent
        agent.add_message("user", "Evaluate pi to 10 digits.")

        turn1 = _tool_call_turn(
            "call_pi", "evaluate_numeric",
            '{"expression": "pi", "precision": 10}',
        )
        turn2 = _content_turn("Pi is approximately 3.141592654.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_stream_completes(events)
        complete = events[-1]
        assert isinstance(complete, StreamComplete)
        assert complete.metrics.tool_calls == 1, (
            f"Expected 1 tool call in metrics, got {complete.metrics.tool_calls}"
        )
        assert complete.metrics.model_calls >= 2, (
            f"Expected >= 2 model calls (tool turn + content turn), "
            f"got {complete.metrics.model_calls}"
        )
