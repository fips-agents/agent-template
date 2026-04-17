"""Integration tests for MCP tool dispatch through LlamaStack.

Unlike other harness tests, these use a **real LLM** (gpt-oss-20b via
LlamaStack) instead of mocked.  The flow:

1. Register calculus-helper MCP tools with LlamaStack's tool_runtime
2. BaseAgent calls LlamaStack for inference with tool schemas
3. The model generates tool calls
4. BaseAgent dispatches to the calculus-helper MCP server (direct path)
5. Tool results flow back through the conversation

This validates that LlamaStack's inference endpoint correctly supports
OpenAI-compatible tool calling with MCP-proxied tool schemas.

Prerequisites:
- LlamaStack running with gpt-oss-20b model
- calculus-helper MCP registered as ``mcp::calculus-helper`` tool group
- Both endpoints reachable from the test runner

Override endpoints with env vars:
- ``LLAMASTACK_URL`` (default: LlamaStack on n7pd5)
- ``CALCULUS_HELPER_MCP_URL`` (default: calculus-helper on n7pd5)
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.config import (
    AgentConfig,
    BackoffConfig,
    LLMConfig,
    LoopConfig,
)
from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    ToolResultEvent,
)
from fipsagents.baseagent.llm import LLMClient

from .conftest import (
    assert_stream_completes,
    assert_tool_call_result_ordering,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LLAMASTACK_URL = os.environ.get(
    "LLAMASTACK_URL",
    "http://llamastack-llamastack.apps.cluster-n7pd5"
    ".n7pd5.sandbox5167.opentlc.com",
)

_CALC_MCP_URL = os.environ.get(
    "CALCULUS_HELPER_MCP_URL",
    "https://mcp-server-calculus-helper-mcp.apps.cluster-n7pd5"
    ".n7pd5.sandbox5167.opentlc.com/mcp/",
)

# gpt-oss-20b generates proper tool calls; granite-3.3-8b does not.
_MODEL_NAME = "openai/RedHatAI/gpt-oss-20b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def llamastack_agent() -> AsyncIterator[BaseAgent]:
    """Agent wired to LlamaStack (real LLM) + calculus-helper MCP (real tools).

    Skips if either endpoint is unreachable.
    """
    config = AgentConfig(
        model=LLMConfig(
            endpoint=f"{_LLAMASTACK_URL}/v1",
            name=_MODEL_NAME,
            temperature=0.0,
            max_tokens=512,
        ),
        loop=LoopConfig(
            max_iterations=5,
            backoff=BackoffConfig(initial=0.1, max=1.0, multiplier=2.0),
        ),
    )
    agent = BaseAgent(config=config)

    # Full setup for LLM (real) but skip filesystem prompts/rules/memory.
    agent.config = config
    agent.llm = LLMClient(config.model)
    agent._reasoning_parser = None

    # Connect to calculus-helper MCP for real tool dispatch.
    await agent.connect_mcp(_CALC_MCP_URL)

    if not agent.tools.get_all():
        await agent.shutdown()
        pytest.skip(
            f"No tools discovered from MCP at {_CALC_MCP_URL} "
            "(server may be unreachable)"
        )

    # Seed system prompt.
    agent.messages = [
        {
            "role": "system",
            "content": (
                "You are a precise calculator. You MUST use the "
                "evaluate_numeric tool for ALL math computations. "
                "Never compute results yourself. After receiving "
                "tool results, present the answer to the user."
            ),
        },
    ]
    agent._setup_done = True

    yield agent

    await agent.shutdown()


def _skip_if_llamastack_down():
    """Module-level check — skip entire file if LlamaStack is unreachable."""
    import httpx

    try:
        resp = httpx.get(f"{_LLAMASTACK_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.llamastack,
    pytest.mark.skipif(
        not _skip_if_llamastack_down(),
        reason=f"LlamaStack not reachable at {_LLAMASTACK_URL}",
    ),
]


# ---------------------------------------------------------------------------
# TestLlamaStackToolCallGeneration
# ---------------------------------------------------------------------------


class TestLlamaStackToolCallGeneration:
    """Verify the LLM (via LlamaStack) generates proper tool calls."""

    async def test_model_generates_tool_call(
        self, llamastack_agent: BaseAgent,
    ) -> None:
        """gpt-oss-20b returns a tool_call for a math expression."""
        agent = llamastack_agent
        agent.add_message(
            "user", "Compute sqrt(pi) * erf(1) / 2 to 15 digits."
        )

        response = await agent.call_model()

        assert response.tool_calls, (
            f"Expected tool calls, got content: {response.content!r}"
        )
        tc = response.tool_calls[0]
        fn = getattr(tc, "function", tc)
        assert fn.name == "evaluate_numeric", (
            f"Expected evaluate_numeric, got {fn.name!r}"
        )
        assert "expression" in fn.arguments, (
            f"Expected 'expression' in arguments: {fn.arguments!r}"
        )


# ---------------------------------------------------------------------------
# TestLlamaStackSyncDispatch
# ---------------------------------------------------------------------------


class TestLlamaStackSyncDispatch:
    """Full sync step() with real LLM + real MCP tool execution."""

    async def test_full_round_trip(
        self, llamastack_agent: BaseAgent,
    ) -> None:
        """LLM calls evaluate_numeric → MCP executes → result in final response."""
        agent = llamastack_agent
        agent.add_message(
            "user", "Compute sqrt(pi) * erf(1) / 2 to 15 digits."
        )

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE, (
            f"Expected DONE, got {result.outcome}"
        )
        # The exact value is ~0.746824132812427
        assert "0.746" in result.result, (
            f"Expected '0.746' in result: {result.result!r}"
        )

        # Verify tool messages in conversation history.
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, "No tool result in message history"
        assert "0.746" in tool_msgs[0]["content"], (
            f"Expected '0.746' in tool result: {tool_msgs[0]['content']!r}"
        )


# ---------------------------------------------------------------------------
# TestLlamaStackStreamingDispatch
# ---------------------------------------------------------------------------


class TestLlamaStackStreamingDispatch:
    """Streaming astep_stream() with real LLM + real MCP tool execution."""

    async def test_streaming_event_ordering(
        self, llamastack_agent: BaseAgent,
    ) -> None:
        """ToolCallDelta → ToolResultEvent → ContentDelta with real LLM."""
        agent = llamastack_agent
        agent.add_message(
            "user", "Compute sqrt(pi) * erf(1) / 2 to 15 digits."
        )

        events = [event async for event in agent.astep_stream()]

        assert_tool_call_result_ordering(events)
        assert_stream_completes(events)

        tc_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert tc_results, "No ToolResultEvent in stream"
        assert "0.746" in tc_results[0].content, (
            f"Expected '0.746' in tool result: {tc_results[0].content!r}"
        )

        content_deltas = [e for e in events if isinstance(e, ContentDelta)]
        assert content_deltas, "No ContentDelta after tool result"

        # Verify final content mentions the result.
        full_content = "".join(e.content for e in content_deltas)
        assert "0.746" in full_content, (
            f"Expected '0.746' in streamed content: {full_content!r}"
        )

    async def test_streaming_metrics(
        self, llamastack_agent: BaseAgent,
    ) -> None:
        """StreamComplete metrics reflect real model calls and tool execution."""
        agent = llamastack_agent
        agent.add_message(
            "user", "Compute sqrt(2) to 10 digits."
        )

        events = [event async for event in agent.astep_stream()]

        assert_stream_completes(events)
        complete = events[-1]
        assert isinstance(complete, StreamComplete)
        assert complete.metrics.model_calls >= 1, (
            f"Expected >= 1 model calls, got {complete.metrics.model_calls}"
        )
        # May or may not use a tool depending on model behavior, so
        # just verify metrics are populated.
        assert complete.metrics.total_time > 0
