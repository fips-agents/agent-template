"""Integration tests for MCP dispatch through Kagenti MCP Gateway.

The Kagenti MCP Gateway (Envoy-based broker) aggregates tools from
multiple upstream MCP servers and exposes them through a single
streamable-http endpoint.  From the client's perspective it looks like
a normal MCP server — tool discovery uses the same FastMCP v3 Client.

**Current limitation (2026-04-17):**  The Kagenti broker supports
``tools/list`` (discovery) but does NOT forward ``tools/call`` —
it returns: "Kagenti MCP Broker doesn't forward tool calls".
Upstream servers (e.g., ``weather-tool-mcp``) are ClusterIP-only,
unreachable from outside the cluster.  Dispatch tests are marked
``xfail`` until the broker gains forwarding support.

Target: Kagenti MCP Gateway on cluster n7pd5.  Override with the
``KAGENTI_MCP_GATEWAY_URL`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.llm import LLMClient

from .conftest import (
    _content_turn,
    _make_mock_stream,
    _tool_call_turn,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_URL = (
    "https://mcp-gateway-gateway-system.apps.cluster-n7pd5"
    ".n7pd5.sandbox5167.opentlc.com/mcp"
)
KAGENTI_URL = os.environ.get("KAGENTI_MCP_GATEWAY_URL", _DEFAULT_URL)


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
async def kagenti_agent() -> AsyncIterator[BaseAgent]:
    """Agent wired to the Kagenti MCP Gateway.

    Skips if the gateway is unreachable or returns no tools.
    """
    config = _make_config()
    agent = BaseAgent(config=config)

    agent.config = config
    agent.llm = MagicMock(spec=LLMClient)
    agent.messages = [{"role": "system", "content": "You are a weather assistant."}]
    agent._reasoning_parser = None
    agent._setup_done = True

    await agent.connect_mcp(KAGENTI_URL)

    if not agent.tools.get_all():
        await agent.shutdown()
        pytest.skip(
            f"No tools discovered from Kagenti Gateway at {KAGENTI_URL} "
            "(gateway may be unreachable)"
        )

    yield agent

    await agent.shutdown()


# ---------------------------------------------------------------------------
# TestKagentiDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.kagenti
class TestKagentiDiscovery:
    """Verify MCP tool discovery through the Kagenti Gateway.

    Discovery works — the broker aggregates tool listings from all
    registered upstream MCP servers.
    """

    async def test_discovers_tools(self, kagenti_agent: BaseAgent) -> None:
        names = {t.name for t in kagenti_agent.tools.get_all()}
        assert names, "No tools discovered from Kagenti Gateway"
        # The demo deployment registers a weather-tool.
        assert "get_weather" in names, (
            f"Expected 'get_weather' in discovered tools: {sorted(names)}"
        )

    async def test_tools_registered_as_llm_only(
        self, kagenti_agent: BaseAgent,
    ) -> None:
        for meta in kagenti_agent.tools.get_all():
            assert meta.visibility == "llm_only", (
                f"Tool {meta.name!r} has visibility {meta.visibility!r}"
            )

    async def test_tool_schemas_generated(
        self, kagenti_agent: BaseAgent,
    ) -> None:
        schemas = kagenti_agent.tools.generate_schemas()
        assert schemas, "No tool schemas generated"
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]

    async def test_weather_tool_has_city_param(
        self, kagenti_agent: BaseAgent,
    ) -> None:
        schemas = kagenti_agent.tools.generate_schemas()
        weather = next(
            (s for s in schemas if s["function"]["name"] == "get_weather"),
            None,
        )
        assert weather is not None, "get_weather schema not found"
        props = weather["function"].get("parameters", {}).get("properties", {})
        assert "city" in props, f"Expected 'city' param, got: {props}"


# ---------------------------------------------------------------------------
# TestKagentiToolExecution
# ---------------------------------------------------------------------------


@pytest.mark.kagenti
class TestKagentiToolExecution:
    """Direct tool execution through the Kagenti Gateway.

    The broker currently does NOT forward tool calls — these tests
    document the limitation and will pass once forwarding is supported.
    """

    @pytest.mark.xfail(
        reason="Kagenti broker does not forward tool calls (2026-04-17)",
        strict=True,
    )
    async def test_call_tool_through_gateway(
        self, kagenti_agent: BaseAgent,
    ) -> None:
        """Calling a tool through the gateway should return a result."""
        result = await kagenti_agent.tools.execute(
            "get_weather", city="London",
        )
        assert not result.is_error, f"Tool error: {result.error}"
        assert result.result, "Empty result from get_weather"


# ---------------------------------------------------------------------------
# TestKagentiSyncDispatch
# ---------------------------------------------------------------------------


@pytest.mark.kagenti
class TestKagentiSyncDispatch:
    """Full sync step() dispatching through Kagenti Gateway.

    Marked xfail until the broker supports tool call forwarding.
    """

    @pytest.mark.xfail(
        reason="Kagenti broker does not forward tool calls (2026-04-17)",
        strict=True,
    )
    async def test_sync_round_trip(self, kagenti_agent: BaseAgent) -> None:
        agent = kagenti_agent
        agent.add_message("user", "What's the weather in London?")

        turn1 = _tool_call_turn(
            "call_weather", "get_weather", '{"city": "London"}',
        )
        turn2 = _content_turn("The weather in London is mild.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs
        # When forwarding works, this should contain weather data,
        # not an error message.
        assert "ERROR" not in tool_msgs[0]["content"]
