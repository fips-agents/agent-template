"""Integration tests for authenticated MCP streamable-http transport.

Validates tool discovery, session authentication, and tool execution
against MemoryHub's MCP server, which requires ``register_session``
before any other tool call will succeed.

Target: MemoryHub MCP on cluster n7pd5.  Override with env vars:
- ``MEMORYHUB_MCP_URL``
- ``MEMORYHUB_API_KEY`` (or reads from ``~/.config/memoryhub/api-key``)

All tests use read-only operations (memory with action=status,
list_projects, search) to avoid polluting the production MemoryHub
instance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.events import ToolResultEvent
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
    "https://memory-hub-mcp-memory-hub-mcp.apps.cluster-n7pd5"
    ".n7pd5.sandbox5167.opentlc.com/mcp/"
)
_MCP_URL = os.environ.get("MEMORYHUB_MCP_URL", _DEFAULT_URL)

_KEY_FILE = Path.home() / ".config" / "memoryhub" / "api-key"


def _load_api_key() -> str | None:
    """Read the API key from env var or key file."""
    key = os.environ.get("MEMORYHUB_API_KEY")
    if key:
        return key.strip()
    if _KEY_FILE.is_file():
        return _KEY_FILE.read_text().strip()
    return None


_API_KEY = _load_api_key()

# Expected tools (subset — enough to validate discovery).
_EXPECTED_TOOLS = {
    "register_session",
    "memory",
}


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.mcp_http,
    pytest.mark.skipif(
        _API_KEY is None,
        reason="MemoryHub API key not found (set MEMORYHUB_API_KEY or "
        f"create {_KEY_FILE})",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> AgentConfig:
    defaults = {
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
async def memoryhub_agent() -> AsyncIterator[BaseAgent]:
    """Agent wired to MemoryHub MCP with session authentication.

    Skips if the server is unreachable or returns no tools.
    After connecting, calls ``register_session`` to authenticate.
    """
    config = _make_config()
    agent = BaseAgent(config=config)

    agent.config = config
    agent.llm = MagicMock(spec=LLMClient)
    agent.messages = [{"role": "system", "content": "You are a memory assistant."}]
    agent._reasoning_parser = None
    agent._setup_done = True

    # Connect to MemoryHub MCP.
    await agent.connect_mcp(_MCP_URL)

    if not agent.tools.get_all():
        await agent.shutdown()
        pytest.skip(
            f"No tools discovered from MemoryHub MCP at {_MCP_URL} "
            "(server may be unreachable)"
        )

    # Authenticate — required before any other tool call.
    auth_result = await agent.tools.execute(
        "register_session", api_key=_API_KEY,
    )
    if auth_result.is_error:
        await agent.shutdown()
        pytest.skip(f"register_session failed: {auth_result.error}")

    yield agent

    await agent.shutdown()


# ---------------------------------------------------------------------------
# TestMemoryHubDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMemoryHubDiscovery:
    """Verify MCP tool discovery from MemoryHub."""

    async def test_discovers_expected_tools(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        names = {t.name for t in memoryhub_agent.tools.get_all()}
        missing = _EXPECTED_TOOLS - names
        assert not missing, (
            f"Expected tools not discovered: {missing} "
            f"(got: {sorted(names)})"
        )

    async def test_tools_registered_as_llm_only(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        for meta in memoryhub_agent.tools.get_all():
            assert meta.visibility == "llm_only"

    async def test_tool_count(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        """MemoryHub exposes 2 tools: register_session + memory."""
        tools = memoryhub_agent.tools.get_all()
        assert len(tools) == 2, (
            f"Expected 2 tools, got {len(tools)}: "
            f"{sorted(t.name for t in tools)}"
        )


# ---------------------------------------------------------------------------
# TestMemoryHubAuthentication
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMemoryHubAuthentication:
    """Verify session authentication via register_session."""

    async def test_get_session_returns_authenticated(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        """After register_session, memory(status) confirms authenticated=true."""
        result = await memoryhub_agent.tools.execute("memory", action="status")
        assert not result.is_error, f"memory(status) error: {result.error}"
        assert "authenticated" in result.result
        assert "true" in result.result.lower() or "True" in result.result

    async def test_get_session_returns_user_id(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        result = await memoryhub_agent.tools.execute("memory", action="status")
        assert not result.is_error
        assert "user_id" in result.result


# ---------------------------------------------------------------------------
# TestMemoryHubToolExecution
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMemoryHubToolExecution:
    """Execute MemoryHub tools directly (read-only operations only)."""

    async def test_manage_project_list(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        result = await memoryhub_agent.tools.execute(
            "memory", action="list_projects",
        )
        assert not result.is_error, f"memory(list_projects) error: {result.error}"
        assert "projects" in result.result

    async def test_search_memory(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        """memory(search) returns results (possibly empty) without error."""
        result = await memoryhub_agent.tools.execute(
            "memory", action="search", query="test query that probably matches nothing",
        )
        assert not result.is_error, f"memory(search) error: {result.error}"
        # Should return a results structure even if empty.
        assert "results" in result.result or "total" in result.result


# ---------------------------------------------------------------------------
# TestMemoryHubSyncDispatch
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMemoryHubSyncDispatch:
    """Full sync step() with mocked LLM calling authenticated MCP tools."""

    async def test_sync_get_session_round_trip(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        """LLM calls get_session; authenticated result flows back."""
        agent = memoryhub_agent
        agent.add_message("user", "Check my session status.")

        turn1 = _tool_call_turn("call_sess", "memory", '{"action": "status"}')
        turn2 = _content_turn("You are authenticated as wjackson.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE

        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, "No tool result in message history"
        assert "authenticated" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# TestMemoryHubStreamingDispatch
# ---------------------------------------------------------------------------


@pytest.mark.mcp_http
class TestMemoryHubStreamingDispatch:
    """Streaming astep_stream() with mocked LLM calling authenticated MCP tools."""

    async def test_streaming_event_ordering(
        self, memoryhub_agent: BaseAgent,
    ) -> None:
        agent = memoryhub_agent
        agent.add_message("user", "List my projects.")

        turn1 = _tool_call_turn(
            "call_proj", "memory", '{"action": "list_projects"}',
        )
        turn2 = _content_turn("Here are your projects.")
        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_tool_call_result_ordering(events)
        assert_stream_completes(events)

        tc_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tc_results) == 1
        assert tc_results[0].name == "memory"
        assert "projects" in tc_results[0].content


# ---------------------------------------------------------------------------
# TestMemoryHubRealLLM
# ---------------------------------------------------------------------------

_LLM_ENDPOINT = os.environ.get(
    "MEMORYHUB_LLM_ENDPOINT",
    "https://gpt-oss-20b-gpt-oss-model.apps.cluster-n7pd5"
    ".n7pd5.sandbox5167.opentlc.com/v1",
)
_LLM_MODEL = os.environ.get("MEMORYHUB_LLM_MODEL", "RedHatAI/gpt-oss-20b")


@pytest.fixture
async def memoryhub_real_llm_agent() -> AsyncIterator[BaseAgent]:
    """Agent with real LLM + real MemoryHub MCP for end-to-end testing.

    Skips if either the MCP server or LLM endpoint is unreachable.
    """
    config = _make_config(
        model=LLMConfig(
            endpoint=_LLM_ENDPOINT,
            name=_LLM_MODEL,
            temperature=0.0,
            max_tokens=256,
        ),
    )
    agent = BaseAgent(config=config)

    agent.config = config
    agent.llm = LLMClient(config.model)
    agent.messages = [
        {
            "role": "system",
            "content": (
                "You are a memory assistant. You have access to a 'memory' tool "
                "that accepts an 'action' parameter. Use it to answer user questions. "
                "When asked about session status, call the memory tool with "
                'action="status". Always use the tools available to you.'
            ),
        }
    ]
    agent._reasoning_parser = None
    agent._setup_done = True

    # Connect to MemoryHub MCP.
    await agent.connect_mcp(_MCP_URL)

    if not agent.tools.get_all():
        await agent.shutdown()
        pytest.skip(
            f"No tools discovered from MemoryHub MCP at {_MCP_URL} "
            "(server may be unreachable)"
        )

    # Authenticate — required before any other tool call.
    auth_result = await agent.tools.execute(
        "register_session", api_key=_API_KEY,
    )
    if auth_result.is_error:
        await agent.shutdown()
        pytest.skip(f"register_session failed: {auth_result.error}")

    yield agent

    await agent.shutdown()


@pytest.mark.mcp_http
class TestMemoryHubRealLLM:
    """End-to-end tests with a real LLM (gpt-oss-20b) and live MemoryHub MCP.

    These tests verify the full round-trip: user message → LLM decides to
    call a MemoryHub tool → BaseAgent dispatches via MCP → result flows
    back into the conversation → LLM produces a final response.
    """

    @pytest.mark.timeout(60)
    async def test_real_llm_calls_memory_status(
        self, memoryhub_real_llm_agent: BaseAgent,
    ) -> None:
        """Real LLM calls memory(status) and returns session info."""
        agent = memoryhub_real_llm_agent
        agent.add_message("user", "What is my current session status? Use the memory tool.")

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE
        # The LLM should have called memory(status) and received auth info.
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, (
            "LLM did not call any tools — expected memory(status). "
            f"Messages: {[m.get('role') for m in agent.messages]}"
        )
        # The tool result should contain session info.
        assert any("authenticated" in m["content"] for m in tool_msgs), (
            f"No tool result contains 'authenticated'. "
            f"Tool results: {[m['content'][:200] for m in tool_msgs]}"
        )
        # The final assistant response should exist.
        assistant_msgs = [m for m in agent.messages if m.get("role") == "assistant"]
        assert assistant_msgs, "No assistant message after tool call"
        # Check that assistant has content (tool call path should produce content)
        assistant_content = [m.get("content") for m in assistant_msgs]
        assert any(c for c in assistant_content), (
            f"Assistant message exists but has no content. "
            f"Messages: {assistant_msgs}"
        )
