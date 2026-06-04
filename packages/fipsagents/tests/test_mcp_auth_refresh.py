"""Tests for MCP token refresh feature (#225).

Covers:
- _McpClientRef — mutable container for MCP client references.
- _is_auth_error — detection of 401/403 auth errors from various formats.
- _register_mcp_tool — auth retry in the tool closure on auth errors.
- _refresh_mcp_auth — hook firing, env var parsing, and client reconnect.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.agent import (
    BaseAgent,
    _McpClientRef,
    _is_auth_error,
    _register_mcp_tool,
)
from fipsagents.baseagent.config import AgentConfig, McpServerConfig, MemoryConfig
from fipsagents.baseagent.hooks import HookEntry, HookRunner
from fipsagents.baseagent.memory import NullMemoryClient
from fipsagents.baseagent.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Stubs for minimal BaseAgent construction (mirrors test_hooks.py pattern)
# ---------------------------------------------------------------------------


class _RenderedPrompt:
    def render(self) -> str:
        return "You are a test agent."


class _PromptStub:
    def get(self, name: str) -> _RenderedPrompt:
        return _RenderedPrompt()


class _RulesStub:
    def get_combined_content(self) -> str:
        return ""


class _SkillsStub:
    def get_manifest(self) -> list:
        return []


def _make_agent(
    *,
    hooks: HookRunner | None = None,
    base_dir: Path | None = None,
) -> BaseAgent:
    """Build a minimal BaseAgent for MCP auth refresh tests."""
    config = AgentConfig(
        model={"name": "test-model", "endpoint": "http://localhost:1234/v1"},
        memory=MemoryConfig(backend="null"),
    )

    class _TestAgent(BaseAgent):
        def __init__(self) -> None:
            self.config = config
            self.memory = NullMemoryClient()
            self.hooks = hooks or HookRunner()
            self.messages: list[dict[str, Any]] = []
            self.prompts = _PromptStub()
            self.rules = _RulesStub()
            self.skills = _SkillsStub()
            self._base_dir = base_dir
            self._config_path = Path("/tmp/agent.yaml")
            self._setup_done = True
            self._mcp_clients: list[_McpClientRef] = []
            self._mcp_prompts: dict[str, tuple[Any, Any]] = {}
            self._mcp_resources: dict[str, tuple[Any, Any]] = {}
            self._mcp_resource_templates: dict[str, tuple[Any, Any]] = {}

    return _TestAgent()


def _make_mcp_tool_stub(
    name: str = "test_tool",
    description: str = "A test tool",
    input_schema: dict | None = None,
) -> MagicMock:
    """Build a mock MCP tool object with the expected attributes."""
    mock_tool = MagicMock()
    mock_tool.name = name
    mock_tool.description = description
    mock_tool.inputSchema = input_schema or {"type": "object", "properties": {}}
    return mock_tool


# ---------------------------------------------------------------------------
# Section 1: _McpClientRef
# ---------------------------------------------------------------------------


def test_client_ref_attributes():
    """_McpClientRef stores client, label, config, header_templates."""
    mock_client = MagicMock()
    config = McpServerConfig(url="http://test:8080")
    ref = _McpClientRef(
        mock_client, "test-server", config, {"Auth": "Bearer ${TOKEN}"}
    )
    assert ref.client is mock_client
    assert ref.label == "test-server"
    assert ref.config is config
    assert ref.header_templates == {"Auth": "Bearer ${TOKEN}"}
    assert isinstance(ref._reconnect_lock, asyncio.Lock)


def test_client_ref_swap():
    """Swapping ref.client is visible to closures that captured the ref."""
    old = MagicMock()
    new = MagicMock()
    ref = _McpClientRef(old, "test")
    captured = ref  # simulates closure capture
    assert captured.client is old
    ref.client = new
    assert captured.client is new


# ---------------------------------------------------------------------------
# Section 2: _is_auth_error
# ---------------------------------------------------------------------------


def test_is_auth_error_401_status():
    """httpx-style 401 response detected."""
    exc = Exception("HTTP error")
    exc.response = MagicMock()
    exc.response.status_code = 401
    assert _is_auth_error(exc) is True


def test_is_auth_error_403_status():
    """httpx-style 403 response detected."""
    exc = Exception("HTTP error")
    exc.response = MagicMock()
    exc.response.status_code = 403
    assert _is_auth_error(exc) is True


def test_is_auth_error_500_not_auth():
    """Non-auth HTTP status is not treated as an auth error."""
    exc = Exception("HTTP error")
    exc.response = MagicMock()
    exc.response.status_code = 500
    assert _is_auth_error(exc) is False


def test_is_auth_error_by_class_name():
    """AuthorizationError class name detected regardless of message."""

    class AuthorizationError(Exception):
        pass

    assert _is_auth_error(AuthorizationError("denied")) is True


def test_is_auth_error_string_match():
    """Auth keywords in exception message string trigger detection."""
    assert _is_auth_error(Exception("401 Unauthorized")) is True
    assert _is_auth_error(Exception("HTTP 403 Forbidden")) is True


def test_is_auth_error_generic():
    """Non-auth exception messages are not false positives."""
    assert _is_auth_error(Exception("connection reset")) is False
    assert _is_auth_error(Exception("timeout after 30s")) is False


# ---------------------------------------------------------------------------
# Section 3: Auth retry in _register_mcp_tool closure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_closure_retry_on_auth_error():
    """Auth error triggers reconnect_fn, retry succeeds on second call."""
    mock_client = AsyncMock()
    # First call raises 401, second call returns success.
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="success")]
    auth_exc = Exception("401 Unauthorized")
    mock_client.call_tool = AsyncMock(side_effect=[auth_exc, mock_result])

    ref = _McpClientRef(mock_client, "test-server")
    reconnect_fn = AsyncMock(return_value=True)

    registry = ToolRegistry()
    _register_mcp_tool(
        registry, ref, _make_mcp_tool_stub("retry_tool"),
        reconnect_fn=reconnect_fn,
    )

    result = await registry.execute("retry_tool")
    assert not result.is_error, f"Expected success, got error: {result.error}"
    assert "success" in result.result
    reconnect_fn.assert_called_once_with(ref)


@pytest.mark.asyncio
async def test_tool_closure_no_retry_on_reconnect_failure():
    """When reconnect returns False, the original auth error propagates."""
    mock_client = AsyncMock()
    auth_exc = Exception("401 Unauthorized")
    mock_client.call_tool = AsyncMock(side_effect=auth_exc)

    ref = _McpClientRef(mock_client, "test-server")
    reconnect_fn = AsyncMock(return_value=False)

    registry = ToolRegistry()
    _register_mcp_tool(
        registry, ref, _make_mcp_tool_stub("fail_tool"),
        reconnect_fn=reconnect_fn,
    )

    result = await registry.execute("fail_tool")
    assert result.is_error
    assert "401" in result.error
    reconnect_fn.assert_called_once_with(ref)


@pytest.mark.asyncio
async def test_tool_closure_no_retry_on_non_auth_error():
    """Non-auth errors propagate immediately without reconnect attempt."""
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(side_effect=Exception("connection reset"))

    ref = _McpClientRef(mock_client, "test-server")
    reconnect_fn = AsyncMock()

    registry = ToolRegistry()
    _register_mcp_tool(
        registry, ref, _make_mcp_tool_stub("conn_tool"),
        reconnect_fn=reconnect_fn,
    )

    result = await registry.execute("conn_tool")
    assert result.is_error
    assert "connection reset" in result.error
    reconnect_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Section 4: _refresh_mcp_auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_parses_hook_stdout_to_env(tmp_path, monkeypatch):
    """Hook stdout KEY=VALUE lines are parsed into os.environ."""
    hook = HookEntry(event="mcp_auth_refresh", command="echo test")
    agent = _make_agent(hooks=HookRunner([hook]), base_dir=tmp_path)

    old_client = AsyncMock()
    config = McpServerConfig(
        url="http://test:8080", headers={"Authorization": "Bearer old"},
    )
    ref = _McpClientRef(
        old_client, "http://test:8080", config,
        {"Authorization": "Bearer ${MCP_AUTH_TOKEN}"},
    )

    # Mock subprocess for hook execution.
    proc = AsyncMock()
    proc.communicate = AsyncMock(
        return_value=(b"MCP_AUTH_TOKEN=refreshed-abc\n", b""),
    )
    proc.returncode = 0
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()

    # Mock FastMCP imports used inside _refresh_mcp_auth.
    new_client_mock = AsyncMock()
    mock_mcp_client_cls = MagicMock(return_value=new_client_mock)
    mock_transport_cls = MagicMock()

    mock_fastmcp = MagicMock()
    mock_fastmcp.Client = mock_mcp_client_cls
    mock_transports = MagicMock()
    mock_transports.StreamableHttpTransport = mock_transport_cls

    with patch("asyncio.create_subprocess_shell", return_value=proc), \
         patch.dict(sys.modules, {
             "fastmcp": mock_fastmcp,
             "fastmcp.client": MagicMock(),
             "fastmcp.client.transports": mock_transports,
         }):
        result = await agent._refresh_mcp_auth(ref)

    assert result is True
    assert os.environ.get("MCP_AUTH_TOKEN") == "refreshed-abc"
    # The client ref should now point to the new client.
    assert ref.client is new_client_mock
    # Transport should have been created with the refreshed header.
    mock_transport_cls.assert_called_once()
    call_kwargs = mock_transport_cls.call_args
    assert call_kwargs[1]["url"] == "http://test:8080"
    resolved_headers = call_kwargs[1]["headers"]
    assert resolved_headers["Authorization"] == "Bearer refreshed-abc"

    # Clean up env pollution.
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)


@pytest.mark.asyncio
async def test_refresh_reconnects_with_fresh_client(tmp_path, monkeypatch):
    """_refresh_mcp_auth swaps the client ref and closes the old one."""
    hook = HookEntry(event="mcp_auth_refresh", command="echo OK")
    agent = _make_agent(hooks=HookRunner([hook]), base_dir=tmp_path)

    old_client = AsyncMock()
    config = McpServerConfig(
        url="http://mcp-server:9090",
        headers={"X-Api-Key": "old-key"},
    )
    ref = _McpClientRef(old_client, "http://mcp-server:9090", config)

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()

    new_client_mock = AsyncMock()
    mock_mcp_client_cls = MagicMock(return_value=new_client_mock)
    mock_transport_cls = MagicMock()

    mock_fastmcp = MagicMock()
    mock_fastmcp.Client = mock_mcp_client_cls
    mock_transports = MagicMock()
    mock_transports.StreamableHttpTransport = mock_transport_cls

    with patch("asyncio.create_subprocess_shell", return_value=proc), \
         patch.dict(sys.modules, {
             "fastmcp": mock_fastmcp,
             "fastmcp.client": MagicMock(),
             "fastmcp.client.transports": mock_transports,
         }):
        result = await agent._refresh_mcp_auth(ref)

    assert result is True
    # Old client should have been closed.
    old_client.__aexit__.assert_called_once_with(None, None, None)
    # New client should have been entered and assigned.
    new_client_mock.__aenter__.assert_called_once()
    assert ref.client is new_client_mock


@pytest.mark.asyncio
async def test_refresh_returns_false_on_reconnect_failure(tmp_path):
    """_refresh_mcp_auth returns False when the new client fails to connect."""
    hook = HookEntry(event="mcp_auth_refresh", command="echo test")
    agent = _make_agent(hooks=HookRunner([hook]), base_dir=tmp_path)

    old_client = AsyncMock()
    config = McpServerConfig(url="http://dead-server:8080")
    ref = _McpClientRef(old_client, "http://dead-server:8080", config)

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()

    # Make the new client construction raise.
    mock_mcp_client_cls = MagicMock(side_effect=ConnectionError("refused"))
    mock_transport_cls = MagicMock()

    mock_fastmcp = MagicMock()
    mock_fastmcp.Client = mock_mcp_client_cls
    mock_transports = MagicMock()
    mock_transports.StreamableHttpTransport = mock_transport_cls

    with patch("asyncio.create_subprocess_shell", return_value=proc), \
         patch.dict(sys.modules, {
             "fastmcp": mock_fastmcp,
             "fastmcp.client": MagicMock(),
             "fastmcp.client.transports": mock_transports,
         }):
        result = await agent._refresh_mcp_auth(ref)

    assert result is False
    # Client ref should NOT have been updated on failure.
    assert ref.client is not mock_mcp_client_cls.return_value
