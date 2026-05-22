"""Tests for stock tool discovery mechanism in fipsagents."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.tools import ToolMeta, ToolRegistry
from fipsagents.baseagent.tools._stock import StockToolSpec


# ---------------------------------------------------------------------------
# Mock agent factory
# ---------------------------------------------------------------------------


def _make_mock_agent(*, subagents=None):
    """Create a minimal mock agent for stock tool discovery."""
    agent = MagicMock()
    agent.subagents = subagents or {}
    agent._question_pending = None
    agent._question_events = []
    agent._subagent_events = []
    agent._subagent_token_usage = []
    agent.config = MagicMock()
    agent.config.subagent_defaults = MagicMock()
    agent.config.subagent_defaults.timeout = 30
    agent.config.subagent_defaults.max_delegation_depth = 3
    return agent


# ---------------------------------------------------------------------------
# StockToolSpec tests
# ---------------------------------------------------------------------------


def test_stock_tool_spec_no_condition():
    """StockToolSpec with condition=None — factory should always proceed."""
    called = []

    def factory(agent):
        called.append(agent)
        return lambda: "tool"

    spec = StockToolSpec(factory=factory)
    assert spec.condition is None
    agent = _make_mock_agent()
    result = spec.factory(agent)
    assert result() == "tool"
    assert called == [agent]


def test_stock_tool_spec_with_true_condition():
    """StockToolSpec where condition returns True — factory should be called."""
    called = []

    def factory(agent):
        called.append(agent)
        return lambda: "tool"

    def condition(agent):
        return True

    spec = StockToolSpec(factory=factory, condition=condition)
    agent = _make_mock_agent()
    assert spec.condition(agent) is True
    result = spec.factory(agent)
    assert result() == "tool"
    assert called == [agent]


def test_stock_tool_spec_with_false_condition():
    """StockToolSpec where condition returns False — factory should not be called."""

    def factory(agent):
        raise AssertionError("Factory should not be called")

    def condition(agent):
        return False

    spec = StockToolSpec(factory=factory, condition=condition)
    agent = _make_mock_agent()
    assert spec.condition(agent) is False


# ---------------------------------------------------------------------------
# ToolRegistry.discover_stock() tests
# ---------------------------------------------------------------------------


def test_discover_stock_registers_question_tool():
    """discover_stock registers ask_user when question attributes present."""
    registry = ToolRegistry()
    agent = _make_mock_agent()

    metas = registry.discover_stock(agent)

    assert len(metas) >= 1
    names = [m.name for m in metas]
    assert "ask_user" in names


def test_discover_stock_skips_delegate_without_subagents():
    """discover_stock skips delegate_to_agent when subagents={} (empty)."""
    registry = ToolRegistry()
    agent = _make_mock_agent(subagents={})

    metas = registry.discover_stock(agent)

    names = [m.name for m in metas]
    assert "delegate_to_agent" not in names
    assert "ask_user" in names


def test_discover_stock_registers_delegate_with_subagents():
    """discover_stock registers both tools when subagents is non-empty."""
    mock_subagent_config = MagicMock()
    mock_subagent_config.when_to_use = "Test subagent"
    mock_subagent_config.transport = MagicMock()
    mock_subagent_config.transport.type = "remote"
    mock_subagent_config.max_depth = 3

    registry = ToolRegistry()
    agent = _make_mock_agent(subagents={"test": mock_subagent_config})

    metas = registry.discover_stock(agent)

    names = [m.name for m in metas]
    assert "delegate_to_agent" in names
    assert "ask_user" in names


def test_discover_stock_returns_tool_metas():
    """Verify discover_stock returns a list of ToolMeta instances."""
    registry = ToolRegistry()
    agent = _make_mock_agent()

    metas = registry.discover_stock(agent)

    assert isinstance(metas, list)
    assert all(isinstance(m, ToolMeta) for m in metas)


def test_discover_stock_idempotent():
    """Calling discover_stock twice raises ValueError on duplicate registration."""
    registry = ToolRegistry()
    agent = _make_mock_agent()

    registry.discover_stock(agent)

    with pytest.raises(ValueError, match="already registered"):
        registry.discover_stock(agent)
