"""Tests for fipsagents.workflow.runner — WorkflowRunner execution engine."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.config import NodeConfig
from fipsagents.workflow.errors import EdgeResolutionError, MaxStepsExceededError
from fipsagents.workflow.graph import Graph
from fipsagents.workflow.node import BaseNode
from fipsagents.workflow.runner import WorkflowRunner
from fipsagents.workflow.state import END, WorkflowState


# ---------------------------------------------------------------------------
# Test state and nodes
# ---------------------------------------------------------------------------


class CountState(WorkflowState):
    count: int = 0
    path: list[str] = []


class IncrementNode(BaseNode):
    async def process(self, state: CountState) -> CountState:
        return state.model_copy(update={
            "count": state.count + 1,
            "path": [*state.path, self.name],
        })


class FailingNode(BaseNode):
    """Fails on first call, succeeds on retry."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    async def process(self, state: CountState) -> CountState:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("transient failure")
        return state.model_copy(update={"path": [*state.path, self.name]})


class AlwaysFailNode(BaseNode):
    async def process(self, state: CountState) -> CountState:
        raise RuntimeError("permanent failure")


class PassthroughNode(BaseNode):
    async def process(self, state: CountState) -> CountState:
        return state


class RecoveryNode(BaseNode):
    async def process(self, state: CountState) -> CountState:
        return state.model_copy(update={
            "path": [*state.path, self.name],
        })


class LifecycleNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_called = False
        self.shutdown_called = False

    async def setup(self):
        self.setup_called = True

    async def shutdown(self):
        self.shutdown_called = True

    async def process(self, state: CountState) -> CountState:
        return state


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _linear_chain(*names: str) -> Graph:
    """Build a linear chain: name[0] -> name[1] -> ... -> END."""
    g = Graph(state_type=CountState)
    nodes = [IncrementNode() for _ in names]
    for name, n in zip(names, nodes):
        g.add_node(name, n)
    for i in range(len(names) - 1):
        g.add_edge(names[i], names[i + 1])
    g.add_edge(names[-1], END)
    g.set_entry_point(names[0])
    return g


# ---------------------------------------------------------------------------
# Linear chain
# ---------------------------------------------------------------------------


class TestLinearChain:
    async def test_three_node_chain(self):
        graph = _linear_chain("a", "b", "c")
        runner = WorkflowRunner(graph)
        result = await runner.start(CountState())

        assert result.count == 3
        assert result.path == ["a", "b", "c"]

    async def test_single_node(self):
        graph = _linear_chain("a")
        runner = WorkflowRunner(graph)
        result = await runner.start(CountState())

        assert result.count == 1
        assert result.path == ["a"]


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


class RouterNode(BaseNode):
    async def process(self, state: CountState) -> CountState:
        return state.model_copy(update={"path": [*state.path, self.name]})


class TestConditionalRouting:
    @pytest.mark.parametrize("initial_count,expected_path", [
        (0, ["router", "low"]),
        (10, ["router", "high"]),
    ])
    async def test_routes_by_count(self, initial_count, expected_path):
        g = Graph(state_type=CountState)
        g.add_node("router", RouterNode())
        g.add_node("low", IncrementNode())
        g.add_node("high", IncrementNode())
        g.set_entry_point("router")
        g.add_conditional_edge(
            "router",
            lambda s: "high" if s.count > 5 else "low",
        )
        g.add_edge("low", END)
        g.add_edge("high", END)

        runner = WorkflowRunner(g)
        result = await runner.start(CountState(count=initial_count))

        assert result.path == expected_path


# ---------------------------------------------------------------------------
# Error edge
# ---------------------------------------------------------------------------


class TestErrorEdge:
    async def test_routes_to_recovery_node(self):
        g = Graph(state_type=CountState)
        g.add_node("bad", AlwaysFailNode())
        g.add_node("recovery", RecoveryNode())
        g.set_entry_point("bad")
        g.add_error_edge("bad", "recovery")
        g.add_edge("recovery", END)

        runner = WorkflowRunner(g, node_retries=1)
        result = await runner.start(CountState())

        assert "recovery" in result.path

    async def test_propagates_without_error_edge(self):
        g = Graph(state_type=CountState)
        g.add_node("bad", AlwaysFailNode())
        g.set_entry_point("bad")
        g.add_edge("bad", END)

        runner = WorkflowRunner(g, node_retries=1)
        with pytest.raises(RuntimeError, match="permanent failure"):
            await runner.start(CountState())


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetry:
    async def test_per_node_retry_succeeds(self):
        g = Graph(state_type=CountState)
        g.add_node("flaky", FailingNode())
        g.add_edge("flaky", END)
        g.set_entry_point("flaky")

        runner = WorkflowRunner(g, node_retries=2)
        result = await runner.start(CountState())

        assert "flaky" in result.path


# ---------------------------------------------------------------------------
# Max steps exceeded
# ---------------------------------------------------------------------------


class TestMaxSteps:
    async def test_exceeds_max_steps(self):
        g = _linear_chain("a", "b", "c", "d", "e")
        runner = WorkflowRunner(g, max_steps=2)

        with pytest.raises(MaxStepsExceededError):
            await runner.start(CountState())


# ---------------------------------------------------------------------------
# Edge resolution error
# ---------------------------------------------------------------------------


class TestEdgeResolution:
    async def test_conditional_returns_nonexistent_node(self):
        g = Graph(state_type=CountState)
        g.add_node("start", PassthroughNode())
        g.set_entry_point("start")
        g.add_conditional_edge("start", lambda s: "nonexistent")

        runner = WorkflowRunner(g)
        with pytest.raises(EdgeResolutionError):
            await runner.start(CountState())


# ---------------------------------------------------------------------------
# Lifecycle (setup / shutdown)
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_calls_setup_and_shutdown(self):
        node = LifecycleNode()
        g = Graph(state_type=CountState)
        g.add_node("lc", node)
        g.add_edge("lc", END)
        g.set_entry_point("lc")

        runner = WorkflowRunner(g)
        await runner.start(CountState())

        assert node.setup_called is True
        assert node.shutdown_called is True


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


class TestLogging:
    async def test_logs_node_transitions(self, caplog):
        graph = _linear_chain("a")
        runner = WorkflowRunner(graph)

        with caplog.at_level(logging.INFO):
            await runner.start(CountState())

        log_messages = " ".join(r.message for r in caplog.records)
        assert "node_transition" in log_messages

    async def test_log_records_contain_node_name(self, caplog):
        graph = _linear_chain("alpha")
        runner = WorkflowRunner(graph)

        with caplog.at_level(logging.INFO):
            await runner.start(CountState())

        extras = [r.__dict__ for r in caplog.records if hasattr(r, "node_name")]
        node_names = [e.get("node_name") for e in extras]
        assert "alpha" in node_names


# ---------------------------------------------------------------------------
# Empty / passthrough state
# ---------------------------------------------------------------------------


class TestPassthroughState:
    async def test_unchanged_state(self):
        g = Graph(state_type=CountState)
        g.add_node("noop", PassthroughNode())
        g.add_edge("noop", END)
        g.set_entry_point("noop")

        initial = CountState(count=42, path=["pre"])
        runner = WorkflowRunner(g)
        result = await runner.start(initial)

        assert result.count == 42
        assert result.path == ["pre"]


# ---------------------------------------------------------------------------
# RemoteNode auto-wrap
# ---------------------------------------------------------------------------


class TestRemoteNodeAutoWrap:
    """WorkflowRunner auto-wraps nodes declared remote in node_configs."""

    async def test_remote_node_is_called_instead_of_local(self):
        """When node_configs declares a node as remote, the runner calls
        RemoteNode.process() instead of the graph node's process()."""
        g = Graph(state_type=CountState)
        local_node = IncrementNode()
        g.add_node("worker", local_node)
        g.add_edge("worker", END)
        g.set_entry_point("worker")

        node_configs = {
            "worker": NodeConfig(
                type="remote",
                endpoint="http://remote:8080",
            ),
        }

        with patch("fipsagents.workflow.runner.RemoteNode") as MockRemoteNode:
            mock_instance = MagicMock()
            mock_instance.process = AsyncMock(
                return_value=CountState(count=99, path=["remote"])
            )
            MockRemoteNode.return_value = mock_instance

            runner = WorkflowRunner(g, node_configs=node_configs)
            result = await runner.start(CountState())

        MockRemoteNode.assert_called_once_with(
            name="worker",
            endpoint="http://remote:8080",
            path="/process",
            timeout=30.0,
            retries=2,
        )
        mock_instance.process.assert_called_once()
        assert result.count == 99

    async def test_local_node_unaffected_by_empty_configs(self):
        """Nodes without remote config execute normally."""
        g = _linear_chain("a")
        runner = WorkflowRunner(g, node_configs={})
        result = await runner.start(CountState())

        assert result.count == 1
        assert result.path == ["a"]

    async def test_mixed_local_and_remote(self):
        """Graph with both local and remote nodes: local nodes run normally,
        remote node is auto-wrapped and its mock state flows to local_c."""
        g = Graph(state_type=CountState)
        g.add_node("local_a", IncrementNode())
        g.add_node("remote_b", IncrementNode())  # will be auto-wrapped
        g.add_node("local_c", IncrementNode())
        g.add_edge("local_a", "remote_b")
        g.add_edge("remote_b", "local_c")
        g.add_edge("local_c", END)
        g.set_entry_point("local_a")

        node_configs = {
            "remote_b": NodeConfig(
                type="remote",
                endpoint="http://b-agent:8080",
            ),
        }

        with patch("fipsagents.workflow.runner.RemoteNode") as MockRemoteNode:
            mock_instance = MagicMock()
            # remote_b returns count=2 (local_a added 1, remote_b "adds" 1)
            mock_instance.process = AsyncMock(
                return_value=CountState(count=2, path=["local_a", "remote_b"])
            )
            MockRemoteNode.return_value = mock_instance

            runner = WorkflowRunner(g, node_configs=node_configs)
            result = await runner.start(CountState())

        # local_a ran (count=1), remote_b was wrapped (count=2),
        # local_c ran normally (count=3).
        assert result.count == 3
        assert "local_a" in result.path
