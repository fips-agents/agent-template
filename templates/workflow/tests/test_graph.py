"""Tests for workflow.graph — Graph definition, edges, and validation."""

from __future__ import annotations

import pytest

from workflow.errors import NodeNotFoundError
from workflow.graph import Graph
from workflow.node import BaseNode
from workflow.state import END, WorkflowState


class SimpleState(WorkflowState):
    value: str = ""


class PassthroughNode(BaseNode):
    async def process(self, state: SimpleState) -> SimpleState:
        return state


def _make_graph() -> Graph:
    return Graph(state_type=SimpleState)


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_registers_node(self):
        g = _make_graph()
        n = PassthroughNode()
        g.add_node("a", n)

        assert "a" in g.nodes
        assert n.name == "a"

    def test_rejects_missing_process(self):
        g = _make_graph()

        class NoProcess:
            pass

        with pytest.raises(TypeError, match="callable 'process'"):
            g.add_node("bad", NoProcess())

    def test_rejects_duplicate_name(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())

        with pytest.raises(ValueError, match="already registered"):
            g.add_node("a", PassthroughNode())


# ---------------------------------------------------------------------------
# add_edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    def test_works_for_registered_nodes(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())
        g.add_node("b", PassthroughNode())
        g.add_edge("a", "b")

        assert g.edges["a"] == "b"

    def test_end_as_target(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())
        g.add_edge("a", END)

        assert g.edges["a"] == END

    def test_unregistered_from_node(self):
        g = _make_graph()
        g.add_node("b", PassthroughNode())

        with pytest.raises(NodeNotFoundError):
            g.add_edge("missing", "b")

    def test_unregistered_to_node(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())

        with pytest.raises(NodeNotFoundError):
            g.add_edge("a", "missing")


# ---------------------------------------------------------------------------
# add_conditional_edge
# ---------------------------------------------------------------------------


class TestAddConditionalEdge:
    def test_stores_edge_function(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())
        fn = lambda s: END  # noqa: E731
        g.add_conditional_edge("a", fn)

        assert g.conditional_edges["a"] is fn

    def test_unregistered_node(self):
        g = _make_graph()

        with pytest.raises(NodeNotFoundError):
            g.add_conditional_edge("missing", lambda s: END)


# ---------------------------------------------------------------------------
# add_error_edge
# ---------------------------------------------------------------------------


class TestAddErrorEdge:
    def test_works_for_registered_nodes(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())
        g.add_node("b", PassthroughNode())
        g.add_error_edge("a", "b")

        assert g.error_edges["a"] == "b"

    def test_unregistered_from_node(self):
        g = _make_graph()
        g.add_node("b", PassthroughNode())

        with pytest.raises(NodeNotFoundError):
            g.add_error_edge("missing", "b")

    def test_unregistered_to_node(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())

        with pytest.raises(NodeNotFoundError):
            g.add_error_edge("a", "missing")


# ---------------------------------------------------------------------------
# set_entry_point
# ---------------------------------------------------------------------------


class TestSetEntryPoint:
    def test_works_for_registered_node(self):
        g = _make_graph()
        g.add_node("start", PassthroughNode())
        g.set_entry_point("start")

        assert g.entry_point == "start"

    def test_unregistered_node(self):
        g = _make_graph()

        with pytest.raises(NodeNotFoundError):
            g.set_entry_point("missing")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_no_entry_point(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())

        with pytest.raises(ValueError, match="No entry point"):
            g.validate()

    def test_linear_and_conditional_edges_conflict(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())
        g.add_node("b", PassthroughNode())
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_conditional_edge("a", lambda s: END)

        with pytest.raises(ValueError, match="both a linear edge and a conditional edge"):
            g.validate()

    def test_valid_graph_passes(self):
        g = _make_graph()
        g.add_node("a", PassthroughNode())
        g.add_edge("a", END)
        g.set_entry_point("a")

        g.validate()  # should not raise


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------


class TestMethodChaining:
    def test_all_methods_return_graph(self):
        g = _make_graph()

        result = g.add_node("a", PassthroughNode())
        assert result is g

        result = g.add_node("b", PassthroughNode())
        assert result is g

        result = g.add_edge("a", "b")
        assert result is g

        result = g.add_conditional_edge("b", lambda s: END)
        assert result is g

        g.add_node("err", PassthroughNode())
        result = g.add_error_edge("a", "err")
        assert result is g

        result = g.set_entry_point("a")
        assert result is g
