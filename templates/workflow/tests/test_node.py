"""Tests for workflow.node — BaseNode lifecycle and interface."""

from __future__ import annotations

import logging

import pytest

from workflow.node import BaseNode
from workflow.state import WorkflowState


class TinyState(WorkflowState):
    value: str = ""


class AppendNode(BaseNode):
    """Node that appends its name to state.value."""

    async def process(self, state: TinyState) -> TinyState:
        return state.model_copy(update={"value": state.value + self.name})


class TestBaseNodeName:
    def test_default_name_is_class_name(self):
        class MyNode(BaseNode):
            pass

        assert MyNode().name == "MyNode"

    def test_custom_name_via_constructor(self):
        node = BaseNode(name="custom")
        assert node.name == "custom"


class TestBaseNodeProcess:
    async def test_process_raises_not_implemented(self):
        node = BaseNode()
        with pytest.raises(NotImplementedError, match="must implement process"):
            await node.process(TinyState())

    async def test_subclass_process_works(self):
        node = AppendNode(name="X")
        result = await node.process(TinyState(value="hello-"))
        assert result.value == "hello-X"


class TestBaseNodeMisc:
    def test_has_logger_attribute(self):
        node = BaseNode(name="test_node")
        assert isinstance(node.logger, logging.Logger)
        assert "test_node" in node.logger.name

    def test_repr_includes_name(self):
        node = BaseNode(name="router")
        r = repr(node)
        assert "router" in r
        assert "BaseNode" in r
