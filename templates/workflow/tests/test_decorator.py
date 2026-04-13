"""Tests for workflow.decorator — @node decorator and NodeMeta."""

from __future__ import annotations

from workflow.decorator import NodeMeta, _NODE_MARKER, node


class TestNodeDecorator:
    def test_bare_decorator(self):
        """@node (no parentheses) attaches NodeMeta using the class name."""

        @node
        class MyNode:
            pass

        meta = getattr(MyNode, _NODE_MARKER)
        assert isinstance(meta, NodeMeta)
        assert meta.name == "MyNode"
        assert meta.error_handler is None

    def test_empty_parens(self):
        """@node() (empty parens) attaches NodeMeta using the class name."""

        @node()
        class AnotherNode:
            pass

        meta = getattr(AnotherNode, _NODE_MARKER)
        assert meta.name == "AnotherNode"
        assert meta.error_handler is None

    def test_custom_name(self):
        @node(name="custom")
        class N:
            pass

        meta = getattr(N, _NODE_MARKER)
        assert meta.name == "custom"

    def test_error_handler(self):
        @node(error_handler="fallback")
        class N:
            pass

        meta = getattr(N, _NODE_MARKER)
        assert meta.error_handler == "fallback"

    def test_does_not_alter_class(self):
        """Decorator returns the same class object, not a wrapper."""

        class Original:
            pass

        decorated = node(Original)
        assert decorated is Original

    def test_nodemeta_fields(self):
        meta = NodeMeta(name="x", error_handler="y")
        assert meta.name == "x"
        assert meta.error_handler == "y"
