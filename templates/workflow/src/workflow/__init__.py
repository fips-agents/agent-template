"""Workflow framework — re-exported from the fipsagents package.

This shim provides backwards compatibility for projects that import
from ``workflow`` directly. New code should import from
``fipsagents.workflow`` instead.
"""

from fipsagents.workflow import (
    END,
    AgentNode,
    BaseNode,
    EdgeResolutionError,
    Graph,
    MaxStepsExceededError,
    NodeNotFoundError,
    StateValidationError,
    WorkflowError,
    WorkflowNode,
    WorkflowRunner,
    WorkflowState,
    node,
)

__all__ = [
    "WorkflowState",
    "END",
    "BaseNode",
    "WorkflowNode",
    "node",
    "Graph",
    "WorkflowRunner",
    "AgentNode",
    "WorkflowError",
    "NodeNotFoundError",
    "EdgeResolutionError",
    "StateValidationError",
    "MaxStepsExceededError",
]
