"""Dynamic discovery of workflow components: state model, graph builder, and tools."""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache

from evals import _TEMPLATE_ROOT


@lru_cache(maxsize=1)
def _discover_build_graph() -> callable:
    """Find the ``build_graph()`` function in agent.py.

    Raises RuntimeError with an actionable message if not found.
    """
    agent_module = importlib.import_module("agent")
    fn = getattr(agent_module, "build_graph", None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            "No build_graph() function found in agent.py. "
            "Run /create-agent to generate your workflow."
        )
    return fn


@lru_cache(maxsize=1)
def _discover_state_class() -> type:
    """Find the WorkflowState subclass in agent.py.

    Returns the first Pydantic model that subclasses WorkflowState.
    Raises RuntimeError if none found.
    """
    from workflow import WorkflowState

    agent_module = importlib.import_module("agent")
    candidates = [
        obj
        for _name, obj in inspect.getmembers(agent_module, inspect.isclass)
        if issubclass(obj, WorkflowState) and obj is not WorkflowState
    ]
    if len(candidates) == 0:
        raise RuntimeError(
            "No WorkflowState subclass found in agent.py. "
            "Run /create-agent to generate your workflow."
        )
    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise RuntimeError(
            f"Multiple WorkflowState subclasses in agent.py: {names}. "
            "The eval runner expects exactly one."
        )
    return candidates[0]


@lru_cache(maxsize=1)
def _discover_llm_tool_name() -> str | None:
    """Find the name of an LLM-visible tool for mock tool call responses.

    Falls back to None if no tools are found, in which case mock responses
    will skip tool call simulation and return text directly.
    """
    try:
        from base_agent.tools import ToolRegistry
        registry = ToolRegistry()
        registry.discover(_TEMPLATE_ROOT / "tools")
        for t in registry.get_all():
            if t.visibility in ("llm_only", "both"):
                return t.name
    except Exception:
        pass
    return None
