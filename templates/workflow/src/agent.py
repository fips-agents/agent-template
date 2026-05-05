"""Workflow definition — typed state, nodes, and graph wiring.

This is a minimal two-node skeleton: one ``BaseNode`` for routing /
transformation (no LLM), one ``AgentNode`` for an LLM-backed step. Wire
them with ``Graph`` and run via ``WorkflowRunner``.

Replace the node and state names with your domain — ``/create-agent``
does this from ``AGENT_PLAN.md``. See CLAUDE.md for the full BaseNode /
AgentNode / Graph reference.
"""

from __future__ import annotations

import asyncio
import logging

from fipsagents.workflow import (
    END,
    AgentNode,
    BaseNode,
    Graph,
    WorkflowRunner,
    WorkflowState,
    node,
)

logger = logging.getLogger(__name__)


# -- State -------------------------------------------------------------------

class MyState(WorkflowState):
    """Typed state flowing through the workflow."""

    query: str
    result: str = ""


# -- Nodes -------------------------------------------------------------------

@node()
class PrepareNode(BaseNode):
    """Lightweight routing / transformation node. No LLM."""

    async def process(self, state: MyState) -> MyState:
        # Replace this with your routing or transformation logic.
        return state


@node()
class RespondNode(AgentNode):
    """Full-agent node — calls the model and writes to state."""

    async def process(self, state: MyState) -> MyState:
        self.add_message("user", state.query)
        response = await self.call_model(include_tools=False)
        return state.model_copy(update={"result": response.content or ""})


# -- Graph -------------------------------------------------------------------

def build_graph() -> Graph:
    """Wire the workflow graph."""
    graph = Graph(state_type=MyState)

    graph.add_node("prepare", PrepareNode())
    graph.add_node("respond", RespondNode())

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "respond")
    graph.add_edge("respond", END)

    return graph


# -- Entry point -------------------------------------------------------------

async def main() -> None:
    """Run the workflow from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    graph = build_graph()
    runner = WorkflowRunner(graph, max_steps=10)

    state = MyState(query="Hello, world.")
    result = await runner.start(state)

    print(f"\nQuery:  {result.query}")
    print(f"Result: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
