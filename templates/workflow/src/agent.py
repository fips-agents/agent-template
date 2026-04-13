"""Example workflow: Research + Summarize pipeline.

Demonstrates a workflow with BaseNode (routing) and AgentNode (LLM) nodes:
  classify -> (if complex) research -> summarize
           -> (if simple) summarize
"""

import asyncio
import json
import logging

from workflow import WorkflowState, BaseNode, Graph, WorkflowRunner, END, node
from workflow.agent_node import AgentNode

logger = logging.getLogger(__name__)


# -- State -------------------------------------------------------------------

class ResearchState(WorkflowState):
    """Typed state flowing through the research workflow."""
    query: str
    complexity: str = ""
    research_results: str = ""
    summary: str = ""


# -- Nodes -------------------------------------------------------------------

@node()
class ClassifyNode(BaseNode):
    """Route based on query complexity. No LLM needed."""

    async def process(self, state: ResearchState) -> ResearchState:
        complexity = "complex" if len(state.query.split()) > 10 else "simple"
        self.logger.info("Classified query as %s", complexity)
        return state.model_copy(update={"complexity": complexity})


@node()
class ResearchNode(AgentNode):
    """Deep research using LLM and tools."""

    async def process(self, state: ResearchState) -> ResearchState:
        self.add_message("user", f"Research the following topic thoroughly: {state.query}")
        response = await self.call_model()

        # Handle tool calls if the LLM wants to use tools
        while response.tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ],
            })
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = await self.tools.execute(tc.function.name, **args)
                self.messages.append({
                    "role": "tool",
                    "content": result.result,
                    "tool_call_id": tc.id,
                })
            response = await self.call_model()

        return state.model_copy(update={
            "research_results": response.content or "",
        })


@node()
class SummarizeNode(AgentNode):
    """Summarize research results or raw query."""

    async def process(self, state: ResearchState) -> ResearchState:
        context = state.research_results or state.query
        self.add_message("user", f"Summarize the following concisely:\n\n{context}")
        response = await self.call_model(include_tools=False)
        return state.model_copy(update={
            "summary": response.content or "",
        })


# -- Graph -------------------------------------------------------------------

def build_graph() -> Graph:
    """Wire the research workflow graph."""
    graph = Graph(state_type=ResearchState)

    graph.add_node("classify", ClassifyNode())
    graph.add_node("research", ResearchNode())
    graph.add_node("summarize", SummarizeNode())

    graph.set_entry_point("classify")
    graph.add_conditional_edge(
        "classify",
        lambda s: "research" if s.complexity == "complex" else "summarize",
    )
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", END)

    return graph


# -- Entry point -------------------------------------------------------------

async def main() -> None:
    """Run the example workflow."""
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    graph = build_graph()
    runner = WorkflowRunner(graph, max_steps=10)

    state = ResearchState(query="Explain quantum computing applications in cryptography")
    result = await runner.start(state)

    print(f"\nQuery: {result.query}")
    print(f"Complexity: {result.complexity}")
    print(f"Summary: {result.summary}")


if __name__ == "__main__":
    asyncio.run(main())
