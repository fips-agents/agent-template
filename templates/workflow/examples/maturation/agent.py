"""Maturation lifecycle workflow example.

Demonstrates stage-gated routing: proto-agents can only suggest skills,
apprentice+ agents can learn them directly. The workflow reads the
agent's maturation stage and routes to the appropriate node.

Run:  python -m examples.maturation.agent
Docs: examples/maturation/README.md
"""

from __future__ import annotations

import asyncio
import logging

from fipsagents.workflow import END, BaseNode, Graph, WorkflowRunner, node

from .state import MaturationState

logger = logging.getLogger(__name__)


@node()
class CheckStageNode(BaseNode):
    """Read the agent's maturation stage and route accordingly.

    In a real deployment, ``maturation_stage`` would be populated from
    the agent's TrustManager via ``MaturationManager.current_stage()``.
    For this example, it is set on the input state.
    """

    async def process(self, state: MaturationState) -> MaturationState:
        stage = state.maturation_stage
        if not stage:
            stage = "proto_agent"
        logger.info("Agent is at maturation stage: %s", stage)
        return state.model_copy(update={"maturation_stage": stage})


@node()
class SuggestSkillNode(BaseNode):
    """Proto-agent path: propose a skill for review without disk writes."""

    async def process(self, state: MaturationState) -> MaturationState:
        logger.info(
            "Proto-agent suggesting skill '%s' for review", state.skill_name
        )
        return state.model_copy(update={
            "action_taken": "suggest_skill",
            "result": (
                f"Proposed skill '{state.skill_name}' for review. "
                f"A review_pending work item has been created."
            ),
        })


@node()
class LearnSkillNode(BaseNode):
    """Apprentice+ path: create or update a learned skill on disk."""

    async def process(self, state: MaturationState) -> MaturationState:
        logger.info(
            "Apprentice+ agent learning skill '%s'", state.skill_name
        )
        return state.model_copy(update={
            "action_taken": "learn_skill",
            "result": (
                f"Learned skill '{state.skill_name}' (version 1). "
                f"Skill written to learned_skills/{state.skill_name}/SKILL.md."
            ),
        })


def route_by_stage(state: MaturationState) -> str:
    """Conditional edge: proto-agents suggest, apprentice+ agents learn."""
    if state.maturation_stage == "proto_agent":
        return "suggest"
    return "learn"


def build_graph() -> Graph:
    """Wire the maturation workflow graph.

    ```
    check_stage ──┬── (proto_agent) ──→ suggest ──→ END
                  └── (apprentice+) ──→ learn   ──→ END
    ```
    """
    graph = Graph(state_type=MaturationState)

    graph.add_node("check_stage", CheckStageNode())
    graph.add_node("suggest", SuggestSkillNode())
    graph.add_node("learn", LearnSkillNode())

    graph.set_entry_point("check_stage")
    graph.add_conditional_edge("check_stage", route_by_stage)
    graph.add_edge("suggest", END)
    graph.add_edge("learn", END)

    return graph


async def main() -> None:
    """Run the maturation workflow from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    graph = build_graph()
    runner = WorkflowRunner(graph, max_steps=10)

    # Proto-agent: can only suggest.
    proto_state = MaturationState(
        query="Learn a new skill for summarizing PDFs",
        skill_name="summarize-pdf",
        skill_description="Summarize PDF documents using Docling",
        skill_content="Use Docling to parse, then summarize key findings.",
        skill_domain="document_processing",
        skill_trigger="summarize a pdf",
        maturation_stage="proto_agent",
    )
    result = await runner.start(proto_state)
    print(f"\n[Proto-agent] Action: {result.action_taken}")
    print(f"[Proto-agent] Result: {result.result}")

    # Apprentice: can learn directly.
    apprentice_state = proto_state.model_copy(
        update={"maturation_stage": "apprentice"}
    )
    result = await runner.start(apprentice_state)
    print(f"\n[Apprentice]  Action: {result.action_taken}")
    print(f"[Apprentice]  Result: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
