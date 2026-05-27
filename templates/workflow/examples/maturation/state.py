"""Typed workflow state for the maturation lifecycle example."""

from __future__ import annotations

from fipsagents.workflow import WorkflowState


class MaturationState(WorkflowState):
    """State flowing through the maturation workflow.

    The workflow routes skill-learning requests differently based on the
    agent's maturation stage: proto-agents can only suggest, apprentice+
    can learn directly.
    """

    query: str
    skill_name: str = ""
    skill_description: str = ""
    skill_content: str = ""
    skill_domain: str = ""
    skill_trigger: str = ""
    maturation_stage: str = ""
    action_taken: str = ""
    result: str = ""
