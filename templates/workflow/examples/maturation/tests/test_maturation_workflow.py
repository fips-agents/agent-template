"""Tests for the maturation lifecycle workflow example.

Exercises stage-gated routing without an LLM — all nodes are BaseNode
(pure logic), so we can run the full graph synchronously.
"""

from __future__ import annotations

import pytest

from fipsagents.workflow import WorkflowRunner

from ..agent import build_graph, route_by_stage
from ..state import MaturationState


@pytest.fixture
def runner() -> WorkflowRunner:
    return WorkflowRunner(build_graph(), max_steps=10)


def _make_state(stage: str) -> MaturationState:
    return MaturationState(
        query="Learn a new skill",
        skill_name="test-skill",
        skill_description="A test skill",
        skill_content="Test content",
        skill_domain="testing",
        skill_trigger="test",
        maturation_stage=stage,
    )


class TestRouteByStage:
    """Unit tests for the conditional edge function."""

    def test_proto_agent_routes_to_suggest(self):
        state = _make_state("proto_agent")
        assert route_by_stage(state) == "suggest"

    def test_apprentice_routes_to_learn(self):
        state = _make_state("apprentice")
        assert route_by_stage(state) == "learn"

    def test_journeyman_routes_to_learn(self):
        state = _make_state("journeyman")
        assert route_by_stage(state) == "learn"

    def test_specialist_routes_to_learn(self):
        state = _make_state("specialist")
        assert route_by_stage(state) == "learn"


class TestMaturationWorkflow:
    """Integration tests running the full graph."""

    @pytest.mark.asyncio
    async def test_proto_agent_suggests(self, runner):
        result = await runner.start(_make_state("proto_agent"))
        assert result.action_taken == "suggest_skill"
        assert "Proposed" in result.result
        assert result.maturation_stage == "proto_agent"

    @pytest.mark.asyncio
    async def test_apprentice_learns(self, runner):
        result = await runner.start(_make_state("apprentice"))
        assert result.action_taken == "learn_skill"
        assert "Learned" in result.result

    @pytest.mark.asyncio
    async def test_journeyman_learns(self, runner):
        result = await runner.start(_make_state("journeyman"))
        assert result.action_taken == "learn_skill"

    @pytest.mark.asyncio
    async def test_specialist_learns(self, runner):
        result = await runner.start(_make_state("specialist"))
        assert result.action_taken == "learn_skill"

    @pytest.mark.asyncio
    async def test_empty_stage_defaults_to_proto(self, runner):
        state = _make_state("")
        result = await runner.start(state)
        assert result.maturation_stage == "proto_agent"
        assert result.action_taken == "suggest_skill"

    @pytest.mark.asyncio
    async def test_skill_name_preserved(self, runner):
        result = await runner.start(_make_state("apprentice"))
        assert result.skill_name == "test-skill"
        assert "test-skill" in result.result
