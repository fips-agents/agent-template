"""Integration tests — end-to-end workflows mixing BaseNode and AgentNode."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.llm import LLMClient, ModelResponse
from fipsagents.baseagent.memory import NullMemoryClient
from fipsagents.workflow.agent_node import AgentNode
from fipsagents.workflow.graph import Graph
from fipsagents.workflow.node import BaseNode
from fipsagents.workflow.runner import WorkflowRunner
from fipsagents.workflow.state import END, WorkflowState


# ---------------------------------------------------------------------------
# State & helpers
# ---------------------------------------------------------------------------


class PipelineState(WorkflowState):
    text: str = ""
    category: str = ""
    summary: str = ""


def _make_config() -> AgentConfig:
    return AgentConfig(
        model=LLMConfig(endpoint="http://fake:8080/v1", name="test-model"),
        loop=LoopConfig(max_iterations=10, backoff=BackoffConfig()),
    )


def _wire_agent(agent: AgentNode) -> None:
    """Manually wire subsystems without real service connections."""
    agent.config = _make_config()
    agent.llm = MagicMock(spec=LLMClient)
    agent.memory = NullMemoryClient()
    agent._setup_done = True


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class ClassifyNode(BaseNode):
    """Pure-logic node that classifies text by length."""

    async def process(self, state: PipelineState) -> PipelineState:
        category = "long" if len(state.text) > 10 else "short"
        return state.model_copy(update={"category": category})


class SummarizeNode(AgentNode):
    """Agent node that 'calls' the LLM to produce a summary."""

    async def process(self, state: PipelineState) -> PipelineState:
        self.add_message("user", f"Summarize: {state.text}")
        response = await self.call_model()
        return state.model_copy(update={"summary": response.content})

    async def setup(self):
        """Override to avoid connecting to real services."""
        pass

    async def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# Mixed pipeline: BaseNode + AgentNode
# ---------------------------------------------------------------------------


class TestMixedPipeline:
    async def test_classify_then_summarize(self):
        summarize = SummarizeNode(config=_make_config())
        _wire_agent(summarize)

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.content = "A concise summary."
        mock_response.tool_calls = None
        summarize.llm.call_model = AsyncMock(return_value=mock_response)

        g = Graph(state_type=PipelineState)
        g.add_node("classify", ClassifyNode())
        g.add_node("summarize", summarize)
        g.set_entry_point("classify")
        g.add_edge("classify", "summarize")
        g.add_edge("summarize", END)

        runner = WorkflowRunner(g)
        result = await runner.start(PipelineState(text="A longer piece of text here"))

        assert result.category == "long"
        assert result.summary == "A concise summary."


# ---------------------------------------------------------------------------
# BaseNode-only pipeline
# ---------------------------------------------------------------------------


class UppercaseNode(BaseNode):
    async def process(self, state: PipelineState) -> PipelineState:
        return state.model_copy(update={"text": state.text.upper()})


class TestBaseNodeOnly:
    async def test_no_agent_setup_needed(self):
        g = Graph(state_type=PipelineState)
        g.add_node("classify", ClassifyNode())
        g.add_node("upper", UppercaseNode())
        g.set_entry_point("classify")
        g.add_edge("classify", "upper")
        g.add_edge("upper", END)

        runner = WorkflowRunner(g)
        result = await runner.start(PipelineState(text="hello"))

        assert result.category == "short"
        assert result.text == "HELLO"


# ---------------------------------------------------------------------------
# State flows correctly through mixed types
# ---------------------------------------------------------------------------


class TagNode(BaseNode):
    """Appends a tag to the text field."""

    async def process(self, state: PipelineState) -> PipelineState:
        return state.model_copy(update={"text": state.text + f"[{self.name}]"})


class TestStateFlow:
    async def test_state_accumulates_through_nodes(self):
        g = Graph(state_type=PipelineState)
        g.add_node("tag1", TagNode())
        g.add_node("tag2", TagNode())
        g.add_node("tag3", TagNode())
        g.set_entry_point("tag1")
        g.add_edge("tag1", "tag2")
        g.add_edge("tag2", "tag3")
        g.add_edge("tag3", END)

        runner = WorkflowRunner(g)
        result = await runner.start(PipelineState(text="start"))

        assert result.text == "start[tag1][tag2][tag3]"
