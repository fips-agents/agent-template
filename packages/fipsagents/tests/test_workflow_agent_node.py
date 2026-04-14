"""Tests for fipsagents.workflow.agent_node — AgentNode bridge between BaseAgent and workflows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.llm import LLMClient, ModelResponse
from fipsagents.baseagent.memory import NullMemoryClient
from fipsagents.workflow.agent_node import AgentNode
from fipsagents.workflow.state import WorkflowState


# ---------------------------------------------------------------------------
# State & helpers
# ---------------------------------------------------------------------------


class NodeState(WorkflowState):
    value: str = ""


def _make_config() -> AgentConfig:
    return AgentConfig(
        model=LLMConfig(endpoint="http://fake:8080/v1", name="test-model"),
        loop=LoopConfig(max_iterations=10, backoff=BackoffConfig()),
    )


def _wire_agent(agent: AgentNode) -> None:
    """Manually wire subsystems so we don't need real services."""
    agent.config = _make_config()
    agent.llm = MagicMock(spec=LLMClient)
    agent.memory = NullMemoryClient()
    agent._setup_done = True


# ---------------------------------------------------------------------------
# step() — not used in workflow context
# ---------------------------------------------------------------------------


class TestStep:
    async def test_step_raises_not_implemented(self):
        node = AgentNode(config=_make_config())
        with pytest.raises(NotImplementedError, match="workflow AgentNode"):
            await node.step()


# ---------------------------------------------------------------------------
# process() — must be overridden
# ---------------------------------------------------------------------------


class TestProcess:
    async def test_process_raises_not_implemented(self):
        node = AgentNode(config=_make_config())
        with pytest.raises(NotImplementedError, match="must implement process"):
            await node.process(NodeState())


# ---------------------------------------------------------------------------
# Name attribute
# ---------------------------------------------------------------------------


class TestName:
    def test_default_name_is_class_name(self):
        class MyAgentNode(AgentNode):
            async def process(self, state):
                return state

        node = MyAgentNode(config=_make_config())
        assert node.name == "MyAgentNode"

    def test_custom_name(self):
        node = AgentNode(name="custom", config=_make_config())
        assert node.name == "custom"


# ---------------------------------------------------------------------------
# Subclass with process() override
# ---------------------------------------------------------------------------


class EchoNode(AgentNode):
    async def process(self, state: NodeState) -> NodeState:
        return state.model_copy(update={"value": f"echo:{state.value}"})


class TestSubclassProcess:
    async def test_override_works(self):
        node = EchoNode(config=_make_config())
        result = await node.process(NodeState(value="hello"))
        assert result.value == "echo:hello"


# ---------------------------------------------------------------------------
# call_model integration (mocked LLM)
# ---------------------------------------------------------------------------


class LLMCallingNode(AgentNode):
    async def process(self, state: NodeState) -> NodeState:
        self.add_message("user", state.value)
        response = await self.call_model()
        return state.model_copy(update={"value": response.content})


class TestCallModel:
    async def test_call_model_in_process(self):
        node = LLMCallingNode(config=_make_config())
        _wire_agent(node)

        mock_response = MagicMock(spec=ModelResponse)
        mock_response.content = "llm says hi"
        mock_response.tool_calls = None
        node.llm.call_model = AsyncMock(return_value=mock_response)

        result = await node.process(NodeState(value="hello"))

        assert result.value == "llm says hi"
        node.llm.call_model.assert_called_once()
