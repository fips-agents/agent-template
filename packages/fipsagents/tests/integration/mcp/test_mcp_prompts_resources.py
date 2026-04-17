"""Integration tests for MCP prompt and resource discovery via BaseAgent.

Validates that connect_mcp() discovers prompts, resources, and resource
templates from a FastMCP server, and that the corresponding BaseAgent
methods (get_mcp_prompt, read_resource, list_mcp_*) work correctly.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.llm import LLMClient

from .calculator_server import mcp as calculator_mcp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> AgentConfig:
    defaults: dict[str, Any] = {
        "model": LLMConfig(
            endpoint="http://test:8321/v1",
            name="test-model",
            temperature=0.0,
            max_tokens=256,
        ),
        "loop": LoopConfig(
            max_iterations=5,
            backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
        ),
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


@pytest.fixture
async def mcp_prompts_resources_agent() -> AsyncIterator[BaseAgent]:
    """Agent wired to the calculator FastMCP server for prompt/resource tests."""
    config = _make_config()
    agent = BaseAgent(config=config)

    agent.config = config
    agent.llm = MagicMock(spec=LLMClient)
    agent.messages = [{"role": "system", "content": "You are a calculator."}]
    agent._reasoning_parser = None
    agent._setup_done = True

    await agent.connect_mcp(calculator_mcp)

    yield agent

    await agent.shutdown()


# ---------------------------------------------------------------------------
# Prompt discovery
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestMcpPromptDiscovery:
    """Verify MCP prompt discovery through connect_mcp."""

    async def test_discovers_explain_result_prompt(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        assert "explain_result" in mcp_prompts_resources_agent._mcp_prompts, (
            f"'explain_result' not found in discovered prompts: "
            f"{sorted(mcp_prompts_resources_agent._mcp_prompts)}"
        )

    async def test_list_mcp_prompts_metadata(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        prompts = mcp_prompts_resources_agent.list_mcp_prompts()
        assert len(prompts) >= 1

        explain = next(p for p in prompts if p["name"] == "explain_result")
        assert explain["description"], "Prompt should have a description"
        assert "arguments" in explain, "Prompt should list its arguments"

        arg_names = {a["name"] for a in explain["arguments"]}
        assert "result" in arg_names, f"'result' arg missing; got {arg_names}"
        assert "operation" in arg_names, f"'operation' arg missing; got {arg_names}"

    async def test_tools_still_discovered(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        """Prompt/resource discovery must not break tool discovery."""
        tool_names = {t.name for t in mcp_prompts_resources_agent.tools.get_all()}
        assert "add" in tool_names, f"'add' not in {sorted(tool_names)}"
        assert "multiply" in tool_names, f"'multiply' not in {sorted(tool_names)}"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestMcpPromptRendering:
    """Verify get_mcp_prompt renders prompts through the MCP server."""

    async def test_get_mcp_prompt_renders(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        result = await mcp_prompts_resources_agent.get_mcp_prompt(
            "explain_result",
            arguments={"result": "42", "operation": "multiplication"},
        )

        assert hasattr(result, "messages"), (
            f"Expected GetPromptResult with .messages, got {type(result)}"
        )
        assert len(result.messages) >= 1, "Prompt should produce at least one message"

        # Extract text from first message — handle both str and TextContent
        content = result.messages[0].content
        text = content.text if hasattr(content, "text") else str(content)
        assert "multiplication" in text, f"Expected 'multiplication' in prompt text: {text}"
        assert "42" in text, f"Expected '42' in prompt text: {text}"

    async def test_get_mcp_prompt_unknown_raises_key_error(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            await mcp_prompts_resources_agent.get_mcp_prompt("nonexistent")


# ---------------------------------------------------------------------------
# Resource discovery
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestMcpResourceDiscovery:
    """Verify MCP resource and resource template discovery."""

    async def test_discovers_help_resource(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        assert "calculator://help" in mcp_prompts_resources_agent._mcp_resources, (
            f"'calculator://help' not found in discovered resources: "
            f"{sorted(mcp_prompts_resources_agent._mcp_resources)}"
        )

    async def test_list_mcp_resources_metadata(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        resources = mcp_prompts_resources_agent.list_mcp_resources()
        assert len(resources) >= 1

        help_res = next(r for r in resources if r["uri"] == "calculator://help")
        assert help_res["name"], "Resource should have a name"

    async def test_discovers_history_template(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        assert "calculator://history/{operation}" in (
            mcp_prompts_resources_agent._mcp_resource_templates
        ), (
            f"'calculator://history/{{operation}}' not found in templates: "
            f"{sorted(mcp_prompts_resources_agent._mcp_resource_templates)}"
        )

    async def test_list_mcp_resource_templates_metadata(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        templates = mcp_prompts_resources_agent.list_mcp_resource_templates()
        assert len(templates) >= 1

        history = next(
            t for t in templates
            if t["uriTemplate"] == "calculator://history/{operation}"
        )
        assert history["name"], "Resource template should have a name"


# ---------------------------------------------------------------------------
# Resource reading
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestMcpResourceReading:
    """Verify read_resource retrieves content from the MCP server."""

    async def test_read_help_resource(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        result = await mcp_prompts_resources_agent.read_resource("calculator://help")

        # read_resource returns a list of content objects.
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert result, "Expected non-empty result from read_resource"
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        assert "addition and multiplication" in text, (
            f"Expected help text about addition and multiplication, got: {text}"
        )

    async def test_read_resource_unknown_raises_key_error(
        self, mcp_prompts_resources_agent: BaseAgent,
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            await mcp_prompts_resources_agent.read_resource("calculator://nonexistent")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestMcpCleanup:
    """Verify shutdown clears prompt and resource registries."""

    async def test_shutdown_clears_prompts_and_resources(self) -> None:
        config = _make_config()
        agent = BaseAgent(config=config)
        agent.config = config
        agent.llm = MagicMock(spec=LLMClient)
        agent.messages = [{"role": "system", "content": "You are a calculator."}]
        agent._reasoning_parser = None
        agent._setup_done = True

        await agent.connect_mcp(calculator_mcp)

        # Sanity: something was discovered
        assert agent._mcp_prompts, "Expected prompts before shutdown"
        assert agent._mcp_resources, "Expected resources before shutdown"
        assert agent._mcp_resource_templates, "Expected resource templates before shutdown"

        await agent.shutdown()

        assert not agent._mcp_prompts, "Prompts should be empty after shutdown"
        assert not agent._mcp_resources, "Resources should be empty after shutdown"
        assert not agent._mcp_resource_templates, "Resource templates should be empty after shutdown"
