"""Tests for the shipped ResearchAssistant example (tools, prompts, skills, rules).

This file is intentionally coupled to the example agent. When /create-agent runs
(Step 10), it replaces this file entirely with tests for the new agent. The generic
framework test suite (test_agent.py, test_tools.py, test_config.py, etc.) is what
survives scaffolding.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.config import AgentConfig, LLMConfig, LoopConfig, BackoffConfig
from fipsagents.baseagent.llm import LLMClient, ModelResponse
from fipsagents.baseagent.prompts import PromptLoader
from fipsagents.baseagent.skills import SkillLoader
from fipsagents.baseagent.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Paths — resolved relative to the template root
# ---------------------------------------------------------------------------

_TEMPLATE_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _TEMPLATE_ROOT / "tools"
_PROMPTS_DIR = _TEMPLATE_ROOT / "prompts"
_SKILLS_DIR = _TEMPLATE_ROOT / "skills"
_RULES_DIR = _TEMPLATE_ROOT / "rules"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> AgentConfig:
    defaults = {
        "model": LLMConfig(
            endpoint="http://test:8321/v1",
            name="test-model",
            temperature=0.5,
            max_tokens=256,
        ),
        "loop": LoopConfig(
            max_iterations=5,
            backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
        ),
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _mock_litellm_response(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> MagicMock:
    """Build a fake litellm response object matching ModelResponse expectations."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_tool_call(name: str, arguments: dict[str, Any]) -> Any:
    """Build a fake tool call object matching OpenAI's format."""
    return SimpleNamespace(
        id="call_test123",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


# ---------------------------------------------------------------------------
# Import the example agent
# ---------------------------------------------------------------------------

# Import after helpers so the module's top-level imports can resolve.
from agent import ResearchAssistant, ResearchReport  # noqa: E402


# ---------------------------------------------------------------------------
# Test: Agent instantiation
# ---------------------------------------------------------------------------


class TestResearchAssistantInstantiation:
    def test_is_base_agent_subclass(self):
        assert issubclass(ResearchAssistant, BaseAgent)

    def test_can_instantiate_with_config(self):
        agent = ResearchAssistant(config=_make_config())
        assert agent.config is None  # not yet set up
        assert isinstance(agent, BaseAgent)

    def test_research_report_schema_fields(self):
        report = ResearchReport(
            answer="Test answer",
            confidence=0.85,
            citations=["https://example.com"],
        )
        assert report.answer == "Test answer"
        assert report.confidence == 0.85
        assert report.citations == ["https://example.com"]

    def test_research_report_confidence_bounds(self):
        with pytest.raises(Exception):
            ResearchReport(answer="x", confidence=1.5, citations=[])
        with pytest.raises(Exception):
            ResearchReport(answer="x", confidence=-0.1, citations=[])


# ---------------------------------------------------------------------------
# Test: step() method with mocked LLM
# ---------------------------------------------------------------------------


class TestResearchAssistantStep:
    """Test the step() method end-to-end with mocked LLM responses."""

    async def _setup_agent(self, tmp_path: Path) -> ResearchAssistant:
        """Create and setup a ResearchAssistant with real tools/prompts."""
        config = _make_config()
        agent = ResearchAssistant(
            config=config,
            base_dir=_TEMPLATE_ROOT,
        )
        # Manual setup to avoid MCP/memory complications
        agent.config = config
        agent.llm = MagicMock(spec=LLMClient)

        # Discover real tools from the tools/ directory
        agent.tools.discover(_TOOLS_DIR)

        # Load real prompts and rules
        if _PROMPTS_DIR.is_dir():
            agent.prompts.load_all(_PROMPTS_DIR)
        if _RULES_DIR.is_dir():
            agent.rules.load_all(_RULES_DIR)
        if _SKILLS_DIR.is_dir():
            agent.skills.load_all(_SKILLS_DIR)

        agent._setup_done = True
        return agent

    async def test_step_no_tool_calls(self, tmp_path: Path):
        """step() completes when the LLM responds without tool calls."""
        agent = await self._setup_agent(tmp_path)
        agent.add_message("user", "What is quantum computing?")

        # First call_model: no tool calls, just a text response
        first_response = ModelResponse(
            _mock_litellm_response(content="Quantum computing uses qubits.")
        )
        # call_model_json: return a structured report
        report_data = ResearchReport(
            answer="Quantum computing is a paradigm that uses qubits.",
            confidence=0.8,
            citations=["https://example.com/quantum"],
        )
        # call_model_validated: validation passes
        validation_response = ModelResponse(
            _mock_litellm_response(content="Yes, the report addresses the query.")
        )

        agent.llm.call_model = AsyncMock(
            side_effect=[first_response, validation_response]
        )
        agent.llm.call_model_json = AsyncMock(return_value=report_data)
        agent.llm.call_model_validated = AsyncMock(
            return_value="The report is relevant."
        )

        result = await agent.step()

        assert result.outcome is StepOutcome.DONE
        assert isinstance(result.result, ResearchReport)
        assert result.result.confidence == 0.8

    async def test_step_with_tool_calls(self, tmp_path: Path):
        """step() processes tool calls from the LLM before producing a report."""
        agent = await self._setup_agent(tmp_path)
        agent.add_message("user", "What is container orchestration?")

        # First call: LLM wants to search
        search_tc = _make_tool_call("web_search", {"query": "container orchestration"})
        first_response = ModelResponse(
            _mock_litellm_response(content=None, tool_calls=[search_tc])
        )
        # Second call: LLM is satisfied, no more tool calls
        second_response = ModelResponse(
            _mock_litellm_response(
                content="Container orchestration manages containerized apps."
            )
        )
        report_data = ResearchReport(
            answer="Container orchestration automates deployment.",
            confidence=0.9,
            citations=["https://example.com/containers"],
        )

        agent.llm.call_model = AsyncMock(
            side_effect=[first_response, second_response]
        )
        agent.llm.call_model_json = AsyncMock(return_value=report_data)
        agent.llm.call_model_validated = AsyncMock(return_value="Relevant.")

        result = await agent.step()

        assert result.outcome is StepOutcome.DONE
        assert result.result.confidence == 0.9
        # The web_search tool was executed (real stub tool)
        assert agent.llm.call_model.call_count == 2

    async def test_tool_call_message_ordering(self, tmp_path: Path):
        """Assistant message with tool_calls must precede the tool result message."""
        agent = await self._setup_agent(tmp_path)
        agent.add_message("user", "What is container orchestration?")

        search_tc = _make_tool_call("web_search", {"query": "container orchestration"})
        first_response = ModelResponse(
            _mock_litellm_response(content=None, tool_calls=[search_tc])
        )
        second_response = ModelResponse(
            _mock_litellm_response(
                content="Container orchestration manages containerized apps."
            )
        )
        report_data = ResearchReport(
            answer="Container orchestration automates deployment.",
            confidence=0.9,
            citations=["https://example.com/containers"],
        )

        agent.llm.call_model = AsyncMock(
            side_effect=[first_response, second_response]
        )
        agent.llm.call_model_json = AsyncMock(return_value=report_data)
        agent.llm.call_model_validated = AsyncMock(return_value="Relevant.")

        await agent.step()

        # Find the indices of the assistant tool-use message and the tool result
        tool_use_idx = next(
            (
                i
                for i, m in enumerate(agent.messages)
                if m.get("role") == "assistant" and m.get("tool_calls")
            ),
            None,
        )
        tool_result_idx = next(
            (
                i
                for i, m in enumerate(agent.messages)
                if m.get("role") == "tool"
            ),
            None,
        )

        assert tool_use_idx is not None, "No assistant message with tool_calls found"
        assert tool_result_idx is not None, "No tool result message found"
        assert tool_use_idx < tool_result_idx, (
            "assistant tool_calls message must precede tool result message"
        )

        # Verify the tool_call_id on the result matches the id in the preceding
        # assistant message's tool_calls list.
        assistant_msg = agent.messages[tool_use_idx]
        tool_result_msg = agent.messages[tool_result_idx]
        assistant_tc_ids = {tc["id"] for tc in assistant_msg["tool_calls"]}
        assert tool_result_msg["tool_call_id"] in assistant_tc_ids, (
            f"tool_call_id {tool_result_msg['tool_call_id']!r} not found in "
            f"assistant tool_calls ids {assistant_tc_ids}"
        )

    async def test_step_formats_citations(self, tmp_path: Path):
        """step() uses the format_citations agent-only tool."""
        agent = await self._setup_agent(tmp_path)
        agent.add_message("user", "Tell me about Kubernetes.")

        first_response = ModelResponse(
            _mock_litellm_response(content="Kubernetes is an orchestrator.")
        )
        report_data = ResearchReport(
            answer="Kubernetes orchestrates containers.",
            confidence=0.95,
            citations=["https://kubernetes.io", "https://example.com/k8s"],
        )

        agent.llm.call_model = AsyncMock(return_value=first_response)
        agent.llm.call_model_json = AsyncMock(return_value=report_data)
        agent.llm.call_model_validated = AsyncMock(return_value="Relevant.")

        result = await agent.step()

        assert result.outcome is StepOutcome.DONE
        # Citations should be formatted by the agent-only tool
        assert len(result.result.citations) == 2
        assert any("kubernetes.io" in c for c in result.result.citations)


# ---------------------------------------------------------------------------
# Test: web_search tool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    """Test the web_search tool independently."""

    def test_tool_discovered_in_registry(self):
        registry = ToolRegistry()
        discovered = registry.discover(_TOOLS_DIR)
        names = [t.name for t in discovered]
        assert "web_search" in names

    def test_web_search_visibility(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        meta = registry.get("web_search")
        assert meta is not None
        assert meta.visibility == "llm_only"

    async def test_web_search_returns_results(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        result = await registry.execute("web_search", query="test query")
        assert not result.is_error
        assert "test query" in result.result
        assert "URL:" in result.result

    async def test_web_search_in_llm_tools(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        llm_tools = registry.get_llm_tools()
        names = [t.name for t in llm_tools]
        assert "web_search" in names

    async def test_web_search_not_in_agent_tools(self):
        """web_search is llm_only, so it should not appear in agent tools."""
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        agent_tools = registry.get_agent_tools()
        names = [t.name for t in agent_tools]
        assert "web_search" not in names

    async def test_web_search_schema_generation(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        schemas = registry.generate_schemas()
        ws_schema = next(
            (s for s in schemas if s["function"]["name"] == "web_search"),
            None,
        )
        assert ws_schema is not None
        assert ws_schema["type"] == "function"
        params = ws_schema["function"].get("parameters", {})
        assert "query" in params.get("properties", {})


# ---------------------------------------------------------------------------
# Test: format_citations tool
# ---------------------------------------------------------------------------


class TestFormatCitationsTool:
    """Test the format_citations tool independently."""

    def test_tool_discovered_in_registry(self):
        registry = ToolRegistry()
        discovered = registry.discover(_TOOLS_DIR)
        names = [t.name for t in discovered]
        assert "format_citations" in names

    def test_format_citations_visibility(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        meta = registry.get("format_citations")
        assert meta is not None
        assert meta.visibility == "agent_only"

    async def test_format_citations_basic(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        result = await registry.execute(
            "format_citations",
            urls=["https://example.com", "https://other.com"],
            titles=["Example", "Other"],
        )
        assert not result.is_error
        assert "[1] Example" in result.result
        assert "https://example.com" in result.result
        assert "[2] Other" in result.result

    async def test_format_citations_empty(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        result = await registry.execute(
            "format_citations", urls=[], titles=[]
        )
        assert not result.is_error
        assert result.result == ""

    async def test_format_citations_skips_empty_urls(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        result = await registry.execute(
            "format_citations",
            urls=["https://example.com", "", "https://other.com"],
            titles=["A", "B", "C"],
        )
        assert not result.is_error
        lines = [line for line in result.result.splitlines() if line.strip()]
        assert len(lines) == 2  # empty URL skipped

    async def test_format_citations_not_in_llm_tools(self):
        """format_citations is agent_only, invisible to the LLM."""
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        llm_tools = registry.get_llm_tools()
        names = [t.name for t in llm_tools]
        assert "format_citations" not in names

    async def test_format_citations_in_agent_tools(self):
        registry = ToolRegistry()
        registry.discover(_TOOLS_DIR)
        agent_tools = registry.get_agent_tools()
        names = [t.name for t in agent_tools]
        assert "format_citations" in names


# ---------------------------------------------------------------------------
# Test: system prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_loads(self):
        loader = PromptLoader()
        loader.load_all(_PROMPTS_DIR)
        assert "system" in loader.names

    def test_system_prompt_metadata(self):
        loader = PromptLoader()
        loader.load_all(_PROMPTS_DIR)
        prompt = loader.get("system")
        assert prompt.name == "system"
        assert prompt.description == "System prompt for the Research Assistant agent"

    def test_system_prompt_has_variables(self):
        loader = PromptLoader()
        loader.load_all(_PROMPTS_DIR)
        prompt = loader.get("system")
        var_names = [v.name for v in prompt.variables]
        # query is passed as a user message, not a system prompt variable
        assert "query" not in var_names
        assert "max_results" in var_names

    def test_system_prompt_renders_with_defaults(self):
        loader = PromptLoader()
        loader.load_all(_PROMPTS_DIR)
        rendered = loader.render("system")
        # max_results has a default of "5"
        assert "5" in rendered
        assert "Research Assistant" in rendered

    def test_system_prompt_renders_with_overrides(self):
        loader = PromptLoader()
        loader.load_all(_PROMPTS_DIR)
        rendered = loader.render("system", max_results="10")
        assert "10" in rendered

    def test_system_prompt_mentions_web_search(self):
        loader = PromptLoader()
        loader.load_all(_PROMPTS_DIR)
        rendered = loader.render("system")
        assert "web_search" in rendered

    def test_system_prompt_requires_citations(self):
        loader = PromptLoader()
        loader.load_all(_PROMPTS_DIR)
        rendered = loader.render("system")
        assert "cite" in rendered.lower() or "citation" in rendered.lower()


# ---------------------------------------------------------------------------
# Test: summarize skill
# ---------------------------------------------------------------------------


class TestSummarizeSkill:
    def test_skill_discovered(self):
        loader = SkillLoader()
        loader.load_all(_SKILLS_DIR)
        assert "summarize" in loader

    def test_skill_metadata(self):
        loader = SkillLoader()
        loader.load_all(_SKILLS_DIR)
        manifest = loader.get_manifest()
        entry = next((e for e in manifest if e.name == "summarize"), None)
        assert entry is not None
        assert "summarize" in entry.triggers
        assert entry.description

    def test_skill_activation(self):
        loader = SkillLoader()
        loader.load_all(_SKILLS_DIR)
        skill = loader.activate("summarize")
        assert skill.activated
        assert skill.content is not None
        assert "Summarize Skill" in skill.content

    def test_skill_version(self):
        loader = SkillLoader()
        loader.load_all(_SKILLS_DIR)
        skill = loader.get("summarize")
        assert skill.version == "1.0"

    def test_skill_deactivation_clears_content(self):
        loader = SkillLoader()
        loader.load_all(_SKILLS_DIR)
        loader.activate("summarize")
        loader.deactivate("summarize")
        skill_obj = loader._skills["summarize"]
        assert not skill_obj.activated
        assert skill_obj.content is None


# ---------------------------------------------------------------------------
# Test: citation_required rule
# ---------------------------------------------------------------------------


class TestCitationRule:
    def test_rule_loads(self):
        from fipsagents.baseagent.rules import RuleLoader

        loader = RuleLoader()
        loader.load_all(_RULES_DIR)
        rule = loader.get("citation_required")
        assert rule.name == "citation_required"

    def test_rule_content(self):
        from fipsagents.baseagent.rules import RuleLoader

        loader = RuleLoader()
        loader.load_all(_RULES_DIR)
        rule = loader.get("citation_required")
        assert "citation" in rule.content.lower()
        assert "fabricate" in rule.content.lower()
