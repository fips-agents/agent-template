"""Tests for fipsagents.baseagent.agent — BaseAgent lifecycle, dispatch, and integration."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome, StepResult
from fipsagents.baseagent.config import AgentConfig, LLMConfig, LoopConfig, BackoffConfig
from fipsagents.baseagent.llm import LLMClient, ModelResponse
from fipsagents.baseagent.memory import NullMemoryClient
from fipsagents.baseagent.tools import tool


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> AgentConfig:
    """Build an AgentConfig suitable for testing (no file I/O needed)."""
    defaults = {
        "model": LLMConfig(
            endpoint="http://test:8321/v1",
            name="test-model",
            temperature=0.5,
            max_tokens=256,
        ),
        "loop": LoopConfig(
            max_iterations=10,
            backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
        ),
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _make_agent_yaml(tmp_path: Path, content: str = "") -> Path:
    """Write a minimal agent.yaml and return its path."""
    if not content:
        content = (
            "model:\n"
            "  endpoint: http://test:8321/v1\n"
            "  name: test-model\n"
            "loop:\n"
            "  max_iterations: 5\n"
            "  backoff:\n"
            "    initial: 0.01\n"
            "    max: 0.05\n"
            "    multiplier: 2.0\n"
        )
    yaml_path = tmp_path / "agent.yaml"
    yaml_path.write_text(content)
    return yaml_path


def _write_system_prompt(tmp_path: Path, content: str = "You are a test agent.") -> None:
    """Write a minimal system prompt file."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / "system.md").write_text(
        f"---\nname: system\ndescription: System prompt\n---\n\n{content}"
    )


def _write_rule(tmp_path: Path, name: str, content: str) -> None:
    """Write a rule file."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(exist_ok=True)
    (rules_dir / f"{name}.md").write_text(content)


def _write_tool(tmp_path: Path, filename: str, code: str) -> None:
    """Write a tool file."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    (tools_dir / filename).write_text(code)


class CountingAgent(BaseAgent):
    """Concrete agent that counts steps and stops after a threshold."""

    def __init__(self, stop_after: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stop_after = stop_after
        self.step_count = 0

    async def step(self) -> StepResult:
        self.step_count += 1
        if self.step_count >= self.stop_after:
            return StepResult.done(result=f"done-after-{self.step_count}")
        return StepResult.continue_()


class ErrorAgent(BaseAgent):
    """Agent that raises on every step."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.step_count = 0

    async def step(self) -> StepResult:
        self.step_count += 1
        raise RuntimeError(f"step-{self.step_count}-failed")


class ErrorThenDoneAgent(BaseAgent):
    """Agent that errors N times then succeeds."""

    def __init__(self, errors_before_done: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.errors_before_done = errors_before_done
        self.step_count = 0

    async def step(self) -> StepResult:
        self.step_count += 1
        if self.step_count <= self.errors_before_done:
            raise RuntimeError(f"transient-error-{self.step_count}")
        return StepResult.done(result="recovered")


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_continue_factory(self):
        r = StepResult.continue_()
        assert r.outcome is StepOutcome.CONTINUE
        assert r.result is None

    def test_done_factory_no_result(self):
        r = StepResult.done()
        assert r.outcome is StepOutcome.DONE
        assert r.result is None

    def test_done_factory_with_result(self):
        r = StepResult.done(result={"answer": 42})
        assert r.outcome is StepOutcome.DONE
        assert r.result == {"answer": 42}

    def test_outcome_enum_values(self):
        assert StepOutcome.CONTINUE.value == "continue"
        assert StepOutcome.DONE.value == "done"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class TestSetup:
    @pytest.mark.asyncio
    async def test_setup_with_provided_config(self, tmp_path: Path):
        """setup() accepts a pre-built config and skips file loading."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        assert agent.config is config
        assert agent.llm is not None
        assert isinstance(agent.llm, LLMClient)
        assert isinstance(agent.memory, NullMemoryClient)
        assert agent._setup_done is True

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_setup_from_yaml(self, tmp_path: Path):
        """setup() loads config from an agent.yaml file."""
        yaml_path = _make_agent_yaml(tmp_path)
        agent = CountingAgent(config_path=yaml_path, base_dir=tmp_path)
        await agent.setup()

        assert agent.config is not None
        assert agent.config.model.name == "test-model"
        assert agent._setup_done is True

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_setup_discovers_tools(self, tmp_path: Path):
        """setup() discovers @tool-decorated functions from the tools dir."""
        _write_tool(tmp_path, "greet.py", (
            "from fipsagents.baseagent.tools import tool\n\n"
            "@tool(description='Say hello', visibility='both')\n"
            "async def greet(name: str) -> str:\n"
            "    return f'hello {name}'\n"
        ))
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        assert agent.tools.get("greet") is not None
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_setup_loads_prompts(self, tmp_path: Path):
        """setup() loads prompt templates from the prompts dir."""
        _write_system_prompt(tmp_path)
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        prompt = agent.prompts.get("system")
        assert "test agent" in prompt.render()
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_setup_loads_rules(self, tmp_path: Path):
        """setup() loads rule files from the rules dir."""
        _write_rule(tmp_path, "safety", "Always be safe.")
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        rules = agent.rules.get_all()
        assert len(rules) == 1
        assert rules[0].name == "safety"
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_setup_loads_skills(self, tmp_path: Path):
        """setup() discovers skill stubs from the skills dir."""
        skills_dir = tmp_path / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\nInstructions."
        )
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        assert "test-skill" in agent.skills.list_skills()
        await agent.shutdown()


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_run_completes_normally(self, tmp_path: Path):
        """run() stops when step() returns DONE."""
        config = _make_config()
        agent = CountingAgent(stop_after=3, config=config, base_dir=tmp_path)
        await agent.setup()
        result = await agent.run()

        assert result == "done-after-3"
        assert agent.step_count == 3
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_run_enforces_max_iterations(self, tmp_path: Path):
        """run() stops after max_iterations even if agent never returns DONE."""
        config = _make_config(
            loop=LoopConfig(
                max_iterations=5,
                backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
            )
        )
        # stop_after=100 means it will never return DONE within 5 steps
        agent = CountingAgent(stop_after=100, config=config, base_dir=tmp_path)
        await agent.setup()
        result = await agent.run()

        assert result is None
        assert agent.step_count == 5
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_run_without_setup_raises(self, tmp_path: Path):
        """run() raises RuntimeError if setup() hasn't been called."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)

        with pytest.raises(RuntimeError, match="setup"):
            await agent.run()

    @pytest.mark.asyncio
    async def test_run_backoff_on_errors(self, tmp_path: Path):
        """run() applies backoff when step() raises, and continues looping."""
        config = _make_config(
            loop=LoopConfig(
                max_iterations=5,
                backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
            )
        )
        agent = ErrorAgent(config=config, base_dir=tmp_path)
        await agent.setup()
        result = await agent.run()

        # All 5 iterations should have been attempted despite errors.
        assert agent.step_count == 5
        assert result is None
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_run_recovers_from_transient_errors(self, tmp_path: Path):
        """run() recovers when step() stops raising."""
        config = _make_config(
            loop=LoopConfig(
                max_iterations=10,
                backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
            )
        )
        agent = ErrorThenDoneAgent(
            errors_before_done=2, config=config, base_dir=tmp_path
        )
        await agent.setup()
        result = await agent.run()

        assert result == "recovered"
        assert agent.step_count == 3  # 2 errors + 1 success
        await agent.shutdown()


# ---------------------------------------------------------------------------
# start() — full lifecycle
# ---------------------------------------------------------------------------


class TestStart:
    @pytest.mark.asyncio
    async def test_start_runs_full_lifecycle(self, tmp_path: Path):
        """start() does setup -> run -> shutdown."""
        config = _make_config()
        agent = CountingAgent(stop_after=2, config=config, base_dir=tmp_path)
        result = await agent.start()

        assert result == "done-after-2"
        # shutdown should have been called (setup_done reset to False)
        assert agent._setup_done is False

    @pytest.mark.asyncio
    async def test_start_calls_shutdown_on_error(self, tmp_path: Path):
        """start() ensures shutdown runs even when run() errors out."""
        config = _make_config(
            loop=LoopConfig(
                max_iterations=3,
                backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
            )
        )
        agent = ErrorAgent(config=config, base_dir=tmp_path)
        # Should not raise — errors are caught in run(), and run returns None
        # when max iterations reached.
        result = await agent.start()
        assert result is None
        assert agent._setup_done is False


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------


class TestConversationState:
    def test_add_message(self):
        config = _make_config()
        agent = CountingAgent(config=config)
        agent.add_message("user", "hello")
        agent.add_message("assistant", "hi there")

        assert len(agent.messages) == 2
        assert agent.messages[0] == {"role": "user", "content": "hello"}
        assert agent.messages[1] == {"role": "assistant", "content": "hi there"}

    def test_get_messages_returns_copy(self):
        config = _make_config()
        agent = CountingAgent(config=config)
        agent.add_message("user", "test")

        msgs = agent.get_messages()
        msgs.append({"role": "user", "content": "extra"})

        # Original should be unaffected.
        assert len(agent.messages) == 1

    def test_clear_messages(self):
        config = _make_config()
        agent = CountingAgent(config=config)
        agent.add_message("user", "test")
        agent.clear_messages()

        assert agent.messages == []


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_use_tool_dispatches(self):
        """use_tool() delegates to the registry's execute method."""
        config = _make_config()
        agent = CountingAgent(config=config)

        @tool(description="Echo input", visibility="both")
        async def echo(msg: str) -> str:
            return f"echo:{msg}"

        agent.tools.register(echo)
        result = await agent.use_tool("echo", msg="hello")

        assert not result.is_error
        assert result.result == "echo:hello"

    @pytest.mark.asyncio
    async def test_use_tool_unknown_tool(self):
        """use_tool() returns an error result for unknown tools."""
        config = _make_config()
        agent = CountingAgent(config=config)
        result = await agent.use_tool("nonexistent")

        assert result.is_error
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_use_tool_logs_call(self, caplog):
        """use_tool() logs the tool call."""
        config = _make_config()
        agent = CountingAgent(config=config)

        @tool(description="noop", visibility="both")
        async def noop() -> str:
            return "ok"

        agent.tools.register(noop)

        with caplog.at_level(logging.INFO):
            await agent.use_tool("noop")

        assert any("noop" in record.message for record in caplog.records)

    def test_get_tool_schemas(self):
        """get_tool_schemas() returns schemas for LLM-visible tools only."""
        config = _make_config()
        agent = CountingAgent(config=config)

        @tool(description="visible", visibility="llm_only")
        async def visible_tool(q: str) -> str:
            return q

        @tool(description="hidden", visibility="agent_only")
        async def hidden_tool(q: str) -> str:
            return q

        agent.tools.register(visible_tool)
        agent.tools.register(hidden_tool)
        schemas = agent.get_tool_schemas()

        names = {s["function"]["name"] for s in schemas}
        assert "visible_tool" in names
        assert "hidden_tool" not in names


# ---------------------------------------------------------------------------
# LLM convenience methods
# ---------------------------------------------------------------------------


class TestLLMConvenience:
    @pytest.mark.asyncio
    async def test_call_model_requires_setup(self):
        """call_model raises when LLM client is not initialised."""
        config = _make_config()
        agent = CountingAgent(config=config)

        with pytest.raises(RuntimeError, match="LLM client not initialised"):
            await agent.call_model()

    @pytest.mark.asyncio
    async def test_call_model_delegates_to_llm(self, tmp_path: Path):
        """call_model() passes messages and tools to self.llm.call_model()."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        # Replace the LLM client with a mock.
        mock_response = MagicMock(spec=ModelResponse)
        mock_response.content = "test response"
        agent.llm.call_model = AsyncMock(return_value=mock_response)

        agent.add_message("user", "hello")
        result = await agent.call_model()

        assert result is mock_response
        # Verify the call was made with our messages.
        call_args = agent.llm.call_model.call_args
        assert call_args[0][0] == agent.messages

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_call_model_uses_explicit_messages(self, tmp_path: Path):
        """call_model() uses explicit messages when provided."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        mock_response = MagicMock(spec=ModelResponse)
        agent.llm.call_model = AsyncMock(return_value=mock_response)

        custom_msgs = [{"role": "user", "content": "custom"}]
        await agent.call_model(messages=custom_msgs)

        call_args = agent.llm.call_model.call_args
        assert call_args[0][0] == custom_msgs

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_call_model_json_delegates(self, tmp_path: Path):
        """call_model_json() delegates to self.llm.call_model_json()."""
        from pydantic import BaseModel as PydanticBase

        class Answer(PydanticBase):
            text: str

        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        expected = Answer(text="hello")
        agent.llm.call_model_json = AsyncMock(return_value=expected)

        result = await agent.call_model_json(Answer)
        assert result.text == "hello"

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_call_model_validated_delegates(self, tmp_path: Path):
        """call_model_validated() delegates to self.llm.call_model_validated()."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        agent.llm.call_model_validated = AsyncMock(return_value="validated-ok")

        def validator(resp: ModelResponse) -> str:
            return "validated"

        result = await agent.call_model_validated(validator)
        assert result == "validated-ok"
        assert agent.llm.call_model_validated.called

        await agent.shutdown()


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    @pytest.mark.asyncio
    async def test_includes_system_prompt(self, tmp_path: Path):
        """build_system_prompt() includes the 'system' prompt template."""
        _write_system_prompt(tmp_path, "You are Agent X.")
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        prompt = agent.build_system_prompt()
        assert "Agent X" in prompt

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_includes_rules(self, tmp_path: Path):
        """build_system_prompt() includes all loaded rules."""
        _write_rule(tmp_path, "safety", "Never harm humans.")
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        prompt = agent.build_system_prompt()
        assert "Never harm humans" in prompt

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_includes_skill_manifest(self, tmp_path: Path):
        """build_system_prompt() includes skill manifests."""
        skills_dir = tmp_path / "skills" / "search"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: search\ndescription: Web search capability\n"
            "triggers:\n  - search the web\n---\n\nFull instructions."
        )
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        prompt = agent.build_system_prompt()
        assert "search" in prompt
        assert "Web search capability" in prompt

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_empty_when_nothing_loaded(self, tmp_path: Path):
        """build_system_prompt() returns empty string with no content loaded."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        prompt = agent.build_system_prompt()
        assert prompt == ""

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_combines_all_sections(self, tmp_path: Path):
        """build_system_prompt() combines prompt, rules, and skills with separators."""
        _write_system_prompt(tmp_path, "System instructions.")
        _write_rule(tmp_path, "tone", "Be professional.")
        skills_dir = tmp_path / "skills" / "code"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: code\ndescription: Write code\n---\n\nCode stuff."
        )
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        prompt = agent.build_system_prompt()
        assert "System instructions" in prompt
        assert "Be professional" in prompt
        assert "Write code" in prompt
        # Sections should be separated.
        assert "---" in prompt

        await agent.shutdown()


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------


class TestMCPIntegration:
    @pytest.mark.asyncio
    async def test_connect_mcp_registers_tools(self, tmp_path: Path):
        """connect_mcp() discovers and registers tools from an MCP server."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        # Build a fake MCP tool descriptor.
        mock_tool = SimpleNamespace(
            name="weather",
            description="Get weather",
            inputSchema={"type": "object", "properties": {"city": {"type": "string"}}},
        )

        # Build a mock FastMCP Client instance that behaves as an async
        # context manager and returns the fake tool from list_tools().
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Patch fastmcp.Client at the location where connect_mcp imports it.
        mock_client_cls = MagicMock(return_value=mock_client)
        with patch.dict("sys.modules", {"fastmcp": MagicMock(Client=mock_client_cls)}):
            # Re-import after patching so the `from fastmcp import Client`
            # inside connect_mcp picks up our mock.
            await agent.connect_mcp("http://fake-mcp:8080/mcp")

        # The tool should now be in the registry.
        assert agent.tools.get("weather") is not None
        meta = agent.tools.get("weather")
        assert meta.visibility == "llm_only"
        assert len(agent._mcp_clients) == 1

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_clears_mcp_clients(self, tmp_path: Path):
        """shutdown() closes MCP client connections."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        agent._mcp_clients.append(mock_client)

        await agent.shutdown()

        mock_client.close.assert_called_once()
        assert len(agent._mcp_clients) == 0


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemoryIntegration:
    @pytest.mark.asyncio
    async def test_default_memory_is_null_client(self, tmp_path: Path):
        """When no .memoryhub.yaml exists, memory is NullMemoryClient."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()

        assert isinstance(agent.memory, NullMemoryClient)
        # Should not raise — NullMemoryClient operations are no-ops.
        result = await agent.memory.search("test")
        assert result == []

        await agent.shutdown()


# ---------------------------------------------------------------------------
# Shutdown idempotency
# ---------------------------------------------------------------------------


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_resets_setup_flag(self, tmp_path: Path):
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()
        assert agent._setup_done is True

        await agent.shutdown()
        assert agent._setup_done is False

    @pytest.mark.asyncio
    async def test_shutdown_without_setup(self):
        """shutdown() does not raise when called before setup."""
        config = _make_config()
        agent = CountingAgent(config=config)
        await agent.shutdown()  # Should not raise.

    @pytest.mark.asyncio
    async def test_double_shutdown(self, tmp_path: Path):
        """shutdown() can be called twice safely."""
        config = _make_config()
        agent = CountingAgent(config=config, base_dir=tmp_path)
        await agent.setup()
        await agent.shutdown()
        await agent.shutdown()  # Should not raise.
