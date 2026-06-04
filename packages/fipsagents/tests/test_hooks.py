"""Tests for the lifecycle hook system (hooks.py, HookEntryConfig, agent integration).

Covers:
- HookResult property semantics (success, blocked, timeout, error).
- HookRunner.hooks_for_event matching and filtering.
- HookRunner.fire subprocess execution with mocked asyncio subprocess.
- load_hooks_from_dir YAML auto-discovery from a directory.
- create_hook_runner combining config and file hooks.
- HookEntryConfig Pydantic validation.
- Agent lifecycle integration (setup_complete, shutdown, custom events).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.config import AgentConfig, HookEntryConfig, MemoryConfig
from fipsagents.baseagent.hooks import (
    HookEntry,
    HookResult,
    HookRunner,
    create_hook_runner,
    load_hooks_from_dir,
)
from fipsagents.baseagent.memory import NullMemoryClient


# ---------------------------------------------------------------------------
# Section 1: HookResult properties
# ---------------------------------------------------------------------------


def test_hook_result_success():
    """exit_code=0, no timeout, no error produces success=True, blocked=False."""
    hook = HookEntry(event="test", command="echo ok")
    result = HookResult(hook=hook, exit_code=0, stdout="ok")
    assert result.success is True
    assert result.blocked is False


def test_hook_result_blocked():
    """exit_code=1 produces blocked=True, success=False."""
    hook = HookEntry(event="test", command="false")
    result = HookResult(hook=hook, exit_code=1, stderr="failed")
    assert result.blocked is True
    assert result.success is False


def test_hook_result_timeout():
    """timed_out=True produces success=False, blocked=False (exit_code is None)."""
    hook = HookEntry(event="test", command="sleep 100")
    result = HookResult(hook=hook, timed_out=True)
    assert result.success is False
    assert result.blocked is False
    assert result.exit_code is None


def test_hook_result_error():
    """error set produces success=False."""
    hook = HookEntry(event="test", command="missing-bin")
    result = HookResult(hook=hook, error="No such file or directory")
    assert result.success is False


# ---------------------------------------------------------------------------
# Section 2: HookRunner.hooks_for_event
# ---------------------------------------------------------------------------


def test_hooks_for_event_basic():
    """Returns hooks matching the event name."""
    h1 = HookEntry(event="setup_complete", command="echo 1")
    h2 = HookEntry(event="shutdown", command="echo 2")
    h3 = HookEntry(event="setup_complete", command="echo 3")
    runner = HookRunner([h1, h2, h3])

    matched = runner.hooks_for_event("setup_complete")
    assert matched == [h1, h3]


def test_hooks_for_event_empty():
    """No registered hooks returns empty list."""
    runner = HookRunner()
    assert runner.hooks_for_event("setup_complete") == []


def test_hooks_for_event_matcher_match():
    """Matcher 'web_*' matches tool_name='web_search'."""
    hook = HookEntry(event="pre_tool_use", command="echo ok", matcher="web_*")
    runner = HookRunner([hook])

    matched = runner.hooks_for_event("pre_tool_use", tool_name="web_search")
    assert matched == [hook]


def test_hooks_for_event_matcher_no_match():
    """Matcher 'web_*' does not match tool_name='db_query'."""
    hook = HookEntry(event="pre_tool_use", command="echo ok", matcher="web_*")
    runner = HookRunner([hook])

    matched = runner.hooks_for_event("pre_tool_use", tool_name="db_query")
    assert matched == []


def test_hooks_for_event_no_matcher_matches_all():
    """Hook without matcher matches regardless of tool_name."""
    hook = HookEntry(event="pre_tool_use", command="echo ok")
    runner = HookRunner([hook])

    assert runner.hooks_for_event("pre_tool_use", tool_name="anything") == [hook]
    assert runner.hooks_for_event("pre_tool_use", tool_name=None) == [hook]
    assert runner.hooks_for_event("pre_tool_use") == [hook]


# ---------------------------------------------------------------------------
# Section 3: HookRunner.fire (mocked subprocess)
# ---------------------------------------------------------------------------


def _mock_process(stdout=b"", stderr=b"", returncode=0):
    """Build a mock asyncio subprocess process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()
    return proc


@pytest.mark.asyncio
async def test_fire_success():
    """Hook exits 0, stdout captured and stripped."""
    proc = _mock_process(stdout=b"  hello world  \n", returncode=0)
    runner = HookRunner([HookEntry(event="test", command="echo hello")])

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        with patch("asyncio.wait_for", return_value=(b"  hello world  \n", b"")):
            proc.communicate = AsyncMock(return_value=(b"  hello world  \n", b""))
            results = await runner.fire("test")

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].stdout == "hello world"


@pytest.mark.asyncio
async def test_fire_timeout():
    """Hook exceeds timeout, process killed, timed_out=True."""
    hook = HookEntry(event="test", command="sleep 100", timeout=0.01)
    runner = HookRunner([hook])

    proc = _mock_process()
    proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            results = await runner.fire("test")

    assert len(results) == 1
    assert results[0].timed_out is True
    assert results[0].success is False


@pytest.mark.asyncio
async def test_fire_nonzero_exit():
    """Hook exits non-zero, stdout/stderr captured, blocked=True."""
    proc = _mock_process(
        stdout=b"partial output", stderr=b"error details", returncode=2
    )
    runner = HookRunner([HookEntry(event="test", command="bad-cmd")])

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        results = await runner.fire("test")

    assert len(results) == 1
    assert results[0].blocked is True
    assert results[0].stderr == "error details"
    assert results[0].stdout == "partial output"


@pytest.mark.asyncio
async def test_fire_oserror():
    """create_subprocess_shell raises OSError, error field populated."""
    runner = HookRunner([HookEntry(event="test", command="/no/such/binary")])

    with patch(
        "asyncio.create_subprocess_shell",
        side_effect=OSError("No such file or directory"),
    ):
        results = await runner.fire("test")

    assert len(results) == 1
    assert results[0].error == "No such file or directory"
    assert results[0].success is False


@pytest.mark.asyncio
async def test_fire_empty_stdout():
    """Hook exits 0 with empty stdout."""
    proc = _mock_process(stdout=b"", returncode=0)
    runner = HookRunner([HookEntry(event="test", command="true")])

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        results = await runner.fire("test")

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].stdout == ""


@pytest.mark.asyncio
async def test_fire_env_vars():
    """HOOK_EVENT and env_extra vars passed to subprocess."""
    proc = _mock_process(stdout=b"ok", returncode=0)
    runner = HookRunner([HookEntry(event="setup_complete", command="env")])

    with patch("asyncio.create_subprocess_shell", return_value=proc) as mock_shell:
        await runner.fire(
            "setup_complete",
            env_extra={"AGENT_NAME": "test-agent", "CUSTOM_VAR": "42"},
        )

    call_kwargs = mock_shell.call_args
    env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
    assert env["HOOK_EVENT"] == "setup_complete"
    assert env["AGENT_NAME"] == "test-agent"
    assert env["CUSTOM_VAR"] == "42"


@pytest.mark.asyncio
async def test_fire_custom_event():
    """Custom event name fires matching hooks."""
    proc = _mock_process(stdout=b"custom", returncode=0)
    runner = HookRunner([
        HookEntry(event="pre_auth", command="echo custom"),
        HookEntry(event="other", command="echo nope"),
    ])

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        results = await runner.fire("pre_auth")

    assert len(results) == 1
    assert results[0].stdout == "custom"


@pytest.mark.asyncio
async def test_fire_multiple_hooks():
    """Multiple hooks for same event all fire in registration order."""
    proc1 = _mock_process(stdout=b"first", returncode=0)
    proc2 = _mock_process(stdout=b"second", returncode=0)

    runner = HookRunner([
        HookEntry(event="test", command="echo first", name="hook-1"),
        HookEntry(event="test", command="echo second", name="hook-2"),
    ])

    call_count = 0

    async def _rotating_shell(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return proc1 if call_count == 1 else proc2

    with patch("asyncio.create_subprocess_shell", side_effect=_rotating_shell):
        results = await runner.fire("test")

    assert len(results) == 2
    assert results[0].stdout == "first"
    assert results[1].stdout == "second"


# ---------------------------------------------------------------------------
# Section 4: load_hooks_from_dir
# ---------------------------------------------------------------------------


def test_discover_yaml_files(tmp_path):
    """Reads *.yaml and *.yml files from directory."""
    (tmp_path / "alpha.yaml").write_text(
        "event: setup_complete\ncommand: echo alpha\n"
    )
    (tmp_path / "beta.yml").write_text(
        "event: shutdown\ncommand: echo beta\ntimeout: 5.0\n"
    )

    entries = load_hooks_from_dir(tmp_path)
    assert len(entries) == 2
    names = {e.name for e in entries}
    assert "alpha" in names
    assert "beta" in names
    assert entries[0].event == "setup_complete"
    assert entries[1].event == "shutdown"
    assert entries[1].timeout == 5.0


def test_discover_skips_underscore(tmp_path):
    """Files starting with _ are ignored."""
    (tmp_path / "_disabled.yaml").write_text(
        "event: test\ncommand: echo hidden\n"
    )
    (tmp_path / "visible.yaml").write_text(
        "event: test\ncommand: echo visible\n"
    )

    entries = load_hooks_from_dir(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "visible"


def test_discover_skips_dotfiles(tmp_path):
    """Files starting with . are ignored."""
    (tmp_path / ".hidden.yaml").write_text(
        "event: test\ncommand: echo hidden\n"
    )
    (tmp_path / "public.yaml").write_text(
        "event: test\ncommand: echo public\n"
    )

    entries = load_hooks_from_dir(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "public"


def test_discover_skips_non_yaml(tmp_path):
    """Shell scripts in directory are not loaded as hooks."""
    (tmp_path / "hook.sh").write_text("#!/bin/bash\necho hi\n")
    (tmp_path / "hook.json").write_text('{"event": "test"}')
    (tmp_path / "real.yaml").write_text(
        "event: test\ncommand: echo real\n"
    )

    entries = load_hooks_from_dir(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "real"


def test_discover_missing_required(tmp_path):
    """YAML without event or command is skipped."""
    (tmp_path / "bad.yaml").write_text("timeout: 5.0\n")
    (tmp_path / "good.yaml").write_text(
        "event: test\ncommand: echo good\n"
    )

    entries = load_hooks_from_dir(tmp_path)
    assert len(entries) == 1
    assert entries[0].command == "echo good"


def test_discover_nonexistent_dir():
    """Non-existent directory returns empty list."""
    entries = load_hooks_from_dir("/nonexistent/hooks/dir")
    assert entries == []


# ---------------------------------------------------------------------------
# Section 5: create_hook_runner
# ---------------------------------------------------------------------------


def test_create_from_config_only():
    """Config hooks loaded when no directory given."""
    configs = [
        HookEntryConfig(event="setup_complete", command="echo config1"),
        HookEntryConfig(event="shutdown", command="echo config2"),
    ]
    runner = create_hook_runner(config_hooks=configs)
    assert len(runner) == 2
    hooks = runner.hooks_for_event("setup_complete")
    assert len(hooks) == 1
    assert hooks[0].source == "config"


def test_create_from_dir_only(tmp_path):
    """Directory hooks loaded when no config given."""
    (tmp_path / "hook.yaml").write_text(
        "event: test\ncommand: echo dir\n"
    )
    runner = create_hook_runner(hooks_dir=tmp_path)
    assert len(runner) == 1
    hooks = runner.hooks_for_event("test")
    assert hooks[0].source.startswith("file:")


def test_create_combined(tmp_path):
    """Config hooks first, then file hooks."""
    (tmp_path / "file_hook.yaml").write_text(
        "event: test\ncommand: echo from-file\n"
    )
    configs = [
        HookEntryConfig(event="test", command="echo from-config"),
    ]
    runner = create_hook_runner(config_hooks=configs, hooks_dir=tmp_path)
    assert len(runner) == 2
    hooks = runner.hooks_for_event("test")
    assert len(hooks) == 2
    assert hooks[0].source == "config"
    assert hooks[1].source.startswith("file:")


# ---------------------------------------------------------------------------
# Section 6: HookEntryConfig validation
# ---------------------------------------------------------------------------


def test_config_valid():
    """Valid config round-trips."""
    cfg = HookEntryConfig(
        event="setup_complete",
        command="echo hello",
        timeout=5.0,
        matcher="web_*",
        name="my-hook",
    )
    assert cfg.event == "setup_complete"
    assert cfg.command == "echo hello"
    assert cfg.timeout == 5.0
    assert cfg.matcher == "web_*"
    assert cfg.name == "my-hook"


def test_config_timeout_positive():
    """Timeout must be > 0."""
    with pytest.raises(ValidationError):
        HookEntryConfig(event="test", command="echo", timeout=0.0)

    with pytest.raises(ValidationError):
        HookEntryConfig(event="test", command="echo", timeout=-1.0)


def test_config_defaults():
    """Default timeout is 10.0, matcher and name are None."""
    cfg = HookEntryConfig(event="test", command="echo")
    assert cfg.timeout == 10.0
    assert cfg.matcher is None
    assert cfg.name is None


# ---------------------------------------------------------------------------
# Section 7: Agent lifecycle integration
# ---------------------------------------------------------------------------


class _RenderedPrompt:
    def render(self) -> str:
        return "You are a test agent."


class _PromptStub:
    def get(self, name: str) -> _RenderedPrompt:
        return _RenderedPrompt()


class _RulesStub:
    def get_combined_content(self) -> str:
        return ""


class _SkillsStub:
    def get_manifest(self) -> list:
        return []


def _make_agent(
    *,
    hooks: HookRunner | None = None,
    max_prefix_chars: int = 8000,
    prefix_role: str = "system",
    base_dir: Path | None = None,
) -> BaseAgent:
    """Build a minimal BaseAgent for lifecycle hook tests."""
    config = AgentConfig(
        model={"name": "test-model", "endpoint": "http://localhost:1234/v1"},
        memory=MemoryConfig(
            backend="null",
            max_prefix_chars=max_prefix_chars,
            prefix_role=prefix_role,
        ),
    )

    class _TestAgent(BaseAgent):
        def __init__(self) -> None:
            self.config = config
            self.memory = NullMemoryClient()
            self.hooks = hooks or HookRunner()
            self.messages: list[dict[str, Any]] = []
            self.prompts = _PromptStub()
            self.rules = _RulesStub()
            self.skills = _SkillsStub()
            self._base_dir = base_dir
            self._config_path = Path("/tmp/agent.yaml")
            self._setup_done = True
            self._mcp_clients: list[tuple[Any, str]] = []
            self._mcp_prompts: dict[str, tuple[Any, Any]] = {}
            self._mcp_resources: dict[str, tuple[Any, Any]] = {}
            self._mcp_resource_templates: dict[str, tuple[Any, Any]] = {}

    return _TestAgent()


@pytest.mark.asyncio
async def test_setup_complete_injects_context():
    """setup_complete hook stdout appears as a message."""
    proc = _mock_process(stdout=b"hello from hook", returncode=0)
    agent = _make_agent(
        hooks=HookRunner([
            HookEntry(event="setup_complete", command="echo 'hello from hook'"),
        ]),
        base_dir=Path("/tmp"),
    )

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        await agent._fire_setup_hooks(Path("/tmp"))

    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] == "hello from hook"


@pytest.mark.asyncio
async def test_setup_complete_truncation():
    """Long stdout is truncated per max_prefix_chars."""
    long_output = "x" * 200
    proc = _mock_process(stdout=long_output.encode(), returncode=0)
    agent = _make_agent(
        hooks=HookRunner([
            HookEntry(event="setup_complete", command="echo long"),
        ]),
        max_prefix_chars=50,
        base_dir=Path("/tmp"),
    )

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        await agent._fire_setup_hooks(Path("/tmp"))

    assert len(agent.messages) == 1
    content = agent.messages[0]["content"]
    assert content == "x" * 50 + "\n\n... [truncated]" or content == "x" * 50 + "\n\n… [truncated]"


@pytest.mark.asyncio
async def test_setup_complete_empty_stdout_no_message():
    """Empty hook output does not add a message."""
    proc = _mock_process(stdout=b"", returncode=0)
    agent = _make_agent(
        hooks=HookRunner([
            HookEntry(event="setup_complete", command="true"),
        ]),
        base_dir=Path("/tmp"),
    )

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        await agent._fire_setup_hooks(Path("/tmp"))

    assert len(agent.messages) == 0


@pytest.mark.asyncio
async def test_shutdown_hook_fires():
    """Shutdown hook fires during shutdown()."""
    proc = _mock_process(stdout=b"bye", returncode=0)
    agent = _make_agent(
        hooks=HookRunner([
            HookEntry(event="shutdown", command="echo bye"),
        ]),
        base_dir=Path("/tmp"),
    )

    with patch("asyncio.create_subprocess_shell", return_value=proc) as mock_shell:
        await agent.shutdown()

    mock_shell.assert_called_once()
    call_kwargs = mock_shell.call_args
    env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
    assert env["HOOK_EVENT"] == "shutdown"


@pytest.mark.asyncio
async def test_custom_event_from_subclass():
    """Subclass can fire custom events via the HookRunner."""
    proc = _mock_process(stdout=b"checking auth", returncode=0)
    hooks = HookRunner([
        HookEntry(event="pre_auth", command="echo 'checking auth'"),
    ])
    agent = _make_agent(hooks=hooks, base_dir=Path("/tmp"))

    with patch("asyncio.create_subprocess_shell", return_value=proc) as mock_shell:
        results = await agent.hooks.fire(
            "pre_auth", env_extra={"USER_ID": "u123"}
        )

    assert len(results) == 1
    assert results[0].success is True
    env = mock_shell.call_args.kwargs.get("env") or mock_shell.call_args[1].get("env")
    assert env["HOOK_EVENT"] == "pre_auth"
    assert env["USER_ID"] == "u123"


@pytest.mark.asyncio
async def test_pre_tool_hook_blocks_on_nonzero():
    """pre_tool_use hook with exit 1 produces a blocked result."""
    proc = _mock_process(
        stdout=b"", stderr=b"blocked by policy", returncode=1
    )
    hooks = HookRunner([
        HookEntry(
            event="pre_tool_use",
            command="exit 1",
            matcher="dangerous_*",
        ),
    ])

    with patch("asyncio.create_subprocess_shell", return_value=proc):
        results = await hooks.fire(
            "pre_tool_use", tool_name="dangerous_delete"
        )

    assert len(results) == 1
    assert results[0].blocked is True
    assert results[0].stderr == "blocked by policy"


@pytest.mark.asyncio
async def test_deferred_memory_eager_returns_early():
    """_inject_deferred_memory returns early for the eager pattern."""
    agent = _make_agent()
    agent.messages = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
    ]
    original_count = len(agent.messages)

    await agent._inject_deferred_memory()

    assert len(agent.messages) == original_count
