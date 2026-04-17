"""Integration tests for the memory-prefix slot through real setup().

Unlike test_memory_prefix.py (which uses FakeMemoryClient and bypasses
setup), these tests exercise the full BaseAgent.setup() path with a real
markdown memory backend, real config files on disk, and real prompt loading.

This verifies:
- agent.yaml ``memory.backend: markdown`` dispatches correctly
- The markdown backend reads a real file and returns content
- setup() step 10 injects the prefix at the correct position
- The prefix_role config propagates through to the message dict
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fipsagents.baseagent import BaseAgent, StepResult
from fipsagents.baseagent.memory_markdown import MarkdownMemoryClient


# ---------------------------------------------------------------------------
# Minimal concrete agent — just enough to call setup()
# ---------------------------------------------------------------------------


class _IntegrationAgent(BaseAgent):
    """Concrete BaseAgent subclass for integration tests."""

    async def step(self) -> StepResult:
        return StepResult.done()


def _write_fixtures(
    tmp_path: Path,
    *,
    memory_content: str,
    prefix_role: str = "system",
    max_prefix_chars: int = 8000,
) -> Path:
    """Write agent.yaml, memory config, and memory file to tmp_path.

    Returns the path to agent.yaml.
    """
    # Memory file (Level 1 — single compound doc).
    (tmp_path / "agent-memory.md").write_text(memory_content)

    # Markdown memory config.
    (tmp_path / ".memory-markdown.yaml").write_text("file: agent-memory.md\n")

    # System prompt.
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text(
        "---\nname: system\n---\nYou are a test agent.\n"
    )

    # Minimal agent.yaml.
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        f"""\
agent:
  name: prefix-integration-test

model:
  name: test-model
  endpoint: http://localhost:1234/v1

memory:
  backend: markdown
  config_path: .memory-markdown.yaml
  prefix_role: "{prefix_role}"
  max_prefix_chars: {max_prefix_chars}

tools:
  local_dir: tools
"""
    )

    # Empty tools dir so discovery doesn't warn.
    (tmp_path / "tools").mkdir()

    return agent_yaml


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_injects_markdown_prefix(tmp_path: Path) -> None:
    """Full setup() with a markdown backend produces the expected prefix."""
    config_path = _write_fixtures(
        tmp_path,
        memory_content=(
            "## Project context\n\n"
            "This agent handles customer support.\n\n"
            "## Key decisions\n\n"
            "Always escalate billing issues.\n"
        ),
    )

    agent = _IntegrationAgent(config_path=config_path)
    await agent.setup()

    try:
        # Verify the backend was correctly created.
        assert isinstance(agent.memory, MarkdownMemoryClient)

        # messages[0] is the system prompt, messages[1] is the prefix.
        assert len(agent.messages) == 2, (
            f"Expected [system, prefix], got {len(agent.messages)} messages"
        )
        assert agent.messages[0]["role"] == "system"
        assert "You are a test agent" in agent.messages[0]["content"]

        prefix_msg = agent.messages[1]
        assert prefix_msg["role"] == "system"
        assert "customer support" in prefix_msg["content"]
        assert "billing issues" in prefix_msg["content"]
        # Two sections joined by separator.
        assert "\n\n---\n\n" in prefix_msg["content"]
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_setup_with_developer_role(tmp_path: Path) -> None:
    """prefix_role=developer propagates through real setup()."""
    config_path = _write_fixtures(
        tmp_path,
        memory_content="## Note\n\nUser prefers metric units.\n",
        prefix_role="developer",
    )

    agent = _IntegrationAgent(config_path=config_path)
    await agent.setup()

    try:
        assert len(agent.messages) == 2
        assert agent.messages[1]["role"] == "developer"
        assert "metric units" in agent.messages[1]["content"]
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_setup_with_empty_memory_file(tmp_path: Path) -> None:
    """Empty memory file produces no prefix message."""
    config_path = _write_fixtures(tmp_path, memory_content="")

    agent = _IntegrationAgent(config_path=config_path)
    await agent.setup()

    try:
        assert len(agent.messages) == 1, (
            "Empty memory file should produce only the system prompt"
        )
        assert agent.messages[0]["role"] == "system"
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_setup_truncates_large_prefix(tmp_path: Path) -> None:
    """max_prefix_chars is respected through real setup()."""
    # One section with lots of content.
    config_path = _write_fixtures(
        tmp_path,
        memory_content=f"## Big section\n\n{'x' * 500}\n",
        max_prefix_chars=100,
    )

    agent = _IntegrationAgent(config_path=config_path)
    await agent.setup()

    try:
        assert len(agent.messages) == 2
        prefix = agent.messages[1]["content"]
        # 100 chars of content + truncation marker.
        assert len(prefix) == 100 + len("\n\n… [truncated]")
        assert prefix.endswith("… [truncated]")
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_prefix_stable_across_conversation(tmp_path: Path) -> None:
    """Prefix at index 1 stays pinned after adding conversation turns."""
    config_path = _write_fixtures(
        tmp_path,
        memory_content="## Stable fact\n\nThe sky is blue.\n",
    )

    agent = _IntegrationAgent(config_path=config_path)
    await agent.setup()

    try:
        original_prefix = agent.messages[1]["content"]

        # Simulate conversation turns.
        for i in range(5):
            agent.add_message("user", f"Question {i}")
            agent.add_message("assistant", f"Answer {i}")

        assert agent.messages[1]["content"] == original_prefix
        assert len(agent.messages) == 12  # system + prefix + 5*(user+assistant)
    finally:
        await agent.shutdown()
