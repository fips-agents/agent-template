"""Tests for ``astep_stream``'s tool-emission gating.

Covers the four-way matrix between ``config.tools.enabled`` and the
per-call ``include_tools`` override:

================================  ================  =====================
config.tools.enabled              include_tools     ``tools=`` to LLM
================================  ================  =====================
True                              None (default)    list of schemas
False                             None (default)    None
any                               True              list of schemas
any                               False             None
================================  ================  =====================

The test mocks ``LLMClient.call_model_stream_raw`` and inspects the
``tools`` kwarg captured on each invocation.  The agent's other
machinery (memory, tool registry, reasoning parser) is bypassed with a
minimal stub so we exercise just the gating logic.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from types import SimpleNamespace

import pytest

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.config import AgentConfig, ToolsConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_FAKE_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo input.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
    }
]


class _SpyLLM:
    """Captures kwargs of ``call_model_stream_raw`` and yields one final
    chunk so the loop terminates cleanly with ``finish_reason='stop'``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call_model_stream_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        # Single chunk: empty content delta + finish_reason=stop ends the
        # loop on the first iteration with no tool calls to dispatch.
        choice = SimpleNamespace(
            delta=SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
        yield SimpleNamespace(choices=[choice], usage=None)


class _StubAgent(BaseAgent):
    """Minimal BaseAgent that bypasses ``setup()``.

    The real ``astep_stream`` calls ``_require_llm``,
    ``_inject_deferred_memory``, ``get_tool_schemas``, and reads
    ``self._reasoning_parser`` / ``self.config``.  We provide just enough
    state for those.
    """

    def __init__(
        self,
        *,
        config: AgentConfig | None,
        schemas: list[dict[str, Any]],
    ) -> None:
        # Skip BaseAgent.__init__ entirely — it expects a config path or
        # an AgentConfig and triggers full setup.  We assemble state by
        # hand so tests stay focused on the gating logic.
        self.llm = _SpyLLM()
        self.config = config
        self.messages: list[dict[str, Any]] = []
        self._reasoning_parser = None
        self._schemas = schemas

    def get_tool_schemas(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return list(self._schemas)

    async def _inject_deferred_memory(self) -> None:  # type: ignore[override]
        return None


def _config_with_tools_enabled(enabled: bool) -> AgentConfig:
    return AgentConfig(tools=ToolsConfig(enabled=enabled))


async def _drain(agent: _StubAgent, **kwargs: Any) -> None:
    """Iterate ``astep_stream`` to completion.  Discards events."""
    async for _ in agent.astep_stream(**kwargs):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_enabled, include_tools, expect_schemas",
    [
        # config drives behavior when include_tools is unset
        (True, None, True),
        (False, None, False),
        # explicit kwarg wins over config in either direction
        (True, False, False),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
async def test_astep_stream_tools_gating(
    config_enabled: bool,
    include_tools: bool | None,
    expect_schemas: bool,
) -> None:
    agent = _StubAgent(
        config=_config_with_tools_enabled(config_enabled),
        schemas=_FAKE_SCHEMA,
    )

    if include_tools is None:
        await _drain(agent)
    else:
        await _drain(agent, include_tools=include_tools)

    assert len(agent.llm.calls) == 1
    tools_arg = agent.llm.calls[0]["tools"]
    if expect_schemas:
        assert tools_arg == _FAKE_SCHEMA
    else:
        assert tools_arg is None


@pytest.mark.asyncio
async def test_astep_stream_with_no_config_defaults_to_emitting_tools() -> None:
    """If ``self.config`` is ``None`` (eg test stub or bare init), the
    legacy behavior — emit registered schemas — is preserved for
    backward compatibility."""
    agent = _StubAgent(config=None, schemas=_FAKE_SCHEMA)
    await _drain(agent)
    assert agent.llm.calls[0]["tools"] == _FAKE_SCHEMA


@pytest.mark.asyncio
async def test_astep_stream_no_registered_tools_sends_none() -> None:
    """Empty schema list always serialises to ``tools=None`` regardless
    of the config switch — ``tools=[]`` would also be valid but the loop
    normalises to None to avoid sending an empty array on the wire."""
    agent = _StubAgent(
        config=_config_with_tools_enabled(True),
        schemas=[],
    )
    await _drain(agent)
    assert agent.llm.calls[0]["tools"] is None
