"""Regression tests for Mistral-format tool_call_id propagation (#37).

Mistral models emit 9-character alphanumeric tool_call_id values (e.g.
``jBwXKcpus``) rather than the ``call_abc123`` style used by OpenAI.
BaseAgent's ``astep_stream`` pipeline must preserve whatever ID format the
provider uses end-to-end:

  streamed delta -> tool_buf -> assembled_calls
  assembled_calls -> assistant message tool_calls[].id
  assembled_calls -> tool message tool_call_id
  assembled_calls -> ToolResultEvent.call_id

All tests mock at the ``llm.call_model_stream_raw`` level — no real LLM calls.
"""

from __future__ import annotations

import pytest

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.events import ToolCallDelta, ToolResultEvent
from fipsagents.baseagent.tools import ToolRegistry, tool


# ---------------------------------------------------------------------------
# Tool fixture
# ---------------------------------------------------------------------------


@tool(description="Add two integers", visibility="llm_only")
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First operand.
        b: Second operand.
    """
    return str(a + b)


# ---------------------------------------------------------------------------
# Mock chunk helpers
# ---------------------------------------------------------------------------


def _chunk(*, content=None, tool_calls=None, finish_reason=None):
    """Build a minimal mock litellm streaming chunk."""
    from unittest.mock import MagicMock

    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = None
    delta.tool_calls = tool_calls
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


def _tool_call_delta(*, index=0, id=None, name=None, arguments=None):
    """Build a mock tool_call delta object as litellm would emit it."""
    from unittest.mock import MagicMock

    tc = MagicMock()
    tc.index = index
    tc.id = id
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    tc.function = fn
    return tc


def _make_agent(call_sequences):
    """Construct a minimal BaseAgent with mocked streaming.

    ``call_sequences`` is a list of chunk lists; each list is replayed for
    one call to ``call_model_stream_raw``.
    """
    agent = BaseAgent.__new__(BaseAgent)
    agent.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 1+2?"},
    ]
    agent.tools = ToolRegistry()
    agent.tools.register(add)
    agent._tool_inspector = None

    call_count = 0

    async def mock_stream_raw(*args, **kwargs):
        nonlocal call_count
        chunks = call_sequences[call_count]
        call_count += 1
        for c in chunks:
            yield c

    from unittest.mock import MagicMock
    agent.llm = MagicMock()
    agent.llm.call_model_stream_raw = mock_stream_raw

    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mistral_format_tool_call_id_preserved_through_pipeline():
    """Mistral 9-char alphanumeric IDs survive the full astep_stream pipeline."""
    mistral_id = "jBwXKcpus"

    # First call: model requests tool execution.
    first_call = [
        _chunk(
            tool_calls=[
                _tool_call_delta(
                    index=0,
                    id=mistral_id,
                    name="add",
                    arguments='{"a": 1, "b": 2}',
                )
            ],
            finish_reason=None,
        ),
        _chunk(finish_reason="tool_calls"),
    ]

    # Second call: model returns the final answer.
    second_call = [
        _chunk(content="3", finish_reason=None),
        _chunk(finish_reason="stop"),
    ]

    agent = _make_agent([first_call, second_call])

    events = []
    async for event in agent.astep_stream():
        events.append(event)

    tool_call_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]

    assert tool_call_deltas, "Expected at least one ToolCallDelta"
    assert tool_call_deltas[0].call_id == mistral_id, (
        f"ToolCallDelta.call_id expected {mistral_id!r}, "
        f"got {tool_call_deltas[0].call_id!r}"
    )

    assert tool_results, "Expected at least one ToolResultEvent"
    assert tool_results[0].call_id == mistral_id, (
        f"ToolResultEvent.call_id expected {mistral_id!r}, "
        f"got {tool_results[0].call_id!r}"
    )

    # Assistant message should carry the tool_calls list with the original ID.
    assistant_msgs = [m for m in agent.messages if m.get("role") == "assistant"
                      and m.get("tool_calls")]
    assert assistant_msgs, "Expected an assistant message with tool_calls"
    tc_id = assistant_msgs[0]["tool_calls"][0]["id"]
    assert tc_id == mistral_id, (
        f"assistant tool_calls[0].id expected {mistral_id!r}, got {tc_id!r}"
    )

    # Tool result message must reference the same ID.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs, "Expected a tool result message"
    assert tool_msgs[0]["tool_call_id"] == mistral_id, (
        f"tool message tool_call_id expected {mistral_id!r}, "
        f"got {tool_msgs[0]['tool_call_id']!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("call_id", [
    "jBwXKcpus",           # Mistral 9-char alphanumeric
    "call_abc123def456",   # OpenAI format
    "chatcmpl-abc",        # another provider format
    "a",                   # minimal single-char
    "tool_call_0_4f8b9c2e",  # Granite-style (hypothetical)
])
async def test_various_provider_id_formats(call_id):
    """ID format is irrelevant — the pipeline preserves whatever arrives."""
    first_call = [
        _chunk(
            tool_calls=[
                _tool_call_delta(
                    index=0, id=call_id, name="add",
                    arguments='{"a": 2, "b": 3}',
                )
            ],
            finish_reason=None,
        ),
        _chunk(finish_reason="tool_calls"),
    ]
    second_call = [
        _chunk(content="5", finish_reason=None),
        _chunk(finish_reason="stop"),
    ]

    agent = _make_agent([first_call, second_call])

    tool_result_events = []
    async for event in agent.astep_stream():
        if isinstance(event, ToolResultEvent):
            tool_result_events.append(event)

    assert tool_result_events, f"No ToolResultEvent for call_id={call_id!r}"
    assert tool_result_events[0].call_id == call_id, (
        f"call_id={call_id!r} not preserved; "
        f"got {tool_result_events[0].call_id!r}"
    )

    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == call_id, (
        f"tool message tool_call_id={call_id!r} not preserved; "
        f"got {tool_msgs[0]['tool_call_id']!r}"
    )


@pytest.mark.asyncio
async def test_multiple_concurrent_tool_calls_preserve_ids():
    """Two concurrent tool calls with distinct Mistral-format IDs stay separate."""
    id_0 = "jBwXKcpus"
    id_1 = "638IxSXTn"

    first_call = [
        # First tool call (index 0).
        _chunk(
            tool_calls=[
                _tool_call_delta(
                    index=0, id=id_0, name="add",
                    arguments='{"a": 1, "b": 2}',
                )
            ],
            finish_reason=None,
        ),
        # Second tool call (index 1) in the same turn.
        _chunk(
            tool_calls=[
                _tool_call_delta(
                    index=1, id=id_1, name="add",
                    arguments='{"a": 3, "b": 4}',
                )
            ],
            finish_reason=None,
        ),
        _chunk(finish_reason="tool_calls"),
    ]
    second_call = [
        _chunk(content="3 and 7", finish_reason=None),
        _chunk(finish_reason="stop"),
    ]

    agent = _make_agent([first_call, second_call])

    tool_call_deltas = []
    tool_results = []
    async for event in agent.astep_stream():
        if isinstance(event, ToolCallDelta):
            tool_call_deltas.append(event)
        elif isinstance(event, ToolResultEvent):
            tool_results.append(event)

    # Both IDs should appear in ToolCallDelta events.
    delta_ids = {e.call_id for e in tool_call_deltas if e.call_id is not None}
    assert id_0 in delta_ids, f"{id_0!r} missing from ToolCallDelta IDs: {delta_ids}"
    assert id_1 in delta_ids, f"{id_1!r} missing from ToolCallDelta IDs: {delta_ids}"

    # Both IDs should appear in ToolResultEvent events.
    result_ids = {e.call_id for e in tool_results}
    assert id_0 in result_ids, f"{id_0!r} missing from ToolResultEvent IDs: {result_ids}"
    assert id_1 in result_ids, f"{id_1!r} missing from ToolResultEvent IDs: {result_ids}"

    # Both tool messages must use the correct IDs.
    tool_msgs = {m["tool_call_id"] for m in agent.messages if m.get("role") == "tool"}
    assert id_0 in tool_msgs, f"{id_0!r} missing from tool messages: {tool_msgs}"
    assert id_1 in tool_msgs, f"{id_1!r} missing from tool messages: {tool_msgs}"
