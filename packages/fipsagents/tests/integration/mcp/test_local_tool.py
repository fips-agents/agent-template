"""Integration tests for local @tool dispatch through BaseAgent.

Covers tool discovery, schema generation, the sync step() path (which
consumes astep_stream() internally), and the astep_stream() event sequence
with real tool execution against mocked LLM responses.
"""

from __future__ import annotations

import pytest

from fipsagents.baseagent.agent import BaseAgent, StepOutcome
from fipsagents.baseagent.events import (
    ContentDelta,
    StreamComplete,
    ToolCallDelta,
    ToolResultEvent,
)

from .conftest import (
    _content_turn,
    _make_mock_stream,
    _make_stream_chunk,
    _make_tc_delta,
    _multi_tool_turn,
    _tool_call_turn,
    assert_stream_completes,
    assert_tool_call_result_ordering,
)


# ---------------------------------------------------------------------------
# TestToolDiscovery
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestToolDiscovery:
    """Verify that tools are registered correctly in the harness agent."""

    def test_tools_registered(self, harness_agent: BaseAgent) -> None:
        names = {t.name for t in harness_agent.tools.get_all()}
        assert "add" in names, f"'add' not in {names}"
        assert "multiply" in names, f"'multiply' not in {names}"
        assert "failing_tool" in names, f"'failing_tool' not in {names}"

    def test_tools_visible_to_llm(self, harness_agent: BaseAgent) -> None:
        llm_tools = {t.name for t in harness_agent.tools.get_llm_tools()}
        assert "add" in llm_tools
        assert "multiply" in llm_tools
        assert "failing_tool" in llm_tools

    def test_tool_schemas_generated(self, harness_agent: BaseAgent) -> None:
        schemas = harness_agent.tools.generate_schemas()
        assert len(schemas) >= 3
        schema_names = {s["function"]["name"] for s in schemas}
        assert {"add", "multiply", "failing_tool"}.issubset(schema_names), (
            f"Missing tools in schemas: {schema_names}"
        )
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]

    def test_tool_schema_parameters(self, harness_agent: BaseAgent) -> None:
        schemas = harness_agent.tools.generate_schemas()
        add_schema = next(
            s for s in schemas if s["function"]["name"] == "add"
        )
        params = add_schema["function"]["parameters"]
        assert params["type"] == "object"
        props = params["properties"]
        assert "a" in props, f"Parameter 'a' missing: {props}"
        assert "b" in props, f"Parameter 'b' missing: {props}"
        assert props["a"].get("type") == "number", (
            f"Expected 'number' for 'a', got {props['a']}"
        )
        assert props["b"].get("type") == "number", (
            f"Expected 'number' for 'b', got {props['b']}"
        )
        assert "required" in params
        assert "a" in params["required"]
        assert "b" in params["required"]


# ---------------------------------------------------------------------------
# TestSyncToolDispatch
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestSyncToolDispatch:
    """Test the sync step() path (which delegates to astep_stream internally)."""

    async def test_single_tool_call(self, harness_agent: BaseAgent) -> None:
        """Model calls add(3, 5), then returns final content '8.0'."""
        agent = harness_agent
        agent.add_message("user", "What is 3 + 5?")

        turn1 = _tool_call_turn("call_1", "add", '{"a": 3, "b": 5}')
        turn2 = _content_turn("The answer is 8.")

        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE, (
            f"Expected DONE, got {result.outcome}"
        )
        assert result.result == "The answer is 8.", (
            f"Unexpected result: {result.result!r}"
        )

    async def test_multi_tool_same_turn(self, harness_agent: BaseAgent) -> None:
        """Model calls add and multiply in one turn, then returns final content."""
        agent = harness_agent
        agent.add_message("user", "Add 2+3 and multiply 4*5.")

        turn1 = _multi_tool_turn([
            ("call_1", "add", '{"a": 2, "b": 3}'),
            ("call_2", "multiply", '{"a": 4, "b": 5}'),
        ])
        turn2 = _content_turn("Results: 5 and 20.")

        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE
        assert "Results: 5 and 20." in result.result

        # Confirm both tool results appear in message history.
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        tool_call_ids = {m["tool_call_id"] for m in tool_msgs}
        assert "call_1" in tool_call_ids, (
            f"add result missing from messages: {agent.messages}"
        )
        assert "call_2" in tool_call_ids, (
            f"multiply result missing from messages: {agent.messages}"
        )

    async def test_tool_error_propagation(self, harness_agent: BaseAgent) -> None:
        """Model calls failing_tool; the error appears in conversation history."""
        agent = harness_agent
        agent.add_message("user", "Trigger an error.")

        turn1 = _tool_call_turn("call_err", "failing_tool", '{"message": "boom"}')
        turn2 = _content_turn("I see an error occurred.")

        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE

        # The error must appear in the tool message in history.
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, "No tool messages found in history"
        error_msg = tool_msgs[0]["content"]
        assert "boom" in error_msg, (
            f"Expected error text 'boom' in tool message: {error_msg!r}"
        )
        assert "ERROR" in error_msg, (
            f"Expected 'ERROR' prefix in tool message: {error_msg!r}"
        )

    async def test_unknown_tool_handled(self, harness_agent: BaseAgent) -> None:
        """Model calls a nonexistent tool; agent records error and doesn't crash."""
        agent = harness_agent
        agent.add_message("user", "Call a ghost tool.")

        turn1 = _tool_call_turn("call_ghost", "nonexistent_tool", '{"x": 1}')
        turn2 = _content_turn("Handled the error gracefully.")

        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        result = await agent.step()

        assert result.outcome == StepOutcome.DONE

        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert tool_msgs, "Expected a tool message for the unknown tool call"
        error_content = tool_msgs[0]["content"]
        assert "nonexistent_tool" in error_content or "Unknown" in error_content, (
            f"Expected tool-not-found error in message: {error_content!r}"
        )


# ---------------------------------------------------------------------------
# TestStreamingToolDispatch
# ---------------------------------------------------------------------------


@pytest.mark.local_tool
class TestStreamingToolDispatch:
    """Test astep_stream() directly, collecting and asserting on emitted events."""

    async def test_streaming_tool_call_event_ordering(
        self, harness_agent: BaseAgent
    ) -> None:
        """ToolCallDelta precedes ToolResultEvent; stream ends with StreamComplete."""
        agent = harness_agent
        agent.add_message("user", "What is 3 + 5?")

        turn1 = [
            _make_stream_chunk(
                tool_calls=[_make_tc_delta(0, call_id="call_1", name="add", arguments_delta='{"a":')]
            ),
            _make_stream_chunk(
                tool_calls=[_make_tc_delta(0, arguments_delta=' 3.0, "b": 5.0}')]
            ),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]
        turn2 = _content_turn("The answer is 8.")

        call_count = 0

        async def mock_stream_raw(messages, *, tools=None):
            nonlocal call_count
            chunks = turn1 if call_count == 0 else turn2
            call_count += 1
            for chunk in chunks:
                yield chunk

        agent.llm.call_model_stream_raw = mock_stream_raw

        events = [event async for event in agent.astep_stream()]

        assert_tool_call_result_ordering(events)
        assert_stream_completes(events)

        tc_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
        tc_results = [e for e in events if isinstance(e, ToolResultEvent)]
        content_deltas = [e for e in events if isinstance(e, ContentDelta)]

        assert len(tc_deltas) >= 1, f"No ToolCallDelta events (events: {events})"
        assert len(tc_results) == 1, (
            f"Expected 1 ToolResultEvent, got {len(tc_results)}"
        )
        assert tc_results[0].name == "add", (
            f"Expected tool name 'add', got {tc_results[0].name!r}"
        )
        assert "8.0" in tc_results[0].content, (
            f"Expected '8.0' in result content: {tc_results[0].content!r}"
        )
        assert len(content_deltas) >= 1, "Expected at least one ContentDelta after tool"

    async def test_streaming_content_after_tool_result(
        self, harness_agent: BaseAgent
    ) -> None:
        """ContentDelta events appear after ToolResultEvent in the event stream."""
        agent = harness_agent
        agent.add_message("user", "Multiply 6 by 7.")

        turn1 = _tool_call_turn("call_m", "multiply", '{"a": 6, "b": 7}')
        turn2 = _content_turn("6 times 7 is 42.")

        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_stream_completes(events)

        # Find the index of the ToolResultEvent and first ContentDelta after it.
        tr_idx = next(
            (i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)),
            None,
        )
        assert tr_idx is not None, "No ToolResultEvent in stream"

        post_tr_content = [
            e for e in events[tr_idx + 1:] if isinstance(e, ContentDelta)
        ]
        assert post_tr_content, (
            f"No ContentDelta after ToolResultEvent "
            f"(events: {[type(e).__name__ for e in events]})"
        )
        full_content = "".join(e.content for e in post_tr_content)
        assert "42" in full_content, (
            f"Expected multiply result in content: {full_content!r}"
        )

    async def test_streaming_multi_tool_events(
        self, harness_agent: BaseAgent
    ) -> None:
        """Two concurrent tool calls produce two ToolCallDelta/ToolResultEvent pairs."""
        agent = harness_agent
        agent.add_message("user", "Add 1+2 and multiply 3*4.")

        turn1 = _multi_tool_turn([
            ("call_a", "add", '{"a": 1, "b": 2}'),
            ("call_m", "multiply", '{"a": 3, "b": 4}'),
        ])
        turn2 = _content_turn("Done: 3 and 12.")

        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        assert_tool_call_result_ordering(events)
        assert_stream_completes(events)

        tc_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tc_results) == 2, (
            f"Expected 2 ToolResultEvents, got {len(tc_results)}: "
            f"{[(e.name, e.content) for e in tc_results]}"
        )

        result_names = {e.name for e in tc_results}
        assert "add" in result_names, f"'add' result missing: {result_names}"
        assert "multiply" in result_names, f"'multiply' result missing: {result_names}"

        # Verify call_ids appear in ToolCallDelta first deltas.
        first_tc_deltas = {
            e.call_id for e in events
            if isinstance(e, ToolCallDelta) and e.call_id is not None
        }
        assert "call_a" in first_tc_deltas, (
            f"call_a not in ToolCallDelta call_ids: {first_tc_deltas}"
        )
        assert "call_m" in first_tc_deltas, (
            f"call_m not in ToolCallDelta call_ids: {first_tc_deltas}"
        )

    async def test_streaming_metrics_populated(
        self, harness_agent: BaseAgent
    ) -> None:
        """StreamComplete.metrics has model_calls >= 1 and matching tool_calls count."""
        agent = harness_agent
        agent.add_message("user", "Add 10 and 20.")

        turn1 = _tool_call_turn("call_metrics", "add", '{"a": 10, "b": 20}')
        turn2 = _content_turn("That's 30.")

        agent.llm.call_model_stream_raw = _make_mock_stream([turn1, turn2])

        events = [event async for event in agent.astep_stream()]

        complete = events[-1]
        assert isinstance(complete, StreamComplete), (
            f"Last event is not StreamComplete: {type(complete).__name__}"
        )

        metrics = complete.metrics
        assert metrics.model_calls >= 1, (
            f"Expected model_calls >= 1, got {metrics.model_calls}"
        )
        assert metrics.tool_calls == 1, (
            f"Expected tool_calls == 1, got {metrics.tool_calls}"
        )
