"""Tests for the Anthropic provider translation layer.

Covers message reconstruction, tool schema translation, request/response
translation, and streaming -- all without touching the Anthropic SDK.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_adapter.models import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
)
from llm_adapter.providers.anthropic import (
    _ANTHROPIC_TO_OPENAI_STOP,
    _build_anthropic_messages,
    _normalize_content,
    _stream_response,
    _translate_request,
    _translate_response,
    _translate_tool_choice,
    _translate_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(type, **kwargs):
    """Build a SimpleNamespace pretending to be an Anthropic streaming event."""
    return SimpleNamespace(type=type, **kwargs)


def _mock_anthropic_response(
    content_blocks,
    stop_reason="end_turn",
    input_tokens=10,
    output_tokens=20,
):
    """Build a SimpleNamespace pretending to be an Anthropic Message."""
    return SimpleNamespace(
        content=content_blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


async def _collect_stream(events):
    """Feed mock events through _stream_response, return all yielded SSE strings."""

    async def _mock_stream():
        for e in events:
            yield e

    chunks = []
    async for chunk in _stream_response(_mock_stream(), "test-model"):
        chunks.append(chunk)
    return chunks


def _parse_sse(sse_string):
    """Parse a 'data: {...}\\n\\n' SSE frame into a dict."""
    payload = sse_string.removeprefix("data: ").strip()
    return json.loads(payload)


# ===================================================================
# _normalize_content
# ===================================================================


class TestNormalizeContent:
    def test_string_content(self):
        result = _normalize_content("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_none_content(self):
        result = _normalize_content(None)
        assert result == [{"type": "text", "text": ""}]

    def test_list_content_passthrough(self):
        blocks = [{"type": "image", "source": {"data": "..."}}]
        result = _normalize_content(blocks)
        assert result is blocks


# ===================================================================
# _build_anthropic_messages
# ===================================================================


class TestBuildAnthropicMessages:
    def test_system_message_extraction(self):
        msgs = [
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Hi"),
        ]
        system, anthropic_msgs = _build_anthropic_messages(msgs)
        assert system == "Be concise."
        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self):
        msgs = [
            ChatMessage(role="system", content="Rule one."),
            ChatMessage(role="system", content="Rule two."),
            ChatMessage(role="user", content="Go"),
        ]
        system, _ = _build_anthropic_messages(msgs)
        assert system == "Rule one.\n\nRule two."

    def test_user_message_content_normalized(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        assert anthropic_msgs[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_assistant_tool_calls_to_tool_use_blocks(self):
        msgs = [
            ChatMessage(role="user", content="Search"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        function=ToolCallFunction(
                            name="search",
                            arguments='{"q":"test"}',
                        ),
                    )
                ],
            ),
        ]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        assistant_msg = anthropic_msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 1
        block = assistant_msg["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "tc_1"
        assert block["name"] == "search"
        assert block["input"] == {"q": "test"}

    def test_tool_results_buffered_into_user_message(self):
        msgs = [
            ChatMessage(role="user", content="go"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        function=ToolCallFunction(name="f", arguments="{}"),
                    )
                ],
            ),
            ChatMessage(role="tool", tool_call_id="tc_1", content="result"),
        ]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        # Tool results flush as the last message (a user turn).
        last = anthropic_msgs[-1]
        assert last["role"] == "user"
        assert last["content"][0]["type"] == "tool_result"
        assert last["content"][0]["tool_use_id"] == "tc_1"
        assert last["content"][0]["content"] == "result"

    def test_parallel_tool_results_single_user_message(self):
        msgs = [
            ChatMessage(role="user", content="go"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=ToolCallFunction(name="f", arguments="{}"),
                    ),
                    ToolCall(
                        id="c2",
                        function=ToolCallFunction(name="g", arguments="{}"),
                    ),
                ],
            ),
            ChatMessage(role="tool", tool_call_id="c1", content="r1"),
            ChatMessage(role="tool", tool_call_id="c2", content="r2"),
        ]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        last = anthropic_msgs[-1]
        assert last["role"] == "user"
        assert len(last["content"]) == 2
        assert last["content"][0]["tool_use_id"] == "c1"
        assert last["content"][1]["tool_use_id"] == "c2"

    def test_trailing_tool_results_flushed(self):
        msgs = [
            ChatMessage(role="user", content="go"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=ToolCallFunction(name="f", arguments="{}"),
                    )
                ],
            ),
            ChatMessage(role="tool", tool_call_id="c1", content="done"),
        ]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        assert anthropic_msgs[-1]["role"] == "user"
        assert anthropic_msgs[-1]["content"][0]["type"] == "tool_result"

    def test_consecutive_same_role_merged(self):
        msgs = [
            ChatMessage(role="user", content="part 1"),
            ChatMessage(role="user", content="part 2"),
        ]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"
        assert len(anthropic_msgs[0]["content"]) == 2

    def test_developer_role_maps_to_user(self):
        msgs = [ChatMessage(role="developer", content="instructions")]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        assert anthropic_msgs[0]["role"] == "user"

    def test_empty_assistant_content_with_tool_calls(self):
        msgs = [
            ChatMessage(role="user", content="go"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        function=ToolCallFunction(name="f", arguments="{}"),
                    )
                ],
            ),
        ]
        _, anthropic_msgs = _build_anthropic_messages(msgs)
        assistant_blocks = anthropic_msgs[1]["content"]
        # Only the tool_use block, no empty text block.
        assert all(b["type"] == "tool_use" for b in assistant_blocks)

    def test_complex_multi_turn(self, conversation_with_tool_results):
        system, anthropic_msgs = _build_anthropic_messages(
            conversation_with_tool_results.messages
        )
        # System extracted.
        assert system == "You are helpful."
        # After merge pass, roles must alternate user/assistant/user.
        roles = [m["role"] for m in anthropic_msgs]
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1], (
                f"Adjacent same-role at index {i}: {roles}"
            )
        # Tool results appear in a user message.
        tool_result_msgs = [
            m
            for m in anthropic_msgs
            if m["role"] == "user"
            and any(
                b.get("type") == "tool_result"
                for b in m["content"]
                if isinstance(b, dict)
            )
        ]
        assert len(tool_result_msgs) >= 1


# ===================================================================
# _translate_tools
# ===================================================================


class TestTranslateTools:
    def test_tool_parameters_to_input_schema(self):
        tools = [
            Tool(
                function=ToolFunction(
                    name="search",
                    description="Search",
                    parameters={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                )
            )
        ]
        result = _translate_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search"
        assert result[0]["input_schema"]["type"] == "object"

    def test_tool_none_returns_none(self):
        assert _translate_tools(None) is None

    def test_tool_empty_list_returns_none(self):
        assert _translate_tools([]) is None

    def test_tool_no_description_defaults_to_empty_string(self):
        tools = [
            Tool(function=ToolFunction(name="noop", parameters={"type": "object"}))
        ]
        result = _translate_tools(tools)
        assert result[0]["description"] == ""

    def test_tool_no_parameters_gets_empty_schema(self):
        tools = [Tool(function=ToolFunction(name="noop", description="Do nothing"))]
        result = _translate_tools(tools)
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}


# ===================================================================
# _translate_request
# ===================================================================


class TestTranslateRequest:
    def test_max_tokens_default(self):
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[ChatMessage(role="user", content="hi")],
        )
        kwargs = _translate_request(req)
        assert kwargs["max_tokens"] == 4096

    def test_max_tokens_forwarded(self):
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=256,
        )
        kwargs = _translate_request(req)
        assert kwargs["max_tokens"] == 256

    def test_system_and_temperature(self, simple_request):
        simple_request.temperature = 0.5
        kwargs = _translate_request(simple_request)
        assert kwargs["system"] == "You are helpful."
        assert kwargs["temperature"] == 0.5

    def test_temperature_none_omitted(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
        )
        kwargs = _translate_request(req)
        assert "temperature" not in kwargs

    def test_tools_forwarded(self, tool_request):
        kwargs = _translate_request(tool_request)
        assert "tools" in kwargs
        assert kwargs["tools"][0]["name"] == "search"

    def test_no_tools_omitted(self, simple_request):
        kwargs = _translate_request(simple_request)
        assert "tools" not in kwargs


# ===================================================================
# _translate_response (non-streaming)
# ===================================================================


class TestTranslateResponse:
    def test_text_response(self):
        resp = _mock_anthropic_response(
            [SimpleNamespace(type="text", text="Hello world")]
        )
        result = _translate_response(resp, "test-model")
        assert result.choices[0].message.content == "Hello world"
        assert result.choices[0].message.tool_calls is None

    def test_tool_use_response(self):
        resp = _mock_anthropic_response(
            [
                SimpleNamespace(
                    type="tool_use",
                    id="tu_1",
                    name="search",
                    input={"query": "weather"},
                )
            ],
            stop_reason="tool_use",
        )
        result = _translate_response(resp, "test-model")
        assert result.choices[0].message.content is None
        tc = result.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0].id == "tu_1"
        assert tc[0].function.name == "search"
        assert json.loads(tc[0].function.arguments) == {"query": "weather"}

    def test_mixed_text_and_tool_use(self):
        resp = _mock_anthropic_response(
            [
                SimpleNamespace(type="text", text="Searching..."),
                SimpleNamespace(
                    type="tool_use",
                    id="tu_1",
                    name="search",
                    input={"q": "test"},
                ),
            ],
            stop_reason="tool_use",
        )
        result = _translate_response(resp, "test-model")
        assert result.choices[0].message.content == "Searching..."
        assert len(result.choices[0].message.tool_calls) == 1

    def test_thinking_block(self):
        resp = _mock_anthropic_response(
            [
                SimpleNamespace(type="thinking", thinking="Let me reason..."),
                SimpleNamespace(type="text", text="The answer is 42."),
            ]
        )
        result = _translate_response(resp, "test-model")
        assert result.choices[0].message.reasoning_content == "Let me reason..."
        assert result.choices[0].message.content == "The answer is 42."

    @pytest.mark.parametrize(
        "anthropic_reason,openai_reason",
        [
            ("end_turn", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
        ],
    )
    def test_stop_reason_mapping(self, anthropic_reason, openai_reason):
        resp = _mock_anthropic_response(
            [SimpleNamespace(type="text", text="ok")],
            stop_reason=anthropic_reason,
        )
        result = _translate_response(resp, "test-model")
        assert result.choices[0].finish_reason == openai_reason

    def test_usage_translation(self):
        resp = _mock_anthropic_response(
            [SimpleNamespace(type="text", text="hi")],
            input_tokens=15,
            output_tokens=25,
        )
        result = _translate_response(resp, "test-model")
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 25
        assert result.usage.total_tokens == 40

    def test_response_model_forwarded(self):
        resp = _mock_anthropic_response(
            [SimpleNamespace(type="text", text="ok")]
        )
        result = _translate_response(resp, "claude-sonnet-4-6")
        assert result.model == "claude-sonnet-4-6"

    def test_response_has_chatcmpl_id(self):
        resp = _mock_anthropic_response(
            [SimpleNamespace(type="text", text="ok")]
        )
        result = _translate_response(resp, "m")
        assert result.id.startswith("chatcmpl-")


# ===================================================================
# _stream_response (streaming)
# ===================================================================


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_stream_role_chunk_first(self):
        events = [
            _event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5)
                ),
            ),
        ]
        chunks = await _collect_stream(events)
        first = _parse_sse(chunks[0])
        assert first["choices"][0]["delta"] == {"role": "assistant"}

    @pytest.mark.asyncio
    async def test_stream_text_deltas(self):
        events = [
            _event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5)
                ),
            ),
            _event(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text"),
            ),
            _event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hello"),
            ),
            _event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text=" world"),
            ),
        ]
        chunks = await _collect_stream(events)
        text_chunks = [
            _parse_sse(c)
            for c in chunks
            if not c.startswith("data: [DONE]")
            and "content" in _parse_sse(c).get("choices", [{}])[0].get("delta", {})
        ]
        assert len(text_chunks) == 2
        assert text_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert text_chunks[1]["choices"][0]["delta"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_stream_thinking_deltas(self):
        events = [
            _event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5)
                ),
            ),
            _event(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(type="thinking"),
            ),
            _event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="hmm"),
            ),
        ]
        chunks = await _collect_stream(events)
        reasoning_chunks = [
            _parse_sse(c)
            for c in chunks
            if not c.startswith("data: [DONE]")
            and "reasoning_content"
            in _parse_sse(c).get("choices", [{}])[0].get("delta", {})
        ]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"] == "hmm"

    @pytest.mark.asyncio
    async def test_stream_tool_call_deltas(self):
        events = [
            _event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5)
                ),
            ),
            _event(
                "content_block_start",
                index=0,
                content_block=SimpleNamespace(
                    type="tool_use", id="toolu_1", name="search"
                ),
            ),
            _event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="input_json_delta", partial_json='{"query":'
                ),
            ),
            _event(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="input_json_delta", partial_json='"test"}'
                ),
            ),
        ]
        chunks = await _collect_stream(events)
        tool_chunks = [
            _parse_sse(c)
            for c in chunks
            if not c.startswith("data: [DONE]")
            and "tool_calls"
            in _parse_sse(c).get("choices", [{}])[0].get("delta", {})
        ]
        # First chunk: tool_use start (id + name + empty arguments).
        assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_1"
        assert (
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "search"
        )
        # Subsequent chunks: argument deltas.
        assert (
            tool_chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ]
            == '{"query":'
        )
        assert (
            tool_chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ]
            == '"test"}'
        )

    @pytest.mark.asyncio
    async def test_stream_finish_reason(self):
        events = [
            _event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5)
                ),
            ),
            _event(
                "message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=10),
            ),
        ]
        chunks = await _collect_stream(events)
        finish_chunks = []
        for c in chunks:
            if c.startswith("data: [DONE]"):
                continue
            parsed = _parse_sse(c)
            choices = parsed.get("choices", [])
            if choices and choices[0].get("finish_reason") is not None:
                finish_chunks.append(parsed)
        assert len(finish_chunks) == 1
        assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self):
        events = [
            _event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5)
                ),
            ),
            _event(
                "message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=10),
            ),
            _event("message_stop"),
        ]
        chunks = await _collect_stream(events)
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_usage_chunk(self):
        events = [
            _event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=15)
                ),
            ),
            _event(
                "message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=25),
            ),
            _event("message_stop"),
        ]
        chunks = await _collect_stream(events)
        # The usage chunk has empty choices and a usage dict.
        usage_chunks = [
            _parse_sse(c)
            for c in chunks
            if not c.startswith("data: [DONE]") and _parse_sse(c).get("usage")
        ]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["prompt_tokens"] == 15
        assert usage_chunks[0]["usage"]["completion_tokens"] == 25
        assert usage_chunks[0]["usage"]["total_tokens"] == 40


# ===================================================================
# Extra body / vLLM params
# ===================================================================


class TestExtraParams:
    def test_extra_params_ignored(self):
        """ChatCompletionRequest with extra='ignore' silently drops unknown fields."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[ChatMessage(role="user", content="hi")],
            top_k=50,
            repetition_penalty=1.1,
        )
        assert req.model == "claude-sonnet-4-6"
        assert not hasattr(req, "top_k")
        assert not hasattr(req, "repetition_penalty")


# ===================================================================
# Stop-reason constant sanity check
# ===================================================================


class TestStopReasonMapping:
    def test_mapping_completeness(self):
        expected_keys = {"end_turn", "tool_use", "max_tokens", "stop_sequence"}
        assert set(_ANTHROPIC_TO_OPENAI_STOP.keys()) == expected_keys


# ===================================================================
# Malformed tool call arguments (review fix)
# ===================================================================


class TestMalformedToolArguments:
    def test_empty_string_arguments_default_to_empty_dict(self):
        msgs = [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="search", arguments=""),
                    )
                ],
            ),
        ]
        _, result = _build_anthropic_messages(msgs)
        tool_block = result[0]["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["input"] == {}

    def test_malformed_json_arguments_default_to_empty_dict(self):
        msgs = [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="search", arguments="{broken"),
                    )
                ],
            ),
        ]
        _, result = _build_anthropic_messages(msgs)
        tool_block = result[0]["content"][0]
        assert tool_block["input"] == {}


# ===================================================================
# tool_choice translation (review fix)
# ===================================================================


class TestToolChoiceTranslation:
    def test_none_returns_none(self):
        assert _translate_tool_choice(None) is None

    def test_auto(self):
        assert _translate_tool_choice("auto") == {"type": "auto"}

    def test_required(self):
        assert _translate_tool_choice("required") == {"type": "any"}

    def test_none_string(self):
        assert _translate_tool_choice("none") is None

    def test_specific_function(self):
        choice = {"type": "function", "function": {"name": "search"}}
        assert _translate_tool_choice(choice) == {"type": "tool", "name": "search"}

    def test_unknown_string(self):
        assert _translate_tool_choice("unknown") is None


# ===================================================================
# top_p forwarding (review fix)
# ===================================================================


class TestTopPForwarding:
    def test_top_p_forwarded_when_set(self):
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[ChatMessage(role="user", content="hi")],
            top_p=0.9,
        )
        kwargs = _translate_request(req)
        assert kwargs["top_p"] == 0.9

    def test_top_p_absent_when_none(self):
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[ChatMessage(role="user", content="hi")],
        )
        kwargs = _translate_request(req)
        assert "top_p" not in kwargs
