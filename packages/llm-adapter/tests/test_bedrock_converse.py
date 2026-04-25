"""Tests for the Bedrock Converse API provider.

Covers message translation, tool schema translation, response mapping,
streaming event processing, and provider wiring -- all with mocked boto3.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from llm_adapter.models import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
)
from llm_adapter.providers.bedrock_converse import (
    _CONVERSE_TO_OPENAI_STOP,
    _build_converse_kwargs,
    _build_converse_messages,
    _stream_response,
    _translate_response,
    _translate_tools,
    BedrockConverseProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_request(**overrides):
    defaults = {
        "model": "mistral.ministral-3-3b-instruct",
        "messages": [ChatMessage(role="user", content="Hello")],
        "max_tokens": 100,
    }
    defaults.update(overrides)
    return ChatCompletionRequest(**defaults)


def _converse_response(
    content_blocks=None,
    stop_reason="end_turn",
    input_tokens=10,
    output_tokens=5,
):
    """Build a dict mimicking a Converse API response."""
    if content_blocks is None:
        content_blocks = [{"text": "Hello!"}]
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content_blocks,
            }
        },
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
        },
    }


def _parse_sse(sse_string):
    """Parse a 'data: {...}\\n\\n' SSE frame into a dict."""
    payload = sse_string.removeprefix("data: ").strip()
    return json.loads(payload)


async def _collect_stream(events):
    """Feed mock events through _stream_response, return all yielded strings."""
    chunks = []
    async for chunk in _stream_response(iter(events), "test-model"):
        chunks.append(chunk)
    return chunks


# ===================================================================
# _build_converse_messages
# ===================================================================


class TestBuildConverseMessages:
    def test_system_message_extraction(self):
        msgs = [
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Hi"),
        ]
        system, converse_msgs = _build_converse_messages(msgs)
        assert system == [{"text": "Be concise."}]
        assert len(converse_msgs) == 1
        assert converse_msgs[0]["role"] == "user"

    def test_multiple_system_messages(self):
        msgs = [
            ChatMessage(role="system", content="Rule one."),
            ChatMessage(role="system", content="Rule two."),
            ChatMessage(role="user", content="Go"),
        ]
        system, _ = _build_converse_messages(msgs)
        assert len(system) == 2
        assert system[0] == {"text": "Rule one."}
        assert system[1] == {"text": "Rule two."}

    def test_user_content_wrapped(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        _, converse_msgs = _build_converse_messages(msgs)
        assert converse_msgs[0]["content"] == [{"text": "Hello"}]

    def test_assistant_text_content(self):
        msgs = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello back"),
        ]
        _, converse_msgs = _build_converse_messages(msgs)
        assert converse_msgs[1]["role"] == "assistant"
        assert converse_msgs[1]["content"] == [{"text": "Hello back"}]

    def test_assistant_tool_calls_to_tool_use(self):
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
        _, converse_msgs = _build_converse_messages(msgs)
        assistant = converse_msgs[1]
        assert assistant["role"] == "assistant"
        block = assistant["content"][0]
        assert "toolUse" in block
        assert block["toolUse"]["toolUseId"] == "tc_1"
        assert block["toolUse"]["name"] == "search"
        assert block["toolUse"]["input"] == {"q": "test"}

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
        _, converse_msgs = _build_converse_messages(msgs)
        last = converse_msgs[-1]
        assert last["role"] == "user"
        tr = last["content"][0]["toolResult"]
        assert tr["toolUseId"] == "tc_1"
        assert tr["content"] == [{"text": "result"}]

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
        _, converse_msgs = _build_converse_messages(msgs)
        last = converse_msgs[-1]
        assert last["role"] == "user"
        assert len(last["content"]) == 2
        assert last["content"][0]["toolResult"]["toolUseId"] == "c1"
        assert last["content"][1]["toolResult"]["toolUseId"] == "c2"

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
        _, converse_msgs = _build_converse_messages(msgs)
        assert converse_msgs[-1]["role"] == "user"
        assert "toolResult" in converse_msgs[-1]["content"][0]

    def test_consecutive_same_role_merged(self):
        msgs = [
            ChatMessage(role="user", content="part 1"),
            ChatMessage(role="user", content="part 2"),
        ]
        _, converse_msgs = _build_converse_messages(msgs)
        assert len(converse_msgs) == 1
        assert converse_msgs[0]["role"] == "user"
        assert len(converse_msgs[0]["content"]) == 2

    def test_developer_role_maps_to_user(self):
        msgs = [ChatMessage(role="developer", content="instructions")]
        _, converse_msgs = _build_converse_messages(msgs)
        assert converse_msgs[0]["role"] == "user"

    def test_malformed_tool_arguments_default_to_empty_dict(self):
        msgs = [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="f", arguments="{broken"),
                    )
                ],
            ),
        ]
        _, converse_msgs = _build_converse_messages(msgs)
        block = converse_msgs[0]["content"][0]
        assert block["toolUse"]["input"] == {}

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
        _, converse_msgs = _build_converse_messages(msgs)
        blocks = converse_msgs[1]["content"]
        assert all("toolUse" in b for b in blocks)


# ===================================================================
# _translate_tools
# ===================================================================


class TestTranslateTools:
    def test_tool_to_tool_spec(self):
        tools = [
            Tool(
                function=ToolFunction(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                )
            )
        ]
        result = _translate_tools(tools)
        assert result is not None
        assert len(result) == 1
        spec = result[0]["toolSpec"]
        assert spec["name"] == "search"
        assert spec["description"] == "Search the web"
        assert spec["inputSchema"]["json"]["type"] == "object"
        assert "q" in spec["inputSchema"]["json"]["properties"]

    def test_none_returns_none(self):
        assert _translate_tools(None) is None

    def test_empty_list_returns_none(self):
        assert _translate_tools([]) is None

    def test_no_description_defaults_to_empty_string(self):
        tools = [
            Tool(function=ToolFunction(name="noop", parameters={"type": "object"}))
        ]
        result = _translate_tools(tools)
        assert result[0]["toolSpec"]["description"] == ""

    def test_no_parameters_gets_empty_schema(self):
        tools = [Tool(function=ToolFunction(name="noop", description="Do nothing"))]
        result = _translate_tools(tools)
        assert result[0]["toolSpec"]["inputSchema"]["json"] == {
            "type": "object",
            "properties": {},
        }


# ===================================================================
# _build_converse_kwargs
# ===================================================================


class TestBuildConverseKwargs:
    def test_model_id_set(self):
        req = _simple_request()
        kwargs = _build_converse_kwargs(req)
        assert kwargs["modelId"] == "mistral.ministral-3-3b-instruct"

    def test_system_included_when_present(self):
        req = _simple_request(
            messages=[
                ChatMessage(role="system", content="Be helpful."),
                ChatMessage(role="user", content="Hi"),
            ]
        )
        kwargs = _build_converse_kwargs(req)
        assert "system" in kwargs
        assert kwargs["system"] == [{"text": "Be helpful."}]

    def test_system_omitted_when_absent(self):
        req = _simple_request()
        kwargs = _build_converse_kwargs(req)
        assert "system" not in kwargs

    def test_inference_config_populated(self):
        req = _simple_request(temperature=0.7, top_p=0.9, max_tokens=256)
        kwargs = _build_converse_kwargs(req)
        ic = kwargs["inferenceConfig"]
        assert ic["maxTokens"] == 256
        assert ic["temperature"] == 0.7
        assert ic["topP"] == 0.9

    def test_inference_config_omitted_when_all_none(self):
        req = _simple_request(max_tokens=None, temperature=None, top_p=None)
        kwargs = _build_converse_kwargs(req)
        assert "inferenceConfig" not in kwargs

    def test_tools_included_as_tool_config(self):
        req = _simple_request(
            tools=[
                Tool(
                    function=ToolFunction(
                        name="search",
                        description="Search",
                        parameters={"type": "object", "properties": {}},
                    )
                )
            ]
        )
        kwargs = _build_converse_kwargs(req)
        assert "toolConfig" in kwargs
        assert len(kwargs["toolConfig"]["tools"]) == 1

    def test_no_tools_omits_tool_config(self):
        req = _simple_request()
        kwargs = _build_converse_kwargs(req)
        assert "toolConfig" not in kwargs


# ===================================================================
# _translate_response (non-streaming)
# ===================================================================


class TestTranslateResponse:
    def test_text_response(self):
        resp = _converse_response([{"text": "Hello world"}])
        result = _translate_response(resp, "test-model")
        assert result.choices[0].message.content == "Hello world"
        assert result.choices[0].message.tool_calls is None

    def test_tool_use_response(self):
        resp = _converse_response(
            [
                {
                    "toolUse": {
                        "toolUseId": "tu_1",
                        "name": "search",
                        "input": {"query": "weather"},
                    }
                }
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
        resp = _converse_response(
            [
                {"text": "Searching..."},
                {
                    "toolUse": {
                        "toolUseId": "tu_1",
                        "name": "search",
                        "input": {"q": "test"},
                    }
                },
            ],
            stop_reason="tool_use",
        )
        result = _translate_response(resp, "test-model")
        assert result.choices[0].message.content == "Searching..."
        assert len(result.choices[0].message.tool_calls) == 1

    @pytest.mark.parametrize(
        "converse_reason,openai_reason",
        [
            ("end_turn", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
        ],
    )
    def test_stop_reason_mapping(self, converse_reason, openai_reason):
        resp = _converse_response(
            [{"text": "ok"}], stop_reason=converse_reason
        )
        result = _translate_response(resp, "test-model")
        assert result.choices[0].finish_reason == openai_reason

    def test_usage_translation(self):
        resp = _converse_response(
            [{"text": "hi"}], input_tokens=15, output_tokens=25
        )
        result = _translate_response(resp, "test-model")
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 25
        assert result.usage.total_tokens == 40

    def test_response_model_forwarded(self):
        resp = _converse_response()
        result = _translate_response(resp, "meta.llama3-70b-instruct-v1:0")
        assert result.model == "meta.llama3-70b-instruct-v1:0"

    def test_response_has_chatcmpl_id(self):
        resp = _converse_response()
        result = _translate_response(resp, "m")
        assert result.id.startswith("chatcmpl-")


# ===================================================================
# _stream_response (streaming)
# ===================================================================


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_role_chunk_first(self):
        events = [{"messageStart": {"role": "assistant"}}]
        chunks = await _collect_stream(events)
        first = _parse_sse(chunks[0])
        assert first["choices"][0]["delta"] == {"role": "assistant"}

    @pytest.mark.asyncio
    async def test_text_deltas(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Hello"},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": " world"},
                }
            },
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
    async def test_tool_call_deltas(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "search",
                        }
                    },
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"query":'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '"test"}'}},
                }
            },
        ]
        chunks = await _collect_stream(events)
        tool_chunks = [
            _parse_sse(c)
            for c in chunks
            if not c.startswith("data: [DONE]")
            and "tool_calls"
            in _parse_sse(c).get("choices", [{}])[0].get("delta", {})
        ]
        # First: opening chunk with id + name
        assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "tu_1"
        assert (
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "search"
        )
        # Subsequent: argument deltas
        assert (
            tool_chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == '{"query":'
        )
        assert (
            tool_chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == '"test"}'
        )

    @pytest.mark.asyncio
    async def test_finish_reason(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        chunks = await _collect_stream(events)
        finish_chunks = [
            _parse_sse(c)
            for c in chunks
            if not c.startswith("data: [DONE]")
            and _parse_sse(c).get("choices", [{}])[0].get("finish_reason") is not None
        ]
        assert len(finish_chunks) == 1
        assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_ends_with_done(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        chunks = await _collect_stream(events)
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_usage_chunk(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 15, "outputTokens": 25},
                }
            },
        ]
        chunks = await _collect_stream(events)
        usage_chunks = [
            _parse_sse(c)
            for c in chunks
            if not c.startswith("data: [DONE]") and _parse_sse(c).get("usage")
        ]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["prompt_tokens"] == 15
        assert usage_chunks[0]["usage"]["completion_tokens"] == 25
        assert usage_chunks[0]["usage"]["total_tokens"] == 40

    @pytest.mark.asyncio
    async def test_content_block_stop_is_noop(self):
        """contentBlockStop events should not produce any SSE output."""
        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Hi"},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ]
        chunks = await _collect_stream(events)
        # role + text + [DONE] = 3
        assert len(chunks) == 3


# ===================================================================
# Provider setup
# ===================================================================


class TestSetup:
    @pytest.mark.asyncio
    async def test_creates_boto3_client_with_region(self):
        provider = BedrockConverseProvider()
        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
            with patch("boto3.client") as mock_client:
                mock_client.return_value = MagicMock()
                await provider.setup()
                mock_client.assert_called_once_with(
                    "bedrock-runtime", region_name="eu-west-1"
                )

    @pytest.mark.asyncio
    async def test_default_region(self):
        provider = BedrockConverseProvider()
        env = {k: v for k, v in os.environ.items() if k != "AWS_REGION"}
        with patch.dict(os.environ, env, clear=True):
            with patch("boto3.client") as mock_client:
                mock_client.return_value = MagicMock()
                await provider.setup()
                mock_client.assert_called_once_with(
                    "bedrock-runtime", region_name="us-east-1"
                )


# ===================================================================
# Provider chat completion (non-streaming)
# ===================================================================


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self):
        provider = BedrockConverseProvider()
        provider._client = MagicMock()
        provider._client.converse.return_value = _converse_response()

        req = _simple_request()
        resp = await provider.chat_completion(req)

        assert resp.choices[0].message.content == "Hello!"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 10
        provider._client.converse.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_name_passed_through(self):
        provider = BedrockConverseProvider()
        provider._client = MagicMock()
        provider._client.converse.return_value = _converse_response()

        model = "meta.llama3-70b-instruct-v1:0"
        req = _simple_request(model=model)
        resp = await provider.chat_completion(req)

        assert resp.model == model
        call_kwargs = provider._client.converse.call_args
        assert call_kwargs.kwargs.get("modelId") or call_kwargs[1].get("modelId") == model


# ===================================================================
# Provider streaming
# ===================================================================


class TestProviderStreaming:
    @pytest.mark.asyncio
    async def test_streaming_produces_sse_chunks(self):
        provider = BedrockConverseProvider()
        provider._client = MagicMock()

        stream_events = iter(
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": "Hi"},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
                {
                    "metadata": {
                        "usage": {"inputTokens": 5, "outputTokens": 1},
                    }
                },
            ]
        )
        provider._client.converse_stream.return_value = {"stream": stream_events}

        req = _simple_request()
        chunks = []
        async for chunk in provider.chat_completion_stream(req):
            chunks.append(chunk)

        assert any('"role": "assistant"' in c for c in chunks)
        assert any('"content": "Hi"' in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"


# ===================================================================
# Provider shutdown
# ===================================================================


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_noop(self):
        provider = BedrockConverseProvider()
        provider._client = MagicMock()
        await provider.shutdown()  # should not raise


# ===================================================================
# Registration
# ===================================================================


class TestRegistration:
    def test_registered_as_bedrock_converse(self):
        from llm_adapter.providers import _REGISTRY

        assert "bedrock-converse" in _REGISTRY
        assert _REGISTRY["bedrock-converse"] is BedrockConverseProvider


# ===================================================================
# Stop-reason constant sanity check
# ===================================================================


class TestStopReasonMapping:
    def test_mapping_completeness(self):
        expected_keys = {"end_turn", "tool_use", "max_tokens", "stop_sequence"}
        assert set(_CONVERSE_TO_OPENAI_STOP.keys()) == expected_keys
