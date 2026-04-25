"""Tests for the Vertex AI / Gemini provider translation layer."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from llm_adapter.models import (
    ChatCompletionRequest, ChatMessage, Tool, ToolCall, ToolCallFunction,
    ToolFunction,
)
from llm_adapter.providers.vertex import (
    _GEMINI_TO_OPENAI_STOP, _build_gemini_messages, _stream_response,
    _translate_request, _translate_response, _translate_tool_choice,
    _translate_tools, VertexProvider,
)

# -- helpers -----------------------------------------------------------------

def _part(text=None, function_call=None):
    ns = SimpleNamespace()
    if text is not None:
        ns.text = text
    if function_call is not None:
        ns.function_call = function_call
    return ns

def _fc(name, args=None):
    return SimpleNamespace(name=name, args=args or {})

def _resp(parts, finish_reason="STOP", prompt_tokens=10, completion_tokens=20):
    return SimpleNamespace(
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=parts), finish_reason=finish_reason,
        )],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens, candidates_token_count=completion_tokens,
        ),
    )

def _chunk(parts, finish_reason=None, usage_metadata=None):
    cand = SimpleNamespace(
        content=SimpleNamespace(parts=parts) if parts is not None else None,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(candidates=[cand], usage_metadata=usage_metadata)

def _usage(p=5, c=10):
    return SimpleNamespace(prompt_token_count=p, candidates_token_count=c)

async def _collect(chunks):
    async def _gen():
        for c in chunks:
            yield c
    out = []
    async for sse in _stream_response(_gen(), "test-model"):
        out.append(sse)
    return out

def _sse(s):
    return json.loads(s.removeprefix("data: ").strip())

def _tc(id, name, args="{}"):
    return ToolCall(id=id, function=ToolCallFunction(name=name, arguments=args))

# -- _build_gemini_messages --------------------------------------------------

class TestBuildGeminiMessages:
    def test_system_message_extraction(self):
        msgs = [ChatMessage(role="system", content="Be concise."),
                ChatMessage(role="user", content="Hi")]
        sys_text, contents, _ = _build_gemini_messages(msgs)
        assert sys_text == "Be concise."
        assert len(contents) == 1 and contents[0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self):
        msgs = [ChatMessage(role="system", content="Rule one."),
                ChatMessage(role="system", content="Rule two."),
                ChatMessage(role="user", content="Go")]
        assert _build_gemini_messages(msgs)[0] == "Rule one.\n\nRule two."

    def test_user_message_mapped(self):
        sys_text, contents, _ = _build_gemini_messages(
            [ChatMessage(role="user", content="Hello")])
        assert sys_text is None
        assert contents[0] == {"role": "user", "parts": [{"text": "Hello"}]}

    def test_developer_role_maps_to_user(self):
        _, contents, _ = _build_gemini_messages(
            [ChatMessage(role="developer", content="instructions")])
        assert contents[0]["role"] == "user"

    def test_assistant_maps_to_model_role(self):
        _, contents, _ = _build_gemini_messages([
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello back")])
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Hello back"}]

    def test_assistant_tool_calls_to_function_call_parts(self):
        _, contents, id_map = _build_gemini_messages([
            ChatMessage(role="user", content="Search"),
            ChatMessage(role="assistant", content=None,
                        tool_calls=[_tc("tc_1", "search", '{"q":"test"}')])])
        fc_part = contents[1]["parts"][0]
        assert fc_part["function_call"]["name"] == "search"
        assert fc_part["function_call"]["args"] == {"q": "test"}
        assert id_map["tc_1"] == "search"

    def test_tool_results_buffered_into_user_message(self):
        _, contents, _ = _build_gemini_messages([
            ChatMessage(role="user", content="go"),
            ChatMessage(role="assistant", content=None, tool_calls=[_tc("tc_1", "f")]),
            ChatMessage(role="tool", tool_call_id="tc_1", content="result")])
        last = contents[-1]
        assert last["role"] == "user"
        fr = last["parts"][0]["function_response"]
        assert fr["name"] == "f" and fr["response"] == {"result": "result"}

    def test_parallel_tool_results_single_user_message(self):
        _, contents, _ = _build_gemini_messages([
            ChatMessage(role="user", content="go"),
            ChatMessage(role="assistant", content=None,
                        tool_calls=[_tc("c1", "f"), _tc("c2", "g")]),
            ChatMessage(role="tool", tool_call_id="c1", content="r1"),
            ChatMessage(role="tool", tool_call_id="c2", content="r2")])
        last = contents[-1]
        assert last["role"] == "user" and len(last["parts"]) == 2
        assert last["parts"][0]["function_response"]["name"] == "f"
        assert last["parts"][1]["function_response"]["name"] == "g"

    def test_trailing_tool_results_flushed(self):
        _, contents, _ = _build_gemini_messages([
            ChatMessage(role="user", content="go"),
            ChatMessage(role="assistant", content=None, tool_calls=[_tc("c1", "f")]),
            ChatMessage(role="tool", tool_call_id="c1", content="done")])
        assert contents[-1]["role"] == "user"
        assert "function_response" in contents[-1]["parts"][0]

    def test_consecutive_same_role_merged(self):
        _, contents, _ = _build_gemini_messages([
            ChatMessage(role="user", content="part 1"),
            ChatMessage(role="user", content="part 2")])
        assert len(contents) == 1 and len(contents[0]["parts"]) == 2

    def test_malformed_json_arguments_default_to_empty_dict(self):
        _, contents, _ = _build_gemini_messages([
            ChatMessage(role="assistant", content=None,
                        tool_calls=[_tc("call_1", "search", "{broken")])])
        assert contents[0]["parts"][0]["function_call"]["args"] == {}

    def test_complex_multi_turn(self, conversation_with_tool_results):
        sys_text, contents, _ = _build_gemini_messages(
            conversation_with_tool_results.messages)
        assert sys_text == "You are helpful."
        roles = [m["role"] for m in contents]
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1], f"Adjacent same-role at {i}: {roles}"
        assert any(
            any("function_response" in p for p in m["parts"] if isinstance(p, dict))
            for m in contents if m["role"] == "user")

# -- _translate_tools --------------------------------------------------------

class TestTranslateTools:
    def test_tool_to_function_declarations(self):
        tools = [Tool(function=ToolFunction(
            name="search", description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}}))]
        result = _translate_tools(tools)
        decls = result[0]["function_declarations"]
        assert len(decls) == 1 and decls[0]["name"] == "search"
        assert decls[0]["description"] == "Search the web"

    def test_none_returns_none(self):
        assert _translate_tools(None) is None

    def test_empty_list_returns_none(self):
        assert _translate_tools([]) is None

    def test_tool_no_description_omitted(self):
        result = _translate_tools(
            [Tool(function=ToolFunction(name="noop", parameters={"type": "object"}))])
        assert "description" not in result[0]["function_declarations"][0]

    def test_tool_no_parameters_omitted(self):
        result = _translate_tools(
            [Tool(function=ToolFunction(name="noop", description="Do nothing"))])
        assert "parameters" not in result[0]["function_declarations"][0]

# -- _translate_tool_choice --------------------------------------------------

class TestTranslateToolChoice:
    def test_none_returns_none(self):
        assert _translate_tool_choice(None) is None

    def test_auto(self):
        assert _translate_tool_choice("auto") == {"function_calling_config": {"mode": "AUTO"}}

    def test_required(self):
        assert _translate_tool_choice("required") == {"function_calling_config": {"mode": "ANY"}}

    def test_none_string(self):
        assert _translate_tool_choice("none") == {"function_calling_config": {"mode": "NONE"}}

    def test_specific_function(self):
        choice = {"type": "function", "function": {"name": "search"}}
        assert _translate_tool_choice(choice) == {
            "function_calling_config": {"mode": "ANY", "allowed_function_names": ["search"]}}

    def test_unknown_string(self):
        assert _translate_tool_choice("unknown") is None

# -- _translate_request ------------------------------------------------------

class TestTranslateRequest:
    def test_system_instruction_in_config(self, simple_request):
        config, _ = _translate_request(simple_request)
        assert config["system_instruction"] == "You are helpful."

    def test_temperature_forwarded(self):
        req = ChatCompletionRequest(model="gemini-2.0-flash",
            messages=[ChatMessage(role="user", content="hi")], temperature=0.7)
        assert _translate_request(req)[0]["temperature"] == 0.7

    def test_max_tokens_to_max_output_tokens(self):
        req = ChatCompletionRequest(model="gemini-2.0-flash",
            messages=[ChatMessage(role="user", content="hi")], max_tokens=256)
        config = _translate_request(req)[0]
        assert config["max_output_tokens"] == 256
        assert "max_tokens" not in config

    def test_top_p_forwarded(self):
        req = ChatCompletionRequest(model="gemini-2.0-flash",
            messages=[ChatMessage(role="user", content="hi")], top_p=0.9)
        assert _translate_request(req)[0]["top_p"] == 0.9

    def test_tools_in_config(self, tool_request):
        config = _translate_request(tool_request)[0]
        assert config["tools"][0]["function_declarations"][0]["name"] == "search"

    def test_empty_config_when_no_options(self):
        req = ChatCompletionRequest(model="gemini-2.0-flash",
            messages=[ChatMessage(role="user", content="hi")])
        config = _translate_request(req)[0]
        for k in ("temperature", "max_output_tokens", "top_p", "tools", "system_instruction"):
            assert k not in config

# -- _translate_response (non-streaming) -------------------------------------

class TestTranslateResponse:
    def test_text_response(self):
        result = _translate_response(_resp([_part(text="Hello world")]), "test-model")
        assert result.choices[0].message.content == "Hello world"
        assert result.choices[0].message.tool_calls is None

    def test_tool_call_response_with_synthetic_ids(self):
        result = _translate_response(
            _resp([_part(function_call=_fc("search", {"query": "weather"}))]), "test-model")
        tc = result.choices[0].message.tool_calls
        assert len(tc) == 1 and tc[0].id.startswith("call_")
        assert tc[0].function.name == "search"

    def test_mixed_text_and_tool_call(self):
        result = _translate_response(_resp([
            _part(text="Searching..."),
            _part(function_call=_fc("search", {"q": "test"}))]), "test-model")
        assert result.choices[0].message.content == "Searching..."
        assert len(result.choices[0].message.tool_calls) == 1

    def test_tool_calls_override_finish_reason(self):
        result = _translate_response(
            _resp([_part(function_call=_fc("f"))], finish_reason="STOP"), "m")
        assert result.choices[0].finish_reason == "tool_calls"

    @pytest.mark.parametrize("gemini,openai", [
        ("STOP", "stop"), ("MAX_TOKENS", "length"), ("SAFETY", "content_filter")])
    def test_finish_reason_mapping(self, gemini, openai):
        result = _translate_response(_resp([_part(text="ok")], finish_reason=gemini), "m")
        assert result.choices[0].finish_reason == openai

    def test_enum_style_finish_reason(self):
        result = _translate_response(
            _resp([_part(text="ok")], finish_reason="FinishReason.STOP"), "m")
        assert result.choices[0].finish_reason == "stop"

    def test_usage_extraction(self):
        result = _translate_response(
            _resp([_part(text="hi")], prompt_tokens=15, completion_tokens=25), "m")
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (15, 25)
        assert result.usage.total_tokens == 40

    def test_response_has_chatcmpl_id(self):
        assert _translate_response(_resp([_part(text="ok")]), "m").id.startswith("chatcmpl-")

    def test_tool_call_args_serialized_as_json_string(self):
        result = _translate_response(
            _resp([_part(function_call=_fc("search", {"query": "test", "limit": 5}))]), "m")
        assert json.loads(result.choices[0].message.tool_calls[0].function.arguments) == {
            "query": "test", "limit": 5}

# -- _stream_response (streaming) -------------------------------------------

def _finish_chunks(result):
    """Extract chunks that carry a non-null finish_reason."""
    out = []
    for c in result:
        if c.startswith("data: [DONE]"):
            continue
        parsed = _sse(c)
        choices = parsed.get("choices", [])
        if choices and choices[0].get("finish_reason") is not None:
            out.append(parsed)
    return out

class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_stream_role_chunk_first(self):
        result = await _collect([_chunk([])])
        assert _sse(result[0])["choices"][0]["delta"] == {"role": "assistant"}

    @pytest.mark.asyncio
    async def test_stream_text_deltas(self):
        result = await _collect([
            _chunk([_part(text="Hello")]), _chunk([_part(text=" world")])])
        text = [_sse(c) for c in result if not c.startswith("data: [DONE]")
                and "content" in _sse(c).get("choices", [{}])[0].get("delta", {})]
        assert len(text) == 2
        assert text[0]["choices"][0]["delta"]["content"] == "Hello"
        assert text[1]["choices"][0]["delta"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_stream_tool_call_chunk(self):
        result = await _collect([_chunk([_part(function_call=_fc("search", {"q": "test"}))])])
        tc_chunks = [_sse(c) for c in result if not c.startswith("data: [DONE]")
                     and "tool_calls" in _sse(c).get("choices", [{}])[0].get("delta", {})]
        assert len(tc_chunks) == 1
        tc = tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["id"].startswith("call_") and tc["index"] == 0
        assert tc["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_stream_finish_reason(self):
        result = await _collect([_chunk([_part(text="done")],
            finish_reason="STOP", usage_metadata=_usage())])
        fc = _finish_chunks(result)
        assert len(fc) == 1 and fc[0]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self):
        result = await _collect([_chunk([], finish_reason="STOP", usage_metadata=_usage())])
        assert result[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_usage_chunk(self):
        result = await _collect([_chunk([], finish_reason="STOP", usage_metadata=_usage(15, 25))])
        usage = [_sse(c) for c in result
                 if not c.startswith("data: [DONE]") and _sse(c).get("usage")]
        assert len(usage) == 1
        assert usage[0]["usage"] == {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}

    @pytest.mark.asyncio
    async def test_stream_tool_calls_override_finish_reason(self):
        result = await _collect([_chunk([_part(function_call=_fc("f"))],
            finish_reason="STOP", usage_metadata=_usage())])
        assert any(c["choices"][0]["finish_reason"] == "tool_calls"
                   for c in _finish_chunks(result))

# -- tool call ID round-trip -------------------------------------------------

class TestToolCallIdRoundTrip:
    def test_round_trip(self):
        _, contents, _ = _build_gemini_messages([
            ChatMessage(role="user", content="go"),
            ChatMessage(role="assistant", content=None, tool_calls=[
                _tc("call_abc", "search", '{"q":"test"}'),
                _tc("call_def", "lookup", '{"id":"42"}')]),
            ChatMessage(role="tool", tool_call_id="call_abc", content="found it"),
            ChatMessage(role="tool", tool_call_id="call_def", content="details")])
        names = [p["function_response"]["name"] for p in contents[-1]["parts"]]
        assert names == ["search", "lookup"]

# -- VertexProvider setup ----------------------------------------------------

class TestVertexSetup:
    def _patch_genai(self, monkeypatch):
        captured = {}
        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)
        mock_genai = SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
        monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=mock_genai))
        return captured

    @pytest.mark.asyncio
    async def test_setup_missing_project_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        self._patch_genai(monkeypatch)
        with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
            await VertexProvider().setup()

    @pytest.mark.asyncio
    async def test_setup_default_location(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        captured = self._patch_genai(monkeypatch)
        await VertexProvider().setup()
        assert captured == {"vertexai": True, "project": "my-project", "location": "us-central1"}

# -- registration & constants ------------------------------------------------

class TestVertexRegistration:
    def test_registered_as_vertex(self):
        from llm_adapter.providers import _REGISTRY
        assert "vertex" in _REGISTRY and _REGISTRY["vertex"] is VertexProvider

class TestStopReasonMapping:
    def test_mapping_completeness(self):
        assert set(_GEMINI_TO_OPENAI_STOP) == {"STOP", "MAX_TOKENS", "SAFETY", "RECITATION"}
