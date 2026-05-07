"""Tests for fipsagents.baseagent.llm — LLMClient, ModelResponse, and helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from fipsagents.baseagent.config import LLMConfig, PlatformConfig, PlatformMcpServer
from fipsagents.baseagent.events import (
    ContentDelta,
    GuardrailFiredEvent,
    StreamComplete,
)
from fipsagents.baseagent.llm import (
    LLMError,
    LLMClient,
    ModelResponse,
    ModerationResult,
    PlatformResponse,
    _extract_refusal,
    _mcp_servers_to_tools,
    _parse_json_response,
    _schema_to_response_format,
    _shield_id_from_refusal,
)


# ---------------------------------------------------------------------------
# Helpers for building fake OpenAI chat completion response objects
# ---------------------------------------------------------------------------


def _make_raw_response(content: str | None = "hello", tool_calls: Any = None) -> MagicMock:
    """Build a MagicMock that looks like an OpenAI ChatCompletion."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    raw = MagicMock()
    raw.choices = [choice]
    return raw


SAMPLE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Say hello"},
]


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_extracts_content(self):
        raw = _make_raw_response(content="some text")
        resp = ModelResponse(raw)
        assert resp.content == "some text"

    def test_content_none_when_empty_string(self):
        raw = _make_raw_response(content="")
        resp = ModelResponse(raw)
        assert resp.content is None

    def test_content_none_when_attribute_missing(self):
        message = MagicMock(spec=[])  # no .content attribute
        choice = MagicMock()
        choice.message = message
        raw = MagicMock()
        raw.choices = [choice]
        resp = ModelResponse(raw)
        assert resp.content is None

    def test_extracts_tool_calls(self):
        tc = [MagicMock(name="call1")]
        raw = _make_raw_response(content=None, tool_calls=tc)
        resp = ModelResponse(raw)
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_tool_calls_none_when_absent(self):
        raw = _make_raw_response(content="hi", tool_calls=None)
        resp = ModelResponse(raw)
        assert resp.tool_calls is None

    def test_str_returns_content(self):
        raw = _make_raw_response(content="result text")
        resp = ModelResponse(raw)
        assert str(resp) == "result text"

    def test_str_returns_empty_when_no_content(self):
        raw = _make_raw_response(content=None)
        resp = ModelResponse(raw)
        assert str(resp) == ""

    def test_raw_stored(self):
        raw = _make_raw_response()
        resp = ModelResponse(raw)
        assert resp.raw is raw


# ---------------------------------------------------------------------------
# _schema_to_response_format
# ---------------------------------------------------------------------------


class TestSchemaToResponseFormat:
    def test_with_pydantic_model(self):
        class MyOutput(BaseModel):
            answer: str
            score: int

        fmt = _schema_to_response_format(MyOutput)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "MyOutput"
        assert "schema" in fmt["json_schema"]

    def test_with_dict_schema(self):
        schema = {"title": "Result", "type": "object", "properties": {"x": {"type": "integer"}}}
        fmt = _schema_to_response_format(schema)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "Result"

    def test_dict_schema_without_title_uses_response(self):
        schema = {"type": "object"}
        fmt = _schema_to_response_format(schema)
        assert fmt["json_schema"]["name"] == "response"

    def test_invalid_type_raises_llm_error(self):
        with pytest.raises(LLMError, match="Pydantic model"):
            _schema_to_response_format("not a schema")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_valid_json_to_pydantic_model(self):
        class Point(BaseModel):
            x: int
            y: int

        result = _parse_json_response('{"x": 1, "y": 2}', Point)
        assert isinstance(result, Point)
        assert result.x == 1
        assert result.y == 2

    def test_valid_json_to_dict(self):
        schema = {"type": "object"}
        result = _parse_json_response('{"key": "value"}', schema)
        assert result == {"key": "value"}

    def test_invalid_json_raises_llm_error(self):
        with pytest.raises(LLMError, match="invalid JSON"):
            _parse_json_response("{not valid json}", {"type": "object"})

    def test_schema_validation_failure_raises_llm_error(self):
        class Strict(BaseModel):
            count: int

        with pytest.raises(LLMError, match="schema validation"):
            _parse_json_response('{"count": "not-an-int"}', Strict)


# ---------------------------------------------------------------------------
# LLMClient._base_kwargs
# ---------------------------------------------------------------------------


class TestLLMClientBaseKwargs:
    @patch("fipsagents.baseagent.llm.AsyncOpenAI")
    def test_includes_model_temperature_max_tokens(self, _mock_cls):
        config = LLMConfig(name="test-model", temperature=0.5, max_tokens=512)
        client = LLMClient(config)
        kwargs = client._base_kwargs()
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 512

    @patch("fipsagents.baseagent.llm.AsyncOpenAI")
    def test_endpoint_passed_to_client_constructor(self, mock_cls):
        """Endpoint is set on the AsyncOpenAI client, not in call kwargs."""
        config = LLMConfig(endpoint="http://localhost:8080")
        LLMClient(config)
        mock_cls.assert_called_once_with(
            base_url="http://localhost:8080",
            api_key=mock_cls.call_args[1]["api_key"],  # don't assert on api_key value
        )

    @patch("fipsagents.baseagent.llm.AsyncOpenAI")
    def test_no_endpoint_passes_none_to_client(self, mock_cls):
        """When endpoint is None, base_url=None is passed to the client."""
        config = LLMConfig(endpoint=None)
        LLMClient(config)
        mock_cls.assert_called_once()
        assert mock_cls.call_args[1]["base_url"] is None

    @patch("fipsagents.baseagent.llm.AsyncOpenAI")
    def test_overrides_applied(self, _mock_cls):
        config = LLMConfig(temperature=0.7)
        client = LLMClient(config)
        kwargs = client._base_kwargs(temperature=0.1)
        assert kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# LLMClient.call_model
# ---------------------------------------------------------------------------


class TestLLMClientCallModel:
    @pytest.mark.asyncio
    async def test_call_model_returns_model_response(self):
        config = LLMConfig(name="test-model")
        raw = _make_raw_response(content="the answer")

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=raw)
            client = LLMClient(config)
            result = await client.call_model([{"role": "user", "content": "hi"}])

        assert isinstance(result, ModelResponse)
        assert result.content == "the answer"

    @pytest.mark.asyncio
    async def test_call_model_passes_tools_kwarg(self):
        config = LLMConfig(name="test-model")
        raw = _make_raw_response()
        tools = [{"type": "function", "function": {"name": "my_tool"}}]

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_create = AsyncMock(return_value=raw)
            mock_client.chat.completions.create = mock_create
            client = LLMClient(config)
            await client.call_model([{"role": "user", "content": "x"}], tools=tools)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_call_model_raises_llm_error_on_failure(self):
        config = LLMConfig(name="test-model")

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=ConnectionError("down"))
            client = LLMClient(config)
            with pytest.raises(LLMError, match="LLM call failed"):
                await client.call_model([{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_kwargs_override_config(self):
        """Caller-provided kwargs take precedence over config defaults."""
        config = LLMConfig(name="test-model", temperature=0.5)
        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_raw_response()
            )
            client = LLMClient(config)
            await client.call_model(SAMPLE_MESSAGES, temperature=0.0)
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# LLMClient.call_model_json
# ---------------------------------------------------------------------------


class TestLLMClientCallModelJson:
    @pytest.mark.asyncio
    async def test_returns_parsed_pydantic_model(self):
        class Answer(BaseModel):
            value: int

        config = LLMConfig(name="test-model")
        raw = _make_raw_response(content='{"value": 42}')

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=raw)
            client = LLMClient(config)
            result = await client.call_model_json(
                [{"role": "user", "content": "what is the answer?"}],
                schema=Answer,
            )

        assert isinstance(result, Answer)
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_raises_llm_error_when_no_content(self):
        config = LLMConfig(name="test-model")
        raw = _make_raw_response(content=None)
        # Need the raw message.content to be None directly
        raw.choices[0].message.content = None

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=raw)
            client = LLMClient(config)
            with pytest.raises(LLMError, match="no content"):
                await client.call_model_json(
                    [{"role": "user", "content": "x"}],
                    schema={"type": "object"},
                )


# ---------------------------------------------------------------------------
# LLMClient.call_model_validated
# ---------------------------------------------------------------------------


class TestLLMClientCallModelValidated:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        config = LLMConfig(name="test-model")
        raw = _make_raw_response(content="valid answer")

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=raw)
            client = LLMClient(config)
            result = await client.call_model_validated(
                [{"role": "user", "content": "x"}],
                validator_fn=lambda r: r.content,
            )

        assert result == "valid answer"

    @pytest.mark.asyncio
    async def test_retries_on_validation_failure(self):
        config = LLMConfig(name="test-model")

        raw_bad = _make_raw_response(content="bad")
        raw_good = _make_raw_response(content="good")
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return raw_bad if call_count < 3 else raw_good

        def validator(resp: ModelResponse) -> str:
            if resp.content == "bad":
                raise ValueError("not good enough")
            return resp.content

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create = mock_create
            client = LLMClient(config)
            with patch("asyncio.sleep", new=AsyncMock()):
                result = await client.call_model_validated(
                    [{"role": "user", "content": "x"}],
                    validator_fn=validator,
                    max_retries=3,
                )

        assert result == "good"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_llm_error_after_max_retries(self):
        config = LLMConfig(name="test-model")
        raw = _make_raw_response(content="always bad")

        def always_fails(resp: ModelResponse) -> str:
            raise ValueError("never valid")

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=raw)
            client = LLMClient(config)
            with patch("asyncio.sleep", new=AsyncMock()):
                with pytest.raises(LLMError, match="Validation failed after"):
                    await client.call_model_validated(
                        [{"role": "user", "content": "x"}],
                        validator_fn=always_fails,
                        max_retries=2,
                    )

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Verify the actual sleep durations follow 2^attempt pattern."""
        attempt = 0

        def always_fail(resp: ModelResponse) -> None:
            nonlocal attempt
            attempt += 1
            raise ValueError("nope")

        with (
            patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls,
            patch("fipsagents.baseagent.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_raw_response(content="x")
            )
            client = LLMClient(LLMConfig(name="test-model"))
            with pytest.raises(LLMError, match="Validation failed after 4 attempts"):
                await client.call_model_validated(
                    SAMPLE_MESSAGES, always_fail, max_retries=3
                )
            # Delays: 2^0=1s, 2^1=2s, 2^2=4s
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert delays == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# LLMClient.call_model_stream_raw — stream_options.include_usage default
# ---------------------------------------------------------------------------


class TestLLMClientCallModelStreamRaw:
    @pytest.mark.asyncio
    async def test_sets_include_usage_by_default(self):
        """Streaming calls must request the terminal usage chunk so the
        server-layer cost-tracking accumulator sees prompt/completion tokens.
        Regression for #118.
        """
        config = LLMConfig(name="test-model")

        async def empty_stream():
            if False:
                yield  # type: ignore[unreachable]

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_create = AsyncMock(return_value=empty_stream())
            mock_client.chat.completions.create = mock_create
            client = LLMClient(config)
            async for _ in client.call_model_stream_raw(
                [{"role": "user", "content": "hi"}],
            ):
                pass

        kwargs = mock_create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_caller_can_override_stream_options(self):
        """Callers passing stream_options explicitly win over the default."""
        config = LLMConfig(name="test-model")

        async def empty_stream():
            if False:
                yield  # type: ignore[unreachable]

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_create = AsyncMock(return_value=empty_stream())
            mock_client.chat.completions.create = mock_create
            client = LLMClient(config)
            async for _ in client.call_model_stream_raw(
                [{"role": "user", "content": "hi"}],
                stream_options={"include_usage": False},
            ):
                pass

        kwargs = mock_create.call_args.kwargs
        assert kwargs["stream_options"] == {"include_usage": False}


# ---------------------------------------------------------------------------
# LLMClient.call_model_stream — high-level wrapper that yields content chunks
# ---------------------------------------------------------------------------


class TestCallModelStream:
    @pytest.mark.asyncio
    async def test_yields_content_chunks(self):
        chunks_data = ["Hello", " ", "world"]

        async def _gen():
            for text in chunks_data:
                delta = SimpleNamespace(content=text)
                choice = SimpleNamespace(delta=delta)
                yield SimpleNamespace(choices=[choice])

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=_gen())
            client = LLMClient(LLMConfig(name="test-model"))
            collected = []
            async for chunk in client.call_model_stream(SAMPLE_MESSAGES):
                collected.append(chunk)
            assert collected == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_skips_none_content(self):
        """Chunks with None content (e.g. role-only deltas) are skipped."""
        async def _gen():
            # First chunk: role only, no content
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            )
            # Second chunk: actual content
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="data"))]
            )

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=_gen())
            client = LLMClient(LLMConfig(name="test-model"))
            collected = []
            async for chunk in client.call_model_stream(SAMPLE_MESSAGES):
                collected.append(chunk)
            assert collected == ["data"]

    @pytest.mark.asyncio
    async def test_stream_kwarg_set(self):
        async def _gen():
            return
            yield  # make it an async generator

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=_gen())
            client = LLMClient(LLMConfig(name="test-model"))
            async for _ in client.call_model_stream(SAMPLE_MESSAGES):
                pass
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "search"}}]

        async def _gen():
            return
            yield

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=_gen())
            client = LLMClient(LLMConfig(name="test-model"))
            async for _ in client.call_model_stream(SAMPLE_MESSAGES, tools=tools):
                pass
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tools"] is tools

    @pytest.mark.asyncio
    async def test_stream_exception_wrapped(self):
        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(
                side_effect=ConnectionError("timeout")
            )
            client = LLMClient(LLMConfig(name="test-model"))
            with pytest.raises(LLMError, match="timeout"):
                async for _ in client.call_model_stream(SAMPLE_MESSAGES):
                    pass


# ---------------------------------------------------------------------------
# Error wrapping — various exception types should all become LLMError
# ---------------------------------------------------------------------------


class TestErrorWrapping:
    """Various OpenAI SDK exception types should all become LLMError."""

    @pytest.mark.parametrize(
        "exc_class, exc_msg",
        [
            (RuntimeError, "generic failure"),
            (ConnectionError, "network unreachable"),
            (TimeoutError, "request timed out"),
            (ValueError, "bad parameter"),
        ],
        ids=["runtime", "connection", "timeout", "value"],
    )
    @pytest.mark.asyncio
    async def test_exception_types_wrapped(self, exc_class, exc_msg):
        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(
                side_effect=exc_class(exc_msg)
            )
            client = LLMClient(LLMConfig(name="test-model"))
            with pytest.raises(LLMError) as exc_info:
                await client.call_model(SAMPLE_MESSAGES)
            assert exc_msg in str(exc_info.value)
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, exc_class)


# ---------------------------------------------------------------------------
# Platform-mode helpers (issue #154)
# ---------------------------------------------------------------------------


class TestMcpServersToTools:
    def test_connector_reference(self):
        srv = PlatformMcpServer(name="weather", connector_id="mcp::weather")
        out = _mcp_servers_to_tools([srv])
        assert out == [
            {"type": "mcp", "server_label": "weather", "connector_id": "mcp::weather"}
        ]

    def test_inline_url(self):
        srv = PlatformMcpServer(name="calculus", url="http://mcp:8080/mcp/")
        out = _mcp_servers_to_tools([srv])
        assert out == [
            {"type": "mcp", "server_label": "calculus", "server_url": "http://mcp:8080/mcp/"}
        ]

    def test_authorization_forwarded(self):
        srv = PlatformMcpServer(
            name="deepwiki",
            url="https://mcp.deepwiki.com/sse",
            authorization="abc123",
        )
        out = _mcp_servers_to_tools([srv])
        assert out[0]["authorization"] == "abc123"

    def test_empty_list(self):
        assert _mcp_servers_to_tools([]) == []

    def test_multiple_servers_mixed_modes(self):
        servers = [
            PlatformMcpServer(name="weather", connector_id="mcp::weather"),
            PlatformMcpServer(name="calculus", url="http://mcp:8080/mcp/"),
        ]
        out = _mcp_servers_to_tools(servers)
        assert len(out) == 2
        assert "connector_id" in out[0]
        assert "server_url" in out[1]


class TestExtractRefusal:
    def test_finds_refusal_in_dict(self):
        output = [
            {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "blocked"}],
            }
        ]
        assert _extract_refusal(output) == "blocked"

    def test_returns_none_when_only_text(self):
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ]
        assert _extract_refusal(output) is None

    def test_returns_none_for_empty(self):
        assert _extract_refusal([]) is None

    def test_finds_refusal_among_mixed_content(self):
        output = [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "partial"},
                    {"type": "refusal", "refusal": "blocked"},
                ],
            }
        ]
        assert _extract_refusal(output) == "blocked"


class TestShieldIdFromRefusal:
    def test_parses_flagged_for_pattern(self):
        # The exact pattern OGX emits in code-scanner refusals.
        msg = (
            "Security concerns detected. Potential code injection due to "
            "eval usage. (flagged for: insecure-eval-use)"
        )
        assert _shield_id_from_refusal(msg, ["code-scanner"]) == "insecure-eval-use"

    def test_parses_multi_value_flagged_for(self):
        msg = "(flagged for: eval-with-expression, insecure-eval-use)"
        assert (
            _shield_id_from_refusal(msg, ["code-scanner"])
            == "eval-with-expression, insecure-eval-use"
        )

    def test_falls_back_to_configured_when_no_pattern(self):
        assert _shield_id_from_refusal("plain text", ["a", "b"]) == "a,b"

    def test_unknown_when_no_pattern_and_no_config(self):
        assert _shield_id_from_refusal("plain text", []) == "unknown"


class TestPlatformResponse:
    def test_text_only_response(self):
        # Shape mirrors the real /v1/responses fixture for a benign call.
        raw = {
            "id": "resp_abc",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {"input_tokens": 76, "output_tokens": 63, "total_tokens": 139},
        }
        resp = PlatformResponse(raw)
        assert resp.content == "Hello!"
        assert resp.refusal is None
        assert resp.response_id == "resp_abc"
        assert resp.usage["input_tokens"] == 76

    def test_refusal_response(self):
        # Shape mirrors a code-scanner-blocked /v1/responses payload.
        raw = {
            "id": "resp_xyz",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "refusal", "refusal": "blocked: eval is unsafe"}
                    ],
                }
            ],
        }
        resp = PlatformResponse(raw)
        assert resp.content is None
        assert resp.refusal == "blocked: eval is unsafe"
        assert str(resp) == "blocked: eval is unsafe"

    def test_joins_multiple_text_parts(self):
        raw = {
            "id": "resp_join",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello "},
                        {"type": "output_text", "text": "world"},
                    ],
                }
            ],
        }
        assert PlatformResponse(raw).content == "Hello world"


# ---------------------------------------------------------------------------
# LLMClient — platform-mode methods
# ---------------------------------------------------------------------------


def _platform_config(enabled: bool = True, **overrides: Any) -> PlatformConfig:
    fields: dict[str, Any] = {"enabled": enabled}
    if enabled:
        fields["endpoint"] = "http://ogx:8321/v1"
    fields.update(overrides)
    return PlatformConfig(**fields)


class TestLLMClientPlatformGuards:
    def test_responses_call_without_platform_raises(self):
        config = LLMConfig(name="test-model")
        with patch("fipsagents.baseagent.llm.AsyncOpenAI"):
            client = LLMClient(config)
            with pytest.raises(LLMError, match="platform.enabled is false"):
                # Drive the lazy guard directly; method is async but the
                # guard fires before any await.
                client._require_platform()

    def test_responses_call_with_disabled_platform_raises(self):
        config = LLMConfig(name="test-model")
        with patch("fipsagents.baseagent.llm.AsyncOpenAI"):
            client = LLMClient(config, platform=_platform_config(enabled=False))
            with pytest.raises(LLMError, match="platform.enabled is false"):
                client._require_platform()


class TestLLMClientCallModelResponses:
    @pytest.mark.asyncio
    async def test_returns_platform_response(self):
        config = LLMConfig(name="vllm/RedHatAI/gpt-oss-20b")
        platform = _platform_config()
        raw = MagicMock()
        raw.id = "resp_abc"
        raw.output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="Hi.")],
            )
        ]
        raw.usage = MagicMock(input_tokens=10, output_tokens=2, total_tokens=12)

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.responses.create = AsyncMock(return_value=raw)
            client = LLMClient(config, platform=platform)
            result = await client.call_model_responses("Say hi.")

        assert isinstance(result, PlatformResponse)
        assert result.content == "Hi."

    @pytest.mark.asyncio
    async def test_defaults_tools_from_platform_mcp(self):
        config = LLMConfig(name="vllm/RedHatAI/gpt-oss-20b")
        platform = _platform_config(
            mcp=[PlatformMcpServer(name="weather", connector_id="mcp::weather")]
        )
        raw = MagicMock(id="r", output=[], usage=None)

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_create = AsyncMock(return_value=raw)
            mock_client.responses.create = mock_create
            client = LLMClient(config, platform=platform)
            await client.call_model_responses("hello")

        kwargs = mock_create.call_args.kwargs
        assert kwargs["tools"] == [
            {"type": "mcp", "server_label": "weather", "connector_id": "mcp::weather"}
        ]
        # Model name passed verbatim — vllm/ prefix preserved (it's an OGX
        # registered model id, not a litellm prefix).
        assert kwargs["model"] == "vllm/RedHatAI/gpt-oss-20b"

    @pytest.mark.asyncio
    async def test_defaults_guardrails_from_platform_config(self):
        config = LLMConfig(name="m")
        platform = _platform_config(guardrails=["code-scanner"])
        raw = MagicMock(id="r", output=[], usage=None)

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_create = AsyncMock(return_value=raw)
            mock_client.responses.create = mock_create
            client = LLMClient(config, platform=platform)
            await client.call_model_responses("hi")

        # guardrails travels via extra_body — OpenAI SDK rejects unknown
        # top-level kwargs.
        kwargs = mock_create.call_args.kwargs
        assert "guardrails" not in kwargs
        assert kwargs["extra_body"] == {"guardrails": ["code-scanner"]}

    @pytest.mark.asyncio
    async def test_extra_body_merges_with_caller_keys(self):
        config = LLMConfig(name="m")
        platform = _platform_config(guardrails=["code-scanner"])
        raw = MagicMock(id="r", output=[], usage=None)

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_create = AsyncMock(return_value=raw)
            mock_client.responses.create = mock_create
            client = LLMClient(config, platform=platform)
            await client.call_model_responses(
                "hi", extra_body={"custom_field": "x"}
            )

        kwargs = mock_create.call_args.kwargs
        assert kwargs["extra_body"] == {
            "custom_field": "x",
            "guardrails": ["code-scanner"],
        }

    @pytest.mark.asyncio
    async def test_per_call_overrides_win(self):
        config = LLMConfig(name="m")
        platform = _platform_config(
            mcp=[PlatformMcpServer(name="weather", connector_id="mcp::weather")],
            guardrails=["code-scanner"],
        )
        raw = MagicMock(id="r", output=[], usage=None)

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_create = AsyncMock(return_value=raw)
            mock_client.responses.create = mock_create
            client = LLMClient(config, platform=platform)
            await client.call_model_responses(
                "hi", tools=[], guardrails=[],
            )

        kwargs = mock_create.call_args.kwargs
        # Empty list per-call → fields omitted (don't override defaults to []).
        assert "tools" not in kwargs
        assert "guardrails" not in kwargs

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_failure(self):
        config = LLMConfig(name="m")
        platform = _platform_config()
        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.responses.create = AsyncMock(
                side_effect=ConnectionError("down")
            )
            client = LLMClient(config, platform=platform)
            with pytest.raises(LLMError, match="Responses API call failed"):
                await client.call_model_responses("hi")


class TestLLMClientCallModelResponsesStream:
    @pytest.mark.asyncio
    async def test_emits_content_deltas_and_terminal(self):
        config = LLMConfig(name="m")
        platform = _platform_config()

        async def fake_stream():
            yield MagicMock(type="response.created")
            yield MagicMock(type="response.in_progress")
            yield MagicMock(type="response.output_text.delta", delta="Hello")
            yield MagicMock(type="response.output_text.delta", delta=" world")
            final = MagicMock()
            final.output = [
                MagicMock(
                    type="message",
                    content=[MagicMock(type="output_text", text="Hello world")],
                )
            ]
            final.usage = MagicMock(
                input_tokens=5, output_tokens=2, total_tokens=7
            )
            yield MagicMock(type="response.completed", response=final)

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.responses.create = AsyncMock(return_value=fake_stream())
            client = LLMClient(config, platform=platform)
            events = [ev async for ev in client.call_model_responses_stream("hi")]

        deltas = [e for e in events if isinstance(e, ContentDelta)]
        complete = [e for e in events if isinstance(e, StreamComplete)]
        assert [d.content for d in deltas] == ["Hello", " world"]
        assert len(complete) == 1
        assert complete[0].finish_reason == "stop"
        assert complete[0].metrics.prompt_tokens == 5
        assert complete[0].metrics.completion_tokens == 2
        assert complete[0].metrics.total_tokens == 7

    @pytest.mark.asyncio
    async def test_emits_guardrail_event_on_refusal(self):
        config = LLMConfig(name="m")
        platform = _platform_config(guardrails=["code-scanner"])

        async def fake_stream():
            yield MagicMock(
                type="response.output_text.delta", delta="Below is"
            )
            final = MagicMock()
            final.output = [
                MagicMock(
                    type="message",
                    content=[
                        MagicMock(
                            type="refusal",
                            refusal=(
                                "Security concerns detected. "
                                "(flagged for: insecure-eval-use)"
                            ),
                        )
                    ],
                )
            ]
            final.usage = None
            yield MagicMock(type="response.completed", response=final)

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.responses.create = AsyncMock(return_value=fake_stream())
            client = LLMClient(config, platform=platform)
            events = [ev async for ev in client.call_model_responses_stream("hi")]

        # Pre-shield delta still passes through (per agreed UX).
        assert any(
            isinstance(e, ContentDelta) and e.content == "Below is" for e in events
        )
        guardrails = [e for e in events if isinstance(e, GuardrailFiredEvent)]
        complete = [e for e in events if isinstance(e, StreamComplete)]
        assert len(guardrails) == 1
        assert guardrails[0].action == "blocked"
        assert guardrails[0].shield_id == "insecure-eval-use"
        assert "(flagged for:" in (guardrails[0].message or "")
        assert complete[0].finish_reason == "guardrail"


class TestLLMClientModerate:
    @pytest.mark.asyncio
    async def test_aggregates_categories_and_scores(self):
        config = LLMConfig(name="m")
        platform = _platform_config()

        raw = MagicMock(model="code-scanner")
        raw.results = [
            MagicMock(
                flagged=False,
                categories={"a": False, "b": True},
                category_scores={"a": 0.1, "b": 0.7},
            ),
            MagicMock(
                flagged=True,
                categories={"a": True, "c": True},
                category_scores={"a": 0.9, "c": 0.6},
            ),
        ]

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.moderations.create = AsyncMock(return_value=raw)
            client = LLMClient(config, platform=platform)
            result = await client.moderate("hello", model="code-scanner")

        assert isinstance(result, ModerationResult)
        assert result.flagged is True
        assert result.categories == {"a": True, "b": True, "c": True}
        # max() across results
        assert result.category_scores["a"] == 0.9
        assert result.category_scores["b"] == 0.7
        assert result.category_scores["c"] == 0.6

    @pytest.mark.asyncio
    async def test_no_results_returns_unflagged(self):
        config = LLMConfig(name="m")
        platform = _platform_config()
        raw = MagicMock(model="code-scanner", results=[])

        with patch("fipsagents.baseagent.llm.AsyncOpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.moderations.create = AsyncMock(return_value=raw)
            client = LLMClient(config, platform=platform)
            result = await client.moderate("hello", model="code-scanner")

        assert result.flagged is False
        assert result.categories == {}
