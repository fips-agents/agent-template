"""Tests for fipsagents.baseagent.llm — LLM client wrapper around litellm."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.config import LLMConfig
from fipsagents.baseagent.llm import (
    LLMClient,
    LLMError,
    ModelResponse,
    _parse_json_response,
    _schema_to_response_format,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> LLMConfig:
    defaults = {
        "endpoint": "http://test:8321/v1",
        "name": "test-model",
        "temperature": 0.5,
        "max_tokens": 2048,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _make_litellm_response(
    content: str | None = "hello",
    tool_calls: list[Any] | None = None,
) -> MagicMock:
    """Build a fake litellm ModelResponse with the given content/tool_calls."""
    message = MagicMock()
    message.content = content
    message.get = lambda key, default=None: (
        content if key == "content" else default
    )
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


SAMPLE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Say hello"},
]


# ---------------------------------------------------------------------------
# _schema_to_response_format
# ---------------------------------------------------------------------------


class TestSchemaToResponseFormat:
    def test_pydantic_model(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        fmt = _schema_to_response_format(Person)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "Person"
        schema = fmt["json_schema"]["schema"]
        assert "name" in schema.get("properties", {})
        assert "age" in schema.get("properties", {})

    def test_dict_schema(self):
        raw_schema = {
            "title": "MySchema",
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        fmt = _schema_to_response_format(raw_schema)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "MySchema"
        assert fmt["json_schema"]["schema"] is raw_schema

    def test_dict_schema_no_title_uses_default(self):
        raw_schema = {"type": "object"}
        fmt = _schema_to_response_format(raw_schema)
        assert fmt["json_schema"]["name"] == "response"

    def test_invalid_schema_type_raises(self):
        with pytest.raises(LLMError, match="must be a Pydantic model class"):
            _schema_to_response_format("not_a_schema")


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_parse_into_pydantic(self):
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        result = _parse_json_response('{"name": "widget"}', Item)
        assert isinstance(result, Item)
        assert result.name == "widget"

    def test_parse_into_dict(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = _parse_json_response('{"x": 42}', schema)
        assert result == {"x": 42}

    def test_invalid_json_raises(self):
        with pytest.raises(LLMError, match="invalid JSON"):
            _parse_json_response("{not json}", {"type": "object"})

    def test_schema_validation_failure_raises(self):
        from pydantic import BaseModel

        class Strict(BaseModel):
            value: int

        with pytest.raises(LLMError, match="failed schema validation"):
            _parse_json_response('{"value": "not_int"}', Strict)


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_content_extraction(self):
        raw = _make_litellm_response(content="world")
        resp = ModelResponse(raw)
        assert resp.content == "world"
        assert resp.tool_calls is None
        assert str(resp) == "world"

    def test_tool_calls_extraction(self):
        tc = [{"id": "call_1", "function": {"name": "foo", "arguments": "{}"}}]
        raw = _make_litellm_response(content=None, tool_calls=tc)
        resp = ModelResponse(raw)
        assert resp.content is None
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert str(resp) == ""

    def test_raw_preserved(self):
        raw = _make_litellm_response(content="hi")
        resp = ModelResponse(raw)
        assert resp.raw is raw


# ---------------------------------------------------------------------------
# LLMClient — config passthrough
# ---------------------------------------------------------------------------


class TestConfigPassthrough:
    """Verify that LLMConfig values reach litellm.acompletion."""

    @pytest.mark.asyncio
    async def test_base_kwargs_forwarded(self):
        config = _make_config(
            name="my-model",
            endpoint="http://host:9000/v1",
            temperature=0.3,
            max_tokens=512,
        )
        client = LLMClient(config)
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response()
            )
            await client.call_model(SAMPLE_MESSAGES)
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["model"] == "my-model"
            assert call_kwargs["api_base"] == "http://host:9000/v1"
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["max_tokens"] == 512
            assert call_kwargs["messages"] is SAMPLE_MESSAGES

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response()
            )
            await client.call_model(SAMPLE_MESSAGES, top_p=0.9, seed=42)
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["seed"] == 42

    @pytest.mark.asyncio
    async def test_kwargs_override_config(self):
        """Caller-provided kwargs take precedence over config defaults."""
        client = LLMClient(_make_config(temperature=0.5))
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response()
            )
            await client.call_model(SAMPLE_MESSAGES, temperature=0.0)
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# LLMClient.call_model
# ---------------------------------------------------------------------------


class TestCallModel:
    @pytest.mark.asyncio
    async def test_returns_model_response(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="hi there")
            )
            result = await client.call_model(SAMPLE_MESSAGES)
            assert isinstance(result, ModelResponse)
            assert result.content == "hi there"

    @pytest.mark.asyncio
    async def test_tool_schemas_passed(self):
        client = LLMClient(_make_config())
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response()
            )
            await client.call_model(SAMPLE_MESSAGES, tools=tools)
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["tools"] is tools

    @pytest.mark.asyncio
    async def test_tools_omitted_when_none(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response()
            )
            await client.call_model(SAMPLE_MESSAGES)
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        """When model returns tool calls, they are accessible on the response."""
        tc = [{"id": "call_99", "function": {"name": "do_thing", "arguments": '{"a": 1}'}}]
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content=None, tool_calls=tc)
            )
            result = await client.call_model(SAMPLE_MESSAGES)
            assert result.tool_calls is not None
            assert result.tool_calls[0]["id"] == "call_99"

    @pytest.mark.asyncio
    async def test_litellm_exception_wrapped(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=RuntimeError("connection refused")
            )
            with pytest.raises(LLMError, match="connection refused"):
                await client.call_model(SAMPLE_MESSAGES)


# ---------------------------------------------------------------------------
# LLMClient.call_model_json
# ---------------------------------------------------------------------------


class TestCallModelJson:
    @pytest.mark.asyncio
    async def test_pydantic_schema(self):
        from pydantic import BaseModel

        class Weather(BaseModel):
            city: str
            temp_f: float

        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(
                    content='{"city": "Portland", "temp_f": 55.0}'
                )
            )
            result = await client.call_model_json(SAMPLE_MESSAGES, Weather)
            assert isinstance(result, Weather)
            assert result.city == "Portland"
            assert result.temp_f == 55.0

            call_kwargs = mock_litellm.acompletion.call_args[1]
            rf = call_kwargs["response_format"]
            assert rf["type"] == "json_schema"
            assert rf["json_schema"]["name"] == "Weather"

    @pytest.mark.asyncio
    async def test_dict_schema(self):
        schema = {"title": "Coord", "type": "object", "properties": {"x": {"type": "integer"}}}
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content='{"x": 7}')
            )
            result = await client.call_model_json(SAMPLE_MESSAGES, schema)
            assert result == {"x": 7}

    @pytest.mark.asyncio
    async def test_no_content_raises(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content=None)
            )
            with pytest.raises(LLMError, match="no content"):
                await client.call_model_json(
                    SAMPLE_MESSAGES, {"type": "object"}
                )

    @pytest.mark.asyncio
    async def test_invalid_json_from_model(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="{broken json")
            )
            with pytest.raises(LLMError, match="invalid JSON"):
                await client.call_model_json(
                    SAMPLE_MESSAGES, {"type": "object"}
                )

    @pytest.mark.asyncio
    async def test_tools_forwarded(self):
        client = LLMClient(_make_config())
        tools = [{"type": "function", "function": {"name": "helper"}}]
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content='{}')
            )
            await client.call_model_json(
                SAMPLE_MESSAGES, {"type": "object"}, tools=tools
            )
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["tools"] is tools


# ---------------------------------------------------------------------------
# LLMClient.call_model_stream
# ---------------------------------------------------------------------------


class TestCallModelStream:
    @pytest.mark.asyncio
    async def test_yields_content_chunks(self):
        chunks_data = ["Hello", " ", "world"]

        async def fake_acompletion(**kwargs):
            async def _gen():
                for text in chunks_data:
                    delta = SimpleNamespace(content=text)
                    choice = SimpleNamespace(delta=delta)
                    yield SimpleNamespace(choices=[choice])
            return _gen()

        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            collected = []
            async for chunk in client.call_model_stream(SAMPLE_MESSAGES):
                collected.append(chunk)
            assert collected == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_skips_none_content(self):
        """Chunks with None content (e.g. role-only deltas) are skipped."""
        async def fake_acompletion(**kwargs):
            async def _gen():
                # First chunk: role only, no content
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
                )
                # Second chunk: actual content
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="data"))]
                )
            return _gen()

        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            collected = []
            async for chunk in client.call_model_stream(SAMPLE_MESSAGES):
                collected.append(chunk)
            assert collected == ["data"]

    @pytest.mark.asyncio
    async def test_stream_kwarg_set(self):
        async def fake_acompletion(**kwargs):
            assert kwargs["stream"] is True

            async def _gen():
                return
                yield  # make it an async generator

            return _gen()

        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            async for _ in client.call_model_stream(SAMPLE_MESSAGES):
                pass

    @pytest.mark.asyncio
    async def test_stream_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "search"}}]

        async def fake_acompletion(**kwargs):
            assert kwargs["tools"] is tools

            async def _gen():
                return
                yield

            return _gen()

        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            async for _ in client.call_model_stream(SAMPLE_MESSAGES, tools=tools):
                pass

    @pytest.mark.asyncio
    async def test_stream_exception_wrapped(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=ConnectionError("timeout")
            )
            with pytest.raises(LLMError, match="timeout"):
                async for _ in client.call_model_stream(SAMPLE_MESSAGES):
                    pass


# ---------------------------------------------------------------------------
# LLMClient.call_model_validated
# ---------------------------------------------------------------------------


class TestCallModelValidated:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="42")
            )
            result = await client.call_model_validated(
                SAMPLE_MESSAGES,
                lambda resp: int(resp.content),
            )
            assert result == 42
            assert mock_litellm.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        """Validator fails twice, then succeeds on the third attempt."""
        call_count = 0

        def flaky_validator(resp: ModelResponse) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"not ready yet (attempt {call_count})")
            return "valid"

        client = LLMClient(_make_config())
        with (
            patch("fipsagents.baseagent.llm.litellm") as mock_litellm,
            patch("fipsagents.baseagent.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="data")
            )
            result = await client.call_model_validated(
                SAMPLE_MESSAGES, flaky_validator, max_retries=3
            )
            assert result == "valid"
            assert mock_litellm.acompletion.call_count == 3
            # Two sleeps: after attempt 0 and attempt 1
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Verify the actual sleep durations follow 2^attempt pattern."""
        attempt = 0

        def always_fail(resp: ModelResponse) -> None:
            nonlocal attempt
            attempt += 1
            raise ValueError("nope")

        client = LLMClient(_make_config())
        with (
            patch("fipsagents.baseagent.llm.litellm") as mock_litellm,
            patch("fipsagents.baseagent.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="x")
            )
            with pytest.raises(LLMError, match="Validation failed after 4 attempts"):
                await client.call_model_validated(
                    SAMPLE_MESSAGES, always_fail, max_retries=3
                )
            # Delays: 2^0=1s, 2^1=2s, 2^2=4s
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert delays == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        client = LLMClient(_make_config())
        with (
            patch("fipsagents.baseagent.llm.litellm") as mock_litellm,
            patch("fipsagents.baseagent.llm.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="bad")
            )
            with pytest.raises(LLMError, match="Validation failed after 2 attempts") as exc_info:
                await client.call_model_validated(
                    SAMPLE_MESSAGES,
                    lambda r: (_ for _ in ()).throw(ValueError("always bad")),
                    max_retries=1,
                )
            # The original ValueError is chained
            assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_max_retries_zero(self):
        """With max_retries=0, only one attempt is made."""
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="ok")
            )
            result = await client.call_model_validated(
                SAMPLE_MESSAGES,
                lambda r: r.content.upper(),
                max_retries=0,
            )
            assert result == "OK"
            assert mock_litellm.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "validate"}}]
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(content="ok")
            )
            await client.call_model_validated(
                SAMPLE_MESSAGES,
                lambda r: r.content,
                tools=tools,
            )
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["tools"] is tools

    @pytest.mark.asyncio
    async def test_llm_error_propagates_immediately(self):
        """If the LLM call itself fails, the error propagates without retry."""
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=RuntimeError("provider down")
            )
            with pytest.raises(LLMError, match="provider down"):
                await client.call_model_validated(
                    SAMPLE_MESSAGES, lambda r: r.content
                )
            # Only one attempt — LLMError from call_model is not retried
            assert mock_litellm.acompletion.call_count == 1


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


class TestErrorWrapping:
    """Various litellm exception types should all become LLMError."""

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
        client = LLMClient(_make_config())
        with patch("fipsagents.baseagent.llm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=exc_class(exc_msg)
            )
            with pytest.raises(LLMError) as exc_info:
                await client.call_model(SAMPLE_MESSAGES)
            assert exc_msg in str(exc_info.value)
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, exc_class)
