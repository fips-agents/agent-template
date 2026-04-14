"""Tests for fipsagents.baseagent.llm — LLMClient, ModelResponse, and helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from fipsagents.baseagent.config import LLMConfig
from fipsagents.baseagent.llm import (
    LLMError,
    LLMClient,
    ModelResponse,
    _parse_json_response,
    _schema_to_response_format,
)


# ---------------------------------------------------------------------------
# Helpers for building fake litellm response objects
# ---------------------------------------------------------------------------


def _make_raw_response(content: str | None = "hello", tool_calls: Any = None) -> MagicMock:
    """Build a MagicMock that looks like a litellm ModelResponse."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    raw = MagicMock()
    raw.choices = [choice]
    return raw


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
    def test_includes_model_temperature_max_tokens(self):
        config = LLMConfig(name="test-model", temperature=0.5, max_tokens=512)
        client = LLMClient(config)
        kwargs = client._base_kwargs()
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 512

    def test_includes_api_base_when_endpoint_set(self):
        config = LLMConfig(endpoint="http://localhost:8080")
        client = LLMClient(config)
        kwargs = client._base_kwargs()
        assert kwargs["api_base"] == "http://localhost:8080"

    def test_no_api_base_when_endpoint_none(self):
        config = LLMConfig(endpoint=None)
        client = LLMClient(config)
        kwargs = client._base_kwargs()
        assert "api_base" not in kwargs

    def test_overrides_applied(self):
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
        client = LLMClient(config)
        raw = _make_raw_response(content="the answer")

        with patch("litellm.acompletion", new=AsyncMock(return_value=raw)):
            result = await client.call_model([{"role": "user", "content": "hi"}])

        assert isinstance(result, ModelResponse)
        assert result.content == "the answer"

    @pytest.mark.asyncio
    async def test_call_model_passes_tools_kwarg(self):
        config = LLMConfig(name="test-model")
        client = LLMClient(config)
        raw = _make_raw_response()
        tools = [{"type": "function", "function": {"name": "my_tool"}}]

        with patch("litellm.acompletion", new=AsyncMock(return_value=raw)) as mock_completion:
            await client.call_model([{"role": "user", "content": "x"}], tools=tools)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_call_model_raises_llm_error_on_failure(self):
        config = LLMConfig(name="test-model")
        client = LLMClient(config)

        with patch("litellm.acompletion", new=AsyncMock(side_effect=ConnectionError("down"))):
            with pytest.raises(LLMError, match="LLM call failed"):
                await client.call_model([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# LLMClient.call_model_json
# ---------------------------------------------------------------------------


class TestLLMClientCallModelJson:
    @pytest.mark.asyncio
    async def test_returns_parsed_pydantic_model(self):
        class Answer(BaseModel):
            value: int

        config = LLMConfig(name="test-model")
        client = LLMClient(config)
        raw = _make_raw_response(content='{"value": 42}')

        with patch("litellm.acompletion", new=AsyncMock(return_value=raw)):
            result = await client.call_model_json(
                [{"role": "user", "content": "what is the answer?"}],
                schema=Answer,
            )

        assert isinstance(result, Answer)
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_raises_llm_error_when_no_content(self):
        config = LLMConfig(name="test-model")
        client = LLMClient(config)
        raw = _make_raw_response(content=None)
        # Need the raw message.content to be None directly
        raw.choices[0].message.content = None

        with patch("litellm.acompletion", new=AsyncMock(return_value=raw)):
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
        client = LLMClient(config)
        raw = _make_raw_response(content="valid answer")

        with patch("litellm.acompletion", new=AsyncMock(return_value=raw)):
            result = await client.call_model_validated(
                [{"role": "user", "content": "x"}],
                validator_fn=lambda r: r.content,
            )

        assert result == "valid answer"

    @pytest.mark.asyncio
    async def test_retries_on_validation_failure(self):
        config = LLMConfig(name="test-model")
        client = LLMClient(config)

        raw_bad = _make_raw_response(content="bad")
        raw_good = _make_raw_response(content="good")
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            return raw_bad if call_count < 3 else raw_good

        def validator(resp: ModelResponse) -> str:
            if resp.content == "bad":
                raise ValueError("not good enough")
            return resp.content

        with patch("litellm.acompletion", new=mock_completion):
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
        client = LLMClient(config)
        raw = _make_raw_response(content="always bad")

        def always_fails(resp: ModelResponse) -> str:
            raise ValueError("never valid")

        with patch("litellm.acompletion", new=AsyncMock(return_value=raw)):
            with patch("asyncio.sleep", new=AsyncMock()):
                with pytest.raises(LLMError, match="Validation failed after"):
                    await client.call_model_validated(
                        [{"role": "user", "content": "x"}],
                        validator_fn=always_fails,
                        max_retries=2,
                    )
