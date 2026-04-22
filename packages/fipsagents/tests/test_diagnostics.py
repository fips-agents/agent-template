"""Tests for fipsagents.baseagent.diagnostics — probe_role_support and RoleProbeResult."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.diagnostics import probe_role_support


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(prompt_tokens: int, content: str = "4") -> MagicMock:
    """Build a MagicMock that looks like an OpenAI ChatCompletion."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.usage.prompt_tokens = prompt_tokens
    return resp


def _make_httpx_client(*, status_code: int = 404, json_data: dict | None = None, raise_exc: Exception | None = None) -> MagicMock:
    """Build an async context manager mock for httpx.AsyncClient."""
    client_mock = AsyncMock()

    if raise_exc is not None:
        client_mock.get = AsyncMock(side_effect=raise_exc)
    else:
        http_resp = MagicMock()
        http_resp.status_code = status_code
        http_resp.json.return_value = json_data or {}
        client_mock.get = AsyncMock(return_value=http_resp)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    cls_mock = MagicMock(return_value=cm)
    return cls_mock, client_mock


# ---------------------------------------------------------------------------
# Canary completion tests
# ---------------------------------------------------------------------------


class TestCanaryCompletion:
    @pytest.mark.asyncio
    async def test_canary_passes_when_token_delta_positive(self):
        """Positive prompt_token delta indicates the role message was consumed."""
        control_resp = _mock_response(prompt_tokens=10)
        test_resp = _mock_response(prompt_tokens=22)

        httpx_cls_mock, _ = _make_httpx_client(status_code=404)
        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=[control_resp, test_resp])
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="developer",
            )

        assert result.canary_passed is True
        assert result.prompt_token_delta == 12
        assert result.template_supported is None

    @pytest.mark.asyncio
    async def test_canary_fails_when_no_token_delta(self):
        """Zero delta means the role message was silently dropped."""
        control_resp = _mock_response(prompt_tokens=10)
        test_resp = _mock_response(prompt_tokens=10)

        httpx_cls_mock, _ = _make_httpx_client(status_code=404)
        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=[control_resp, test_resp])
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="developer",
            )

        assert result.canary_passed is False
        assert result.prompt_token_delta == 0

    @pytest.mark.asyncio
    async def test_canary_handles_completion_error(self):
        """An exception from the OpenAI client marks the canary as failed with no delta."""
        httpx_cls_mock, _ = _make_httpx_client(status_code=404)
        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("model down"))
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="developer",
            )

        assert result.canary_passed is False
        assert result.prompt_token_delta is None


# ---------------------------------------------------------------------------
# Template inspection tests
# ---------------------------------------------------------------------------


class TestTemplateInspection:
    @pytest.mark.asyncio
    async def test_template_supported_when_role_in_template(self):
        """Role name present in chat_template Jinja2 source means the template handles it."""
        template_json = {
            "id": "granite-8b",
            "chat_template": "{% if role == 'developer' %}...{% endif %}",
        }
        httpx_cls_mock, _ = _make_httpx_client(status_code=200, json_data=template_json)
        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=[
                _mock_response(10),
                _mock_response(22),
            ])
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="developer",
            )

        assert result.template_supported is True

    @pytest.mark.asyncio
    async def test_template_not_supported_when_role_absent(self):
        """chat_template present but does not mention the role."""
        template_json = {
            "id": "some-model",
            "chat_template": "{% if role == 'system' %}...{% endif %}",
        }
        httpx_cls_mock, _ = _make_httpx_client(status_code=200, json_data=template_json)
        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=[
                _mock_response(10),
                _mock_response(22),
            ])
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="developer",
            )

        assert result.template_supported is False

    @pytest.mark.asyncio
    async def test_template_inconclusive_on_http_error(self):
        """An HTTP exception leaves template_supported as None (inconclusive)."""
        httpx_cls_mock, _ = _make_httpx_client(raise_exc=RuntimeError("connection refused"))
        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=[
                _mock_response(10),
                _mock_response(22),
            ])
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="developer",
            )

        assert result.template_supported is None

    @pytest.mark.asyncio
    async def test_template_inconclusive_when_no_chat_template_field(self):
        """Response JSON lacks a chat_template field entirely."""
        httpx_cls_mock, _ = _make_httpx_client(status_code=200, json_data={"id": "model-1"})
        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = AsyncMock(side_effect=[
                _mock_response(10),
                _mock_response(22),
            ])
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="developer",
            )

        assert result.template_supported is None


# ---------------------------------------------------------------------------
# Custom role parameter
# ---------------------------------------------------------------------------


class TestCustomRoleParameter:
    @pytest.mark.asyncio
    async def test_custom_role_reflected_in_result_and_messages(self):
        """Passing role='assistant-prefill' flows through to the result and test call."""
        control_resp = _mock_response(prompt_tokens=10)
        test_resp = _mock_response(prompt_tokens=25)

        httpx_cls_mock, _ = _make_httpx_client(status_code=404)
        captured_calls: list = []

        async def capturing_create(*, messages, **kwargs):
            captured_calls.append(messages)
            return control_resp if len(captured_calls) == 1 else test_resp

        with patch("fipsagents.baseagent.diagnostics.AsyncOpenAI") as mock_openai_cls, \
             patch("httpx.AsyncClient", httpx_cls_mock):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create = capturing_create
            result = await probe_role_support(
                model="test-model",
                endpoint="http://localhost:8080",
                role="assistant-prefill",
            )

        assert result.role == "assistant-prefill"

        # The test call (second completion) must include a message with the custom role.
        assert len(captured_calls) == 2
        test_messages = captured_calls[1]
        roles_in_test = {msg["role"] for msg in test_messages}
        assert "assistant-prefill" in roles_in_test
