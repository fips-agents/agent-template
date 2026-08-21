"""Tests for FallbackProvider — model fallback chain (#194)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from fipsagents.baseagent.config import FallbackLLMConfig, LLMConfig
from fipsagents.baseagent.llm import LLMError
from fipsagents.baseagent.providers._base import LLMProvider
from fipsagents.baseagent.providers._fallback import FallbackProvider, _is_retriable
from fipsagents.baseagent.providers import create_provider


class _FakeStatusError(Exception):
    """Exception with a status_code attribute for testing."""
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _make_response(content: str = "ok") -> Any:
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _make_chunk(content: str | None, finish: str | None = None) -> Any:
    delta = SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=None)


class MockProvider(LLMProvider):
    """Test provider that can be configured to succeed or fail."""

    def __init__(self, name: str, fail: Exception | None = None):
        config = LLMConfig(provider="openai", name=name)
        super().__init__(config)
        self._name = name
        self._fail = fail
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def complete(self, messages, *, tools=None, **kwargs):
        self.call_count += 1
        if self._fail:
            raise self._fail
        return _make_response(f"from {self._name}")

    async def stream_raw(self, messages, *, tools=None, **kwargs):
        self.call_count += 1
        if self._fail:
            raise self._fail
        yield _make_chunk(f"from {self._name}")
        yield _make_chunk(None, finish="stop")


class TestIsRetriable:
    def test_connection_error_is_retriable(self):
        cause = ConnectionError("refused")
        exc = LLMError("fail")
        exc.__cause__ = cause
        assert _is_retriable(exc) is True

    def test_timeout_is_retriable(self):
        cause = TimeoutError("timed out")
        exc = LLMError("fail")
        exc.__cause__ = cause
        assert _is_retriable(exc) is True

    def test_5xx_is_retriable(self):
        cause = _FakeStatusError(503)
        exc = LLMError("fail")
        exc.__cause__ = cause
        assert _is_retriable(exc) is True

    def test_4xx_is_not_retriable(self):
        cause = _FakeStatusError(400)
        exc = LLMError("fail")
        exc.__cause__ = cause
        assert _is_retriable(exc) is False

    def test_401_is_not_retriable(self):
        cause = _FakeStatusError(401)
        exc = LLMError("fail")
        exc.__cause__ = cause
        assert _is_retriable(exc) is False

    def test_no_cause_is_retriable(self):
        exc = LLMError("fail")
        assert _is_retriable(exc) is True


class TestFallbackComplete:
    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self):
        primary = MockProvider("primary")
        fb = MockProvider("fallback")
        provider = FallbackProvider(primary, [fb])

        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.choices[0].message.content == "from primary"
        assert primary.call_count == 1
        assert fb.call_count == 0

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_used(self):
        err = LLMError("connection refused")
        err.__cause__ = ConnectionError("refused")
        primary = MockProvider("primary", fail=err)
        fb = MockProvider("fallback")
        provider = FallbackProvider(primary, [fb])

        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.choices[0].message.content == "from fallback"
        assert primary.call_count == 1
        assert fb.call_count == 1

    @pytest.mark.asyncio
    async def test_chain_exhausted_raises_last_error(self):
        err1 = LLMError("primary down")
        err1.__cause__ = ConnectionError()
        err2 = LLMError("fallback down")
        err2.__cause__ = ConnectionError()
        primary = MockProvider("primary", fail=err1)
        fb = MockProvider("fallback", fail=err2)
        provider = FallbackProvider(primary, [fb])

        with pytest.raises(LLMError, match="fallback down"):
            await provider.complete([{"role": "user", "content": "hi"}])
        assert primary.call_count == 1
        assert fb.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retriable_error_no_fallback(self):
        err = LLMError("bad request")
        err.__cause__ = _FakeStatusError(400)
        primary = MockProvider("primary", fail=err)
        fb = MockProvider("fallback")
        provider = FallbackProvider(primary, [fb])

        with pytest.raises(LLMError, match="bad request"):
            await provider.complete([{"role": "user", "content": "hi"}])
        assert fb.call_count == 0

    @pytest.mark.asyncio
    async def test_second_fallback_used(self):
        err = LLMError("down")
        err.__cause__ = ConnectionError()
        primary = MockProvider("primary", fail=err)
        fb1_err = LLMError("also down")
        fb1_err.__cause__ = ConnectionError()
        fb1 = MockProvider("fb1", fail=fb1_err)
        fb2 = MockProvider("fb2")
        provider = FallbackProvider(primary, [fb1, fb2])

        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.choices[0].message.content == "from fb2"
        assert primary.call_count == 1
        assert fb1.call_count == 1
        assert fb2.call_count == 1


class TestFallbackStream:
    @pytest.mark.asyncio
    async def test_primary_stream_succeeds(self):
        primary = MockProvider("primary")
        fb = MockProvider("fallback")
        provider = FallbackProvider(primary, [fb])

        chunks = []
        async for chunk in provider.stream_raw(
            [{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "from primary"
        assert fb.call_count == 0

    @pytest.mark.asyncio
    async def test_stream_fallback_on_connection_error(self):
        err = LLMError("connection refused")
        err.__cause__ = ConnectionError("refused")
        primary = MockProvider("primary", fail=err)
        fb = MockProvider("fallback")
        provider = FallbackProvider(primary, [fb])

        chunks = []
        async for chunk in provider.stream_raw(
            [{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "from fallback"

    @pytest.mark.asyncio
    async def test_stream_non_retriable_no_fallback(self):
        err = LLMError("bad request")
        err.__cause__ = _FakeStatusError(400)
        primary = MockProvider("primary", fail=err)
        fb = MockProvider("fallback")
        provider = FallbackProvider(primary, [fb])

        with pytest.raises(LLMError, match="bad request"):
            async for _ in provider.stream_raw(
                [{"role": "user", "content": "hi"}]
            ):
                pass
        assert fb.call_count == 0


class TestFallbackProviderName:
    def test_provider_name_format(self):
        primary = MockProvider("main-model")
        provider = FallbackProvider(primary, [MockProvider("fb1"), MockProvider("fb2")])
        assert provider.provider_name == "main-model+2 fallback(s)"


class TestConfigIntegration:
    def test_no_fallback_returns_single_provider(self):
        config = LLMConfig(provider="openai", name="test")
        provider = create_provider(config)
        assert not isinstance(provider, FallbackProvider)

    def test_fallback_config_creates_fallback_provider(self):
        config = LLMConfig(
            provider="openai",
            name="primary-model",
            fallback=[
                FallbackLLMConfig(name="fallback-model"),
                FallbackLLMConfig(
                    provider="openai",
                    endpoint="http://other:8080/v1",
                    name="other-model",
                ),
            ],
        )
        provider = create_provider(config)
        assert isinstance(provider, FallbackProvider)
        assert len(provider._chain) == 3

    def test_fallback_inherits_parent_fields(self):
        config = LLMConfig(
            provider="openai",
            name="primary",
            temperature=0.5,
            max_tokens=2048,
            fallback=[FallbackLLMConfig(name="fallback-only-name")],
        )
        provider = create_provider(config)
        assert isinstance(provider, FallbackProvider)
        fb = provider._fallbacks[0]
        assert fb._config.temperature == 0.5
        assert fb._config.max_tokens == 2048
        assert fb._config.name == "fallback-only-name"
        assert fb._config.provider == "openai"

    def test_fallback_overrides_parent_fields(self):
        config = LLMConfig(
            provider="openai",
            name="primary",
            temperature=0.7,
            fallback=[
                FallbackLLMConfig(
                    name="fb", temperature=0.1, max_tokens=512
                ),
            ],
        )
        provider = create_provider(config)
        fb = provider._fallbacks[0]
        assert fb._config.temperature == 0.1
        assert fb._config.max_tokens == 512
