"""Tests for provider-based endpoint rewrite in BaseAgent.setup().

Verifies that off-platform providers (anthropic, bedrock, azure) get their
endpoint rewritten to the adapter sidecar, while openai passes through
unchanged.
"""

from __future__ import annotations

import pytest

from fipsagents.baseagent.config import (
    AgentConfig,
    LLMConfig,
    LoopConfig,
    BackoffConfig,
    _ADAPTER_ENDPOINT,
    _OFF_PLATFORM_PROVIDERS,
)
from fipsagents.baseagent.llm import LLMClient


def _make_config(**overrides) -> AgentConfig:
    defaults = {
        "model": LLMConfig(
            endpoint="http://test:8321/v1",
            name="test-model",
            temperature=0.0,
            max_tokens=256,
        ),
        "loop": LoopConfig(
            max_iterations=5,
            backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
        ),
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


class TestProviderEndpointRewriteIntegration:
    """Verify that the endpoint rewrite logic produces correct LLMClient config.

    These tests exercise the same model_copy path that setup() uses,
    applied to real LLMConfig objects.
    """

    def test_openai_provider_preserves_endpoint(self):
        """openai provider should pass the endpoint through unchanged."""
        config = _make_config(
            model=LLMConfig(
                provider="openai",
                endpoint="http://vllm:8000/v1",
                name="test-model",
            ),
        )
        # Simulate what setup() does: no rewrite for openai.
        effective = config.model
        if config.model.provider in _OFF_PLATFORM_PROVIDERS:
            effective = config.model.model_copy(
                update={"endpoint": _ADAPTER_ENDPOINT},
            )
        client = LLMClient(effective)
        assert str(client._client.base_url) == "http://vllm:8000/v1/"

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock", "azure"])
    def test_off_platform_provider_rewrites_endpoint(self, provider):
        """Non-openai providers rewrite endpoint to adapter sidecar."""
        config = _make_config(
            model=LLMConfig(
                provider=provider,
                endpoint="http://should-be-ignored:8000/v1",
                name="claude-sonnet-4-6",
            ),
        )
        effective = config.model
        if config.model.provider in _OFF_PLATFORM_PROVIDERS:
            effective = config.model.model_copy(
                update={"endpoint": _ADAPTER_ENDPOINT},
            )
        client = LLMClient(effective)
        # The base_url should be the adapter sidecar, not the original.
        assert str(client._client.base_url) == "http://localhost:8081/v1/"
        # Original config should be unchanged.
        assert config.model.endpoint == "http://should-be-ignored:8000/v1"

    def test_off_platform_provider_with_no_endpoint(self):
        """Provider set but no endpoint — rewrite still applies."""
        config = _make_config(
            model=LLMConfig(
                provider="anthropic",
                name="claude-sonnet-4-6",
                # endpoint defaults to None
            ),
        )
        effective = config.model
        if config.model.provider in _OFF_PLATFORM_PROVIDERS:
            effective = config.model.model_copy(
                update={"endpoint": _ADAPTER_ENDPOINT},
            )
        client = LLMClient(effective)
        assert str(client._client.base_url) == "http://localhost:8081/v1/"

    def test_model_name_preserved_through_rewrite(self):
        """The model name must survive the endpoint rewrite."""
        config = _make_config(
            model=LLMConfig(
                provider="bedrock",
                endpoint="http://ignored:8000/v1",
                name="anthropic.claude-v2",
            ),
        )
        effective = config.model.model_copy(
            update={"endpoint": _ADAPTER_ENDPOINT},
        )
        assert effective.name == "anthropic.claude-v2"
        assert effective.provider == "bedrock"
        assert effective.endpoint == _ADAPTER_ENDPOINT
