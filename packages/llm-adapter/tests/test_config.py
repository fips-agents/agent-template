"""Tests for AdapterConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_adapter.config import AdapterConfig


class TestAdapterConfigDefaults:
    def test_defaults(self):
        cfg = AdapterConfig()
        assert cfg.provider == "anthropic"
        assert cfg.port == 8081
        assert cfg.log_level == "INFO"


class TestAdapterConfigFromEnv:
    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("ADAPTER_PROVIDER", "bedrock")
        cfg = AdapterConfig.from_env()
        assert cfg.provider == "bedrock"

    def test_port_from_env(self, monkeypatch):
        monkeypatch.setenv("ADAPTER_PORT", "9090")
        cfg = AdapterConfig.from_env()
        assert cfg.port == 9090

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "debug")
        cfg = AdapterConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_from_env_uses_defaults_when_unset(self, monkeypatch):
        for var in ("ADAPTER_PROVIDER", "ADAPTER_PORT", "LOG_LEVEL"):
            monkeypatch.delenv(var, raising=False)
        cfg = AdapterConfig.from_env()
        assert cfg.provider == "anthropic"
        assert cfg.port == 8081
        assert cfg.log_level == "INFO"


class TestLogLevelValidation:
    def test_invalid_level_raises(self):
        with pytest.raises(ValidationError, match="log_level"):
            AdapterConfig(log_level="TRACE")

    def test_case_insensitive(self):
        cfg = AdapterConfig(log_level="warning")
        assert cfg.log_level == "WARNING"

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_all_valid_levels(self, level):
        cfg = AdapterConfig(log_level=level)
        assert cfg.log_level == level
