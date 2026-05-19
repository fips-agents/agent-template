"""Tests for event source/sink config models and ServerConfig integration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fipsagents.baseagent.config import (
    AgentConfig,
    CronSourceConfig,
    EventRetryConfig,
    HttpCallbackSinkConfig,
    LogSinkConfig,
    NullSinkConfig,
    NullSourceConfig,
    ServerConfig,
    WebhookSourceConfig,
    load_config_from_string,
)


class TestEventRetryConfig:
    def test_defaults(self):
        cfg = EventRetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.backoff_base == 2.0
        assert cfg.backoff_max == 60.0
        assert "TimeoutError" in cfg.retriable_errors

    def test_custom(self):
        cfg = EventRetryConfig(max_attempts=5, backoff_base=1.5)
        assert cfg.max_attempts == 5

    def test_max_attempts_ge_1(self):
        with pytest.raises(ValidationError):
            EventRetryConfig(max_attempts=0)


class TestWebhookSourceConfig:
    def test_minimal(self):
        cfg = WebhookSourceConfig(type="webhook", path="/hooks/github")
        assert cfg.type == "webhook"
        assert cfg.path == "/hooks/github"
        assert cfg.source_id is None
        assert cfg.secret is None
        assert cfg.session_ttl_hours == 168

    def test_empty_string_coercion(self):
        cfg = WebhookSourceConfig(
            type="webhook", path="/hooks/x", source_id="", secret="  ",
        )
        assert cfg.source_id is None
        assert cfg.secret is None

    def test_with_all_fields(self):
        cfg = WebhookSourceConfig(
            type="webhook",
            path="/hooks/gitlab",
            source_id="gitlab-main",
            secret="s3cr3t",
            event_type_header="X-Gitlab-Event",
            signature_header="X-Gitlab-Token",
            session_ttl_hours=24,
            max_events_per_second=5.0,
        )
        assert cfg.source_id == "gitlab-main"
        assert cfg.secret == "s3cr3t"


class TestCronSourceConfig:
    def test_minimal(self):
        cfg = CronSourceConfig(
            type="cron", schedule="0 9 * * *", event_type="daily_check",
        )
        assert cfg.type == "cron"
        assert cfg.schedule == "0 9 * * *"
        assert cfg.event_type == "daily_check"
        assert cfg.source_id is None

    def test_empty_source_id_coercion(self):
        cfg = CronSourceConfig(
            type="cron", schedule="*/5 * * * *",
            event_type="tick", source_id="",
        )
        assert cfg.source_id is None


class TestNullSourceConfig:
    def test_construct(self):
        cfg = NullSourceConfig(type="null")
        assert cfg.type == "null"
        assert cfg.source_id == "null"


class TestNullSinkConfig:
    def test_construct(self):
        cfg = NullSinkConfig(type="null")
        assert cfg.type == "null"


class TestLogSinkConfig:
    def test_defaults(self):
        cfg = LogSinkConfig(type="log")
        assert cfg.level == "INFO"

    def test_custom_level(self):
        cfg = LogSinkConfig(type="log", level="DEBUG")
        assert cfg.level == "DEBUG"


class TestHttpCallbackSinkConfig:
    def test_minimal(self):
        cfg = HttpCallbackSinkConfig(
            type="http_callback", url="https://example.com/callback",
        )
        assert cfg.url == "https://example.com/callback"
        assert cfg.timeout_seconds == 30.0

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            HttpCallbackSinkConfig(
                type="http_callback", url="https://x.com", timeout_seconds=0,
            )


class TestServerConfigBackwardCompat:
    """ServerConfig with no event fields must remain backward compatible."""

    def test_empty_event_sources(self):
        cfg = ServerConfig()
        assert cfg.event_sources == []
        assert cfg.event_sink is None

    def test_agent_config_round_trip(self):
        cfg = AgentConfig()
        assert cfg.server.event_sources == []
        assert cfg.server.event_sink is None


class TestDiscriminatedUnionParsing:
    """Verify discriminated union correctly selects the right config type."""

    def test_webhook_source_via_yaml(self):
        yaml_str = """
server:
  event_sources:
    - type: webhook
      path: /hooks/github
      secret: my-secret
"""
        cfg = load_config_from_string(yaml_str)
        assert len(cfg.server.event_sources) == 1
        src = cfg.server.event_sources[0]
        assert isinstance(src, WebhookSourceConfig)
        assert src.path == "/hooks/github"
        assert src.secret == "my-secret"

    def test_cron_source_via_yaml(self):
        yaml_str = """
server:
  event_sources:
    - type: cron
      schedule: "0 9 * * *"
      event_type: daily_check
"""
        cfg = load_config_from_string(yaml_str)
        assert len(cfg.server.event_sources) == 1
        src = cfg.server.event_sources[0]
        assert isinstance(src, CronSourceConfig)
        assert src.schedule == "0 9 * * *"

    def test_null_source_via_yaml(self):
        yaml_str = """
server:
  event_sources:
    - type: "null"
"""
        cfg = load_config_from_string(yaml_str)
        assert len(cfg.server.event_sources) == 1
        src = cfg.server.event_sources[0]
        assert isinstance(src, NullSourceConfig)

    def test_log_sink_via_yaml(self):
        yaml_str = """
server:
  event_sink:
    type: log
    level: DEBUG
"""
        cfg = load_config_from_string(yaml_str)
        assert isinstance(cfg.server.event_sink, LogSinkConfig)
        assert cfg.server.event_sink.level == "DEBUG"

    def test_http_callback_sink_via_yaml(self):
        yaml_str = """
server:
  event_sink:
    type: http_callback
    url: https://example.com/cb
    timeout_seconds: 15.0
"""
        cfg = load_config_from_string(yaml_str)
        assert isinstance(cfg.server.event_sink, HttpCallbackSinkConfig)
        assert cfg.server.event_sink.timeout_seconds == 15.0

    def test_null_sink_via_yaml(self):
        yaml_str = """
server:
  event_sink:
    type: "null"
"""
        cfg = load_config_from_string(yaml_str)
        assert isinstance(cfg.server.event_sink, NullSinkConfig)

    def test_multiple_sources(self):
        yaml_str = """
server:
  event_sources:
    - type: webhook
      path: /hooks/github
    - type: cron
      schedule: "*/5 * * * *"
      event_type: heartbeat
    - type: "null"
"""
        cfg = load_config_from_string(yaml_str)
        assert len(cfg.server.event_sources) == 3
        assert isinstance(cfg.server.event_sources[0], WebhookSourceConfig)
        assert isinstance(cfg.server.event_sources[1], CronSourceConfig)
        assert isinstance(cfg.server.event_sources[2], NullSourceConfig)
