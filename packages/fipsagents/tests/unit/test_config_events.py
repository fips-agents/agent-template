"""Tests for event source/sink config models and ServerConfig integration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fipsagents.baseagent.config import (
    AgentConfig,
    CronSourceConfig,
    EventRetryConfig,
    HttpCallbackSinkConfig,
    KafkaSinkConfig,
    KafkaSourceConfig,
    LogSinkConfig,
    NullSinkConfig,
    NullSourceConfig,
    RedisSinkConfig,
    RedisSourceConfig,
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

    def test_kafka_source_via_yaml(self):
        yaml_str = """
server:
  event_sources:
    - type: kafka
      bootstrap_servers: "localhost:9092"
      topic: my-events
      consumer_group: my-agent
"""
        cfg = load_config_from_string(yaml_str)
        assert len(cfg.server.event_sources) == 1
        src = cfg.server.event_sources[0]
        assert isinstance(src, KafkaSourceConfig)
        assert src.bootstrap_servers == "localhost:9092"
        assert src.topic == "my-events"
        assert src.consumer_group == "my-agent"
        assert src.auto_offset_reset == "latest"
        assert src.source_id is None

    def test_kafka_source_with_sasl(self):
        yaml_str = """
server:
  event_sources:
    - type: kafka
      bootstrap_servers: "broker:9093"
      topic: secure-events
      consumer_group: agent-group
      security_protocol: SASL_SSL
      sasl_mechanism: PLAIN
      sasl_username: user1
      sasl_password: secret
"""
        cfg = load_config_from_string(yaml_str)
        src = cfg.server.event_sources[0]
        assert isinstance(src, KafkaSourceConfig)
        assert src.security_protocol == "SASL_SSL"
        assert src.sasl_mechanism == "PLAIN"
        assert src.sasl_username == "user1"
        assert src.sasl_password == "secret"

    def test_kafka_source_empty_source_id_coercion(self):
        cfg = KafkaSourceConfig(
            type="kafka",
            bootstrap_servers="localhost:9092",
            topic="t",
            consumer_group="g",
            source_id="",
        )
        assert cfg.source_id is None

    def test_redis_source_via_yaml(self):
        yaml_str = """
server:
  event_sources:
    - type: redis
      url: "redis://localhost:6379"
      stream: events
      consumer_group: my-agent
"""
        cfg = load_config_from_string(yaml_str)
        assert len(cfg.server.event_sources) == 1
        src = cfg.server.event_sources[0]
        assert isinstance(src, RedisSourceConfig)
        assert src.url == "redis://localhost:6379"
        assert src.stream == "events"
        assert src.consumer_group == "my-agent"
        assert src.consumer_name == "worker-0"
        assert src.block_ms == 5000

    def test_redis_source_custom_fields(self):
        yaml_str = """
server:
  event_sources:
    - type: redis
      url: "redis://redis:6379"
      stream: jobs
      consumer_group: workers
      consumer_name: worker-3
      block_ms: 10000
      session_ttl_hours: 48
"""
        cfg = load_config_from_string(yaml_str)
        src = cfg.server.event_sources[0]
        assert isinstance(src, RedisSourceConfig)
        assert src.consumer_name == "worker-3"
        assert src.block_ms == 10000
        assert src.session_ttl_hours == 48

    def test_redis_source_empty_source_id_coercion(self):
        cfg = RedisSourceConfig(
            type="redis",
            url="redis://localhost",
            stream="s",
            consumer_group="g",
            source_id="  ",
        )
        assert cfg.source_id is None

    def test_kafka_sink_via_yaml(self):
        yaml_str = """
server:
  event_sink:
    type: kafka
    bootstrap_servers: "localhost:9092"
    topic: results
"""
        cfg = load_config_from_string(yaml_str)
        assert isinstance(cfg.server.event_sink, KafkaSinkConfig)
        assert cfg.server.event_sink.bootstrap_servers == "localhost:9092"
        assert cfg.server.event_sink.topic == "results"
        assert cfg.server.event_sink.security_protocol is None

    def test_kafka_sink_with_sasl(self):
        yaml_str = """
server:
  event_sink:
    type: kafka
    bootstrap_servers: "broker:9093"
    topic: output
    security_protocol: SASL_SSL
    sasl_mechanism: SCRAM-SHA-256
    sasl_username: producer
    sasl_password: pw
"""
        cfg = load_config_from_string(yaml_str)
        sink = cfg.server.event_sink
        assert isinstance(sink, KafkaSinkConfig)
        assert sink.security_protocol == "SASL_SSL"
        assert sink.sasl_mechanism == "SCRAM-SHA-256"

    def test_redis_sink_via_yaml(self):
        yaml_str = """
server:
  event_sink:
    type: redis
    url: "redis://localhost:6379"
    stream: results
"""
        cfg = load_config_from_string(yaml_str)
        assert isinstance(cfg.server.event_sink, RedisSinkConfig)
        assert cfg.server.event_sink.url == "redis://localhost:6379"
        assert cfg.server.event_sink.stream == "results"
        assert cfg.server.event_sink.maxlen is None

    def test_redis_sink_with_maxlen(self):
        yaml_str = """
server:
  event_sink:
    type: redis
    url: "redis://localhost"
    stream: bounded
    maxlen: 50000
"""
        cfg = load_config_from_string(yaml_str)
        sink = cfg.server.event_sink
        assert isinstance(sink, RedisSinkConfig)
        assert sink.maxlen == 50000

    def test_mixed_kafka_redis_sources(self):
        yaml_str = """
server:
  event_sources:
    - type: kafka
      bootstrap_servers: "localhost:9092"
      topic: topic-a
      consumer_group: grp-a
    - type: redis
      url: "redis://localhost"
      stream: stream-b
      consumer_group: grp-b
    - type: webhook
      path: /hooks/gh
"""
        cfg = load_config_from_string(yaml_str)
        assert len(cfg.server.event_sources) == 3
        assert isinstance(cfg.server.event_sources[0], KafkaSourceConfig)
        assert isinstance(cfg.server.event_sources[1], RedisSourceConfig)
        assert isinstance(cfg.server.event_sources[2], WebhookSourceConfig)
