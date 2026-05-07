"""Tests for SubagentConfig and its integration with AgentConfig.

Covers: transport discriminated union, validator behaviour, env-var
substitution, defaults, backward compatibility, and duplicate-name
detection on AgentConfig.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fipsagents.baseagent.config import (
    AgentConfig,
    ConfigError,
    IdentityServiceAccount,
    InProcessTransportConfig,
    RemoteTransportConfig,
    load_config_from_string,
    parse_yaml_with_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_SUBAGENTS_YAML = """\
subagents:
  - name: research_helper
    description: "Searches internal knowledge base and returns a synthesised brief."
    when_to_use: "Use when the user asks an open-ended policy question."
    transport:
      type: remote
      url: ${RESEARCH_HELPER_URL:-http://research-helper:8080/v1}
      timeout_seconds: 60
    permission_scope: research-readonly
    identity: inherit
    max_depth: 3

  - name: account_lookup
    description: "Looks up an account by ID and returns relevant fields."
    when_to_use: "Use when the user references an account by ID."
    transport:
      type: remote
      url: ${ACCOUNT_LOOKUP_URL:-http://account-lookup:8080/v1}
      timeout_seconds: 15
    permission_scope: account-read
    identity:
      service_account: account-reader
    max_depth: 1
"""


# ---------------------------------------------------------------------------
# Parsing a fully-specified subagents block
# ---------------------------------------------------------------------------


class TestFullSubagentsBlock:
    def test_parses_two_subagents(self):
        cfg = load_config_from_string(_FULL_SUBAGENTS_YAML, env={})
        assert len(cfg.subagents) == 2

    def test_first_subagent_fields(self):
        cfg = load_config_from_string(_FULL_SUBAGENTS_YAML, env={})
        sa = cfg.subagents[0]
        assert sa.name == "research_helper"
        assert "knowledge base" in sa.description
        assert "policy question" in sa.when_to_use
        assert sa.permission_scope == "research-readonly"
        assert sa.identity == "inherit"
        assert sa.max_depth == 3

    def test_first_subagent_transport(self):
        cfg = load_config_from_string(_FULL_SUBAGENTS_YAML, env={})
        t = cfg.subagents[0].transport
        assert isinstance(t, RemoteTransportConfig)
        assert t.type == "remote"
        assert t.url == "http://research-helper:8080/v1"
        assert t.timeout_seconds == 60.0

    def test_second_subagent_service_account(self):
        cfg = load_config_from_string(_FULL_SUBAGENTS_YAML, env={})
        sa = cfg.subagents[1]
        assert isinstance(sa.identity, IdentityServiceAccount)
        assert sa.identity.service_account == "account-reader"
        assert sa.max_depth == 1
        assert sa.transport.timeout_seconds == 15.0


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestSubagentDefaults:
    _MINIMAL_YAML = """\
subagents:
  - name: helper_agent
    description: "A helper."
    when_to_use: "Always."
    transport:
      type: remote
      url: http://helper:8080/v1
"""

    def test_max_depth_default(self):
        cfg = load_config_from_string(self._MINIMAL_YAML)
        assert cfg.subagents[0].max_depth == 3

    def test_identity_default(self):
        cfg = load_config_from_string(self._MINIMAL_YAML)
        assert cfg.subagents[0].identity == "inherit"

    def test_timeout_default(self):
        cfg = load_config_from_string(self._MINIMAL_YAML)
        assert cfg.subagents[0].transport.timeout_seconds == 60.0

    def test_permission_scope_default_none(self):
        cfg = load_config_from_string(self._MINIMAL_YAML)
        assert cfg.subagents[0].permission_scope is None


# ---------------------------------------------------------------------------
# Transport discriminated union
# ---------------------------------------------------------------------------


class TestTransportDiscrimination:
    def test_remote_transport_routed_correctly(self):
        cfg = load_config_from_string("""\
subagents:
  - name: remote_agent
    description: "Remote."
    when_to_use: "Always."
    transport:
      type: remote
      url: http://example.com/v1
""")
        t = cfg.subagents[0].transport
        assert isinstance(t, RemoteTransportConfig)
        assert t.type == "remote"

    def test_inprocess_transport_routed_correctly(self):
        cfg = load_config_from_string("""\
subagents:
  - name: inproc_agent
    description: "In process."
    when_to_use: "Never remote."
    transport:
      type: inprocess
      class_path: mypackage.agents.MyAgent
""")
        t = cfg.subagents[0].transport
        assert isinstance(t, InProcessTransportConfig)
        assert t.type == "inprocess"
        assert t.class_path == "mypackage.agents.MyAgent"
        assert t.config_path is None

    def test_inprocess_config_path_optional(self):
        cfg = load_config_from_string("""\
subagents:
  - name: inproc_agent
    description: "In process with config."
    when_to_use: "Use it."
    transport:
      type: inprocess
      class_path: mypackage.agents.MyAgent
      config_path: ./subagent.yaml
""")
        t = cfg.subagents[0].transport
        assert isinstance(t, InProcessTransportConfig)
        assert t.config_path == "./subagent.yaml"

    def test_unknown_transport_type_raises(self):
        with pytest.raises((ValidationError, ConfigError)):
            load_config_from_string("""\
subagents:
  - name: bad_agent
    description: "Bad transport."
    when_to_use: "Never."
    transport:
      type: grpc
      url: grpc://example.com
""")


# ---------------------------------------------------------------------------
# Validator: inprocess + service_account is forbidden
# ---------------------------------------------------------------------------


class TestInprocessIdentityForbidden:
    def test_inprocess_with_service_account_raises(self):
        yaml_str = """\
subagents:
  - name: inproc_agent
    description: "In process."
    when_to_use: "Use it."
    transport:
      type: inprocess
      class_path: mypackage.agents.MyAgent
    identity:
      service_account: my-sa
"""
        with pytest.raises((ValidationError, ConfigError)) as exc_info:
            load_config_from_string(yaml_str)
        msg = str(exc_info.value)
        assert "inprocess" in msg
        assert "service_account" in msg

    def test_inprocess_with_inherit_is_allowed(self):
        cfg = load_config_from_string("""\
subagents:
  - name: inproc_agent
    description: "In process."
    when_to_use: "Use it."
    transport:
      type: inprocess
      class_path: mypackage.agents.MyAgent
    identity: inherit
""")
        assert cfg.subagents[0].identity == "inherit"

    def test_remote_with_service_account_is_allowed(self):
        cfg = load_config_from_string("""\
subagents:
  - name: remote_agent
    description: "Remote."
    when_to_use: "Use it."
    transport:
      type: remote
      url: http://example.com/v1
    identity:
      service_account: my-sa
""")
        assert isinstance(cfg.subagents[0].identity, IdentityServiceAccount)


# ---------------------------------------------------------------------------
# Validator: empty transport.url
# ---------------------------------------------------------------------------


class TestEmptyTransportUrl:
    def test_empty_url_raises_helpful_error(self):
        with pytest.raises((ValidationError, ConfigError)) as exc_info:
            load_config_from_string("""\
subagents:
  - name: bad_agent
    description: "Bad url."
    when_to_use: "Use it."
    transport:
      type: remote
      url: ""
""")
        msg = str(exc_info.value)
        assert "transport.url" in msg

    def test_whitespace_only_url_raises(self):
        with pytest.raises((ValidationError, ConfigError)):
            load_config_from_string("""\
subagents:
  - name: bad_agent
    description: "Bad url."
    when_to_use: "Use it."
    transport:
      type: remote
      url: "   "
""")

    def test_empty_url_via_env_substitution_raises(self, monkeypatch):
        """${UNSET_VAR:-} produces an empty string after substitution — must fail."""
        with pytest.raises((ValidationError, ConfigError)) as exc_info:
            load_config_from_string(
                """\
subagents:
  - name: bad_agent
    description: "Missing url."
    when_to_use: "Use it."
    transport:
      type: remote
      url: ${UNSET_SUBAGENT_URL:-}
""",
                env={},
            )
        msg = str(exc_info.value)
        assert "transport.url" in msg


# ---------------------------------------------------------------------------
# Validator: invalid name format
# ---------------------------------------------------------------------------


class TestSubagentNameValidation:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "1starts_with_digit",
            "has spaces",
            "has-hyphen",
            "has.dot",
            "",
        ],
    )
    def test_invalid_names_raise(self, bad_name):
        yaml_str = f"""\
subagents:
  - name: "{bad_name}"
    description: "Bad name."
    when_to_use: "Use it."
    transport:
      type: remote
      url: http://example.com/v1
"""
        with pytest.raises((ValidationError, ConfigError)):
            load_config_from_string(yaml_str)

    @pytest.mark.parametrize(
        "good_name",
        [
            "helper",
            "helper_agent",
            "researchHelper",
            "Agent2",
            "a",
        ],
    )
    def test_valid_names_accepted(self, good_name):
        yaml_str = f"""\
subagents:
  - name: "{good_name}"
    description: "Good name."
    when_to_use: "Use it."
    transport:
      type: remote
      url: http://example.com/v1
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.subagents[0].name == good_name


# ---------------------------------------------------------------------------
# AgentConfig validator: duplicate subagent names
# ---------------------------------------------------------------------------


class TestDuplicateSubagentNames:
    def test_duplicate_names_raise_with_offending_name(self):
        yaml_str = """\
subagents:
  - name: helper
    description: "First."
    when_to_use: "Use it."
    transport:
      type: remote
      url: http://example.com/v1
  - name: helper
    description: "Duplicate."
    when_to_use: "Use it."
    transport:
      type: remote
      url: http://other.com/v1
"""
        with pytest.raises((ValidationError, ConfigError)) as exc_info:
            load_config_from_string(yaml_str)
        msg = str(exc_info.value)
        assert "helper" in msg

    def test_two_different_names_ok(self):
        cfg = load_config_from_string("""\
subagents:
  - name: agent_a
    description: "A."
    when_to_use: "Use A."
    transport:
      type: remote
      url: http://a.example.com/v1
  - name: agent_b
    description: "B."
    when_to_use: "Use B."
    transport:
      type: remote
      url: http://b.example.com/v1
""")
        assert len(cfg.subagents) == 2


# ---------------------------------------------------------------------------
# Env-var substitution through to subagent fields
# ---------------------------------------------------------------------------


class TestEnvVarSubstitution:
    def test_url_resolved_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_SUBAGENT_URL", "http://resolved:9000/v1")
        cfg = load_config_from_string("""\
subagents:
  - name: env_agent
    description: "Env URL."
    when_to_use: "Use it."
    transport:
      type: remote
      url: ${TEST_SUBAGENT_URL}
""")
        assert cfg.subagents[0].transport.url == "http://resolved:9000/v1"

    def test_url_uses_default_when_env_unset(self):
        cfg = load_config_from_string(
            """\
subagents:
  - name: env_agent
    description: "Env URL with default."
    when_to_use: "Use it."
    transport:
      type: remote
      url: ${TOTALLY_UNSET_SUBAGENT_URL:-http://fallback:8080/v1}
""",
            env={},
        )
        assert cfg.subagents[0].transport.url == "http://fallback:8080/v1"

    def test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("SA_TIMEOUT", "120")
        cfg = load_config_from_string("""\
subagents:
  - name: timed_agent
    description: "Timeout from env."
    when_to_use: "Use it."
    transport:
      type: remote
      url: http://example.com/v1
      timeout_seconds: ${SA_TIMEOUT:-60}
""")
        assert cfg.subagents[0].transport.timeout_seconds == 120.0

    def test_end_to_end_parse_yaml_with_env(self):
        """parse_yaml_with_env resolves placeholders before Pydantic sees them."""
        raw = """\
subagents:
  - name: direct_agent
    description: "Direct test."
    when_to_use: "Use it."
    transport:
      type: remote
      url: ${MY_DIRECT_URL:-http://direct:8080/v1}
"""
        data = parse_yaml_with_env(raw, env={"MY_DIRECT_URL": "http://injected:9999/v1"})
        cfg = AgentConfig.model_validate(data)
        assert cfg.subagents[0].transport.url == "http://injected:9999/v1"


# ---------------------------------------------------------------------------
# Backward compatibility: no subagents key
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_missing_subagents_key_yields_empty_list(self):
        cfg = load_config_from_string("")
        assert cfg.subagents == []

    def test_existing_agent_yaml_unaffected(self):
        """A typical agent.yaml without subagents: still parses cleanly."""
        raw = """\
model:
  name: meta-llama/Llama-3.3-70B-Instruct
  temperature: 0.7
loop:
  max_iterations: 50
logging:
  level: INFO
"""
        cfg = load_config_from_string(raw)
        assert cfg.subagents == []
        assert cfg.model.name == "meta-llama/Llama-3.3-70B-Instruct"
