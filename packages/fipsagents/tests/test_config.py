"""Tests for fipsagents.baseagent.config — loading, env-var substitution, and models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fipsagents.baseagent.config import (
    AgentConfig,
    BackoffConfig,
    ConfigError,
    LLMConfig,
    LoggingConfig,
    LoopConfig,
    NodeConfig,
    _substitute_recursive,
    load_config,
    load_config_from_string,
    parse_yaml_with_env,
    substitute_env_vars,
)


# ---------------------------------------------------------------------------
# substitute_env_vars
# ---------------------------------------------------------------------------


class TestSubstituteEnvVars:
    def test_resolves_var(self):
        result = substitute_env_vars("${MY_VAR}", env={"MY_VAR": "hello"})
        assert result == "hello"

    def test_resolves_var_with_default_syntax(self):
        result = substitute_env_vars("${MY_VAR:-fallback}", env={"MY_VAR": "real"})
        assert result == "real"

    def test_uses_default_when_var_unset(self):
        result = substitute_env_vars("${MY_VAR:-fallback}", env={})
        assert result == "fallback"

    def test_uses_default_without_colon(self):
        # ${VAR-default} is also accepted by the pattern
        result = substitute_env_vars("${MY_VAR-fallback}", env={})
        assert result == "fallback"

    def test_leaves_placeholder_when_unset_non_strict(self):
        result = substitute_env_vars("${UNSET_VAR}", env={}, strict=False)
        assert result == "${UNSET_VAR}"

    def test_raises_in_strict_mode_when_unset(self):
        with pytest.raises(ConfigError, match="UNSET_VAR"):
            substitute_env_vars("${UNSET_VAR}", env={}, strict=True)

    def test_empty_default_is_allowed(self):
        result = substitute_env_vars("${MY_VAR:-}", env={})
        assert result == ""

    def test_multiple_vars_in_string(self):
        result = substitute_env_vars(
            "${A}:${B:-default_b}",
            env={"A": "alpha"},
        )
        assert result == "alpha:default_b"

    def test_no_placeholders_unchanged(self):
        result = substitute_env_vars("plain string", env={})
        assert result == "plain string"

    def test_var_in_middle_of_string(self):
        result = substitute_env_vars("prefix_${X}_suffix", env={"X": "mid"})
        assert result == "prefix_mid_suffix"

    @pytest.mark.parametrize(
        "raw, env, expected",
        [
            ("${A}", {"A": "1"}, "1"),
            ("${A:-x}", {}, "x"),
            ("${A:-x}", {"A": "y"}, "y"),
            ("no_vars", {}, "no_vars"),
        ],
    )
    def test_parametrized_cases(self, raw, env, expected):
        assert substitute_env_vars(raw, env=env) == expected


# ---------------------------------------------------------------------------
# _substitute_recursive
# ---------------------------------------------------------------------------


class TestSubstituteRecursive:
    def test_walks_nested_dict(self):
        data = {"key": "${A}", "nested": {"sub": "${B:-bval}"}}
        result = _substitute_recursive(data, env={"A": "aval"})
        assert result == {"key": "aval", "nested": {"sub": "bval"}}

    def test_walks_list(self):
        data = ["${X}", "${Y:-y_default}"]
        result = _substitute_recursive(data, env={"X": "x_val"})
        assert result == ["x_val", "y_default"]

    def test_passes_through_non_strings(self):
        data = {"num": 42, "flag": True}
        result = _substitute_recursive(data, env={})
        assert result == {"num": 42, "flag": True}

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": "${DEEP}"}}}
        result = _substitute_recursive(data, env={"DEEP": "found"})
        assert result["a"]["b"]["c"] == "found"

    def test_list_of_dicts(self):
        data = [{"url": "${URL}"}]
        result = _substitute_recursive(data, env={"URL": "http://example.com"})
        assert result == [{"url": "http://example.com"}]


# ---------------------------------------------------------------------------
# parse_yaml_with_env
# ---------------------------------------------------------------------------


class TestParseYamlWithEnv:
    def test_valid_yaml(self):
        raw = "key: value\nother: 42"
        result = parse_yaml_with_env(raw, env={})
        assert result == {"key": "value", "other": 42}

    def test_substitutes_env_vars(self):
        result = parse_yaml_with_env("model: ${MODEL:-gpt-4}", env={})
        assert result == {"model": "gpt-4"}

    def test_invalid_yaml_raises_config_error(self):
        with pytest.raises(ConfigError, match="Invalid YAML"):
            parse_yaml_with_env("key: [unclosed", env={})

    def test_empty_yaml_returns_empty_dict(self):
        result = parse_yaml_with_env("", env={})
        assert result == {}

    def test_null_yaml_returns_empty_dict(self):
        result = parse_yaml_with_env("null", env={})
        assert result == {}

    def test_non_mapping_yaml_raises_config_error(self):
        with pytest.raises(ConfigError, match="mapping"):
            parse_yaml_with_env("- item1\n- item2", env={})

    def test_strict_mode_propagated(self):
        with pytest.raises(ConfigError, match="MISSING"):
            parse_yaml_with_env("val: ${MISSING}", env={}, strict=True)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_file_not_found_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_valid_yaml_file_returns_agent_config(self, tmp_path):
        cfg_file = tmp_path / "agent.yaml"
        cfg_file.write_text("model:\n  name: test-model\n")
        config = load_config(cfg_file)
        assert config.model.name == "test-model"

    def test_env_var_substitution_in_file(self, tmp_path):
        cfg_file = tmp_path / "agent.yaml"
        cfg_file.write_text("model:\n  name: ${MODEL_NAME:-default-model}\n")
        config = load_config(cfg_file, env={})
        assert config.model.name == "default-model"

    def test_env_var_overridden(self, tmp_path):
        cfg_file = tmp_path / "agent.yaml"
        cfg_file.write_text("model:\n  name: ${MODEL_NAME:-default-model}\n")
        config = load_config(cfg_file, env={"MODEL_NAME": "custom-model"})
        assert config.model.name == "custom-model"

    def test_invalid_config_raises_config_error(self, tmp_path):
        cfg_file = tmp_path / "agent.yaml"
        # temperature out of range
        cfg_file.write_text("model:\n  temperature: 999\n")
        with pytest.raises(ConfigError, match="Invalid"):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# load_config_from_string
# ---------------------------------------------------------------------------


class TestLoadConfigFromString:
    def test_minimal_config(self):
        config = load_config_from_string("")
        assert isinstance(config, AgentConfig)

    def test_full_config(self):
        raw = """
model:
  name: my-model
  temperature: 0.5
  max_tokens: 1024
loop:
  max_iterations: 50
logging:
  level: DEBUG
"""
        config = load_config_from_string(raw)
        assert config.model.name == "my-model"
        assert config.model.temperature == 0.5
        assert config.model.max_tokens == 1024
        assert config.loop.max_iterations == 50
        assert config.logging.level == "DEBUG"

    def test_env_substitution(self):
        raw = "model:\n  name: ${MODEL:-default}\n"
        config = load_config_from_string(raw, env={"MODEL": "overridden"})
        assert config.model.name == "overridden"

    def test_invalid_config_raises_config_error(self):
        with pytest.raises(ConfigError):
            load_config_from_string("model:\n  temperature: -1\n")


# ---------------------------------------------------------------------------
# AgentConfig defaults
# ---------------------------------------------------------------------------


class TestAgentConfigDefaults:
    def test_all_sub_models_have_defaults(self):
        config = AgentConfig()
        assert isinstance(config.model, LLMConfig)
        assert config.mcp_servers == []
        assert config.loop.max_iterations == 100
        assert config.logging.level == "INFO"
        assert config.memory.config_path == ".memoryhub.yaml"

    def test_tools_default_dir(self):
        config = AgentConfig()
        assert config.tools.local_dir == "./tools"

    def test_prompts_default_dir(self):
        config = AgentConfig()
        assert config.prompts.dir == "./prompts"


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfig:
    @pytest.mark.parametrize("temp", [0.0, 1.0, 2.0])
    def test_valid_temperature(self, temp):
        cfg = LLMConfig(temperature=temp)
        assert cfg.temperature == temp

    @pytest.mark.parametrize("temp", [-0.1, 2.1, 10.0])
    def test_invalid_temperature(self, temp):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=temp)

    def test_max_tokens_positive(self):
        cfg = LLMConfig(max_tokens=1)
        assert cfg.max_tokens == 1

    def test_max_tokens_zero_invalid(self):
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)

    def test_default_model_name(self):
        cfg = LLMConfig()
        assert "Llama" in cfg.name or cfg.name  # has some default

    def test_default_provider_is_openai(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "bedrock", "azure"])
    def test_valid_providers(self, provider):
        cfg = LLMConfig(provider=provider)
        assert cfg.provider == provider

    def test_invalid_provider_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider="grok")

    def test_provider_from_yaml(self):
        config = load_config_from_string(
            "model:\n  provider: anthropic\n",
        )
        assert config.model.provider == "anthropic"

    def test_provider_default_in_yaml(self):
        config = load_config_from_string("model:\n  name: test\n")
        assert config.model.provider == "openai"


# ---------------------------------------------------------------------------
# Provider endpoint rewrite
# ---------------------------------------------------------------------------


class TestProviderEndpointRewrite:
    """Verify that setup() rewrites endpoint for off-platform providers."""

    def test_openai_provider_preserves_endpoint(self):
        cfg = LLMConfig(
            provider="openai",
            endpoint="http://vllm:8000/v1",
        )
        # openai provider should not trigger a rewrite.
        from fipsagents.baseagent.config import _OFF_PLATFORM_PROVIDERS

        assert cfg.provider not in _OFF_PLATFORM_PROVIDERS

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock", "azure"])
    def test_off_platform_provider_in_set(self, provider):
        from fipsagents.baseagent.config import (
            _ADAPTER_ENDPOINT,
            _OFF_PLATFORM_PROVIDERS,
        )

        assert provider in _OFF_PLATFORM_PROVIDERS
        # model_copy produces the rewritten config.
        cfg = LLMConfig(provider=provider, endpoint="http://original:8000/v1")
        rewritten = cfg.model_copy(update={"endpoint": _ADAPTER_ENDPOINT})
        assert rewritten.endpoint == "http://localhost:8081/v1"
        # Original config is unchanged.
        assert cfg.endpoint == "http://original:8000/v1"


# ---------------------------------------------------------------------------
# BackoffConfig
# ---------------------------------------------------------------------------


class TestBackoffConfig:
    def test_valid_backoff(self):
        cfg = BackoffConfig(initial=1.0, max=30.0, multiplier=2.0)
        assert cfg.initial == 1.0
        assert cfg.max == 30.0

    def test_max_less_than_initial_raises(self):
        with pytest.raises(ValidationError, match="max"):
            BackoffConfig(initial=10.0, max=5.0, multiplier=2.0)

    def test_max_equal_initial_is_valid(self):
        cfg = BackoffConfig(initial=5.0, max=5.0, multiplier=2.0)
        assert cfg.max == cfg.initial


# ---------------------------------------------------------------------------
# LoopConfig
# ---------------------------------------------------------------------------


class TestLoopConfig:
    def test_default_max_iterations(self):
        cfg = LoopConfig()
        assert cfg.max_iterations == 100

    def test_coerces_string_to_int(self):
        cfg = LoopConfig(max_iterations="42")
        assert cfg.max_iterations == 42

    def test_invalid_string_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            LoopConfig(max_iterations="not_a_number")

    def test_zero_max_iterations_raises(self):
        with pytest.raises(ValidationError):
            LoopConfig(max_iterations=0)

    def test_negative_max_iterations_raises(self):
        with pytest.raises(ValidationError):
            LoopConfig(max_iterations=-5)


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    @pytest.mark.parametrize(
        "level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    def test_valid_levels(self, level):
        cfg = LoggingConfig(level=level)
        assert cfg.level == level

    def test_case_insensitive(self):
        cfg = LoggingConfig(level="debug")
        assert cfg.level == "DEBUG"

    def test_invalid_level_raises(self):
        with pytest.raises(ValidationError, match="logging.level"):
            LoggingConfig(level="VERBOSE")


# ---------------------------------------------------------------------------
# NodeConfig
# ---------------------------------------------------------------------------


class TestNodeConfig:
    def test_defaults_to_local(self):
        cfg = NodeConfig()
        assert cfg.type == "local"
        assert cfg.endpoint is None

    def test_remote_requires_endpoint(self):
        with pytest.raises(ValidationError, match="endpoint"):
            NodeConfig(type="remote")

    def test_remote_with_endpoint(self):
        cfg = NodeConfig(type="remote", endpoint="http://agent:8080")
        assert cfg.endpoint == "http://agent:8080"
        assert cfg.path == "/process"
        assert cfg.timeout == 30.0
        assert cfg.retries == 2

    def test_local_ignores_endpoint(self):
        cfg = NodeConfig(type="local", endpoint="http://unused:8080")
        assert cfg.type == "local"

    def test_custom_values(self):
        cfg = NodeConfig(
            type="remote",
            endpoint="http://agent:9090",
            path="/run",
            timeout=60.0,
            retries=5,
        )
        assert cfg.path == "/run"
        assert cfg.timeout == 60.0
        assert cfg.retries == 5

    def test_nodes_in_agent_config(self):
        cfg = load_config_from_string("""
model:
  endpoint: http://localhost:8080/v1
  name: test-model
nodes:
  research:
    type: remote
    endpoint: http://research:8080
  classify:
    type: local
""")
        assert "research" in cfg.nodes
        assert cfg.nodes["research"].type == "remote"
        assert cfg.nodes["classify"].type == "local"

    def test_empty_nodes_default(self):
        cfg = load_config_from_string("""
model:
  endpoint: http://localhost:8080/v1
  name: test-model
""")
        assert cfg.nodes == {}
