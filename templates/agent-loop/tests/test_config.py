"""Tests for fipsagents.baseagent.config — env var substitution, YAML parsing, validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fipsagents.baseagent.config import (
    AgentConfig,
    BackoffConfig,
    ConfigError,
    LLMConfig,
    LoggingConfig,
    LoopConfig,
    McpServerConfig,
    MemoryConfig,
    ToolsConfig,
    load_config,
    load_config_from_string,
    parse_yaml_with_env,
    substitute_env_vars,
)

# ── Environment variable substitution ────────────────────────────────────


class TestSubstituteEnvVars:
    """Unit tests for the low-level ``substitute_env_vars`` function."""

    @pytest.mark.parametrize(
        "template, env, expected",
        [
            # Variable with default, env set → use env value
            ("${FOO:-fallback}", {"FOO": "bar"}, "bar"),
            # Variable with default, env not set → use default
            ("${FOO:-fallback}", {}, "fallback"),
            # Variable without default, env set → use env value
            ("${FOO}", {"FOO": "bar"}, "bar"),
            # Empty default is still a valid default
            ("${FOO:-}", {}, ""),
            # Multiple substitutions in one string
            (
                "${HOST:-localhost}:${PORT:-8080}",
                {"HOST": "example.com"},
                "example.com:8080",
            ),
            # No placeholders at all
            ("plain string", {}, "plain string"),
            # Default contains special characters
            (
                "${URL:-http://localhost:8321/v1}",
                {},
                "http://localhost:8321/v1",
            ),
            # Dash-only syntax (${VAR-default} without colon)
            ("${FOO-fallback}", {}, "fallback"),
            # Env value overrides even with dash-only syntax
            ("${FOO-fallback}", {"FOO": "bar"}, "bar"),
        ],
        ids=[
            "env-overrides-default",
            "falls-back-to-default",
            "no-default-env-set",
            "empty-default",
            "multiple-vars",
            "no-placeholders",
            "url-in-default",
            "dash-only-syntax",
            "dash-only-env-override",
        ],
    )
    def test_substitution(self, template: str, env: dict, expected: str):
        assert substitute_env_vars(template, env=env) == expected

    def test_missing_var_no_default_non_strict(self):
        """Without strict mode, unresolved placeholders stay in the string."""
        result = substitute_env_vars("${MISSING_VAR}", env={})
        assert result == "${MISSING_VAR}"

    def test_missing_var_no_default_strict(self):
        """In strict mode, unresolved variables without defaults raise."""
        with pytest.raises(ConfigError, match="MISSING_VAR.*required.*not set"):
            substitute_env_vars("${MISSING_VAR}", env={}, strict=True)

    def test_mixed_resolved_and_unresolved_non_strict(self):
        result = substitute_env_vars(
            "${A:-hello} ${B}", env={}
        )
        assert result == "hello ${B}"


# ── YAML parsing with env substitution ───────────────────────────────────


class TestParseYamlWithEnv:
    def test_basic_yaml(self):
        raw = textwrap.dedent("""\
            model:
              endpoint: http://localhost:8321/v1
              name: test-model
        """)
        data = parse_yaml_with_env(raw, env={})
        assert data["model"]["endpoint"] == "http://localhost:8321/v1"
        assert data["model"]["name"] == "test-model"

    def test_env_substitution_in_yaml(self):
        raw = textwrap.dedent("""\
            model:
              endpoint: ${MODEL_ENDPOINT:-http://fallback:8321/v1}
              name: ${MODEL_NAME}
        """)
        env = {"MODEL_NAME": "my-model", "MODEL_ENDPOINT": "http://real:9999/v1"}
        data = parse_yaml_with_env(raw, env=env)
        assert data["model"]["endpoint"] == "http://real:9999/v1"
        assert data["model"]["name"] == "my-model"

    def test_empty_yaml_returns_empty_dict(self):
        assert parse_yaml_with_env("", env={}) == {}

    def test_invalid_yaml_raises(self):
        with pytest.raises(ConfigError, match="Invalid YAML"):
            parse_yaml_with_env("{{not valid yaml", env={})

    def test_non_mapping_raises(self):
        with pytest.raises(ConfigError, match="must be a YAML mapping"):
            parse_yaml_with_env("- item1\n- item2", env={})


# ── Pydantic model validation ────────────────────────────────────────────


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.endpoint is None
        assert cfg.name == "meta-llama/Llama-3.3-70B-Instruct"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_custom_values(self):
        cfg = LLMConfig(
            endpoint="http://custom:9999/v1",
            name="my-model",
            temperature=0.0,
            max_tokens=1024,
        )
        assert cfg.endpoint == "http://custom:9999/v1"
        assert cfg.temperature == 0.0

    def test_temperature_out_of_range(self):
        with pytest.raises(Exception, match="less than or equal to 2"):
            LLMConfig(temperature=3.0)

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(Exception, match="greater than 0"):
            LLMConfig(max_tokens=0)


class TestBackoffConfig:
    def test_defaults(self):
        cfg = BackoffConfig()
        assert cfg.initial == 1.0
        assert cfg.max == 30.0
        assert cfg.multiplier == 2.0

    def test_max_less_than_initial_raises(self):
        with pytest.raises(ValueError, match="must be >= backoff.initial"):
            BackoffConfig(initial=10.0, max=5.0)

    def test_multiplier_must_exceed_one(self):
        with pytest.raises(Exception, match="greater than 1"):
            BackoffConfig(multiplier=0.5)


class TestLoopConfig:
    def test_string_max_iterations_coerced(self):
        """Env var substitution may deliver max_iterations as a string."""
        cfg = LoopConfig(max_iterations="50")
        assert cfg.max_iterations == 50

    def test_invalid_string_max_iterations(self):
        with pytest.raises(Exception, match="must be an integer"):
            LoopConfig(max_iterations="not_a_number")


class TestLoggingConfig:
    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_levels(self, level: str):
        cfg = LoggingConfig(level=level)
        assert cfg.level == level

    def test_case_insensitive(self):
        cfg = LoggingConfig(level="debug")
        assert cfg.level == "DEBUG"

    def test_invalid_level(self):
        with pytest.raises(Exception, match="must be one of"):
            LoggingConfig(level="TRACE")


class TestToolsConfig:
    def test_defaults(self):
        cfg = ToolsConfig()
        assert cfg.local_dir == "./tools"
        assert cfg.visibility_default == "agent_only"

    def test_invalid_visibility(self):
        with pytest.raises(Exception, match="visibility_default"):
            ToolsConfig(visibility_default="invalid")


class TestMcpServerConfig:
    def test_requires_url_or_command(self):
        with pytest.raises(Exception, match="requires either"):
            McpServerConfig()

    def test_rejects_both_url_and_command(self):
        with pytest.raises(Exception, match="cannot have both"):
            McpServerConfig(url="http://mcp:8080/mcp", command="python")

    def test_valid_http(self):
        cfg = McpServerConfig(url="http://mcp:8080/mcp")
        assert cfg.url == "http://mcp:8080/mcp"
        assert cfg.command is None

    def test_valid_stdio(self):
        cfg = McpServerConfig(command="python", args=["server.py"])
        assert cfg.command == "python"
        assert cfg.args == ["server.py"]
        assert cfg.url is None

    def test_stdio_optional_fields(self):
        cfg = McpServerConfig(
            command="python",
            args=["server.py"],
            env={"LOG_LEVEL": "debug"},
            cwd="/tmp",
        )
        assert cfg.env == {"LOG_LEVEL": "debug"}
        assert cfg.cwd == "/tmp"


class TestMemoryConfig:
    def test_default_config_path(self):
        cfg = MemoryConfig()
        assert cfg.config_path == ".memoryhub.yaml"

    def test_custom_config_path(self):
        cfg = MemoryConfig(config_path="/etc/agent/memoryhub.yaml")
        assert cfg.config_path == "/etc/agent/memoryhub.yaml"

    def test_memory_optional_in_agent_config(self):
        """AgentConfig works with no memory section; defaults are applied."""
        cfg = AgentConfig.model_validate({})
        assert cfg.memory.config_path == ".memoryhub.yaml"


# ── Full AgentConfig ─────────────────────────────────────────────────────


class TestAgentConfig:
    def test_all_defaults(self):
        """A completely empty dict yields a valid config with all defaults."""
        cfg = AgentConfig()
        assert cfg.model.name == "meta-llama/Llama-3.3-70B-Instruct"
        assert cfg.mcp_servers == []
        assert cfg.loop.max_iterations == 100
        assert cfg.logging.level == "INFO"

    def test_from_partial_dict(self):
        cfg = AgentConfig.model_validate({"model": {"temperature": 0.2}})
        assert cfg.model.temperature == 0.2
        # Other fields keep defaults
        assert cfg.model.endpoint is None

    def test_mcp_servers_list(self):
        cfg = AgentConfig.model_validate({
            "mcp_servers": [
                {"url": "http://a:8080/mcp"},
                {"url": "http://b:8080/mcp"},
            ]
        })
        assert len(cfg.mcp_servers) == 2
        assert cfg.mcp_servers[1].url == "http://b:8080/mcp"


# ── load_config_from_string ──────────────────────────────────────────────


class TestLoadConfigFromString:
    def test_full_config(self):
        # Validates the canonical agent.yaml example documented in docs/architecture.md.
        raw = textwrap.dedent("""\
            model:
              endpoint: ${MODEL_ENDPOINT:-http://llamastack:8321/v1}
              name: ${MODEL_NAME:-meta-llama/Llama-3.3-70B-Instruct}
              temperature: 0.7
              max_tokens: 4096

            mcp_servers:
              - url: ${MCP_WEATHER_URL:-http://weather-mcp:8080/mcp}

            tools:
              local_dir: ./tools
              visibility_default: agent_only

            prompts:
              dir: ./prompts

            loop:
              max_iterations: ${MAX_ITERATIONS:-100}
              backoff:
                initial: 1.0
                max: 30.0
                multiplier: 2.0

            logging:
              level: ${LOG_LEVEL:-INFO}
        """)
        cfg = load_config_from_string(raw, env={})
        assert cfg.model.endpoint == "http://llamastack:8321/v1"
        assert cfg.model.name == "meta-llama/Llama-3.3-70B-Instruct"
        assert cfg.mcp_servers[0].url == "http://weather-mcp:8080/mcp"
        assert cfg.loop.max_iterations == 100
        assert cfg.logging.level == "INFO"

    def test_env_override(self):
        raw = textwrap.dedent("""\
            model:
              endpoint: ${MODEL_ENDPOINT:-http://default:8080}
            logging:
              level: ${LOG_LEVEL:-INFO}
        """)
        env = {"MODEL_ENDPOINT": "http://prod:9999", "LOG_LEVEL": "DEBUG"}
        cfg = load_config_from_string(raw, env=env)
        assert cfg.model.endpoint == "http://prod:9999"
        assert cfg.logging.level == "DEBUG"

    def test_empty_yaml_gives_defaults(self):
        cfg = load_config_from_string("", env={})
        assert cfg.model.name == "meta-llama/Llama-3.3-70B-Instruct"

    def test_invalid_validation_raises_config_error(self):
        raw = textwrap.dedent("""\
            model:
              temperature: 999
        """)
        with pytest.raises(ConfigError, match="Invalid agent configuration"):
            load_config_from_string(raw, env={})

    def test_unresolved_placeholder_passes_through_non_strict(self):
        """With strict=False, an unresolved placeholder is treated as a literal string."""
        raw = textwrap.dedent("""\
            model:
              endpoint: ${UNSET_ENDPOINT}
              name: my-model
        """)
        # Should not raise; the placeholder stays in the endpoint string.
        cfg = load_config_from_string(raw, env={}, strict=False)
        assert "${UNSET_ENDPOINT}" in cfg.model.endpoint


# ── load_config (file-based) ─────────────────────────────────────────────


class TestLoadConfig:
    def test_loads_from_file(self, tmp_path: Path):
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(textwrap.dedent("""\
            model:
              name: file-model
              temperature: 0.1
            loop:
              max_iterations: 10
        """))
        cfg = load_config(config_file, env={})
        assert cfg.model.name == "file-model"
        assert cfg.model.temperature == 0.1
        assert cfg.loop.max_iterations == 10

    def test_file_with_env_substitution(self, tmp_path: Path):
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(textwrap.dedent("""\
            model:
              endpoint: ${ENDPOINT:-http://local:8000}
        """))
        cfg = load_config(config_file, env={"ENDPOINT": "http://remote:9000"})
        assert cfg.model.endpoint == "http://remote:9000"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config(tmp_path / "nonexistent.yaml", env={})

    def test_file_not_found_message_includes_path(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ConfigError) as exc_info:
            load_config(missing, env={})
        assert "does_not_exist.yaml" in str(exc_info.value)

    def test_invalid_yaml_in_file(self, tmp_path: Path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml content")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(config_file, env={})

    def test_strict_mode_with_file(self, tmp_path: Path):
        config_file = tmp_path / "strict.yaml"
        config_file.write_text(textwrap.dedent("""\
            model:
              endpoint: ${REQUIRED_ENDPOINT}
        """))
        with pytest.raises(ConfigError, match="REQUIRED_ENDPOINT.*required"):
            load_config(config_file, env={}, strict=True)


# ── Edge cases and integration ───────────────────────────────────────────


class TestEdgeCases:
    def test_numeric_env_var_in_yaml_context(self):
        """When YAML parses a substituted value as int, it should still work."""
        raw = textwrap.dedent("""\
            loop:
              max_iterations: ${ITERS:-50}
        """)
        cfg = load_config_from_string(raw, env={})
        assert cfg.loop.max_iterations == 50

    def test_numeric_env_var_override(self):
        raw = textwrap.dedent("""\
            loop:
              max_iterations: ${ITERS:-50}
        """)
        cfg = load_config_from_string(raw, env={"ITERS": "200"})
        assert cfg.loop.max_iterations == 200

    def test_extra_keys_are_ignored(self):
        """Unknown top-level keys should not cause a crash (forward compat)."""
        raw = textwrap.dedent("""\
            model:
              name: test
            future_feature:
              something: true
        """)
        # Should not raise — Pydantic ignores extra by default
        cfg = load_config_from_string(raw, env={})
        assert cfg.model.name == "test"

    def test_nested_env_vars_in_list(self):
        raw = textwrap.dedent("""\
            mcp_servers:
              - url: ${MCP_1:-http://a:8080}
              - url: ${MCP_2:-http://b:8080}
        """)
        cfg = load_config_from_string(
            raw, env={"MCP_1": "http://custom:1111"}
        )
        assert cfg.mcp_servers[0].url == "http://custom:1111"
        assert cfg.mcp_servers[1].url == "http://b:8080"
