"""Tests for fipsagents.baseagent.prompts — Prompt loading and rendering."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fipsagents.baseagent.prompts import (
    Prompt,
    PromptError,
    PromptLoader,
    PromptNotFoundError,
    PromptParameters,
    PromptParseError,
    PromptVariableError,
    VariableDefinition,
    _parse_parameters,
    _parse_variable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_prompt(tmp_path: Path, filename: str, content: str) -> Path:
    """Write a prompt file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# VariableDefinition
# ---------------------------------------------------------------------------


class TestVariableDefinition:
    def test_valid_construction(self):
        var = VariableDefinition(name="query", required=True)
        assert var.name == "query"
        assert var.required is True
        assert var.type == "string"
        assert var.default is None

    def test_optional_fields_have_defaults(self):
        var = VariableDefinition(name="x")
        assert var.description == ""
        assert var.default is None

    def test_empty_name_raises_prompt_parse_error(self):
        with pytest.raises(PromptParseError, match="non-empty"):
            VariableDefinition(name="")


# ---------------------------------------------------------------------------
# PromptParameters.as_kwargs
# ---------------------------------------------------------------------------


class TestPromptParametersAsKwargs:
    def test_all_none_returns_empty_dict(self):
        params = PromptParameters()
        assert params.as_kwargs() == {}

    def test_only_set_values_returned(self):
        params = PromptParameters(temperature=0.5)
        assert params.as_kwargs() == {"temperature": 0.5}

    def test_all_set(self):
        params = PromptParameters(model="gpt-4", temperature=0.3, max_tokens=512)
        result = params.as_kwargs()
        assert result == {"model": "gpt-4", "temperature": 0.3, "max_tokens": 512}


# ---------------------------------------------------------------------------
# Prompt.render
# ---------------------------------------------------------------------------


class TestPromptRender:
    def _simple_prompt(self, body: str, vars: list[VariableDefinition] | None = None) -> Prompt:
        return Prompt(
            name="test",
            description="",
            variables=tuple(vars or []),
            parameters=PromptParameters(),
            raw_content=body,
        )

    def test_substitutes_variable(self):
        prompt = self._simple_prompt(
            "Hello, {name}!",
            vars=[VariableDefinition(name="name")],
        )
        result = prompt.render(name="World")
        assert result == "Hello, World!"

    def test_uses_default_when_not_provided(self):
        prompt = self._simple_prompt(
            "Length: {max_length}",
            vars=[VariableDefinition(name="max_length", required=False, default="500 words")],
        )
        result = prompt.render()
        assert result == "Length: 500 words"

    def test_raises_for_missing_required_variable(self):
        prompt = self._simple_prompt(
            "Hello, {name}!",
            vars=[VariableDefinition(name="name", required=True)],
        )
        with pytest.raises(PromptVariableError, match="name"):
            prompt.render()

    def test_extra_kwargs_silently_ignored(self):
        prompt = self._simple_prompt(
            "Hello, {name}!",
            vars=[VariableDefinition(name="name")],
        )
        result = prompt.render(name="Alice", unused_extra="ignored")
        assert result == "Hello, Alice!"

    def test_undeclared_brace_pairs_left_unchanged(self):
        """Braces in code blocks (like {json}) should not raise KeyError."""
        prompt = self._simple_prompt(
            "Use JSON like: {json}\nQuery: {query}",
            vars=[VariableDefinition(name="query")],
        )
        result = prompt.render(query="search this")
        assert "{json}" in result
        assert "search this" in result

    def test_multiple_variables(self):
        prompt = self._simple_prompt(
            "{greeting}, {name}. You have {count} messages.",
            vars=[
                VariableDefinition(name="greeting"),
                VariableDefinition(name="name"),
                VariableDefinition(name="count", default="0", required=False),
            ],
        )
        result = prompt.render(greeting="Hi", name="Alice")
        assert result == "Hi, Alice. You have 0 messages."


# ---------------------------------------------------------------------------
# _parse_variable
# ---------------------------------------------------------------------------


class TestParseVariable:
    def test_string_shorthand(self):
        var = _parse_variable("query", "test_prompt", 0)
        assert var.name == "query"
        assert var.required is True
        assert var.default is None

    def test_dict_with_name_and_default(self):
        raw = {"name": "max_length", "default": "500 words"}
        var = _parse_variable(raw, "test_prompt", 0)
        assert var.name == "max_length"
        assert var.default == "500 words"
        assert var.required is False

    def test_dict_explicitly_required_with_default(self):
        raw = {"name": "x", "default": "d", "required": True}
        var = _parse_variable(raw, "test_prompt", 0)
        assert var.required is True

    def test_dict_with_type_and_description(self):
        raw = {"name": "count", "type": "integer", "description": "The count"}
        var = _parse_variable(raw, "test_prompt", 0)
        assert var.type == "integer"
        assert var.description == "The count"

    def test_invalid_type_raises_prompt_parse_error(self):
        with pytest.raises(PromptParseError, match="string or mapping"):
            _parse_variable(42, "test_prompt", 0)

    def test_dict_without_name_raises_prompt_parse_error(self):
        with pytest.raises(PromptParseError, match="name"):
            _parse_variable({"description": "no name here"}, "test_prompt", 0)


# ---------------------------------------------------------------------------
# _parse_parameters
# ---------------------------------------------------------------------------


class TestParseParameters:
    def test_from_top_level_keys(self):
        meta = {"model": "gpt-4", "temperature": 0.3, "max_tokens": 1024}
        params = _parse_parameters(meta)
        assert params.model == "gpt-4"
        assert params.temperature == 0.3
        assert params.max_tokens == 1024

    def test_from_nested_parameters_dict(self):
        meta = {"parameters": {"model": "gpt-3.5", "temperature": 0.7}}
        params = _parse_parameters(meta)
        assert params.model == "gpt-3.5"
        assert params.temperature == 0.7

    def test_top_level_takes_precedence(self):
        meta = {
            "model": "top-level-model",
            "parameters": {"model": "nested-model"},
        }
        params = _parse_parameters(meta)
        assert params.model == "top-level-model"

    def test_empty_meta_returns_defaults(self):
        params = _parse_parameters({})
        assert params.model is None
        assert params.temperature is None
        assert params.max_tokens is None


# ---------------------------------------------------------------------------
# PromptLoader.load_all
# ---------------------------------------------------------------------------


class TestPromptLoaderLoadAll:
    def test_loads_valid_prompt_files(self, tmp_path):
        _write_prompt(
            tmp_path,
            "summarize.md",
            """\
            ---
            name: summarize
            description: Summarize a document
            variables:
              - document
            ---
            Summarize: {document}
            """,
        )
        loader = PromptLoader()
        prompts = loader.load_all(tmp_path)
        assert len(prompts) == 1
        assert prompts[0].name == "summarize"

    def test_clears_previously_loaded(self, tmp_path):
        _write_prompt(tmp_path, "a.md", "---\nname: prompt_a\n---\nHello")
        loader = PromptLoader()
        loader.load_all(tmp_path)

        # Replace file content
        (tmp_path / "a.md").write_text("---\nname: prompt_b\n---\nWorld")
        loader.load_all(tmp_path)

        assert "prompt_a" not in loader.names
        assert "prompt_b" in loader.names

    def test_handles_malformed_file_gracefully(self, tmp_path):
        _write_prompt(tmp_path, "good.md", "---\nname: good\n---\nOK")
        _write_prompt(
            tmp_path,
            "bad.md",
            # variables as a dict instead of a list — triggers PromptParseError
            "---\nname: bad\nvariables:\n  key: value\n---\nbody",
        )
        loader = PromptLoader()
        prompts = loader.load_all(tmp_path)
        names = [p.name for p in prompts]
        assert "good" in names
        assert "bad" not in names

    def test_raises_when_all_files_malformed(self, tmp_path):
        _write_prompt(
            tmp_path,
            "bad.md",
            "---\nname: bad\nvariables:\n  key: value\n---\nbody",
        )
        loader = PromptLoader()
        with pytest.raises(PromptParseError, match="failed to parse"):
            loader.load_all(tmp_path)

    def test_raises_when_directory_does_not_exist(self):
        loader = PromptLoader()
        with pytest.raises(PromptError, match="does not exist"):
            loader.load_all("/nonexistent/prompts/dir")

    def test_names_property_sorted(self, tmp_path):
        _write_prompt(tmp_path, "zzz.md", "---\nname: zzz\n---\n")
        _write_prompt(tmp_path, "aaa.md", "---\nname: aaa\n---\n")
        loader = PromptLoader()
        loader.load_all(tmp_path)
        assert loader.names == ["aaa", "zzz"]


# ---------------------------------------------------------------------------
# PromptLoader.get
# ---------------------------------------------------------------------------


class TestPromptLoaderGet:
    def test_returns_prompt(self, tmp_path):
        _write_prompt(tmp_path, "greet.md", "---\nname: greet\n---\nHi!")
        loader = PromptLoader()
        loader.load_all(tmp_path)
        prompt = loader.get("greet")
        assert prompt.name == "greet"

    def test_raises_for_unknown_name(self, tmp_path):
        loader = PromptLoader()
        loader.load_all(tmp_path)  # empty dir is fine since no .md files
        with pytest.raises(PromptNotFoundError, match="unknown_prompt"):
            loader.get("unknown_prompt")


# ---------------------------------------------------------------------------
# PromptLoader.render
# ---------------------------------------------------------------------------


class TestPromptLoaderRender:
    def test_render_convenience_method(self, tmp_path):
        _write_prompt(
            tmp_path,
            "hello.md",
            """\
            ---
            name: hello
            variables:
              - name: subject
            ---
            Hello, {subject}!
            """,
        )
        loader = PromptLoader()
        loader.load_all(tmp_path)
        result = loader.render("hello", subject="World")
        assert "Hello, World!" in result


# ---------------------------------------------------------------------------
# PromptLoader.load_file
# ---------------------------------------------------------------------------


class TestPromptLoaderLoadFile:
    def test_loads_single_file(self, tmp_path):
        p = _write_prompt(tmp_path, "single.md", "---\nname: single\n---\nContent")
        loader = PromptLoader()
        prompt = loader.load_file(p)
        assert prompt.name == "single"
        assert loader.get("single").name == "single"

    def test_name_falls_back_to_stem(self, tmp_path):
        """When frontmatter has no name, the file stem is used."""
        p = _write_prompt(tmp_path, "my_prompt.md", "---\n---\nBody text")
        loader = PromptLoader()
        prompt = loader.load_file(p)
        assert prompt.name == "my_prompt"
