"""Tests for fipsagents.baseagent.prompts — loading, parsing, rendering prompt templates."""

from __future__ import annotations

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
    _parse_prompt_file,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _write_prompt(directory: Path, filename: str, content: str) -> Path:
    """Write a prompt file into *directory* and return the path."""
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


SUMMARIZE_PROMPT = """\
---
name: summarize
description: Summarize a document for the user
model: default
temperature: 0.3
variables:
  - name: document
    required: true
  - name: max_length
    default: "500 words"
---
Summarize the following document in {max_length} or less.

## Document

{document}
"""

GREETING_PROMPT = """\
---
name: greeting
description: Greet the user
---
Hello! How can I help you today?
"""

NO_FRONTMATTER_PROMPT = """\
This prompt has no frontmatter at all.
Just plain markdown content with no variables.
"""


# ── VariableDefinition ──────────────────────────────────────────────────────


class TestVariableDefinition:
    def test_defaults(self):
        v = VariableDefinition(name="foo")
        assert v.name == "foo"
        assert v.type == "string"
        assert v.required is True
        assert v.default is None

    def test_empty_name_raises(self):
        with pytest.raises(PromptParseError, match="non-empty"):
            VariableDefinition(name="")


# ── PromptParameters ────────────────────────────────────────────────────────


class TestPromptParameters:
    def test_as_kwargs_only_set_values(self):
        p = PromptParameters(temperature=0.5)
        assert p.as_kwargs() == {"temperature": 0.5}

    def test_as_kwargs_all_set(self):
        p = PromptParameters(model="gpt-4", temperature=0.0, max_tokens=100)
        assert p.as_kwargs() == {"model": "gpt-4", "temperature": 0.0, "max_tokens": 100}

    def test_as_kwargs_empty(self):
        assert PromptParameters().as_kwargs() == {}


# ── Prompt.render ───────────────────────────────────────────────────────────


class TestPromptRender:
    """Tests for the Prompt dataclass render method, independent of file I/O."""

    @pytest.fixture()
    def prompt_with_vars(self) -> Prompt:
        return Prompt(
            name="test",
            description="test prompt",
            variables=(
                VariableDefinition(name="subject", required=True),
                VariableDefinition(name="tone", required=False, default="formal"),
            ),
            parameters=PromptParameters(),
            raw_content="Write about {subject} in a {tone} tone.",
        )

    def test_happy_path(self, prompt_with_vars: Prompt):
        result = prompt_with_vars.render(subject="AI safety")
        assert result == "Write about AI safety in a formal tone."

    def test_override_default(self, prompt_with_vars: Prompt):
        result = prompt_with_vars.render(subject="AI safety", tone="casual")
        assert result == "Write about AI safety in a casual tone."

    def test_missing_required_raises(self, prompt_with_vars: Prompt):
        with pytest.raises(PromptVariableError, match="subject"):
            prompt_with_vars.render()

    def test_extra_variables_ignored(self, prompt_with_vars: Prompt):
        result = prompt_with_vars.render(subject="testing", extra="ignored")
        assert "testing" in result

    def test_no_variables(self):
        p = Prompt(
            name="static",
            description="",
            variables=(),
            parameters=PromptParameters(),
            raw_content="Hello world.",
        )
        assert p.render() == "Hello world."

    def test_braces_in_content_preserved(self):
        """Brace pairs that aren't declared variables survive rendering."""
        p = Prompt(
            name="code",
            description="",
            variables=(VariableDefinition(name="lang"),),
            parameters=PromptParameters(),
            raw_content="Use {lang}. Example: `dict = {{}}`",
        )
        result = p.render(lang="Python")
        # Doubled braces become single braces via str.format_map
        assert "Python" in result


# ── Parsing a single file ──────────────────────────────────────────────────


class TestParsePromptFile:
    def test_full_frontmatter(self, tmp_path: Path):
        path = _write_prompt(tmp_path, "summarize.md", SUMMARIZE_PROMPT)
        prompt = _parse_prompt_file(path)

        assert prompt.name == "summarize"
        assert prompt.description == "Summarize a document for the user"
        assert prompt.parameters.model == "default"
        assert prompt.parameters.temperature == 0.3
        assert len(prompt.variables) == 2

        doc_var = prompt.variables[0]
        assert doc_var.name == "document"
        assert doc_var.required is True

        length_var = prompt.variables[1]
        assert length_var.name == "max_length"
        assert length_var.default == "500 words"
        # Having a default makes it optional unless explicitly required
        assert length_var.required is False

    def test_prompt_with_no_variables(self, tmp_path: Path):
        path = _write_prompt(tmp_path, "greeting.md", GREETING_PROMPT)
        prompt = _parse_prompt_file(path)
        assert prompt.name == "greeting"
        assert prompt.variables == ()
        assert prompt.render() == "Hello! How can I help you today?"

    def test_no_frontmatter_uses_filename(self, tmp_path: Path):
        path = _write_prompt(tmp_path, "fallback_name.md", NO_FRONTMATTER_PROMPT)
        prompt = _parse_prompt_file(path)
        assert prompt.name == "fallback_name"
        assert prompt.description == ""

    def test_variables_as_shorthand_strings(self, tmp_path: Path):
        content = """\
---
name: shorthand
variables:
  - query
  - context
---
Search for {query} using {context}.
"""
        path = _write_prompt(tmp_path, "shorthand.md", content)
        prompt = _parse_prompt_file(path)
        assert len(prompt.variables) == 2
        assert prompt.variables[0].name == "query"
        assert prompt.variables[0].required is True

    def test_parameters_block(self, tmp_path: Path):
        """Model parameters in a grouped 'parameters' dict."""
        content = """\
---
name: grouped
parameters:
  temperature: 0.0
  max_tokens: 2000
---
Body.
"""
        path = _write_prompt(tmp_path, "grouped.md", content)
        prompt = _parse_prompt_file(path)
        assert prompt.parameters.temperature == 0.0
        assert prompt.parameters.max_tokens == 2000

    def test_top_level_params_override_grouped(self, tmp_path: Path):
        """Top-level model/temperature take precedence over parameters block."""
        content = """\
---
name: override
temperature: 0.9
parameters:
  temperature: 0.1
  max_tokens: 500
---
Body.
"""
        path = _write_prompt(tmp_path, "override.md", content)
        prompt = _parse_prompt_file(path)
        assert prompt.parameters.temperature == 0.9
        assert prompt.parameters.max_tokens == 500

    def test_invalid_variables_type_raises(self, tmp_path: Path):
        content = """\
---
name: bad
variables: "not a list"
---
Body.
"""
        path = _write_prompt(tmp_path, "bad.md", content)
        with pytest.raises(PromptParseError, match="must be a list"):
            _parse_prompt_file(path)

    def test_variable_missing_name_raises(self, tmp_path: Path):
        content = """\
---
name: bad
variables:
  - description: "no name field"
---
Body.
"""
        path = _write_prompt(tmp_path, "bad.md", content)
        with pytest.raises(PromptParseError, match="must have a string 'name'"):
            _parse_prompt_file(path)

    def test_variable_wrong_type_raises(self, tmp_path: Path):
        content = """\
---
name: bad
variables:
  - 42
---
Body.
"""
        path = _write_prompt(tmp_path, "bad.md", content)
        with pytest.raises(PromptParseError, match="must be a string or mapping"):
            _parse_prompt_file(path)

    def test_source_path_recorded(self, tmp_path: Path):
        path = _write_prompt(tmp_path, "test.md", GREETING_PROMPT)
        prompt = _parse_prompt_file(path)
        assert prompt.source_path == path


# ── PromptLoader ────────────────────────────────────────────────────────────


class TestPromptLoaderLoadAll:
    def test_load_directory(self, tmp_path: Path):
        _write_prompt(tmp_path, "summarize.md", SUMMARIZE_PROMPT)
        _write_prompt(tmp_path, "greeting.md", GREETING_PROMPT)

        loader = PromptLoader()
        loaded = loader.load_all(tmp_path)

        assert len(loaded) == 2
        assert set(loader.names) == {"summarize", "greeting"}

    def test_empty_directory(self, tmp_path: Path):
        loader = PromptLoader()
        loaded = loader.load_all(tmp_path)
        assert loaded == []
        assert loader.names == []

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        loader = PromptLoader()
        with pytest.raises(PromptError, match="does not exist"):
            loader.load_all(tmp_path / "nonexistent")

    def test_ignores_non_md_files(self, tmp_path: Path):
        _write_prompt(tmp_path, "real.md", GREETING_PROMPT)
        (tmp_path / "notes.txt").write_text("not a prompt")
        (tmp_path / "data.json").write_text("{}")

        loader = PromptLoader()
        loaded = loader.load_all(tmp_path)
        assert len(loaded) == 1

    def test_reload_clears_previous(self, tmp_path: Path):
        """Calling load_all again replaces all previously loaded prompts."""
        _write_prompt(tmp_path, "a.md", GREETING_PROMPT)

        loader = PromptLoader()
        loader.load_all(tmp_path)
        assert len(loader.names) == 1

        # Create a second directory with different prompts
        other = tmp_path / "other"
        other.mkdir()
        _write_prompt(other, "b.md", SUMMARIZE_PROMPT)

        loader.load_all(other)
        assert "greeting" not in loader.names
        assert "summarize" in loader.names


class TestPromptLoaderGet:
    def test_get_existing(self, tmp_path: Path):
        _write_prompt(tmp_path, "greeting.md", GREETING_PROMPT)
        loader = PromptLoader()
        loader.load_all(tmp_path)

        prompt = loader.get("greeting")
        assert prompt.name == "greeting"

    def test_get_unknown_raises(self, tmp_path: Path):
        loader = PromptLoader()
        loader.load_all(tmp_path)

        with pytest.raises(PromptNotFoundError, match="no_such_prompt"):
            loader.get("no_such_prompt")

    def test_get_unknown_lists_available(self, tmp_path: Path):
        _write_prompt(tmp_path, "greeting.md", GREETING_PROMPT)
        loader = PromptLoader()
        loader.load_all(tmp_path)

        with pytest.raises(PromptNotFoundError, match="greeting"):
            loader.get("missing")


class TestPromptLoaderRender:
    def test_render_end_to_end(self, tmp_path: Path):
        _write_prompt(tmp_path, "summarize.md", SUMMARIZE_PROMPT)
        loader = PromptLoader()
        loader.load_all(tmp_path)

        result = loader.render("summarize", document="Hello world")
        assert "Hello world" in result
        assert "500 words" in result  # default applied

    def test_render_with_override(self, tmp_path: Path):
        _write_prompt(tmp_path, "summarize.md", SUMMARIZE_PROMPT)
        loader = PromptLoader()
        loader.load_all(tmp_path)

        result = loader.render(
            "summarize", document="content", max_length="100 words"
        )
        assert "100 words" in result
        assert "500 words" not in result

    def test_render_missing_required(self, tmp_path: Path):
        _write_prompt(tmp_path, "summarize.md", SUMMARIZE_PROMPT)
        loader = PromptLoader()
        loader.load_all(tmp_path)

        with pytest.raises(PromptVariableError, match="document"):
            loader.render("summarize")

    def test_render_unknown_prompt(self):
        loader = PromptLoader()
        with pytest.raises(PromptNotFoundError):
            loader.render("nonexistent")


class TestPromptLoaderLoadFile:
    def test_load_single_file(self, tmp_path: Path):
        path = _write_prompt(tmp_path, "one.md", GREETING_PROMPT)
        loader = PromptLoader()
        prompt = loader.load_file(path)

        assert prompt.name == "greeting"
        assert loader.get("greeting") is prompt


class TestPromptLoaderListPrompts:
    def test_list_prompts_metadata(self, tmp_path: Path):
        _write_prompt(tmp_path, "summarize.md", SUMMARIZE_PROMPT)
        _write_prompt(tmp_path, "greeting.md", GREETING_PROMPT)
        loader = PromptLoader()
        loader.load_all(tmp_path)

        listing = loader.list_prompts()
        assert len(listing) == 2

        # Sorted by name
        assert listing[0]["name"] == "greeting"
        assert listing[1]["name"] == "summarize"
        assert listing[1]["variables"][0]["name"] == "document"


# ── Data-driven render scenarios ────────────────────────────────────────────


_RENDER_CASES = [
    pytest.param(
        "Hello {name}!",
        [{"name": "name", "required": True}],
        {"name": "World"},
        "Hello World!",
        id="simple-substitution",
    ),
    pytest.param(
        "{a} and {b}",
        [{"name": "a", "required": True}, {"name": "b", "required": True}],
        {"a": "one", "b": "two"},
        "one and two",
        id="multiple-variables",
    ),
    pytest.param(
        "Default is {val}",
        [{"name": "val", "default": "42"}],
        {},
        "Default is 42",
        id="default-applied",
    ),
    pytest.param(
        "Static content only.",
        [],
        {},
        "Static content only.",
        id="no-variables",
    ),
    pytest.param(
        "{x} with extra",
        [{"name": "x", "required": True}],
        {"x": "value", "unused": "ignored"},
        "value with extra",
        id="extra-vars-ignored",
    ),
]


@pytest.mark.parametrize("template,var_defs,variables,expected", _RENDER_CASES)
def test_render_data_driven(
    template: str,
    var_defs: list[dict],
    variables: dict[str, str],
    expected: str,
):
    prompt = Prompt(
        name="test",
        description="",
        variables=tuple(
            VariableDefinition(
                name=v["name"],
                required=v.get("required", "default" not in v),
                default=v.get("default"),
            )
            for v in var_defs
        ),
        parameters=PromptParameters(),
        raw_content=template,
    )
    assert prompt.render(**variables) == expected
