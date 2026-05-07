"""Tests for fipsagents.baseagent.rules — plain Markdown rule loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from fipsagents.baseagent.rules import Rule, RuleLoadError, RuleLoader, RuleNotFoundError


# ── Rule data model ─────────────────────────────────────────────────────


class TestRule:
    def test_fields(self):
        rule = Rule(name="safety", content="Do not harm users.")
        assert rule.name == "safety"
        assert rule.content == "Do not harm users."

    def test_frozen(self):
        rule = Rule(name="safety", content="Do not harm users.")
        with pytest.raises(AttributeError):
            rule.name = "other"  # type: ignore[misc]


# ── RuleLoader.load_all ─────────────────────────────────────────────────


class TestLoadAll:
    def test_single_rule(self, tmp_path: Path):
        (tmp_path / "safety.md").write_text("Be safe.\n")
        loader = RuleLoader()
        rules = loader.load_all(tmp_path)
        assert len(rules) == 1
        assert rules[0].name == "safety"
        assert rules[0].content == "Be safe.\n"

    def test_multiple_rules_sorted_by_name(self, tmp_path: Path):
        (tmp_path / "zebra.md").write_text("Z content")
        (tmp_path / "alpha.md").write_text("A content")
        (tmp_path / "middle.md").write_text("M content")
        loader = RuleLoader()
        rules = loader.load_all(tmp_path)
        names = [r.name for r in rules]
        assert names == ["alpha", "middle", "zebra"]

    def test_empty_directory(self, tmp_path: Path):
        loader = RuleLoader()
        rules = loader.load_all(tmp_path)
        assert rules == []

    def test_nonexistent_directory(self, tmp_path: Path):
        loader = RuleLoader()
        rules = loader.load_all(tmp_path / "does_not_exist")
        assert rules == []

    def test_non_md_files_ignored(self, tmp_path: Path):
        (tmp_path / "valid.md").write_text("A rule")
        (tmp_path / "notes.txt").write_text("Not a rule")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "script.py").write_text("print('hi')")
        loader = RuleLoader()
        rules = loader.load_all(tmp_path)
        assert len(rules) == 1
        assert rules[0].name == "valid"

    def test_name_derived_from_filename(self, tmp_path: Path):
        (tmp_path / "no-harm-policy.md").write_text("content")
        loader = RuleLoader()
        rules = loader.load_all(tmp_path)
        assert rules[0].name == "no-harm-policy"

    def test_reload_clears_previous(self, tmp_path: Path):
        """Calling load_all again replaces the previous set of rules."""
        (tmp_path / "first.md").write_text("First")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        assert loader.get("first").content == "First"

        # Create a second directory with different rules
        other = tmp_path / "other"
        other.mkdir()
        (other / "second.md").write_text("Second")
        loader.load_all(other)

        with pytest.raises(RuleNotFoundError):
            loader.get("first")
        assert loader.get("second").content == "Second"

    def test_unreadable_file_raises(self, tmp_path: Path):
        bad_file = tmp_path / "broken.md"
        bad_file.write_text("content")
        bad_file.chmod(0o000)
        loader = RuleLoader()
        try:
            with pytest.raises(RuleLoadError, match="Cannot read rule file"):
                loader.load_all(tmp_path)
        finally:
            bad_file.chmod(0o644)  # restore so tmp_path cleanup succeeds


# ── RuleLoader.get ──────────────────────────────────────────────────────


class TestGet:
    def test_known_name(self, tmp_path: Path):
        (tmp_path / "safety.md").write_text("Be safe.\n")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        rule = loader.get("safety")
        assert rule is not None
        assert rule.name == "safety"
        assert rule.content == "Be safe.\n"

    def test_unknown_name_raises(self, tmp_path: Path):
        (tmp_path / "safety.md").write_text("Be safe.\n")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        with pytest.raises(RuleNotFoundError, match="No rule named 'nonexistent'"):
            loader.get("nonexistent")

    def test_error_message_lists_available_rules(self, tmp_path: Path):
        (tmp_path / "alpha.md").write_text("A")
        (tmp_path / "beta.md").write_text("B")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        with pytest.raises(RuleNotFoundError, match="alpha.*beta"):
            loader.get("nonexistent")

    def test_get_before_load_raises(self):
        loader = RuleLoader()
        with pytest.raises(RuleNotFoundError, match="Available rules: \\(none\\)"):
            loader.get("anything")


# ── RuleLoader.get_all ──────────────────────────────────────────────────


class TestGetAll:
    def test_returns_all_rules(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("Alpha")
        (tmp_path / "b.md").write_text("Beta")
        (tmp_path / "c.md").write_text("Charlie")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        rules = loader.get_all()
        assert len(rules) == 3
        assert {r.name for r in rules} == {"a", "b", "c"}

    def test_empty_when_no_rules(self):
        loader = RuleLoader()
        assert loader.get_all() == []

    def test_sorted_order(self, tmp_path: Path):
        for name in ["z", "m", "a"]:
            (tmp_path / f"{name}.md").write_text(f"{name} content")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        names = [r.name for r in loader.get_all()]
        assert names == ["a", "m", "z"]


# ── RuleLoader.get_combined_content ─────────────────────────────────────


class TestGetCombinedContent:
    def test_single_rule(self, tmp_path: Path):
        (tmp_path / "safety.md").write_text("Be safe.")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        combined = loader.get_combined_content()
        assert "# Rule: safety" in combined
        assert "Be safe." in combined

    def test_multiple_rules_with_separators(self, tmp_path: Path):
        (tmp_path / "alpha.md").write_text("Alpha content")
        (tmp_path / "beta.md").write_text("Beta content")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        combined = loader.get_combined_content()
        assert "# Rule: alpha" in combined
        assert "# Rule: beta" in combined
        assert "Alpha content" in combined
        assert "Beta content" in combined
        assert "\n\n---\n\n" in combined

    def test_rules_appear_in_sorted_order(self, tmp_path: Path):
        (tmp_path / "zebra.md").write_text("Z")
        (tmp_path / "alpha.md").write_text("A")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        combined = loader.get_combined_content()
        alpha_pos = combined.index("# Rule: alpha")
        zebra_pos = combined.index("# Rule: zebra")
        assert alpha_pos < zebra_pos

    def test_custom_separator(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("A")
        (tmp_path / "b.md").write_text("B")
        loader = RuleLoader()
        loader.load_all(tmp_path)
        combined = loader.get_combined_content(separator="\n===\n")
        assert "\n===\n" in combined
        assert "\n\n---\n\n" not in combined

    def test_empty_returns_empty_string(self):
        loader = RuleLoader()
        assert loader.get_combined_content() == ""

    def test_empty_directory_returns_empty_string(self, tmp_path: Path):
        loader = RuleLoader()
        loader.load_all(tmp_path)
        assert loader.get_combined_content() == ""
