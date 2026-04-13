"""Tests for fipsagents.baseagent.skills — progressive-disclosure skill loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fipsagents.baseagent.skills import (
    SkillError,
    SkillLoader,
    SkillManifestEntry,
    SkillNotFoundError,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _write_skill(
    skills_dir: Path,
    dirname: str,
    *,
    name: str = "test-skill",
    description: str = "A test skill.",
    version: str | None = "1.0",
    triggers: list[str] | None = None,
    body: str = "Full instructions go here.",
    extra_frontmatter: str = "",
) -> Path:
    """Create a minimal skill directory with a SKILL.md file."""
    skill_dir = skills_dir / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        "---",
        f'name: "{name}"',
        f'description: "{description}"',
    ]
    if version is not None:
        parts.append(f'version: "{version}"')
    if triggers is not None:
        parts.append("triggers:")
        for t in triggers:
            parts.append(f'  - "{t}"')
    if extra_frontmatter:
        parts.append(extra_frontmatter)
    parts.append("---")
    parts.append(body)

    (skill_dir / "SKILL.md").write_text("\n".join(parts))
    return skill_dir


# ── Discovery ───────────────────────────────────────────────────────────────


class TestLoadAll:
    def test_discovers_skill_directories(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha", name="alpha", description="First skill")
        _write_skill(tmp_path, "beta", name="beta", description="Second skill")

        loader = SkillLoader()
        skills = loader.load_all(tmp_path)

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"alpha", "beta"}

    def test_ignores_directories_without_skill_md(self, tmp_path: Path):
        _write_skill(tmp_path, "valid", name="valid", description="Has SKILL.md")
        (tmp_path / "no-skill-file").mkdir()

        loader = SkillLoader()
        skills = loader.load_all(tmp_path)

        assert len(skills) == 1
        assert skills[0].name == "valid"

    def test_ignores_regular_files_in_root(self, tmp_path: Path):
        _write_skill(tmp_path, "real-skill", name="real", description="Real skill")
        (tmp_path / "README.md").write_text("Not a skill")

        loader = SkillLoader()
        skills = loader.load_all(tmp_path)

        assert len(skills) == 1

    def test_nonexistent_directory_returns_empty(self, tmp_path: Path):
        loader = SkillLoader()
        skills = loader.load_all(tmp_path / "does-not-exist")
        assert skills == []

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        loader = SkillLoader()
        skills = loader.load_all(tmp_path)
        assert skills == []

    def test_load_all_clears_previous_state(self, tmp_path: Path):
        """Calling load_all again replaces the previous skill set."""
        _write_skill(tmp_path, "first", name="first", description="First")
        loader = SkillLoader()
        loader.load_all(tmp_path)
        assert len(loader) == 1

        # Create a second skills directory with different content
        other = tmp_path / "other_skills"
        other.mkdir()
        _write_skill(other, "second", name="second", description="Second")
        loader.load_all(other)

        assert len(loader) == 1
        assert "second" in loader
        assert "first" not in loader


# ── Frontmatter-only loading ────────────────────────────────────────────────


class TestProgressiveDisclosure:
    def test_content_is_none_before_activation(self, tmp_path: Path):
        _write_skill(
            tmp_path, "lazy",
            name="lazy", description="Lazy skill", body="Secret instructions"
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)

        skills = loader.get_manifest()
        assert len(skills) == 1

        # The underlying Skill object should have content=None
        skill = loader._skills["lazy"]
        assert skill.content is None
        assert skill.activated is False

    def test_frontmatter_fields_populated(self, tmp_path: Path):
        _write_skill(
            tmp_path, "full-meta",
            name="full-meta",
            description="Fully specified",
            version="2.1",
            triggers=["user says hello", "greeting detected"],
            extra_frontmatter=textwrap.dedent("""\
                dependencies:
                  - other-skill
                parameters:
                  max_retries: 3"""),
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)

        skill = loader._skills["full-meta"]
        assert skill.name == "full-meta"
        assert skill.description == "Fully specified"
        assert skill.version == "2.1"
        assert skill.triggers == ["user says hello", "greeting detected"]
        assert skill.dependencies == ["other-skill"]
        assert skill.parameters == {"max_retries": 3}
        assert skill.content is None

    def test_optional_fields_default(self, tmp_path: Path):
        """Skills with only required fields should still load cleanly."""
        _write_skill(
            tmp_path, "minimal",
            name="minimal", description="Bare minimum", version=None
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)

        skill = loader._skills["minimal"]
        assert skill.version is None
        assert skill.triggers == []
        assert skill.dependencies == []
        assert skill.parameters == {}


# ── Activation ──────────────────────────────────────────────────────────────


class TestActivate:
    def test_loads_full_content(self, tmp_path: Path):
        body = "Step 1: Do the thing.\n\nStep 2: Verify it worked."
        _write_skill(
            tmp_path, "activatable",
            name="activatable", description="Can activate", body=body,
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)

        skill = loader.activate("activatable")

        assert skill.activated is True
        assert skill.content is not None
        assert "Step 1" in skill.content
        assert "Step 2" in skill.content

    def test_double_activation_is_idempotent(self, tmp_path: Path):
        _write_skill(tmp_path, "idem", name="idem", description="Idempotent")
        loader = SkillLoader()
        loader.load_all(tmp_path)

        first = loader.activate("idem")
        second = loader.activate("idem")
        assert first is second
        assert first.activated is True

    def test_activate_unknown_skill_raises(self, tmp_path: Path):
        loader = SkillLoader()
        loader.load_all(tmp_path)

        with pytest.raises(SkillNotFoundError, match="Unknown skill 'ghost'"):
            loader.activate("ghost")

    def test_error_message_lists_available_skills(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha", name="alpha", description="A")
        _write_skill(tmp_path, "beta", name="beta", description="B")
        loader = SkillLoader()
        loader.load_all(tmp_path)

        with pytest.raises(SkillNotFoundError, match="alpha.*beta"):
            loader.activate("nonexistent")


# ── Deactivation ────────────────────────────────────────────────────────────


class TestDeactivate:
    def test_clears_content_and_flag(self, tmp_path: Path):
        _write_skill(
            tmp_path, "d",
            name="deact", description="Deactivatable", body="Full content",
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)
        loader.activate("deact")

        skill = loader._skills["deact"]
        assert skill.activated is True
        assert skill.content is not None

        loader.deactivate("deact")

        assert skill.activated is False
        assert skill.content is None

    def test_deactivate_then_reactivate(self, tmp_path: Path):
        _write_skill(
            tmp_path, "r",
            name="reactivate", description="Reactivatable", body="Content here",
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)
        loader.activate("reactivate")
        loader.deactivate("reactivate")

        skill = loader.activate("reactivate")
        assert skill.activated is True
        assert "Content here" in skill.content

    def test_deactivate_already_inactive_is_safe(self, tmp_path: Path):
        _write_skill(tmp_path, "i", name="inactive", description="Never activated")
        loader = SkillLoader()
        loader.load_all(tmp_path)

        # Should not raise
        loader.deactivate("inactive")
        skill = loader._skills["inactive"]
        assert skill.activated is False
        assert skill.content is None

    def test_deactivate_unknown_skill_raises(self, tmp_path: Path):
        loader = SkillLoader()
        loader.load_all(tmp_path)

        with pytest.raises(SkillNotFoundError, match="Unknown skill 'ghost'"):
            loader.deactivate("ghost")


# ── Manifest ────────────────────────────────────────────────────────────────


class TestGetManifest:
    def test_returns_summaries(self, tmp_path: Path):
        _write_skill(
            tmp_path, "s1",
            name="summarizer", description="Summarizes text",
            triggers=["summarize this"],
        )
        _write_skill(
            tmp_path, "s2",
            name="translator", description="Translates text",
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)

        manifest = loader.get_manifest()

        assert len(manifest) == 2
        assert all(isinstance(e, SkillManifestEntry) for e in manifest)

        by_name = {e.name: e for e in manifest}
        assert by_name["summarizer"].description == "Summarizes text"
        assert by_name["summarizer"].triggers == ["summarize this"]
        assert by_name["translator"].triggers == []

    def test_manifest_does_not_activate_skills(self, tmp_path: Path):
        _write_skill(tmp_path, "stays-lazy", name="stays-lazy", description="Lazy")
        loader = SkillLoader()
        loader.load_all(tmp_path)

        loader.get_manifest()

        skill = loader._skills["stays-lazy"]
        assert skill.activated is False
        assert skill.content is None


# ── get() auto-activation ──────────────────────────────────────────────────


class TestGet:
    def test_auto_activates(self, tmp_path: Path):
        _write_skill(
            tmp_path, "auto",
            name="auto", description="Auto skill", body="Auto content",
        )
        loader = SkillLoader()
        loader.load_all(tmp_path)

        skill = loader.get("auto")

        assert skill.activated is True
        assert skill.content is not None
        assert "Auto content" in skill.content

    def test_returns_already_activated(self, tmp_path: Path):
        _write_skill(tmp_path, "pre", name="pre", description="Pre-activated")
        loader = SkillLoader()
        loader.load_all(tmp_path)
        loader.activate("pre")

        skill = loader.get("pre")
        assert skill.activated is True

    def test_unknown_skill_raises(self, tmp_path: Path):
        loader = SkillLoader()
        loader.load_all(tmp_path)

        with pytest.raises(SkillNotFoundError, match="Unknown skill 'nope'"):
            loader.get("nope")


# ── Error handling ──────────────────────────────────────────────────────────


class TestErrors:
    def test_missing_skill_md_skipped_silently(self, tmp_path: Path):
        """Directories without SKILL.md are silently skipped, not errors."""
        (tmp_path / "no-skill").mkdir()
        loader = SkillLoader()
        skills = loader.load_all(tmp_path)
        assert skills == []

    def test_missing_name_field_raises(self, tmp_path: Path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            description: "Has description but no name"
            ---
            Body text.
        """))

        loader = SkillLoader()
        with pytest.raises(SkillError, match="missing required frontmatter.*name"):
            loader.load_all(tmp_path)

    def test_missing_description_field_raises(self, tmp_path: Path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: "has-name"
            ---
            Body text.
        """))

        loader = SkillLoader()
        with pytest.raises(SkillError, match="missing required frontmatter.*description"):
            loader.load_all(tmp_path)

    def test_missing_both_required_fields_lists_all(self, tmp_path: Path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            version: "1.0"
            ---
            Body text.
        """))

        loader = SkillLoader()
        with pytest.raises(SkillError, match="name.*description"):
            loader.load_all(tmp_path)

    def test_unparseable_frontmatter_raises(self, tmp_path: Path):
        skill_dir = tmp_path / "corrupt"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\n{{invalid yaml\n---\nBody")

        loader = SkillLoader()
        with pytest.raises(SkillError, match="Failed to parse"):
            loader.load_all(tmp_path)


# ── Utility methods ─────────────────────────────────────────────────────────


class TestUtilities:
    def test_contains(self, tmp_path: Path):
        _write_skill(tmp_path, "present", name="present", description="Here")
        loader = SkillLoader()
        loader.load_all(tmp_path)

        assert "present" in loader
        assert "absent" not in loader

    def test_len(self, tmp_path: Path):
        _write_skill(tmp_path, "a", name="a", description="A")
        _write_skill(tmp_path, "b", name="b", description="B")
        _write_skill(tmp_path, "c", name="c", description="C")
        loader = SkillLoader()
        loader.load_all(tmp_path)

        assert len(loader) == 3

    def test_list_skills(self, tmp_path: Path):
        _write_skill(tmp_path, "x", name="x", description="X")
        _write_skill(tmp_path, "y", name="y", description="Y")
        loader = SkillLoader()
        loader.load_all(tmp_path)

        names = loader.list_skills()
        assert sorted(names) == ["x", "y"]
