"""Tests for named-layer prompt assembly with precedence and audit logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from fipsagents.baseagent.config import (
    AgentConfig,
    IdentityConfig,
    PersonalityConfig,
    PromptAssemblyConfig,
)
from fipsagents.baseagent.prompt_assembly import PromptAssembler
from fipsagents.baseagent.prompts import PromptLoader
from fipsagents.baseagent.rules import RuleLoader
from fipsagents.baseagent.skills import SkillLoader


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def base_dir(tmp_path):
    """Create a base directory with prompts/, rules/, and skills/ subdirs."""
    (tmp_path / "prompts").mkdir()
    (tmp_path / "rules").mkdir()
    (tmp_path / "skills").mkdir()
    return tmp_path


def _write_prompt(base_dir: Path, name: str, content: str):
    """Write a prompt file to prompts/ directory."""
    (base_dir / "prompts" / f"{name}.md").write_text(content)


def _write_identity(base_dir: Path, content: str):
    """Write identity.md to base directory."""
    (base_dir / "identity.md").write_text(content)


def _write_personality(base_dir: Path, content: str):
    """Write personality.md to base directory."""
    (base_dir / "personality.md").write_text(content)


def _write_rule(base_dir: Path, name: str, content: str):
    """Write a rule file to rules/ directory."""
    (base_dir / "rules" / f"{name}.md").write_text(content)


def _write_skill(base_dir: Path, name: str, description: str, triggers: list[str]):
    """Write a skill directory with SKILL.md to skills/ directory."""
    skill_dir = base_dir / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    trigger_str = "\n".join(f"  - {t}" for t in triggers) if triggers else "  - none"
    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
triggers:
{trigger_str}
---
Skill body content here.
""")


def _make_assembler(base_dir: Path, **kwargs) -> PromptAssembler:
    """Create a PromptAssembler with real loaders."""
    prompts = PromptLoader()
    prompts_dir = base_dir / "prompts"
    if prompts_dir.is_dir():
        prompts.load_all(prompts_dir)

    rules = RuleLoader()
    rules_dir = base_dir / "rules"
    if rules_dir.is_dir():
        rules.load_all(rules_dir)

    skills = SkillLoader()
    skills_dir = base_dir / "skills"
    if skills_dir.is_dir():
        skills.load_all(skills_dir)

    defaults = dict(
        base_dir=base_dir,
        prompts=prompts,
        rules=rules,
        skills=skills,
    )
    defaults.update(kwargs)
    return PromptAssembler(**defaults)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPromptAssemblyConfig:
    def test_identity_config_defaults(self):
        cfg = IdentityConfig()
        assert cfg.source == "identity.md"
        assert cfg.inline is None
        assert cfg.enabled is True

    def test_personality_config_defaults(self):
        cfg = PersonalityConfig()
        assert cfg.source == "personality.md"
        assert cfg.enabled is False

    def test_prompt_assembly_config_defaults(self):
        cfg = PromptAssemblyConfig()
        assert cfg.governance_enabled is True
        assert cfg.capabilities_enabled is True
        assert cfg.identity.source == "identity.md"
        assert cfg.personality.enabled is False

    def test_agent_config_with_prompt_assembly_none_means_legacy(self):
        cfg = AgentConfig(name="test", model={"name": "test-model"})
        assert cfg.prompt_assembly is None


# ---------------------------------------------------------------------------
# Identity layer tests
# ---------------------------------------------------------------------------


class TestIdentityLayer:
    def test_identity_from_file(self, base_dir):
        _write_identity(base_dir, "I am an identity from file.")
        assembler = _make_assembler(base_dir)
        result = assembler.assemble()
        assert "I am an identity from file." in result

        audit = assembler.get_audit()
        assert audit is not None
        identity_layer = next(ly for ly in audit.layers if ly.name == "identity")
        assert identity_layer.skipped is False
        assert identity_layer.content == "I am an identity from file."
        assert str(base_dir / "identity.md") in identity_layer.source

    def test_identity_inline_takes_precedence(self, base_dir):
        _write_identity(base_dir, "I am from file.")
        assembler = _make_assembler(
            base_dir,
            identity_inline="I am inline identity.",
        )
        result = assembler.assemble()
        assert "I am inline identity." in result
        assert "I am from file." not in result

        audit = assembler.get_audit()
        identity_layer = next(ly for ly in audit.layers if ly.name == "identity")
        assert identity_layer.source == "inline"

    def test_identity_fallback_to_system_prompt(self, base_dir):
        _write_prompt(base_dir, "system", "You are a helpful assistant.")
        assembler = _make_assembler(base_dir)
        result = assembler.assemble()
        assert "You are a helpful assistant." in result

        audit = assembler.get_audit()
        identity_layer = next(ly for ly in audit.layers if ly.name == "identity")
        assert identity_layer.source == "prompt_loader"

    def test_identity_missing_everywhere_skips_layer(self, base_dir):
        assembler = _make_assembler(base_dir)
        assembler.assemble()
        audit = assembler.get_audit()
        identity_layer = next(ly for ly in audit.layers if ly.name == "identity")
        assert identity_layer.skipped is True
        assert identity_layer.skip_reason == "no identity source found"
        assert "identity" in audit.skipped_layers

    def test_identity_disabled_skips_layer(self, base_dir):
        _write_identity(base_dir, "I exist but am disabled.")
        assembler = _make_assembler(base_dir, identity_enabled=False)
        result = assembler.assemble()
        assert "I exist but am disabled." not in result

        audit = assembler.get_audit()
        identity_layer = next(ly for ly in audit.layers if ly.name == "identity")
        assert identity_layer.skipped is True
        assert identity_layer.skip_reason == "disabled"


# ---------------------------------------------------------------------------
# Personality layer tests
# ---------------------------------------------------------------------------


class TestPersonalityLayer:
    def test_personality_disabled_by_default(self, base_dir):
        _write_personality(base_dir, "I am a personality.")
        assembler = _make_assembler(base_dir)
        result = assembler.assemble()
        assert "I am a personality." not in result

        audit = assembler.get_audit()
        personality_layer = next(ly for ly in audit.layers if ly.name == "personality")
        assert personality_layer.skipped is True
        assert personality_layer.skip_reason == "disabled"

    def test_personality_enabled_with_file(self, base_dir):
        _write_personality(base_dir, "I am friendly and helpful.")
        assembler = _make_assembler(base_dir, personality_enabled=True)
        result = assembler.assemble()
        assert "I am friendly and helpful." in result

        audit = assembler.get_audit()
        personality_layer = next(ly for ly in audit.layers if ly.name == "personality")
        assert personality_layer.skipped is False
        assert str(base_dir / "personality.md") in personality_layer.source

    def test_personality_enabled_but_file_missing(self, base_dir):
        assembler = _make_assembler(base_dir, personality_enabled=True)
        assembler.assemble()

        audit = assembler.get_audit()
        personality_layer = next(ly for ly in audit.layers if ly.name == "personality")
        assert personality_layer.skipped is True
        assert personality_layer.skip_reason == "file not found"


# ---------------------------------------------------------------------------
# Governance layer tests
# ---------------------------------------------------------------------------


class TestGovernanceLayer:
    def test_governance_uses_rule_loader(self, base_dir):
        _write_rule(base_dir, "security", "# Security\n\nNever share secrets.")
        _write_rule(base_dir, "style", "# Style\n\nBe concise.")
        assembler = _make_assembler(base_dir)
        result = assembler.assemble()
        assert "Never share secrets." in result
        assert "Be concise." in result

        audit = assembler.get_audit()
        governance_layer = next(ly for ly in audit.layers if ly.name == "governance")
        assert governance_layer.skipped is False
        assert governance_layer.source == "rules"

    def test_governance_disabled_skips_layer(self, base_dir):
        _write_rule(base_dir, "security", "# Security\n\nNever share secrets.")
        assembler = _make_assembler(base_dir, governance_enabled=False)
        result = assembler.assemble()
        assert "Never share secrets." not in result

        audit = assembler.get_audit()
        governance_layer = next(ly for ly in audit.layers if ly.name == "governance")
        assert governance_layer.skipped is True
        assert governance_layer.skip_reason == "disabled"

    def test_governance_no_rules_skips_layer(self, base_dir):
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        governance_layer = next(ly for ly in audit.layers if ly.name == "governance")
        assert governance_layer.skipped is True
        assert governance_layer.skip_reason == "no rules loaded"


# ---------------------------------------------------------------------------
# Capabilities layer tests
# ---------------------------------------------------------------------------


class TestCapabilitiesLayer:
    def test_capabilities_uses_skill_loader(self, base_dir):
        _write_skill(base_dir, "search", "Search the web", ["search", "find"])
        _write_skill(base_dir, "code", "Write code", ["code", "implement"])
        assembler = _make_assembler(base_dir)
        result = assembler.assemble()
        assert "# Available Skills" in result
        assert "**search**: Search the web (triggers: search, find)" in result
        assert "**code**: Write code (triggers: code, implement)" in result

        audit = assembler.get_audit()
        capabilities_layer = next(ly for ly in audit.layers if ly.name == "capabilities")
        assert capabilities_layer.skipped is False
        assert capabilities_layer.source == "skills"

    def test_capabilities_disabled_skips_layer(self, base_dir):
        _write_skill(base_dir, "search", "Search the web", ["search"])
        assembler = _make_assembler(base_dir, capabilities_enabled=False)
        result = assembler.assemble()
        assert "# Available Skills" not in result

        audit = assembler.get_audit()
        capabilities_layer = next(ly for ly in audit.layers if ly.name == "capabilities")
        assert capabilities_layer.skipped is True
        assert capabilities_layer.skip_reason == "disabled"

    def test_capabilities_no_skills_skips_layer(self, base_dir):
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        capabilities_layer = next(ly for ly in audit.layers if ly.name == "capabilities")
        assert capabilities_layer.skipped is True
        assert capabilities_layer.skip_reason == "no skills loaded"

    def test_capabilities_formats_skills_same_as_legacy(self, base_dir):
        _write_skill(base_dir, "calculator", "Do math", ["calc", "math"])
        assembler = _make_assembler(base_dir)
        result = assembler.assemble()
        # Format should match legacy build_system_prompt()
        assert "- **calculator**: Do math (triggers: calc, math)" in result


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAssemblyIntegration:
    def test_all_layers_assembled_in_precedence_order(self, base_dir):
        _write_identity(base_dir, "Identity layer.")
        _write_personality(base_dir, "Personality layer.")
        _write_rule(base_dir, "rule1", "Governance layer.")
        _write_skill(base_dir, "skill1", "Capabilities layer.", ["test"])

        assembler = _make_assembler(base_dir, personality_enabled=True)
        result = assembler.assemble()

        # Check layers are in correct order with separators
        parts = result.split("\n\n---\n\n")
        assert len(parts) == 4
        assert "Identity layer." in parts[0]
        assert "Personality layer." in parts[1]
        assert "Governance layer." in parts[2]
        assert "Capabilities layer." in parts[3]

    def test_skipped_layers_excluded_from_output(self, base_dir):
        _write_identity(base_dir, "Identity layer.")
        _write_rule(base_dir, "rule1", "Governance layer.")
        # Personality is disabled by default
        # No skills, so capabilities will be skipped

        assembler = _make_assembler(base_dir)
        result = assembler.assemble()

        parts = result.split("\n\n---\n\n")
        assert len(parts) == 2  # Only identity and governance
        assert "Identity layer." in parts[0]
        assert "Governance layer." in parts[1]

        audit = assembler.get_audit()
        assert set(audit.skipped_layers) == {"personality", "capabilities"}

    def test_only_active_layers_with_content_in_result(self, base_dir):
        _write_identity(base_dir, "Identity.")
        # Everything else skipped or missing

        assembler = _make_assembler(base_dir)
        result = assembler.assemble()

        assert result == "Identity."
        assert "---" not in result  # No separators when only one layer

    def test_empty_assembler_returns_empty_string(self, base_dir):
        assembler = _make_assembler(
            base_dir,
            identity_enabled=False,
            personality_enabled=False,
            governance_enabled=False,
            capabilities_enabled=False,
        )
        result = assembler.assemble()
        assert result == ""

        audit = assembler.get_audit()
        assert len(audit.assembly_order) == 0
        assert len(audit.skipped_layers) == 4


# ---------------------------------------------------------------------------
# Audit logging tests
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_audit_available_after_assemble(self, base_dir):
        _write_identity(base_dir, "Identity.")
        assembler = _make_assembler(base_dir)

        assert assembler.get_audit() is None
        assembler.assemble()
        audit = assembler.get_audit()
        assert audit is not None

    def test_audit_records_assembly_order(self, base_dir):
        _write_identity(base_dir, "Identity.")
        _write_rule(base_dir, "rule1", "Governance.")
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        assert audit.assembly_order == ["identity", "governance"]

    def test_audit_records_skipped_layers(self, base_dir):
        _write_identity(base_dir, "Identity.")
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        # Personality disabled by default, no rules/skills loaded
        assert "personality" in audit.skipped_layers
        assert "governance" in audit.skipped_layers
        assert "capabilities" in audit.skipped_layers

    def test_audit_records_token_estimates(self, base_dir):
        _write_identity(base_dir, "x" * 400)  # ~100 tokens
        _write_rule(base_dir, "rule1", "y" * 800)  # ~200 tokens
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        # Token estimate is len(content) // 4
        identity_layer = next(ly for ly in audit.layers if ly.name == "identity")
        assert identity_layer.token_estimate == 100

        governance_layer = next(ly for ly in audit.layers if ly.name == "governance")
        # Governance adds header "# Rule: rule1\n\n" which adds ~14 chars
        # So total content is 800 + 14 = 814, token estimate = 814 // 4 = 203
        assert governance_layer.token_estimate == 203

        # Total tokens is sum of active layers
        assert audit.total_tokens == 100 + 203

    def test_audit_records_external_layers(self, base_dir):
        _write_identity(base_dir, "Identity.")
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        assert audit.external_layers == ["knowledge", "operational_context", "ephemeral"]

    def test_audit_timestamp_is_iso_format(self, base_dir):
        _write_identity(base_dir, "Identity.")
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        # Should parse without error
        from datetime import datetime
        datetime.fromisoformat(audit.timestamp)

    def test_info_log_emitted_with_summary(self, base_dir, caplog):
        _write_identity(base_dir, "Identity.")
        _write_rule(base_dir, "rule1", "Governance.")
        assembler = _make_assembler(base_dir)

        with caplog.at_level(logging.INFO, logger="fipsagents.baseagent.prompt_assembly"):
            assembler.assemble()

        # Check that info log contains expected data
        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert "prompt_assembly" in record.message
        assert "layers=['identity', 'governance']" in record.message
        assert "skipped=['personality', 'capabilities']" in record.message


# ---------------------------------------------------------------------------
# Layer metadata tests
# ---------------------------------------------------------------------------


class TestLayerMetadata:
    def test_layer_precedence_is_correct(self, base_dir):
        _write_identity(base_dir, "I")
        _write_personality(base_dir, "P")
        _write_rule(base_dir, "r", "G")
        _write_skill(base_dir, "s", "C", ["test"])

        assembler = _make_assembler(base_dir, personality_enabled=True)
        assembler.assemble()

        audit = assembler.get_audit()
        assert audit.layers[0].name == "identity"
        assert audit.layers[0].precedence == 0
        assert audit.layers[1].name == "personality"
        assert audit.layers[1].precedence == 1
        assert audit.layers[2].name == "governance"
        assert audit.layers[2].precedence == 2
        assert audit.layers[3].name == "capabilities"
        assert audit.layers[3].precedence == 3

    def test_layer_mutability_is_immutable(self, base_dir):
        _write_identity(base_dir, "Identity.")
        assembler = _make_assembler(base_dir)
        assembler.assemble()

        audit = assembler.get_audit()
        for layer in audit.layers:
            assert layer.mutability == "immutable"


# ---------------------------------------------------------------------------
# Data-driven tests for skip reasons
# ---------------------------------------------------------------------------


class TestSkipReasons:
    @pytest.mark.parametrize("layer_name,enabled_param,skip_reason", [
        ("identity", "identity_enabled", "disabled"),
        ("personality", "personality_enabled", "disabled"),
        ("governance", "governance_enabled", "disabled"),
        ("capabilities", "capabilities_enabled", "disabled"),
    ])
    def test_disabled_layers_have_correct_skip_reason(
        self, base_dir, layer_name, enabled_param, skip_reason
    ):
        kwargs = {enabled_param: False}
        assembler = _make_assembler(base_dir, **kwargs)
        assembler.assemble()

        audit = assembler.get_audit()
        layer = next(ly for ly in audit.layers if ly.name == layer_name)
        assert layer.skipped is True
        assert layer.skip_reason == skip_reason


# ---------------------------------------------------------------------------
# Learned skills in capabilities layer
# ---------------------------------------------------------------------------


def _write_learned_skill(
    base_dir: Path, name: str, description: str, triggers: list[str],
):
    """Write a learned skill with ``author: agent`` frontmatter."""
    skill_dir = base_dir / "learned_skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    trigger_str = "\n".join(f"  - {t}" for t in triggers) if triggers else "  - none"
    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
author: agent
triggers:
{trigger_str}
---
Learned skill body content.
""")


class TestLearnedSkillsInCapabilities:
    """Verify that learned skills loaded via ``SkillLoader.load_learned()``
    appear in the capabilities layer with the ``[learned]`` tag."""

    @pytest.fixture
    def base_dir_with_learned(self, tmp_path):
        """Directory tree with bundled + learned skills."""
        (tmp_path / "prompts").mkdir()
        (tmp_path / "rules").mkdir()
        (tmp_path / "skills").mkdir()
        (tmp_path / "learned_skills").mkdir()
        return tmp_path

    def test_learned_skill_appears_in_capabilities(self, base_dir_with_learned):
        base = base_dir_with_learned
        _write_skill(base, "bundled-search", "Search the web", ["search"])
        _write_learned_skill(base, "data-cleanup", "Clean messy data", ["clean"])

        skills = SkillLoader()
        skills.load_all(base / "skills")
        skills.load_learned(base / "learned_skills")

        assembler = _make_assembler(base, skills=skills)
        result = assembler.assemble()

        assert "**bundled-search**: Search the web" in result
        assert "**data-cleanup**: [learned] Clean messy data" in result

        audit = assembler.get_audit()
        cap_layer = next(ly for ly in audit.layers if ly.name == "capabilities")
        assert cap_layer.skipped is False
        assert cap_layer.source == "skills"

    def test_learned_skill_only_no_bundled(self, base_dir_with_learned):
        """Capabilities layer works when all skills are learned."""
        base = base_dir_with_learned
        _write_learned_skill(base, "auto-fix", "Fix common errors", ["fix", "repair"])

        skills = SkillLoader()
        skills.load_all(base / "skills")  # empty
        skills.load_learned(base / "learned_skills")

        assembler = _make_assembler(base, skills=skills)
        result = assembler.assemble()

        assert "# Available Skills" in result
        assert "**auto-fix**: [learned] Fix common errors (triggers: fix, repair)" in result

        audit = assembler.get_audit()
        cap_layer = next(ly for ly in audit.layers if ly.name == "capabilities")
        assert cap_layer.skipped is False

    def test_learned_skill_tagged_in_manifest(self, base_dir_with_learned):
        """SkillLoader.get_manifest() tags learned skills with [learned]."""
        base = base_dir_with_learned
        _write_skill(base, "bundled", "Bundled skill", ["b"])
        _write_learned_skill(base, "learned", "Learned skill", ["l"])

        skills = SkillLoader()
        skills.load_all(base / "skills")
        skills.load_learned(base / "learned_skills")

        manifest = skills.get_manifest()
        by_name = {e.name: e for e in manifest}

        assert "bundled" in by_name
        assert "[learned]" not in by_name["bundled"].description
        assert "learned" in by_name
        assert by_name["learned"].description.startswith("[learned]")

    def test_audit_log_with_learned_skills(self, base_dir_with_learned, caplog):
        """Assembly audit log reflects capabilities layer when learned skills
        are present."""
        base = base_dir_with_learned
        _write_learned_skill(base, "summarize", "Summarize text", ["summarize"])

        skills = SkillLoader()
        skills.load_all(base / "skills")
        skills.load_learned(base / "learned_skills")

        assembler = _make_assembler(base, skills=skills)
        with caplog.at_level(logging.INFO, logger="fipsagents.baseagent.prompt_assembly"):
            assembler.assemble()

        assert any("capabilities" in r.message for r in caplog.records)
        audit = assembler.get_audit()
        assert "capabilities" in audit.assembly_order
        assert "capabilities" not in audit.skipped_layers
