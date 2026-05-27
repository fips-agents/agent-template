"""Tests for self-healing stock tools and learned-skill loading."""
from __future__ import annotations

import json
from pathlib import Path

import frontmatter
import pytest

from fipsagents.baseagent.config import AgentConfig, SelfHealingConfig
from fipsagents.baseagent.events import SkillEdited, SkillLearned, SkillRolledBack
from fipsagents.baseagent.skills import SkillLoader
from fipsagents.baseagent.tools.self_healing import (
    STOCK_TOOL_SPEC,
    make_self_healing_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_stub(
    base_dir: Path,
    trust_level: int = 2,
    trust_domains: list[str] | None = None,
    review_policy: str = "audit_only",
    max_skills: int = 50,
):
    """Build a minimal agent-like object for the factory."""
    agent = type("Agent", (), {})()
    agent.config = AgentConfig(
        self_healing=SelfHealingConfig(
            enabled=True,
            trust_level=trust_level,
            trust_domains=trust_domains or ["document_processing"],
            review_policy=review_policy,
            learned_skills_dir=str(base_dir / "learned_skills"),
            max_skills=max_skills,
        )
    )
    agent._base_dir = base_dir
    agent._self_healing_events = []
    return agent


def _write_skill(skill_dir: Path, name: str, version: int = 1, **extra_meta):
    """Write a SKILL.md with frontmatter into a skill directory."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name,
        "description": f"Test skill {name}",
        "version": version,
        "triggers": [f"trigger-{name}"],
        **extra_meta,
    }
    post = frontmatter.Post(f"Content for {name}", **meta)
    (skill_dir / "SKILL.md").write_text(frontmatter.dumps(post), encoding="utf-8")


# ---------------------------------------------------------------------------
# STOCK_TOOL_SPEC condition
# ---------------------------------------------------------------------------


class TestStockToolSpecCondition:
    def test_condition_true_when_enabled(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=2)
        assert STOCK_TOOL_SPEC.condition(agent) is True

    def test_condition_false_when_disabled(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        agent.config.self_healing.enabled = False
        assert STOCK_TOOL_SPEC.condition(agent) is False

    def test_condition_false_when_no_config(self):
        agent = type("Agent", (), {})()
        assert STOCK_TOOL_SPEC.condition(agent) is False


# ---------------------------------------------------------------------------
# learn_skill
# ---------------------------------------------------------------------------


class TestLearnSkill:
    @pytest.mark.asyncio
    async def test_creates_new_skill(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        result = json.loads(await learn(
            name="summarize-pdf",
            description="Summarize PDF documents",
            content="Use Docling to parse, then summarize.",
            domain="document_processing",
            trigger="summarize a pdf",
        ))

        assert result["skill_name"] == "summarize-pdf"
        assert result["version"] == 1
        assert result["review_status"] == "auto_approved"
        assert result["domain"] == "document_processing"

        # Verify file exists.
        skill_path = tmp_path / "learned_skills" / "summarize-pdf" / "SKILL.md"
        assert skill_path.exists()
        post = frontmatter.load(str(skill_path))
        assert post.metadata["name"] == "summarize-pdf"
        assert post.metadata["version"] == 1
        assert post.metadata["author"] == "agent"

    @pytest.mark.asyncio
    async def test_updates_existing_skill_archives_old(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        # Create initial version.
        await learn(
            name="my-skill",
            description="v1",
            content="Version 1 content",
            domain="document_processing",
            trigger="do the thing",
        )

        # Update to v2.
        result = json.loads(await learn(
            name="my-skill",
            description="v2",
            content="Version 2 content",
            domain="document_processing",
            trigger="do the thing better",
        ))

        assert result["version"] == 2

        # Check archive exists.
        archive = tmp_path / "learned_skills" / "my-skill" / ".versions" / "v1.md"
        assert archive.exists()

        # Current SKILL.md should be v2.
        skill_path = tmp_path / "learned_skills" / "my-skill" / "SKILL.md"
        post = frontmatter.load(str(skill_path))
        assert post.metadata["version"] == 2

    @pytest.mark.asyncio
    async def test_trust_level_zero_rejected(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=0)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        result = json.loads(await learn(
            name="blocked",
            description="Should fail",
            content="Nope",
            domain="document_processing",
            trigger="test",
        ))

        assert "error" in result
        assert result["current"] == 0

    @pytest.mark.asyncio
    async def test_invalid_name_rejected(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        result = json.loads(await learn(
            name="Invalid Name!",
            description="Bad name",
            content="Nope",
            domain="document_processing",
            trigger="test",
        ))

        assert "error" in result
        assert "kebab-case" in result["detail"]

    @pytest.mark.asyncio
    async def test_domain_not_permitted(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=2, trust_domains=["allowed"])
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        result = json.loads(await learn(
            name="bad-domain",
            description="Wrong domain",
            content="Nope",
            domain="forbidden_domain",
            trigger="test",
        ))

        assert result["error"] == "Domain not permitted"
        assert result["domain"] == "forbidden_domain"

    @pytest.mark.asyncio
    async def test_trust_level_4_bypasses_domain_check(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=4, trust_domains=["allowed"])
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        result = json.loads(await learn(
            name="any-domain",
            description="Trust 4 can write any domain",
            content="Works",
            domain="exotic_domain",
            trigger="test",
        ))

        assert result["skill_name"] == "any-domain"
        assert result["domain"] == "exotic_domain"

    @pytest.mark.asyncio
    async def test_review_policy_pending_review(self, tmp_path):
        agent = _make_agent_stub(tmp_path, review_policy="human_review")
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        result = json.loads(await learn(
            name="needs-review",
            description="Pending",
            content="Content",
            domain="document_processing",
            trigger="test",
        ))

        assert result["review_status"] == "pending_review"

    @pytest.mark.asyncio
    async def test_emits_skill_learned_event(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        await learn(
            name="event-test",
            description="Test event",
            content="Content",
            domain="document_processing",
            trigger="test",
        )

        assert len(agent._self_healing_events) == 1
        event = agent._self_healing_events[0]
        assert isinstance(event, SkillLearned)
        assert event.skill_name == "event-test"
        assert event.version == 1

    @pytest.mark.asyncio
    async def test_new_skill_does_not_emit_skill_edited(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        await learn(
            name="fresh-skill",
            description="Brand new",
            content="Content",
            domain="document_processing",
            trigger="test",
        )

        assert len(agent._self_healing_events) == 1
        assert isinstance(agent._self_healing_events[0], SkillLearned)
        assert not any(isinstance(e, SkillEdited) for e in agent._self_healing_events)

    @pytest.mark.asyncio
    async def test_update_skill_emits_skill_edited_event(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        # Create v1.
        await learn(
            name="evolving-skill",
            description="v1",
            content="Version 1",
            domain="document_processing",
            trigger="test",
        )
        agent._self_healing_events.clear()

        # Update to v2.
        await learn(
            name="evolving-skill",
            description="v2",
            content="Version 2",
            domain="document_processing",
            trigger="test",
        )

        learned_events = [e for e in agent._self_healing_events if isinstance(e, SkillLearned)]
        edited_events = [e for e in agent._self_healing_events if isinstance(e, SkillEdited)]

        assert len(learned_events) == 1
        assert learned_events[0].version == 2

        assert len(edited_events) == 1
        assert edited_events[0].skill_name == "evolving-skill"
        assert edited_events[0].from_version == 1
        assert edited_events[0].to_version == 2

    @pytest.mark.asyncio
    async def test_maturation_proto_agent_blocks_learn(self, tmp_path):
        """Maturation gate blocks learn_skill even if trust_level config passes."""
        agent = _make_agent_stub(tmp_path, trust_level=1)  # passes static check
        from fipsagents.baseagent.maturation import MaturationManager
        from fipsagents.baseagent.trust import TrustManager, TrustState

        tm = TrustManager(state=TrustState(level=0, score=0.0))
        agent._maturation_manager = MaturationManager(tm)

        tools = make_self_healing_tools(agent)
        learn = tools[0]
        result = json.loads(await learn(
            name="blocked-skill",
            description="Should fail",
            content="Content",
            domain="document_processing",
            trigger="test",
        ))
        assert "error" in result
        assert "suggest_skill" in result["error"]
        assert "proto_agent" in result["error"]

    @pytest.mark.asyncio
    async def test_max_skills_cap_rejects_new_skill(self, tmp_path):
        agent = _make_agent_stub(tmp_path, max_skills=2)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        # Create 2 skills to fill the cap.
        await learn(
            name="skill-one",
            description="First",
            content="Content",
            domain="document_processing",
            trigger="test",
        )
        await learn(
            name="skill-two",
            description="Second",
            content="Content",
            domain="document_processing",
            trigger="test",
        )

        # Third skill should be rejected.
        result = json.loads(await learn(
            name="skill-three",
            description="Over the cap",
            content="Content",
            domain="document_processing",
            trigger="test",
        ))

        assert "error" in result
        assert "cap reached" in result["error"]
        # Verify no directory was created for the rejected skill.
        assert not (tmp_path / "learned_skills" / "skill-three").exists()

    @pytest.mark.asyncio
    async def test_max_skills_allows_update_of_existing(self, tmp_path):
        agent = _make_agent_stub(tmp_path, max_skills=2)
        tools = make_self_healing_tools(agent)
        learn = tools[0]

        # Create 2 skills to fill the cap.
        await learn(
            name="skill-one",
            description="First",
            content="Content v1",
            domain="document_processing",
            trigger="test",
        )
        await learn(
            name="skill-two",
            description="Second",
            content="Content v1",
            domain="document_processing",
            trigger="test",
        )

        # Updating an existing skill should succeed despite the cap.
        result = json.loads(await learn(
            name="skill-one",
            description="First updated",
            content="Content v2",
            domain="document_processing",
            trigger="test",
        ))

        assert result["skill_name"] == "skill-one"
        assert result["version"] == 2


# ---------------------------------------------------------------------------
# suggest_skill
# ---------------------------------------------------------------------------


class TestSuggestSkill:
    @pytest.mark.asyncio
    async def test_returns_proposed_status(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=0)
        tools = make_self_healing_tools(agent)
        suggest = tools[1]

        result = json.loads(await suggest(
            name="suggested-skill",
            description="A suggestion",
            content="Proposed content",
            domain="any",
            trigger="test",
        ))

        assert result["status"] == "proposed"
        assert result["review_status"] == "pending_review"

    @pytest.mark.asyncio
    async def test_does_not_write_to_disk(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=0)
        tools = make_self_healing_tools(agent)
        suggest = tools[1]

        await suggest(
            name="no-disk",
            description="Should not persist",
            content="Content",
            domain="any",
            trigger="test",
        )

        skill_path = tmp_path / "learned_skills" / "no-disk" / "SKILL.md"
        assert not skill_path.exists()

    @pytest.mark.asyncio
    async def test_emits_event(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=0)
        tools = make_self_healing_tools(agent)
        suggest = tools[1]

        await suggest(
            name="event-test",
            description="Test",
            content="Content",
            domain="test",
            trigger="test",
        )

        assert len(agent._self_healing_events) == 1
        event = agent._self_healing_events[0]
        assert isinstance(event, SkillLearned)
        assert event.review_status == "pending_review"
        assert event.version == 0

    @pytest.mark.asyncio
    async def test_invalid_name_rejected(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        suggest = tools[1]

        result = json.loads(await suggest(
            name="BAD",
            description="Bad name",
            content="Nope",
            domain="any",
            trigger="test",
        ))

        assert "error" in result


# ---------------------------------------------------------------------------
# rollback_skill
# ---------------------------------------------------------------------------


class TestRollbackSkill:
    @pytest.mark.asyncio
    async def test_rollback_restores_archived_version(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=3)
        tools = make_self_healing_tools(agent)
        learn = tools[0]
        rollback = tools[2]

        # Create v1.
        await learn(
            name="rollback-test",
            description="v1",
            content="Version 1",
            domain="document_processing",
            trigger="test",
        )

        # Create v2 (archives v1).
        await learn(
            name="rollback-test",
            description="v2",
            content="Version 2",
            domain="document_processing",
            trigger="test",
        )

        # Rollback to v1.
        result = json.loads(await rollback(
            name="rollback-test",
            to_version=1,
            reason="v2 was bad",
        ))

        assert result["from_version"] == 2
        assert result["to_version"] == 1
        assert result["new_version"] == 3

        # Verify SKILL.md is now v3 with v1's content restored.
        skill_path = tmp_path / "learned_skills" / "rollback-test" / "SKILL.md"
        post = frontmatter.load(str(skill_path))
        assert post.metadata["version"] == 3

    @pytest.mark.asyncio
    async def test_rollback_insufficient_trust(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=2)
        tools = make_self_healing_tools(agent)
        rollback = tools[2]

        result = json.loads(await rollback(
            name="any",
            to_version=1,
        ))

        assert result["error"] == "Insufficient trust level for rollback"
        assert result["required"] == 3

    @pytest.mark.asyncio
    async def test_rollback_version_not_found(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=3)
        tools = make_self_healing_tools(agent)
        learn = tools[0]
        rollback = tools[2]

        await learn(
            name="no-archive",
            description="v1",
            content="Only version",
            domain="document_processing",
            trigger="test",
        )

        result = json.loads(await rollback(
            name="no-archive",
            to_version=99,
        ))

        assert result["error"] == "Version not found"

    @pytest.mark.asyncio
    async def test_rollback_skill_not_found(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=3)
        tools = make_self_healing_tools(agent)
        rollback = tools[2]

        # Create the .versions dir but no SKILL.md.
        versions_dir = tmp_path / "learned_skills" / "gone" / ".versions"
        versions_dir.mkdir(parents=True)
        (versions_dir / "v1.md").write_text("archived content")

        result = json.loads(await rollback(name="gone", to_version=1))
        assert result["error"] == "Skill not found"

    @pytest.mark.asyncio
    async def test_rollback_emits_event(self, tmp_path):
        agent = _make_agent_stub(tmp_path, trust_level=3)
        tools = make_self_healing_tools(agent)
        learn = tools[0]
        rollback = tools[2]

        await learn(
            name="event-rb",
            description="v1",
            content="V1",
            domain="document_processing",
            trigger="test",
        )
        await learn(
            name="event-rb",
            description="v2",
            content="V2",
            domain="document_processing",
            trigger="test",
        )
        agent._self_healing_events.clear()

        await rollback(name="event-rb", to_version=1, reason="oops")

        assert len(agent._self_healing_events) == 1
        event = agent._self_healing_events[0]
        assert isinstance(event, SkillRolledBack)
        assert event.from_version == 2
        assert event.to_version == 1
        assert event.reason == "oops"


# ---------------------------------------------------------------------------
# SkillLoader.load_learned
# ---------------------------------------------------------------------------


class TestSkillLoaderLearnedSkills:
    def test_load_learned_basic(self, tmp_path):
        loader = SkillLoader()

        learned_dir = tmp_path / "learned_skills"
        _write_skill(learned_dir / "my-learned", "my-learned")

        loaded = loader.load_learned(learned_dir)
        assert loaded == ["my-learned"]
        assert "my-learned" in loader
        assert loader._skills["my-learned"].learned is True

    def test_load_learned_skips_bundled_conflict(self, tmp_path):
        loader = SkillLoader()

        # Simulate a bundled skill.
        bundled_dir = tmp_path / "skills"
        _write_skill(bundled_dir / "overlap", "overlap")
        loader.load_all(bundled_dir)
        assert loader._skills["overlap"].learned is False

        # Try loading a learned skill with the same name.
        learned_dir = tmp_path / "learned_skills"
        _write_skill(learned_dir / "overlap", "overlap", version=5)

        loaded = loader.load_learned(learned_dir)
        assert loaded == []
        # Bundled version is preserved.
        assert loader._skills["overlap"].learned is False

    def test_load_learned_nonexistent_dir(self, tmp_path):
        loader = SkillLoader()
        loaded = loader.load_learned(tmp_path / "does-not-exist")
        assert loaded == []

    def test_load_learned_skips_invalid_skill(self, tmp_path):
        loader = SkillLoader()
        learned_dir = tmp_path / "learned_skills"

        # Write a skill missing required fields.
        bad_dir = learned_dir / "bad-skill"
        bad_dir.mkdir(parents=True)
        (bad_dir / "SKILL.md").write_text("---\nauthor: test\n---\nBad", encoding="utf-8")

        loaded = loader.load_learned(learned_dir)
        assert loaded == []


class TestSkillLoaderQuarantine:
    def test_quarantined_skills_excluded_from_load(self, tmp_path):
        """Quarantined skills are not loaded by SkillLoader.load_learned()."""
        learned_dir = tmp_path / "learned_skills"

        # Active skill.
        active_dir = learned_dir / "active-skill"
        active_dir.mkdir(parents=True)
        (active_dir / "SKILL.md").write_text(
            "---\nname: active-skill\ndomain: nlp\nversion: 1\n"
            "description: Active skill\n---\nActive content\n"
        )

        # Quarantined skill.
        q_dir = learned_dir / "quarantined-skill"
        q_dir.mkdir(parents=True)
        (q_dir / "SKILL.md").write_text(
            "---\nname: quarantined-skill\ndomain: nlp\nquarantined: true\n"
            "version: 1\ndescription: Quarantined skill\n---\nQuarantined\n"
        )

        loader = SkillLoader()
        loaded = loader.load_learned(learned_dir)
        assert "active-skill" in loaded
        assert "quarantined-skill" not in loaded


class TestSkillLoaderManifest:
    def test_manifest_tags_learned_skills(self, tmp_path):
        loader = SkillLoader()

        bundled_dir = tmp_path / "skills"
        _write_skill(bundled_dir / "bundled", "bundled")
        loader.load_all(bundled_dir)

        learned_dir = tmp_path / "learned_skills"
        _write_skill(learned_dir / "learned", "learned")
        loader.load_learned(learned_dir)

        manifest = loader.get_manifest()
        descs = {e.name: e.description for e in manifest}

        assert not descs["bundled"].startswith("[learned]")
        assert descs["learned"].startswith("[learned]")


# ---------------------------------------------------------------------------
# Factory returns correct number of tools
# ---------------------------------------------------------------------------


class TestMakeSelfHealingTools:
    def test_returns_three_tools(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        assert len(tools) == 3

    def test_tool_names(self, tmp_path):
        agent = _make_agent_stub(tmp_path)
        tools = make_self_healing_tools(agent)
        from fipsagents.baseagent.tools._registry import _TOOL_MARKER
        names = [getattr(t, _TOOL_MARKER).name for t in tools]
        assert names == ["learn_skill", "suggest_skill", "rollback_skill"]
