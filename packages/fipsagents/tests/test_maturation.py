"""Tests for agent maturation: stage derivation, permissions, progress, quarantine."""
from __future__ import annotations

import frontmatter
import pytest

from fipsagents.baseagent.maturation import (
    MaturationManager,
    MaturationStage,
    StagePermissions,
    quarantine_out_of_scope_skills,
)
from fipsagents.baseagent.trust import TrustManager, TrustState


def _make_trust(level: int = 0, score: float = 0.0) -> TrustManager:
    """Create a TrustManager pre-set to a given level and score."""
    state = TrustState(level=level, score=score)
    return TrustManager(state=state)


# ---------------------------------------------------------------------------
# Stage derivation
# ---------------------------------------------------------------------------


class TestMaturationStageDerivation:
    @pytest.mark.parametrize(
        "level,expected",
        [
            (0, MaturationStage.PROTO_AGENT),
            (1, MaturationStage.APPRENTICE),
            (2, MaturationStage.JOURNEYMAN),
            (3, MaturationStage.JOURNEYMAN),
            (4, MaturationStage.SPECIALIST),
        ],
    )
    def test_stage_from_trust_level(self, level, expected):
        tm = _make_trust(level=level, score=float(level * 100))
        mm = MaturationManager(tm)
        assert mm.current_stage() == expected

    def test_custom_boundaries(self):
        tm = _make_trust(level=2, score=50.0)
        mm = MaturationManager(tm, apprentice_max_trust=2, specialist_min_trust=4)
        assert mm.current_stage() == MaturationStage.APPRENTICE

        tm2 = _make_trust(level=3, score=200.0)
        mm2 = MaturationManager(tm2, apprentice_max_trust=2, specialist_min_trust=4)
        assert mm2.current_stage() == MaturationStage.JOURNEYMAN

    def test_stage_values_are_strings(self):
        """MaturationStage is a str enum, so values should be plain strings."""
        assert MaturationStage.PROTO_AGENT == "proto_agent"
        assert MaturationStage.SPECIALIST == "specialist"


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


class TestPermissions:
    def test_proto_agent_cannot_create_skills(self):
        tm = _make_trust(level=0)
        mm = MaturationManager(tm)
        perms = mm.get_permissions()
        assert perms.can_create_skills is False
        assert perms.can_edit_own is False
        assert perms.can_delete_own is False
        assert perms.review_gate == "none"

    def test_apprentice_needs_human_review(self):
        tm = _make_trust(level=1, score=10.0)
        mm = MaturationManager(tm)
        perms = mm.get_permissions()
        assert perms.can_create_skills is True
        assert perms.review_gate == "human_review"
        assert perms.can_edit_own is False

    def test_journeyman_peer_review(self):
        tm = _make_trust(level=2, score=50.0)
        mm = MaturationManager(tm)
        perms = mm.get_permissions()
        assert perms.review_gate == "peer_review"
        assert perms.can_edit_own is True
        assert perms.can_delete_own is False

    def test_specialist_full_autonomy(self):
        tm = _make_trust(level=4, score=500.0)
        mm = MaturationManager(tm)
        perms = mm.get_permissions()
        assert perms.review_gate == "audit_only"
        assert perms.can_edit_own is True
        assert perms.can_delete_own is True

    def test_explicit_stage_parameter(self):
        tm = _make_trust(level=0)
        mm = MaturationManager(tm)
        perms = mm.get_permissions(MaturationStage.SPECIALIST)
        assert perms.can_delete_own is True

    def test_permissions_are_frozen(self):
        perms = StagePermissions(
            can_create_skills=True,
            review_gate="human_review",
            can_edit_own=False,
            can_delete_own=False,
        )
        with pytest.raises(AttributeError):
            perms.can_create_skills = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Promotion progress
# ---------------------------------------------------------------------------


class TestPromotionProgress:
    def test_proto_agent_progress(self):
        tm = _make_trust(level=0, score=5.0)
        mm = MaturationManager(tm)
        progress = mm.promotion_progress()
        assert progress["current_stage"] == "proto_agent"
        assert progress["next_stage"] == "apprentice"
        assert progress["pct_complete"] == 50.0

    def test_apprentice_progress(self):
        tm = _make_trust(level=1, score=25.0)
        mm = MaturationManager(tm)
        progress = mm.promotion_progress()
        assert progress["current_stage"] == "apprentice"
        assert progress["next_stage"] == "journeyman"
        # Default thresholds: level_2 = 50.0, so 25/50 = 50%
        assert progress["threshold_for_next"] == 50.0
        assert progress["pct_complete"] == 50.0

    def test_journeyman_progress(self):
        tm = _make_trust(level=3, score=400.0)
        mm = MaturationManager(tm)
        progress = mm.promotion_progress()
        assert progress["current_stage"] == "journeyman"
        assert progress["next_stage"] == "specialist"
        # Default thresholds: level_4 = 500.0, so 400/500 = 80%
        assert progress["threshold_for_next"] == 500.0
        assert progress["pct_complete"] == 80.0

    def test_specialist_at_max(self):
        tm = _make_trust(level=4, score=500.0)
        mm = MaturationManager(tm)
        progress = mm.promotion_progress()
        assert progress["current_stage"] == "specialist"
        assert progress["next_stage"] is None
        assert progress["pct_complete"] == 100.0
        assert progress["threshold_for_next"] is None

    def test_progress_capped_at_100(self):
        """Score exceeding the threshold should cap at 100%."""
        tm = _make_trust(level=0, score=15.0)
        mm = MaturationManager(tm)
        progress = mm.promotion_progress()
        assert progress["pct_complete"] == 100.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_format(self):
        tm = _make_trust(level=2, score=80.0)
        mm = MaturationManager(tm)
        summary = mm.get_summary()
        assert summary["stage"] == "journeyman"
        assert "permissions" in summary
        assert "trust" in summary
        assert "progress" in summary
        assert summary["trust"]["level"] == 2
        assert summary["trust"]["score"] == 80.0

    def test_summary_trust_counters(self):
        tm = _make_trust(level=0, score=0.0)
        tm.record_completion(reason="test")
        tm.record_completion(reason="test")
        tm.record_failure(reason="oops")
        mm = MaturationManager(tm)
        summary = mm.get_summary()
        assert summary["trust"]["completions"] == 2
        assert summary["trust"]["failures"] == 1
        assert summary["trust"]["violations"] == 0

    def test_summary_permissions_match_stage(self):
        tm = _make_trust(level=4, score=500.0)
        mm = MaturationManager(tm)
        summary = mm.get_summary()
        assert summary["permissions"]["can_create_skills"] is True
        assert summary["permissions"]["review_gate"] == "audit_only"
        assert summary["permissions"]["can_edit_own"] is True
        assert summary["permissions"]["can_delete_own"] is True


# ---------------------------------------------------------------------------
# Events in StreamEvent union
# ---------------------------------------------------------------------------


class TestMaturationEvents:
    def test_stage_promoted_in_union(self):
        from fipsagents.baseagent.events import StagePromoted, StreamEvent
        import typing

        args = typing.get_args(StreamEvent)
        assert StagePromoted in args

    def test_stage_demoted_in_union(self):
        from fipsagents.baseagent.events import StageDemoted, StreamEvent
        import typing

        args = typing.get_args(StreamEvent)
        assert StageDemoted in args

    def test_stage_promoted_fields(self):
        from fipsagents.baseagent.events import StagePromoted

        event = StagePromoted(
            from_stage="apprentice",
            to_stage="journeyman",
            trust_level=2,
            reason="trust level increased",
        )
        assert event.from_stage == "apprentice"
        assert event.to_stage == "journeyman"
        assert event.trust_level == 2

    def test_stage_demoted_fields(self):
        from fipsagents.baseagent.events import StageDemoted

        event = StageDemoted(
            from_stage="journeyman",
            to_stage="apprentice",
            trust_level=1,
            reason="trust level decreased",
        )
        assert event.from_stage == "journeyman"
        assert event.to_stage == "apprentice"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestMaturationConfig:
    def test_defaults(self):
        from fipsagents.baseagent.config import MaturationConfig

        cfg = MaturationConfig()
        assert cfg.enabled is False
        assert cfg.apprentice_max_trust == 1
        assert cfg.journeyman_max_trust == 3
        assert cfg.specialist_min_trust == 4
        assert cfg.promotion_requires == "auto"

    def test_agent_config_includes_maturation(self):
        from fipsagents.baseagent.config import AgentConfig, MaturationConfig

        cfg = AgentConfig()
        assert isinstance(cfg.maturation, MaturationConfig)
        assert cfg.maturation.enabled is False

    def test_custom_values(self):
        from fipsagents.baseagent.config import MaturationConfig

        cfg = MaturationConfig(
            enabled=True,
            apprentice_max_trust=2,
            journeyman_max_trust=3,
            specialist_min_trust=4,
            promotion_requires="human_approval",
        )
        assert cfg.enabled is True
        assert cfg.apprentice_max_trust == 2
        assert cfg.promotion_requires == "human_approval"

    def test_validation_bounds(self):
        from pydantic import ValidationError
        from fipsagents.baseagent.config import MaturationConfig

        with pytest.raises(ValidationError):
            MaturationConfig(apprentice_max_trust=5)  # max is 4

        with pytest.raises(ValidationError):
            MaturationConfig(specialist_min_trust=0)  # min is 1


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


class TestQuarantine:
    def test_quarantine_marks_out_of_scope(self, tmp_path):
        """Skills outside trust domains get quarantined."""
        skill_dir = tmp_path / "learned_skills" / "nlp-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: nlp-skill\ndomain: nlp\nversion: 1\n"
            "description: NLP skill\n---\nContent\n"
        )

        events = quarantine_out_of_scope_skills(
            tmp_path / "learned_skills",
            trust_level=1,
            trust_domains=["document_processing"],
        )

        assert len(events) == 1
        assert events[0].skill_name == "nlp-skill"

        post = frontmatter.load(str(skill_dir / "SKILL.md"))
        assert post.metadata["quarantined"] is True
        assert "nlp" in post.metadata["quarantine_reason"]

    def test_quarantine_skips_in_scope(self, tmp_path):
        """Skills within trust domains are not quarantined."""
        skill_dir = tmp_path / "learned_skills" / "doc-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: doc-skill\ndomain: document_processing\nversion: 1\n"
            "description: Doc skill\n---\nContent\n"
        )

        events = quarantine_out_of_scope_skills(
            tmp_path / "learned_skills",
            trust_level=1,
            trust_domains=["document_processing"],
        )

        assert len(events) == 0

    def test_quarantine_skips_already_quarantined(self, tmp_path):
        """Already-quarantined skills are not re-processed."""
        skill_dir = tmp_path / "learned_skills" / "old-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: old-skill\ndomain: nlp\nquarantined: true\n"
            "version: 1\ndescription: Old skill\n---\nContent\n"
        )

        events = quarantine_out_of_scope_skills(
            tmp_path / "learned_skills",
            trust_level=1,
            trust_domains=["document_processing"],
        )

        assert len(events) == 0

    def test_specialist_never_quarantined(self, tmp_path):
        """Trust level 4+ (specialist) bypasses domain restriction."""
        skill_dir = tmp_path / "learned_skills" / "any-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: any-skill\ndomain: random\nversion: 1\n"
            "description: Any skill\n---\nContent\n"
        )

        events = quarantine_out_of_scope_skills(
            tmp_path / "learned_skills",
            trust_level=4,
            trust_domains=["document_processing"],
        )

        assert len(events) == 0

    def test_quarantine_nonexistent_dir(self, tmp_path):
        """Non-existent directory returns empty list without error."""
        events = quarantine_out_of_scope_skills(
            tmp_path / "nonexistent",
            trust_level=1,
            trust_domains=["document_processing"],
        )
        assert events == []
