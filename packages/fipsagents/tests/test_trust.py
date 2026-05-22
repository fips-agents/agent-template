"""Tests for trust accumulation, decay, and level transitions."""
from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest

from fipsagents.baseagent.config import (
    SelfHealingConfig,
    TrustThresholdsConfig,
)
from fipsagents.baseagent.events import TrustLevelChanged
from fipsagents.baseagent.trust import TrustEvent, TrustManager, TrustState


# ---------------------------------------------------------------------------
# TrustState
# ---------------------------------------------------------------------------


class TestTrustState:
    def test_defaults(self):
        state = TrustState()
        assert state.level == 0
        assert state.score == 0.0
        assert state.completions == 0
        assert state.failures == 0
        assert state.violations == 0
        assert state.last_promotion is None
        assert state.last_decay is None
        assert state.history == []

    def test_serialization_roundtrip(self):
        state = TrustState(level=2, score=55.0, completions=10, failures=1)
        d = asdict(state)
        assert d["level"] == 2
        assert d["score"] == 55.0
        assert d["completions"] == 10
        assert d["failures"] == 1
        # Verify JSON-serializable.
        serialized = json.dumps(d)
        restored = json.loads(serialized)
        assert restored["level"] == 2
        assert restored["score"] == 55.0


# ---------------------------------------------------------------------------
# TrustEvent
# ---------------------------------------------------------------------------


class TestTrustEvent:
    def test_fields(self):
        event = TrustEvent(
            timestamp="2026-01-01T00:00:00+00:00",
            event_type="completion",
            delta=1.0,
            reason="test completion",
            resulting_level=0,
            resulting_score=1.0,
        )
        assert event.event_type == "completion"
        assert event.delta == 1.0
        assert event.resulting_level == 0
        assert event.resulting_score == 1.0

    def test_serialization(self):
        event = TrustEvent(
            timestamp="2026-01-01T00:00:00+00:00",
            event_type="failure",
            delta=-5.0,
            reason="broke things",
            resulting_level=1,
            resulting_score=45.0,
        )
        d = asdict(event)
        assert d["event_type"] == "failure"
        assert d["delta"] == -5.0
        # JSON-serializable.
        json.dumps(d)


# ---------------------------------------------------------------------------
# TrustManager — basic operations
# ---------------------------------------------------------------------------


class TestTrustManagerBasics:
    def test_starts_at_level_0_score_0(self):
        tm = TrustManager()
        assert tm.level == 0
        assert tm.score == 0.0
        state = tm.get_state()
        assert state.level == 0
        assert state.score == 0.0

    def test_record_completion_increases_score(self):
        tm = TrustManager()
        tm.record_completion(quality_score=1.0)
        assert tm.score == 1.0
        assert tm.get_state().completions == 1

    def test_record_completion_quality_scaling(self):
        tm = TrustManager()
        tm.record_completion(quality_score=0.5)
        assert tm.score == 0.5
        tm.record_completion(quality_score=2.0)
        assert tm.score == 2.5

    def test_record_completion_negative_quality_clamped(self):
        tm = TrustManager()
        tm.record_completion(quality_score=-1.0)
        # max(0.0, -1.0) = 0.0, so delta = 0.0
        assert tm.score == 0.0

    def test_record_failure_decreases_score(self):
        tm = TrustManager()
        # Build up some score first.
        for _ in range(10):
            tm.record_completion()
        assert tm.score == 10.0
        tm.record_failure(severity=1.0)
        assert tm.score == 5.0
        assert tm.get_state().failures == 1

    def test_record_failure_score_clamped_at_zero(self):
        tm = TrustManager()
        tm.record_completion()  # score = 1.0
        tm.record_failure(severity=1.0)  # delta = -5.0
        assert tm.score == 0.0  # clamped, not -4.0

    def test_record_violation_steep_decay(self):
        tm = TrustManager()
        for _ in range(60):
            tm.record_completion()  # score = 60.0
        assert tm.score == 60.0
        tm.record_violation(severity=1.0)
        assert tm.score == 10.0  # 60 - 50 = 10
        assert tm.get_state().violations == 1

    def test_record_violation_clamped_at_zero(self):
        tm = TrustManager()
        tm.record_completion()  # score = 1.0
        tm.record_violation()  # delta = -50.0
        assert tm.score == 0.0

    def test_failure_sets_last_decay(self):
        tm = TrustManager()
        assert tm.get_state().last_decay is None
        tm.record_failure()
        assert tm.get_state().last_decay is not None

    def test_violation_sets_last_decay(self):
        tm = TrustManager()
        tm.record_violation()
        assert tm.get_state().last_decay is not None

    def test_custom_reason_recorded(self):
        tm = TrustManager()
        tm.record_completion(reason="good job")
        assert tm.get_state().history[-1].reason == "good job"


# ---------------------------------------------------------------------------
# TrustManager — promotion
# ---------------------------------------------------------------------------


class TestTrustManagerPromotion:
    def test_promotion_at_threshold(self):
        """10 completions at quality 1.0 => score 10.0 => level 1."""
        tm = TrustManager()
        for _ in range(9):
            tm.record_completion()
        assert tm.level == 0
        tm.record_completion()
        assert tm.level == 1
        assert tm.get_state().last_promotion is not None

    def test_promotion_records_event(self):
        tm = TrustManager()
        for _ in range(10):
            tm.record_completion()
        promotion_events = [
            e for e in tm.get_state().history if e.event_type == "promotion"
        ]
        assert len(promotion_events) == 1
        assert "promoted from level 0 to 1" in promotion_events[0].reason

    def test_multi_level_promotion(self):
        """Score large enough for level 2 in one step (via high quality)."""
        tm = TrustManager()
        # 50 completions at quality 1.0 => score 50.0 => should pass
        # threshold for level 1 (10) and level 2 (50).
        # But promotion checks one level at a time per record_completion.
        for _ in range(50):
            tm.record_completion()
        assert tm.level == 2

    def test_no_promotion_beyond_level_4(self):
        tm = TrustManager(thresholds=(1.0, 2.0, 3.0, 4.0))
        for _ in range(100):
            tm.record_completion()
        assert tm.level == 4

    def test_custom_thresholds(self):
        tm = TrustManager(thresholds=(5.0, 15.0, 30.0, 50.0))
        for _ in range(5):
            tm.record_completion()
        assert tm.level == 1
        for _ in range(10):
            tm.record_completion()
        assert tm.level == 2


# ---------------------------------------------------------------------------
# TrustManager — demotion
# ---------------------------------------------------------------------------


class TestTrustManagerDemotion:
    def test_demotion_below_half_threshold(self):
        """Level 1 threshold is 10.0; demote at score < 5.0."""
        tm = TrustManager()
        # Reach level 1.
        for _ in range(10):
            tm.record_completion()
        assert tm.level == 1
        assert tm.score == 10.0

        # Failure brings score down: 10 - 5 = 5.0, at boundary, no demotion.
        tm.record_failure(severity=1.0)
        assert tm.score == 5.0
        assert tm.level == 1  # exactly at 50%, no demotion

        # Another failure: 5 - 5 = 0, below 50% threshold.
        tm.record_failure(severity=1.0)
        assert tm.score == 0.0
        assert tm.level == 0

    def test_demotion_records_event(self):
        tm = TrustManager()
        for _ in range(10):
            tm.record_completion()
        assert tm.level == 1
        # Heavy failure to force demotion.
        tm.record_failure(severity=2.0)  # delta = -10, score = 0
        demotion_events = [
            e for e in tm.get_state().history if e.event_type == "demotion"
        ]
        assert len(demotion_events) == 1
        assert "demoted from level 1 to 0" in demotion_events[0].reason

    def test_no_demotion_at_level_0(self):
        tm = TrustManager()
        tm.record_failure()
        assert tm.level == 0  # Can't go below 0.

    def test_violation_causes_demotion(self):
        tm = TrustManager()
        for _ in range(50):
            tm.record_completion()
        assert tm.level == 2
        # Violation: score = 50 - 50 = 0.
        tm.record_violation()
        assert tm.score == 0.0
        # Should demote (possibly multiple levels via cascading checks).
        assert tm.level < 2


# ---------------------------------------------------------------------------
# TrustManager — history cap
# ---------------------------------------------------------------------------


class TestTrustManagerHistory:
    def test_history_capped_at_max(self):
        tm = TrustManager()
        for i in range(TrustManager.MAX_HISTORY + 50):
            tm.record_completion(reason=f"completion-{i}")
        # History includes completion events + promotion events.
        assert len(tm.get_state().history) <= TrustManager.MAX_HISTORY

    def test_history_preserves_recent(self):
        tm = TrustManager()
        for i in range(TrustManager.MAX_HISTORY + 10):
            tm.record_completion(reason=f"c-{i}")
        history = tm.get_state().history
        # Last event should be from the most recent completions, not the oldest.
        last_completion = [e for e in history if e.event_type == "completion"][-1]
        assert "c-" in last_completion.reason


# ---------------------------------------------------------------------------
# TrustManager — pre-existing state
# ---------------------------------------------------------------------------


class TestTrustManagerWithState:
    def test_loads_existing_state(self):
        existing = TrustState(level=2, score=55.0, completions=20)
        tm = TrustManager(state=existing)
        assert tm.level == 2
        assert tm.score == 55.0
        assert tm.get_state().completions == 20

    def test_continues_from_existing_state(self):
        existing = TrustState(level=1, score=45.0, completions=45)
        tm = TrustManager(state=existing)
        for _ in range(5):
            tm.record_completion()
        assert tm.score == 50.0
        assert tm.level == 2  # crossed threshold


# ---------------------------------------------------------------------------
# TrustThresholdsConfig
# ---------------------------------------------------------------------------


class TestTrustThresholdsConfig:
    def test_defaults(self):
        cfg = TrustThresholdsConfig()
        assert cfg.level_1 == 10.0
        assert cfg.level_2 == 50.0
        assert cfg.level_3 == 200.0
        assert cfg.level_4 == 500.0

    def test_custom_values(self):
        cfg = TrustThresholdsConfig(level_1=5.0, level_2=25.0, level_3=100.0, level_4=250.0)
        assert cfg.level_1 == 5.0
        assert cfg.level_2 == 25.0
        assert cfg.level_3 == 100.0
        assert cfg.level_4 == 250.0

    def test_selfhealing_config_includes_thresholds(self):
        cfg = SelfHealingConfig(enabled=True)
        assert isinstance(cfg.trust_thresholds, TrustThresholdsConfig)
        assert cfg.trust_thresholds.level_1 == 10.0

    def test_selfhealing_config_custom_thresholds(self):
        cfg = SelfHealingConfig(
            enabled=True,
            trust_thresholds=TrustThresholdsConfig(level_1=20.0),
        )
        assert cfg.trust_thresholds.level_1 == 20.0
        assert cfg.trust_thresholds.level_2 == 50.0  # default kept


# ---------------------------------------------------------------------------
# TrustLevelChanged event
# ---------------------------------------------------------------------------


class TestTrustLevelChangedEvent:
    def test_fields(self):
        event = TrustLevelChanged(
            from_level=0,
            to_level=1,
            score=10.0,
            reason="promoted",
        )
        assert event.from_level == 0
        assert event.to_level == 1
        assert event.score == 10.0
        assert event.reason == "promoted"

    def test_in_stream_event_union(self):
        """TrustLevelChanged should be part of the StreamEvent union."""
        from fipsagents.baseagent.events import StreamEvent
        import typing

        args = typing.get_args(StreamEvent)
        assert TrustLevelChanged in args


# ---------------------------------------------------------------------------
# Work-item integration — trust recording on completion
# ---------------------------------------------------------------------------


class TestWorkItemTrustIntegration:
    @pytest.mark.asyncio
    async def test_complete_records_trust(self):
        """complete_work_item should call trust.record_completion."""
        from fipsagents.baseagent.tools.work_items import make_work_item_tools
        from fipsagents.server.work_items import (
            WorkItem,
            WorkItemStatus,
        )

        # Build a minimal agent stub.
        agent = type("Agent", (), {})()
        agent._work_item_store = AsyncMock()
        agent._work_item_actor_id = "test-agent"
        agent._work_item_events = []
        agent._checked_out_work_item = None

        tm = TrustManager()
        agent._trust_manager = tm

        completed_item = WorkItem(
            id="wi_1",
            title="Test task",
            status=WorkItemStatus.completed,
        )
        agent._work_item_store.complete = AsyncMock(return_value=completed_item)

        tools = make_work_item_tools(agent)
        complete_fn = tools[2]  # complete_work_item

        await complete_fn(
            item_id="wi_1",
            result_summary="Done",
            accomplished=["did it"],
        )

        assert tm.score == 1.0
        assert tm.get_state().completions == 1

    @pytest.mark.asyncio
    async def test_complete_works_without_trust_manager(self):
        """complete_work_item should work fine when no trust manager is set."""
        from fipsagents.baseagent.tools.work_items import make_work_item_tools
        from fipsagents.server.work_items import WorkItem, WorkItemStatus

        agent = type("Agent", (), {})()
        agent._work_item_store = AsyncMock()
        agent._work_item_actor_id = "test-agent"
        agent._work_item_events = []
        agent._checked_out_work_item = None
        # No _trust_manager attribute.

        completed_item = WorkItem(
            id="wi_1",
            title="Test task",
            status=WorkItemStatus.completed,
        )
        agent._work_item_store.complete = AsyncMock(return_value=completed_item)

        tools = make_work_item_tools(agent)
        complete_fn = tools[2]

        # Should not raise.
        result = await complete_fn(
            item_id="wi_1",
            result_summary="Done",
            accomplished=["did it"],
        )
        assert "wi_1" in result

    @pytest.mark.asyncio
    async def test_release_does_not_record_trust(self):
        """release_work_item should NOT record trust (release is handoff, not failure)."""
        from fipsagents.baseagent.tools.work_items import make_work_item_tools
        from fipsagents.server.work_items import WorkItem, WorkItemStatus

        agent = type("Agent", (), {})()
        agent._work_item_store = AsyncMock()
        agent._work_item_actor_id = "test-agent"
        agent._work_item_events = []
        agent._checked_out_work_item = None

        tm = TrustManager()
        agent._trust_manager = tm

        released_item = WorkItem(
            id="wi_2",
            title="Released task",
            status=WorkItemStatus.available,
        )
        agent._work_item_store.release = AsyncMock(return_value=released_item)

        tools = make_work_item_tools(agent)
        release_fn = tools[3]  # release_work_item

        await release_fn(
            item_id="wi_2",
            accomplished=["partial work"],
            remaining=["finish it"],
        )

        # Trust should be unchanged.
        assert tm.score == 0.0
        assert tm.get_state().completions == 0
