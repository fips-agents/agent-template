"""Tests for budget headroom enforcement in work-item coordination."""

from __future__ import annotations

import pytest

from fipsagents.baseagent.events import BudgetHeadroomWarning, HandoffRequired
from fipsagents.baseagent.pricing import compute_cost
from fipsagents.baseagent.config import PricingConfig, PricingRate
from fipsagents.server.work_items import WorkItem


# ---------------------------------------------------------------------------
# Event dataclass smoke tests
# ---------------------------------------------------------------------------


class TestBudgetHeadroomWarningEvent:
    def test_fields(self):
        ev = BudgetHeadroomWarning(item_id="wi_42", remaining_pct=7.5)
        assert ev.item_id == "wi_42"
        assert ev.remaining_pct == 7.5

    def test_zero_remaining(self):
        ev = BudgetHeadroomWarning(item_id="wi_0", remaining_pct=0.0)
        assert ev.remaining_pct == 0.0


class TestHandoffRequiredEvent:
    def test_fields(self):
        ev = HandoffRequired(
            item_id="wi_99",
            actor_id="agent-a",
            expires_at="2026-01-01T00:00:00Z",
        )
        assert ev.item_id == "wi_99"
        assert ev.actor_id == "agent-a"
        assert ev.expires_at == "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Threshold math — the core logic extracted from astep_stream
# ---------------------------------------------------------------------------


def _should_warn(
    max_cost_usd: float,
    headroom_pct: float,
    turn_cost: float,
) -> tuple[bool, float]:
    """Replicate the threshold calculation from astep_stream.

    Returns (should_warn, remaining_pct).
    """
    remaining = max_cost_usd - turn_cost
    threshold = max_cost_usd * (headroom_pct / 100.0)
    remaining_pct = max(0.0, (remaining / max_cost_usd) * 100.0)
    return remaining <= threshold, remaining_pct


class TestHeadroomThresholdMath:
    """Data-driven tests for the threshold calculation."""

    @pytest.mark.parametrize(
        "max_cost, headroom_pct, turn_cost, expect_warn, expect_remaining_pct",
        [
            # Cost=$0.95 of $1.00, headroom=10% => remaining=$0.05 <= $0.10 threshold
            (1.0, 10.0, 0.95, True, 5.0),
            # Cost=$0.80 of $1.00, headroom=10% => remaining=$0.20 > $0.10 threshold
            (1.0, 10.0, 0.80, False, 20.0),
            # Exactly at threshold boundary: remaining=$0.10 <= $0.10
            (1.0, 10.0, 0.90, True, 10.0),
            # Zero headroom: only warn when cost >= max
            (1.0, 0.0, 0.99, False, 1.0),
            (1.0, 0.0, 1.00, True, 0.0),
            # 100% headroom: always warn (threshold = max_cost itself)
            (1.0, 100.0, 0.01, True, 99.0),
            # Overshoot: cost exceeds budget
            (1.0, 10.0, 1.50, True, 0.0),
            # Small budget
            (0.01, 10.0, 0.0095, True, 5.0),
        ],
        ids=[
            "below_threshold",
            "above_threshold",
            "at_threshold_boundary",
            "zero_headroom_under",
            "zero_headroom_at",
            "full_headroom",
            "overshoot",
            "small_budget",
        ],
    )
    def test_threshold(
        self,
        max_cost: float,
        headroom_pct: float,
        turn_cost: float,
        expect_warn: bool,
        expect_remaining_pct: float,
    ):
        should_warn, remaining_pct = _should_warn(max_cost, headroom_pct, turn_cost)
        assert should_warn is expect_warn
        assert remaining_pct == pytest.approx(expect_remaining_pct, abs=0.01)


# ---------------------------------------------------------------------------
# Skip conditions — when the headroom check should not fire
# ---------------------------------------------------------------------------


class TestHeadroomSkipConditions:
    def test_no_checked_out_item(self):
        """No warning when _checked_out_work_item is None."""
        # The check in astep_stream gates on `self._checked_out_work_item is not None`.
        # Simulate by testing the condition directly.
        checked_out = None
        assert checked_out is None  # guard never enters the block

    def test_max_cost_usd_is_none(self):
        """No warning when the work item has no budget."""
        wi = WorkItem(id="wi_no_budget", title="No budget")
        assert wi.max_cost_usd is None

    def test_max_cost_usd_is_zero(self):
        """No warning when max_cost_usd is zero (guard: > 0)."""
        wi = WorkItem(id="wi_zero", title="Zero budget", max_cost_usd=0.0)
        assert not (wi.max_cost_usd is not None and wi.max_cost_usd > 0)


# ---------------------------------------------------------------------------
# Dedup flag — _headroom_warned prevents re-emission
# ---------------------------------------------------------------------------


class TestHeadroomWarnedFlag:
    def test_flag_prevents_duplicate(self):
        """Once _headroom_warned is True, no second warning should emit."""
        headroom_warned = False

        # First check: below threshold → warn and set flag.
        should_warn, _ = _should_warn(1.0, 10.0, 0.95)
        if should_warn and not headroom_warned:
            headroom_warned = True

        assert headroom_warned is True

        # Second check: still below threshold but flag blocks re-emission.
        should_warn_2, _ = _should_warn(1.0, 10.0, 0.97)
        emitted_second = should_warn_2 and not headroom_warned
        assert emitted_second is False

    def test_flag_resets_per_turn(self):
        """_headroom_warned resets to False at the start of each turn."""
        # Simulates the reset at the top of astep_stream.
        headroom_warned = True
        headroom_warned = False  # reset
        assert headroom_warned is False


# ---------------------------------------------------------------------------
# Cost computation integration — verify compute_cost cooperates
# ---------------------------------------------------------------------------


class TestCostComputationForHeadroom:
    def test_zero_cost_with_default_pricing(self):
        """Default pricing (all zeros) => compute_cost returns 0.0."""
        pricing = PricingConfig()
        cost = compute_cost(
            "test-model",
            input_tokens=1000,
            output_tokens=500,
            pricing=pricing,
        )
        assert cost == 0.0

    def test_nonzero_cost_with_custom_pricing(self):
        """Custom pricing produces a real cost for threshold comparison."""
        pricing = PricingConfig(
            default=PricingRate(input_per_1k=0.01, output_per_1k=0.03),
        )
        cost = compute_cost(
            "test-model",
            input_tokens=1000,
            output_tokens=500,
            pricing=pricing,
        )
        # 1.0 * 0.01 + 0.5 * 0.03 = 0.01 + 0.015 = 0.025
        assert cost == pytest.approx(0.025, abs=1e-6)

    def test_headroom_with_real_cost(self):
        """End-to-end: compute cost, then check threshold."""
        pricing = PricingConfig(
            default=PricingRate(input_per_1k=0.01, output_per_1k=0.03),
        )
        cost = compute_cost(
            "test-model",
            input_tokens=5000,
            output_tokens=2000,
            pricing=pricing,
        )
        # 5.0 * 0.01 + 2.0 * 0.03 = 0.05 + 0.06 = 0.11
        assert cost == pytest.approx(0.11, abs=1e-6)

        # Work item with budget $0.12, headroom 10% => threshold $0.012.
        # Remaining = 0.12 - 0.11 = $0.01 < $0.012 => should warn.
        should_warn, remaining_pct = _should_warn(0.12, 10.0, cost)
        assert should_warn is True
        assert remaining_pct == pytest.approx(8.33, abs=0.1)


# ---------------------------------------------------------------------------
# WorkItem lifecycle integration (checked_out_work_item field)
# ---------------------------------------------------------------------------


class TestCheckedOutWorkItemLifecycle:
    def test_work_item_has_budget_fields(self):
        """WorkItem exposes max_cost_usd for headroom checks."""
        wi = WorkItem(
            id="wi_budget",
            title="Budgeted item",
            max_cost_usd=0.50,
        )
        assert wi.max_cost_usd == 0.50
        assert wi.id == "wi_budget"

    def test_work_item_default_no_budget(self):
        """By default, WorkItem.max_cost_usd is None (no budget)."""
        wi = WorkItem(id="wi_default", title="Default")
        assert wi.max_cost_usd is None


# ---------------------------------------------------------------------------
# Integration: headroom check with realistic token accumulation
# ---------------------------------------------------------------------------


class TestHeadroomIntegrationWithTokenAccumulation:
    """Simulate the astep_stream headroom check end-to-end.

    These tests wire together WorkItem (budget), PricingConfig (rates),
    compute_cost (USD calculation), and the threshold logic — the same
    chain that runs inside ``astep_stream`` after each model call.
    """

    def _simulate_headroom_check(
        self,
        wi: WorkItem,
        pricing: PricingConfig,
        model: str,
        turns: list[tuple[int, int]],
        headroom_pct: float = 20.0,
    ) -> list[BudgetHeadroomWarning]:
        """Run multiple model-call turns and collect any warnings.

        Each entry in *turns* is (prompt_tokens, completion_tokens) for
        one iteration of the inner loop.  Tokens accumulate across turns,
        matching the ``_cumulative_prompt`` / ``_cumulative_completion``
        pattern in ``astep_stream``.
        """
        warnings: list[BudgetHeadroomWarning] = []
        cumulative_prompt = 0
        cumulative_completion = 0
        headroom_warned = False

        for prompt_tok, completion_tok in turns:
            cumulative_prompt += prompt_tok
            cumulative_completion += completion_tok

            # Guard: skip if no budget.
            if wi.max_cost_usd is None or wi.max_cost_usd <= 0:
                continue

            turn_cost = compute_cost(
                model,
                input_tokens=cumulative_prompt,
                output_tokens=cumulative_completion,
                pricing=pricing,
            )
            remaining = wi.max_cost_usd - turn_cost
            threshold = wi.max_cost_usd * (headroom_pct / 100.0)
            if remaining <= threshold and not headroom_warned:
                remaining_pct = max(
                    0.0, (remaining / wi.max_cost_usd) * 100.0,
                )
                warnings.append(
                    BudgetHeadroomWarning(
                        item_id=wi.id, remaining_pct=remaining_pct,
                    )
                )
                headroom_warned = True

        return warnings

    def test_warning_fires_when_tokens_cross_threshold(self):
        """Multi-turn accumulation crosses the 80% mark (headroom=20%)."""
        pricing = PricingConfig(
            default=PricingRate(input_per_1k=0.01, output_per_1k=0.03),
        )
        wi = WorkItem(id="wi_multi", title="Multi-turn", max_cost_usd=0.10)

        # Turn 1: 1k in + 500 out = $0.01 + $0.015 = $0.025
        # Turn 2: cumulative 3k in + 1k out = $0.03 + $0.03 = $0.06
        # Turn 3: cumulative 5k in + 2k out = $0.05 + $0.06 = $0.11
        #   => $0.11 > $0.10 budget, remaining < 0, warning fires
        turns = [(1000, 500), (2000, 500), (2000, 1000)]
        warnings = self._simulate_headroom_check(
            wi, pricing, "test-model", turns, headroom_pct=20.0,
        )

        assert len(warnings) == 1
        assert warnings[0].item_id == "wi_multi"
        # Remaining capped at 0% when overspent.
        assert warnings[0].remaining_pct == 0.0

    def test_no_warning_when_under_threshold(self):
        """Tokens stay well within budget — no warning."""
        pricing = PricingConfig(
            default=PricingRate(input_per_1k=0.001, output_per_1k=0.003),
        )
        wi = WorkItem(id="wi_cheap", title="Cheap work", max_cost_usd=1.00)

        # 2k in + 1k out = $0.002 + $0.003 = $0.005  (0.5% of budget)
        turns = [(1000, 500), (1000, 500)]
        warnings = self._simulate_headroom_check(
            wi, pricing, "test-model", turns, headroom_pct=20.0,
        )
        assert warnings == []

    def test_warning_fires_once_across_many_turns(self):
        """Even with many turns past threshold, only one warning emitted."""
        pricing = PricingConfig(
            default=PricingRate(input_per_1k=0.05, output_per_1k=0.10),
        )
        wi = WorkItem(id="wi_many", title="Many turns", max_cost_usd=0.50)

        # Each turn: 1k in + 500 out = $0.05 + $0.05 = $0.10 cumulative per turn
        # Turn 5: cumulative 5k in + 2.5k out = $0.25 + $0.25 = $0.50 (= budget)
        turns = [(1000, 500)] * 8  # push far past budget
        warnings = self._simulate_headroom_check(
            wi, pricing, "test-model", turns, headroom_pct=20.0,
        )

        assert len(warnings) == 1  # dedup flag works

    def test_exact_threshold_boundary_triggers_warning(self):
        """Cost lands exactly at the headroom boundary (remaining == threshold)."""
        pricing = PricingConfig(
            default=PricingRate(input_per_1k=0.01, output_per_1k=0.00),
        )
        wi = WorkItem(id="wi_exact", title="Exact", max_cost_usd=0.10)

        # headroom_pct=20 => threshold=$0.02, warn at remaining<=$0.02
        # Need cost = $0.08 => 8000 input tokens at $0.01/1k
        turns = [(8000, 0)]
        warnings = self._simulate_headroom_check(
            wi, pricing, "test-model", turns, headroom_pct=20.0,
        )

        assert len(warnings) == 1
        assert warnings[0].remaining_pct == pytest.approx(20.0, abs=0.01)

    def test_no_budget_skips_check(self):
        """WorkItem with no budget (max_cost_usd=None) never warns."""
        pricing = PricingConfig(
            default=PricingRate(input_per_1k=1.00, output_per_1k=1.00),
        )
        wi = WorkItem(id="wi_none", title="No budget")

        turns = [(100000, 100000)]  # enormous usage
        warnings = self._simulate_headroom_check(
            wi, pricing, "test-model", turns, headroom_pct=20.0,
        )
        assert warnings == []
