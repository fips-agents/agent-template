"""Tests for the pricing helper and PricingConfig schema."""

from __future__ import annotations

import pytest

from fipsagents.baseagent.config import PricingConfig, PricingRate
from fipsagents.server.pricing import compute_cost, rate_for_model


class TestPricingRate:
    def test_defaults_are_zero(self):
        rate = PricingRate()
        assert rate.input_per_1k == 0.0
        assert rate.output_per_1k == 0.0
        assert rate.cached_input_per_1k is None
        assert rate.per_request == 0.0

    def test_negative_rates_rejected(self):
        with pytest.raises(ValueError):
            PricingRate(input_per_1k=-0.1)
        with pytest.raises(ValueError):
            PricingRate(output_per_1k=-0.1)
        with pytest.raises(ValueError):
            PricingRate(cached_input_per_1k=-0.1)
        with pytest.raises(ValueError):
            PricingRate(per_request=-0.1)


class TestPricingConfig:
    def test_default_lookup_table_empty(self):
        cfg = PricingConfig()
        assert cfg.default == PricingRate()
        assert cfg.models == {}

    def test_yaml_round_trip(self):
        cfg = PricingConfig.model_validate({
            "default": {"input_per_1k": 0.001, "output_per_1k": 0.002},
            "models": {
                "gpt-4o": {
                    "input_per_1k": 0.0025,
                    "output_per_1k": 0.01,
                    "cached_input_per_1k": 0.00125,
                },
                "self-hosted": {},
            },
        })
        assert cfg.default.input_per_1k == 0.001
        assert cfg.models["gpt-4o"].cached_input_per_1k == 0.00125
        assert cfg.models["self-hosted"].input_per_1k == 0.0


class TestRateForModel:
    def test_exact_match_wins(self):
        cfg = PricingConfig(
            default=PricingRate(input_per_1k=99.0),
            models={"gpt-4o": PricingRate(input_per_1k=0.0025)},
        )
        rate = rate_for_model("gpt-4o", cfg)
        assert rate.input_per_1k == 0.0025

    def test_unknown_model_falls_back_to_default(self):
        cfg = PricingConfig(default=PricingRate(input_per_1k=0.001))
        rate = rate_for_model("never-heard-of-it", cfg)
        assert rate.input_per_1k == 0.001

    def test_none_model_falls_back_to_default(self):
        cfg = PricingConfig(default=PricingRate(input_per_1k=0.5))
        assert rate_for_model(None, cfg).input_per_1k == 0.5
        assert rate_for_model("", cfg).input_per_1k == 0.5


class TestComputeCost:
    def test_zero_pricing_returns_zero(self):
        cfg = PricingConfig()  # all zeros
        cost = compute_cost(
            "any-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing=cfg,
        )
        assert cost == 0.0

    def test_input_and_output_basic(self):
        cfg = PricingConfig(
            default=PricingRate(input_per_1k=0.001, output_per_1k=0.002),
        )
        cost = compute_cost(
            "any",
            input_tokens=1000,
            output_tokens=500,
            pricing=cfg,
        )
        # 1000/1000 * 0.001 + 500/1000 * 0.002 = 0.001 + 0.001
        assert cost == pytest.approx(0.002)

    def test_per_request_added_once(self):
        cfg = PricingConfig(default=PricingRate(per_request=0.01))
        cost = compute_cost("any", pricing=cfg)
        assert cost == pytest.approx(0.01)

    def test_cached_tokens_discounted(self):
        cfg = PricingConfig(
            default=PricingRate(
                input_per_1k=0.01,
                output_per_1k=0.0,
                cached_input_per_1k=0.001,  # 90% discount
            ),
        )
        cost = compute_cost(
            "any",
            input_tokens=1000,
            cached_tokens=500,  # half the input was cached
            pricing=cfg,
        )
        # 500 input @ 0.01/1k = 0.005, 500 cached @ 0.001/1k = 0.0005
        assert cost == pytest.approx(0.0055)

    def test_cached_without_cache_rate_uses_full_rate(self):
        cfg = PricingConfig(default=PricingRate(input_per_1k=0.01))
        cost_with_cache = compute_cost(
            "any", input_tokens=1000, cached_tokens=500, pricing=cfg,
        )
        cost_without_cache = compute_cost(
            "any", input_tokens=1000, pricing=cfg,
        )
        assert cost_with_cache == cost_without_cache

    def test_cached_clamped_to_input(self):
        cfg = PricingConfig(
            default=PricingRate(
                input_per_1k=0.01, cached_input_per_1k=0.001,
            ),
        )
        # Provider reports more cached than input; should clamp.
        cost = compute_cost(
            "any",
            input_tokens=1000,
            cached_tokens=5000,
            pricing=cfg,
        )
        # Same as if cached==input==1000.
        expected = compute_cost(
            "any", input_tokens=1000, cached_tokens=1000, pricing=cfg,
        )
        assert cost == expected

    def test_per_model_override(self):
        cfg = PricingConfig(
            default=PricingRate(input_per_1k=0.001),
            models={"gpt-4o": PricingRate(input_per_1k=0.0025)},
        )
        default_cost = compute_cost(
            "self-hosted", input_tokens=1000, pricing=cfg,
        )
        gpt_cost = compute_cost(
            "gpt-4o", input_tokens=1000, pricing=cfg,
        )
        assert default_cost == pytest.approx(0.001)
        assert gpt_cost == pytest.approx(0.0025)

    def test_negative_token_counts_treated_as_zero(self):
        cfg = PricingConfig(default=PricingRate(input_per_1k=0.01))
        # Defensive: provider sends -1 by mistake.
        cost = compute_cost("any", input_tokens=-100, pricing=cfg)
        assert cost == 0.0
