"""Tests for BudgetEnforcer + BudgetConfig + the 402 wire shape."""

from __future__ import annotations

import logging

import pytest

from fipsagents.baseagent.config import (
    BudgetConfig,
    BudgetLimits,
    PricingConfig,
    PricingRate,
)
from fipsagents.server.budget import (
    BudgetEnforcer,
    BudgetExceededError,
    NullBudgetEnforcer,
    create_budget_enforcer,
)
from fipsagents.server.sessions import NullSessionStore, SqliteSessionStore


# ---------------------------------------------------------------------------
# BudgetConfig schema
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    def test_defaults_are_inactive(self):
        cfg = BudgetConfig()
        assert cfg.mode == "enforce"
        assert cfg.is_active() is False

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError):
            BudgetLimits(warn_usd=-0.01)
        with pytest.raises(ValueError):
            BudgetLimits(limit_usd=-0.01)

    def test_any_threshold_makes_active(self):
        assert BudgetConfig(
            per_session=BudgetLimits(warn_usd=0.5),
        ).is_active()
        assert BudgetConfig(
            per_tenant=BudgetLimits(limit_usd=10.0),
        ).is_active()


# ---------------------------------------------------------------------------
# create_budget_enforcer factory
# ---------------------------------------------------------------------------


class TestCreateBudgetEnforcer:
    def test_inactive_config_returns_null(self):
        enforcer = create_budget_enforcer(
            BudgetConfig(),
            pricing=PricingConfig(),
            session_store=NullSessionStore(),
        )
        assert isinstance(enforcer, NullBudgetEnforcer)

    def test_active_config_returns_real(self):
        enforcer = create_budget_enforcer(
            BudgetConfig(per_session=BudgetLimits(limit_usd=1.0)),
            pricing=PricingConfig(),
            session_store=NullSessionStore(),
        )
        assert isinstance(enforcer, BudgetEnforcer)


# ---------------------------------------------------------------------------
# BudgetEnforcer behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
async def sqlite_store(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "budget.db"))
    yield store
    await store.close()


@pytest.fixture
def pricing():
    # 1 cent per 1k input tokens => 1000 input tokens = $0.01.
    return PricingConfig(default=PricingRate(input_per_1k=0.01))


async def _seed_session(store, sid: str, *, input_tokens: int, model: str):
    await store.create(sid)
    await store.update(
        sid,
        cost_data={
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cached_tokens": 0,
            "model": model,
            "turn_count": 1,
        },
    )


class TestPerSessionEnforcement:
    @pytest.mark.asyncio
    async def test_under_limit_passes(self, sqlite_store, pricing):
        # 10000 tokens * 0.01/1k = $0.10 cost; limit $1.00 -> ok.
        await _seed_session(
            sqlite_store, "sess1", input_tokens=10_000, model="m",
        )
        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                per_session=BudgetLimits(limit_usd=1.00),
            ),
            pricing=pricing,
            session_store=sqlite_store,
        )
        await enforcer.check_before_request(
            session_id="sess1", tenant_id="acme",
        )

    @pytest.mark.asyncio
    async def test_at_limit_raises_in_enforce_mode(self, sqlite_store, pricing):
        # 100_000 tokens * 0.01/1k = $1.00 = limit. Hit -> raise.
        await _seed_session(
            sqlite_store, "sess1", input_tokens=100_000, model="m",
        )
        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                mode="enforce",
                per_session=BudgetLimits(limit_usd=1.00),
            ),
            pricing=pricing,
            session_store=sqlite_store,
        )
        with pytest.raises(BudgetExceededError) as excinfo:
            await enforcer.check_before_request(
                session_id="sess1", tenant_id=None,
            )
        assert excinfo.value.scope == "session"
        assert excinfo.value.identifier == "sess1"
        assert excinfo.value.current_usd == pytest.approx(1.00)
        assert excinfo.value.limit_usd == pytest.approx(1.00)

    @pytest.mark.asyncio
    async def test_observe_mode_does_not_raise(self, sqlite_store, pricing, caplog):
        await _seed_session(
            sqlite_store, "sess1", input_tokens=200_000, model="m",
        )
        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                mode="observe",
                per_session=BudgetLimits(limit_usd=1.00),
            ),
            pricing=pricing,
            session_store=sqlite_store,
        )
        with caplog.at_level(logging.WARNING):
            await enforcer.check_before_request(
                session_id="sess1", tenant_id=None,
            )
        assert any(
            "Budget session limit hit" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_get_cost_data_not_implemented_treated_as_zero(self, pricing):
        class _NoRead(NullSessionStore):
            async def get_cost_data(self, session_id):  # noqa: ARG002
                raise NotImplementedError

        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                per_session=BudgetLimits(limit_usd=0.01),
            ),
            pricing=pricing,
            session_store=_NoRead(),
        )
        # Should NOT raise — degraded backend reads as zero.
        await enforcer.check_before_request(
            session_id="anything", tenant_id=None,
        )


class TestPerTenantEnforcement:
    @pytest.mark.asyncio
    async def test_tenant_accumulator_starts_empty(self, sqlite_store, pricing):
        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                per_tenant=BudgetLimits(limit_usd=1.00),
            ),
            pricing=pricing,
            session_store=sqlite_store,
        )
        await enforcer.check_before_request(
            session_id=None, tenant_id="acme",
        )

    @pytest.mark.asyncio
    async def test_tenant_accumulates_across_sessions(self, sqlite_store, pricing):
        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                per_tenant=BudgetLimits(limit_usd=10.00),
            ),
            pricing=pricing,
            session_store=sqlite_store,
        )

        # Seed two sessions for the same tenant.
        await _seed_session(
            sqlite_store, "sess_a", input_tokens=100_000, model="m",
        )
        await _seed_session(
            sqlite_store, "sess_b", input_tokens=100_000, model="m",
        )

        # First record: $1.00 from sess_a.
        await enforcer.record_after_request(
            session_id="sess_a", tenant_id="acme",
        )
        assert enforcer._tenant_costs["acme"] == pytest.approx(1.00)

        # Second record: +$1.00 from sess_b -> $2.00 total.
        await enforcer.record_after_request(
            session_id="sess_b", tenant_id="acme",
        )
        assert enforcer._tenant_costs["acme"] == pytest.approx(2.00)

    @pytest.mark.asyncio
    async def test_tenant_limit_raises_after_accumulation(
        self, sqlite_store, pricing,
    ):
        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                mode="enforce",
                per_tenant=BudgetLimits(limit_usd=1.00),
            ),
            pricing=pricing,
            session_store=sqlite_store,
        )
        await _seed_session(
            sqlite_store, "sess1", input_tokens=150_000, model="m",
        )
        await enforcer.record_after_request(
            session_id="sess1", tenant_id="acme",
        )
        with pytest.raises(BudgetExceededError) as excinfo:
            await enforcer.check_before_request(
                session_id="sess2", tenant_id="acme",
            )
        assert excinfo.value.scope == "tenant"
        assert excinfo.value.identifier == "acme"


class TestSoftWarnings:
    @pytest.mark.asyncio
    async def test_soft_warning_logged_once_per_session(
        self, sqlite_store, pricing, caplog,
    ):
        enforcer = BudgetEnforcer(
            config=BudgetConfig(
                per_session=BudgetLimits(warn_usd=0.50, limit_usd=10.00),
            ),
            pricing=pricing,
            session_store=sqlite_store,
        )
        await _seed_session(
            sqlite_store, "sess1", input_tokens=80_000, model="m",  # $0.80
        )
        with caplog.at_level(logging.WARNING):
            await enforcer.record_after_request(
                session_id="sess1", tenant_id=None,
            )
            await enforcer.record_after_request(
                session_id="sess1", tenant_id=None,
            )

        warns = [
            r for r in caplog.records
            if "crossed soft budget warning" in r.message
        ]
        assert len(warns) == 1, "Soft warning should fire only once per session"


# ---------------------------------------------------------------------------
# 402 wire shape via TestClient
# ---------------------------------------------------------------------------


def test_chat_completion_returns_402_when_session_over_limit(tmp_path):
    """When per_session.limit_usd is hit, /chat/completions returns 402."""
    pytest.importorskip("fastapi")

    import asyncio
    import types

    from fastapi.testclient import TestClient

    from fipsagents.baseagent.events import (
        ContentDelta,
        StreamComplete,
        StreamMetrics,
    )
    from fipsagents.server import OpenAIChatServer

    # Stub agent imports happen lazily because tests/conftest.py builds the
    # path; reuse the helper from test_server_openai by replicating the
    # minimal surface here so this test file stays independent.
    from tests.test_server_openai import _make_agent_class  # type: ignore

    metrics = StreamMetrics(prompt_tokens=10, completion_tokens=4)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    AgentClass = _make_agent_class(events, model_name="m1")
    db_path = str(tmp_path / "budget.db")

    class _A(AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.server.storage = types.SimpleNamespace(
                backend="sqlite",
                sqlite_path=db_path,
                database_url="",
                platform_url="",
                platform_token="",
            )
            self.config.server.sessions = types.SimpleNamespace(
                enabled=True, max_age_hours=0, backend=None,
            )
            self.config.pricing = PricingConfig(
                default=PricingRate(input_per_1k=1.0),
            )
            self.config.budget = BudgetConfig(
                mode="enforce",
                per_session=BudgetLimits(limit_usd=0.01),
            )

    server = OpenAIChatServer(_A)

    with TestClient(server.app) as client:
        # Pre-seed a session with cost over the limit.
        async def seed():
            await server._session_store.create("sess_over")
            await server._session_store.update(
                "sess_over",
                cost_data={
                    "input_tokens": 1000,  # 1.0/1k * 1000 = $1.00
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "model": "m1",
                    "turn_count": 1,
                },
            )
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "session_id": "sess_over",
            },
        )

    assert resp.status_code == 402
    body = resp.json()
    assert body["detail"]["error"] == "budget_exceeded"
    assert body["detail"]["scope"] == "session"
    assert body["detail"]["identifier"] == "sess_over"
    assert body["detail"]["limit_usd"] == 0.01


def test_chat_completion_allowed_when_no_budget_configured(tmp_path):
    """Without any budget config, requests are never blocked."""
    pytest.importorskip("fastapi")

    from fastapi.testclient import TestClient

    from fipsagents.baseagent.events import (
        ContentDelta,
        StreamComplete,
        StreamMetrics,
    )
    from fipsagents.server import OpenAIChatServer
    from tests.test_server_openai import _make_agent_class  # type: ignore

    metrics = StreamMetrics(prompt_tokens=10)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    AgentClass = _make_agent_class(events, model_name="m1")

    class _A(AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Default BudgetConfig() -> inactive -> NullBudgetEnforcer.
            self.config.budget = BudgetConfig()

    server = OpenAIChatServer(_A)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
