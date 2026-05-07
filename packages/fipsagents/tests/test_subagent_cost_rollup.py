"""Test that OpenAIChatServer._persist_cost_data folds subagent tokens
into the parent session's cost_data, so BudgetEnforcer (which reads
cost_data) sees the rolled-up totals.

This is the v1 mechanism for "parent's BudgetEnforcer shows child's
tokens rolled up" from issue #165's Definition of Done.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fipsagents.baseagent.events import StreamMetrics  # noqa: E402
from fipsagents.server import OpenAIChatServer  # noqa: E402


class _RecordingSessionStore:
    """Minimal SessionStore stub that records update() calls."""

    def __init__(self) -> None:
        self.updates: list[dict] = []
        self._cost_data: dict[str, dict] = {}

    async def get_cost_data(self, session_id: str) -> dict:
        return dict(self._cost_data.get(session_id, {}))

    async def update(self, session_id: str, *, cost_data: dict | None = None, **_: object) -> None:
        if cost_data is not None:
            self._cost_data[session_id] = dict(cost_data)
            self.updates.append(dict(cost_data))


def _make_server_with_stub_agent(*, subagent_usage: list[dict] | None = None) -> tuple[OpenAIChatServer, _RecordingSessionStore]:
    """Build an OpenAIChatServer with the minimum surface _persist_cost_data reads."""

    class _Agent:
        async def setup(self) -> None:
            pass

        async def shutdown(self) -> None:
            pass

    server = OpenAIChatServer(_Agent)
    server._agent = _Agent()
    server._agent._subagent_token_usage = list(subagent_usage or [])
    store = _RecordingSessionStore()
    server._session_store = store
    return server, store


@pytest.mark.asyncio
async def test_persist_cost_data_folds_subagent_tokens_into_session_total() -> None:
    """Subagent tokens are added to the turn's totals before persistence.

    BudgetEnforcer reads cost_data on the next request and sees the
    rolled-up totals, so its limit checks reflect subagent usage.
    """
    server, store = _make_server_with_stub_agent(
        subagent_usage=[
            {"input": 100, "output": 200, "cached": 5},
            {"input": 50, "output": 75, "cached": 0},
        ],
    )

    metrics = StreamMetrics(prompt_tokens=10, completion_tokens=20)

    await server._persist_cost_data(
        session_id="sess-1",
        metrics=metrics,
        model_name="parent-model",
    )

    assert len(store.updates) == 1
    cost_data = store.updates[0]
    # Parent turn (10 + 20) plus both subagent entries (150 + 275 + 5).
    assert cost_data["input_tokens"] == 10 + 100 + 50
    assert cost_data["output_tokens"] == 20 + 200 + 75
    assert cost_data["cached_tokens"] == 0 + 5 + 0
    assert cost_data["model"] == "parent-model"
    assert cost_data["turn_count"] == 1


@pytest.mark.asyncio
async def test_persist_cost_data_drains_subagent_buffer() -> None:
    """The buffer is cleared after draining so the next turn doesn't double-count."""
    server, _ = _make_server_with_stub_agent(
        subagent_usage=[{"input": 10, "output": 20, "cached": 0}],
    )

    await server._persist_cost_data(
        session_id="sess-1",
        metrics=StreamMetrics(prompt_tokens=1, completion_tokens=1),
        model_name="m",
    )

    assert server._agent._subagent_token_usage == []


@pytest.mark.asyncio
async def test_persist_cost_data_without_subagents_unchanged() -> None:
    """When the buffer is empty, the original cost-data path is unchanged."""
    server, store = _make_server_with_stub_agent(subagent_usage=[])

    await server._persist_cost_data(
        session_id="sess-1",
        metrics=StreamMetrics(prompt_tokens=42, completion_tokens=58),
        model_name="m",
    )

    cost_data = store.updates[0]
    assert cost_data["input_tokens"] == 42
    assert cost_data["output_tokens"] == 58
    assert cost_data["cached_tokens"] == 0


@pytest.mark.asyncio
async def test_persist_cost_data_accumulates_across_turns() -> None:
    """Two consecutive turns each fold their own subagent tokens into the running total."""
    server, store = _make_server_with_stub_agent(
        subagent_usage=[{"input": 100, "output": 200, "cached": 0}],
    )

    await server._persist_cost_data(
        session_id="sess-1",
        metrics=StreamMetrics(prompt_tokens=10, completion_tokens=20),
        model_name="m",
    )

    server._agent._subagent_token_usage = [
        {"input": 30, "output": 40, "cached": 0},
    ]

    await server._persist_cost_data(
        session_id="sess-1",
        metrics=StreamMetrics(prompt_tokens=5, completion_tokens=5),
        model_name="m",
    )

    last = store.updates[-1]
    assert last["input_tokens"] == 10 + 100 + 5 + 30
    assert last["output_tokens"] == 20 + 200 + 5 + 40
    assert last["turn_count"] == 2
