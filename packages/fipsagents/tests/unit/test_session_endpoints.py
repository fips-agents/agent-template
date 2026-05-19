"""Tests for fork/revert session REST endpoints and request/response models."""

from __future__ import annotations

import pytest
import pytest_asyncio
from pydantic import ValidationError

from fipsagents.server.models import (
    ForkSessionRequest,
    ForkSessionResponse,
    RevertSessionRequest,
)
from fipsagents.server.sessions import NullSessionStore, SqliteSessionStore


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestForkSessionRequest:
    def test_default_index_is_none(self):
        req = ForkSessionRequest()
        assert req.from_message_index is None

    def test_explicit_index(self):
        req = ForkSessionRequest(from_message_index=3)
        assert req.from_message_index == 3

    def test_zero_index(self):
        req = ForkSessionRequest(from_message_index=0)
        assert req.from_message_index == 0


class TestForkSessionResponse:
    def test_fields(self):
        resp = ForkSessionResponse(
            session_id="new-1",
            parent_session_id="parent-1",
            message_count=5,
        )
        assert resp.session_id == "new-1"
        assert resp.parent_session_id == "parent-1"
        assert resp.message_count == 5

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            ForkSessionResponse(session_id="x", parent_session_id="y")


class TestRevertSessionRequest:
    def test_valid_index(self):
        req = RevertSessionRequest(to_message_index=4)
        assert req.to_message_index == 4

    def test_zero_index(self):
        req = RevertSessionRequest(to_message_index=0)
        assert req.to_message_index == 0

    def test_negative_index_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal"):
            RevertSessionRequest(to_message_index=-1)

    def test_missing_index_rejected(self):
        with pytest.raises(ValidationError):
            RevertSessionRequest()


# ---------------------------------------------------------------------------
# Endpoint logic — exercised via the session store layer
# ---------------------------------------------------------------------------

# The fork/revert endpoints delegate to SessionStore.fork() / .revert().
# Rather than spinning up a full ASGI app with a mocked BaseAgent, we test
# the store operations the handlers call and the error paths they guard.


@pytest_asyncio.fixture
async def sqlite_store(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "test_endpoints.db"))
    yield store
    await store.close()


async def _seed(store, session_id, n=5):
    """Seed alternating user/assistant messages."""
    msgs = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}
        for i in range(n)
    ]
    await store.save(session_id, msgs)
    return msgs


class TestForkEndpointLogic:
    """Mirrors the fork handler's control flow against a real store."""

    @pytest.mark.asyncio
    async def test_fork_returns_new_session(self, sqlite_store):
        sid = await sqlite_store.create()
        original = await _seed(sqlite_store, sid)

        new_id = await sqlite_store.fork(sid)
        forked = await sqlite_store.load(new_id)

        assert forked == original
        assert new_id != sid

    @pytest.mark.asyncio
    async def test_fork_with_index_truncates(self, sqlite_store):
        sid = await sqlite_store.create()
        await _seed(sqlite_store, sid, n=5)

        new_id = await sqlite_store.fork(sid, from_message_index=2)
        forked = await sqlite_store.load(new_id)
        assert len(forked) == 2

    @pytest.mark.asyncio
    async def test_fork_invalid_index_raises_value_error(self, sqlite_store):
        sid = await sqlite_store.create()
        await _seed(sqlite_store, sid, n=3)

        with pytest.raises(ValueError, match="out of range"):
            await sqlite_store.fork(sid, from_message_index=99)

    @pytest.mark.asyncio
    async def test_fork_response_model_round_trip(self, sqlite_store):
        sid = await sqlite_store.create()
        await _seed(sqlite_store, sid, n=4)

        new_id = await sqlite_store.fork(sid, from_message_index=2)
        forked = await sqlite_store.load(new_id)

        resp = ForkSessionResponse(
            session_id=new_id,
            parent_session_id=sid,
            message_count=len(forked) if forked else 0,
        )
        d = resp.model_dump()
        assert d["session_id"] == new_id
        assert d["parent_session_id"] == sid
        assert d["message_count"] == 2

    @pytest.mark.asyncio
    async def test_null_store_fork_raises_not_implemented(self):
        store = NullSessionStore()
        with pytest.raises(NotImplementedError, match="NullSessionStore"):
            await store.fork("any-id")


class TestRevertEndpointLogic:
    """Mirrors the revert handler's control flow against a real store."""

    @pytest.mark.asyncio
    async def test_revert_truncates(self, sqlite_store):
        sid = await sqlite_store.create()
        await _seed(sqlite_store, sid, n=5)

        await sqlite_store.revert(sid, to_message_index=2)
        msgs = await sqlite_store.load(sid)
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_revert_to_zero_empties(self, sqlite_store):
        sid = await sqlite_store.create()
        await _seed(sqlite_store, sid)

        await sqlite_store.revert(sid, to_message_index=0)
        msgs = await sqlite_store.load(sid)
        assert msgs == []

    @pytest.mark.asyncio
    async def test_revert_invalid_index_raises_value_error(self, sqlite_store):
        sid = await sqlite_store.create()
        await _seed(sqlite_store, sid, n=3)

        with pytest.raises(ValueError, match="out of range"):
            await sqlite_store.revert(sid, to_message_index=99)

    @pytest.mark.asyncio
    async def test_revert_nonexistent_session_raises(self, sqlite_store):
        with pytest.raises(ValueError, match="not found"):
            await sqlite_store.revert("no-such-session", to_message_index=0)

    @pytest.mark.asyncio
    async def test_null_store_revert_raises_not_implemented(self):
        store = NullSessionStore()
        with pytest.raises(NotImplementedError, match="NullSessionStore"):
            await store.revert("any-id", to_message_index=0)
