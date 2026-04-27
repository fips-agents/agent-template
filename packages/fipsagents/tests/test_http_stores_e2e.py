"""End-to-end tests for the Http*Store implementations.

Boots the real ``fipsagents_platform`` ASGI app in-process against a
fresh SQLite-backed store and points the agent-side ``Http*Store`` at
it via ``httpx.ASGITransport``.  Verifies the full ABC surface
round-trips through the REST veneer with no mocks.

Skipped automatically when ``fipsagents_platform`` is not installed —
the suite is opt-in and only runs when the sibling repo is checked out
and pip-installed (eg via ``pip install -e ../../fipsagents-platform``).
"""

from __future__ import annotations

import os
import tempfile

import httpx
import pytest

pytest.importorskip(
    "fipsagents_platform",
    reason="fipsagents_platform not installed — pip install the sibling "
    "repo to exercise the e2e suite.",
)

from fipsagents.server.feedback import FeedbackRecord
from fipsagents.server.http import (
    HttpFeedbackStore,
    HttpSessionStore,
    HttpTraceStore,
)
from fipsagents.server.tracing import Span, Trace


@pytest.fixture
def platform_db(monkeypatch: pytest.MonkeyPatch):
    fd, path = tempfile.mkstemp(prefix="e2e-platform-", suffix=".db")
    os.close(fd)
    os.unlink(path)
    monkeypatch.setenv("PLATFORM_BACKEND", "sqlite")
    monkeypatch.setenv("PLATFORM_SQLITE_PATH", path)
    monkeypatch.setenv("PLATFORM_AUTH_MODE", "none")
    from fipsagents_platform.config import reset_settings_for_tests

    reset_settings_for_tests()
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
async def platform_transport(platform_db):
    """Yield an ASGITransport bound to the platform app, with lifespan running."""
    from fipsagents_platform.app import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        yield httpx.ASGITransport(app=app)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_full_lifecycle(platform_transport) -> None:
    store = HttpSessionStore(
        "http://platform.test", transport=platform_transport,
    )
    sid = await store.create()
    assert sid.startswith("sess_")

    assert await store.exists(sid) is True
    assert await store.load(sid) == []

    await store.save(sid, [{"role": "user", "content": "hello"}])
    msgs = await store.load(sid)
    assert msgs == [{"role": "user", "content": "hello"}]

    # Save again to verify upsert semantics.
    await store.save(sid, [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])
    assert len(await store.load(sid)) == 2

    assert await store.delete(sid) is True
    assert await store.exists(sid) is False
    assert await store.load(sid) is None
    assert await store.delete(sid) is False
    await store.close()


@pytest.mark.asyncio
async def test_session_create_with_explicit_id(platform_transport) -> None:
    store = HttpSessionStore(
        "http://platform.test", transport=platform_transport,
    )
    sid = await store.create("sess_e2e_explicit")
    assert sid == "sess_e2e_explicit"
    assert await store.exists(sid) is True
    await store.close()


@pytest.mark.asyncio
async def test_session_save_creates_when_missing(platform_transport) -> None:
    """PUT /v1/sessions/{id} upserts — verifies HttpSessionStore.save() works
    against a session that was never explicitly created."""
    store = HttpSessionStore(
        "http://platform.test", transport=platform_transport,
    )
    await store.save("sess_e2e_upsert", [{"role": "user", "content": "x"}])
    assert await store.exists("sess_e2e_upsert") is True
    await store.close()


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def _trace(trace_id: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        started_at="2026-04-27T12:00:00+00:00",
        ended_at="2026-04-27T12:00:01+00:00",
        model="test-model",
        session_id="sess_e2e",
        status="ok",
        spans=[
            Span(
                trace_id=trace_id, span_id="root",
                name="request", start_time=0.0, end_time=1.0,
                attributes={"prompt_tokens": 10, "completion_tokens": 5},
            ),
        ],
    )


@pytest.mark.asyncio
async def test_trace_save_and_get_roundtrip(platform_transport) -> None:
    store = HttpTraceStore(
        "http://platform.test", transport=platform_transport,
    )
    await store.save_trace(_trace("trace_e2e_1"))
    fetched = await store.get_trace("trace_e2e_1")
    assert fetched is not None
    assert fetched.model == "test-model"
    assert len(fetched.spans) == 1
    assert fetched.spans[0].span_id == "root"
    await store.close()


@pytest.mark.asyncio
async def test_trace_get_missing_returns_none(platform_transport) -> None:
    store = HttpTraceStore(
        "http://platform.test", transport=platform_transport,
    )
    assert await store.get_trace("trace_does_not_exist") is None
    await store.close()


@pytest.mark.asyncio
async def test_trace_list_paginates(platform_transport) -> None:
    store = HttpTraceStore(
        "http://platform.test", transport=platform_transport,
    )
    for i in range(5):
        await store.save_trace(_trace(f"trace_e2e_list_{i}"))

    page1 = await store.list_traces(limit=2, offset=0)
    page2 = await store.list_traces(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    # No overlap between pages.
    ids1 = {s.trace_id for s in page1}
    ids2 = {s.trace_id for s in page2}
    assert ids1.isdisjoint(ids2)
    await store.close()


@pytest.mark.asyncio
async def test_trace_save_upserts(platform_transport) -> None:
    store = HttpTraceStore(
        "http://platform.test", transport=platform_transport,
    )
    t1 = _trace("trace_e2e_upsert")
    await store.save_trace(t1)
    t1.status = "error"
    await store.save_trace(t1)
    fetched = await store.get_trace("trace_e2e_upsert")
    assert fetched is not None
    assert fetched.status == "error"
    await store.close()


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def _record() -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id="fb_local_ignored",  # platform regenerates
        trace_id="trace_e2e_fb",
        session_id="sess_e2e",
        rating=1,
        comment="great",
        correction=None,
        model_id="test-model",
        latency_ms=12.5,
        turn_index=0,
        agent_type="calculus",
        created_at="2026-04-27T12:00:00+00:00",
        user_id="anonymous",  # platform overwrites with bearer subject
    )


@pytest.mark.asyncio
async def test_feedback_add_and_get(platform_transport) -> None:
    store = HttpFeedbackStore(
        "http://platform.test", transport=platform_transport,
    )
    fb_id = await store.add(_record())
    assert fb_id.startswith("fb_")
    assert fb_id != "fb_local_ignored"  # platform-generated
    fetched = await store.get(fb_id)
    assert fetched is not None
    assert fetched.feedback_id == fb_id
    assert fetched.rating == 1
    assert fetched.comment == "great"
    await store.close()


@pytest.mark.asyncio
async def test_feedback_get_missing_returns_none(platform_transport) -> None:
    store = HttpFeedbackStore(
        "http://platform.test", transport=platform_transport,
    )
    assert await store.get("fb_does_not_exist") is None
    await store.close()


@pytest.mark.asyncio
async def test_feedback_query_filters(platform_transport) -> None:
    store = HttpFeedbackStore(
        "http://platform.test", transport=platform_transport,
    )
    rec_a = _record()
    rec_a.trace_id = "trace_filter_a"
    rec_b = _record()
    rec_b.trace_id = "trace_filter_b"
    await store.add(rec_a)
    await store.add(rec_a)
    await store.add(rec_b)

    matches = await store.query(trace_id="trace_filter_a")
    assert len(matches) == 2
    assert all(m.trace_id == "trace_filter_a" for m in matches)
    await store.close()


@pytest.mark.asyncio
async def test_feedback_update(platform_transport) -> None:
    store = HttpFeedbackStore(
        "http://platform.test", transport=platform_transport,
    )
    fb_id = await store.add(_record())
    updated = await store.update(fb_id, comment="updated")
    assert updated is not None
    assert updated.comment == "updated"
    assert updated.rating == 1  # rating unchanged
    await store.close()


@pytest.mark.asyncio
async def test_feedback_update_missing_returns_none(platform_transport) -> None:
    store = HttpFeedbackStore(
        "http://platform.test", transport=platform_transport,
    )
    assert await store.update("fb_missing", comment="x") is None
    await store.close()


@pytest.mark.asyncio
async def test_feedback_stats_aggregates(platform_transport) -> None:
    store = HttpFeedbackStore(
        "http://platform.test", transport=platform_transport,
    )
    up = _record()
    up.trace_id = "trace_stats"
    up.rating = 1
    down = _record()
    down.trace_id = "trace_stats"
    down.rating = -1
    for _ in range(3):
        await store.add(up)
    await store.add(down)

    stats = await store.stats(window="day", agent_type="calculus")
    assert len(stats) >= 1
    bucket = stats[-1]
    assert bucket.thumbs_up >= 3
    assert bucket.thumbs_down >= 1
    await store.close()


# ---------------------------------------------------------------------------
# delete_before is intentionally a no-op against the platform.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_stores_delete_before_is_noop(platform_transport) -> None:
    from datetime import datetime, timezone

    far_future = datetime.now(timezone.utc).replace(year=2099)
    sess = HttpSessionStore("http://platform.test", transport=platform_transport)
    trc = HttpTraceStore("http://platform.test", transport=platform_transport)
    fb = HttpFeedbackStore("http://platform.test", transport=platform_transport)
    try:
        # Seed something the platform would otherwise sweep.
        sid = await sess.create("sess_noop_check")
        await fb.add(_record())

        assert await sess.delete_before(far_future) == 0
        assert await trc.delete_before(far_future) == 0
        assert await fb.delete_before(far_future) == 0

        # Data still present.
        assert await sess.exists(sid) is True
    finally:
        await sess.close()
        await trc.close()
        await fb.close()
