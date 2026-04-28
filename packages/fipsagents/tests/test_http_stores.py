"""Unit tests for the Http*Store implementations.

Uses ``httpx.MockTransport`` to drive each store against a recorded
wire shape — verifies request method/path/body/params/headers without
needing a real platform service. End-to-end coverage against the live
platform app lives in ``test_http_stores_e2e.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from fipsagents.server.feedback import FeedbackRecord
from fipsagents.server.http import (
    HttpFeedbackStore,
    HttpSessionStore,
    HttpTraceStore,
    PlatformError,
    reset_request_context,
    set_request_context,
)
from fipsagents.server.tracing import Span, Trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Recorder:
    """Collects every request the transport sees, returns canned responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            return httpx.Response(500, json={"detail": "no canned response"})
        return self._responses.pop(0)


def _ok(status: int, body: Any = None) -> httpx.Response:
    if body is None:
        return httpx.Response(status)
    return httpx.Response(status, json=body)


# ---------------------------------------------------------------------------
# HttpSessionStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_create_posts_and_returns_id() -> None:
    rec = _Recorder([_ok(201, {"session_id": "sess_abc"})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    sid = await store.create("sess_abc")
    assert sid == "sess_abc"
    assert rec.requests[0].method == "POST"
    assert rec.requests[0].url.path == "/v1/sessions"
    assert json.loads(rec.requests[0].content) == {"session_id": "sess_abc"}
    await store.close()


@pytest.mark.asyncio
async def test_session_create_omits_id_when_none() -> None:
    rec = _Recorder([_ok(201, {"session_id": "sess_gen"})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    sid = await store.create()
    assert sid == "sess_gen"
    assert json.loads(rec.requests[0].content) == {}
    await store.close()


@pytest.mark.asyncio
async def test_session_load_returns_messages() -> None:
    rec = _Recorder([
        _ok(200, {"session_id": "sess_1", "messages": [{"role": "user", "content": "hi"}]}),
    ])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    msgs = await store.load("sess_1")
    assert msgs == [{"role": "user", "content": "hi"}]
    assert rec.requests[0].method == "GET"
    assert rec.requests[0].url.path == "/v1/sessions/sess_1"
    await store.close()


@pytest.mark.asyncio
async def test_session_load_404_returns_none() -> None:
    rec = _Recorder([_ok(404, {"detail": "not found"})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    assert await store.load("missing") is None
    await store.close()


@pytest.mark.asyncio
async def test_session_save_puts_messages() -> None:
    rec = _Recorder([_ok(200, {"session_id": "sess_1", "saved": True})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    await store.save("sess_1", [{"role": "user", "content": "hi"}])
    assert rec.requests[0].method == "PUT"
    assert rec.requests[0].url.path == "/v1/sessions/sess_1"
    assert json.loads(rec.requests[0].content) == {
        "messages": [{"role": "user", "content": "hi"}],
    }
    await store.close()


@pytest.mark.asyncio
async def test_session_exists_uses_HEAD() -> None:
    rec = _Recorder([_ok(200), _ok(404)])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    assert await store.exists("sess_1") is True
    assert await store.exists("missing") is False
    assert rec.requests[0].method == "HEAD"
    await store.close()


@pytest.mark.asyncio
async def test_session_delete_returns_existed_flag() -> None:
    rec = _Recorder([_ok(200, {"deleted": True}), _ok(404)])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    assert await store.delete("sess_1") is True
    assert await store.delete("missing") is False
    assert rec.requests[0].method == "DELETE"
    await store.close()


@pytest.mark.asyncio
async def test_session_delete_before_is_noop() -> None:
    rec = _Recorder([])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    deleted = await store.delete_before(datetime.now(timezone.utc))
    assert deleted == 0
    assert rec.requests == []  # no HTTP call made
    await store.close()


@pytest.mark.asyncio
async def test_session_update_sends_patch() -> None:
    rec = _Recorder([_ok(200, {
        "session_id": "sess_abc",
        "messages": [],
        "created_at": "2026-04-27T12:00:00+00:00",
        "updated_at": "2026-04-27T12:00:01+00:00",
        "cost_data": {"input_tokens": 100, "output_tokens": 50},
    })])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    tokens = set_request_context(
        authorization="Bearer per-request-jwt", traceparent=None,
    )
    try:
        result = await store.update(
            "sess_abc",
            cost_data={"input_tokens": 100, "output_tokens": 50},
        )
    finally:
        reset_request_context(tokens)
    assert result is True
    assert rec.requests[0].method == "PATCH"
    assert rec.requests[0].url.path == "/v1/sessions/sess_abc"
    assert json.loads(rec.requests[0].content) == {
        "cost_data": {"input_tokens": 100, "output_tokens": 50},
    }
    assert rec.requests[0].headers["authorization"] == "Bearer per-request-jwt"
    await store.close()


@pytest.mark.asyncio
async def test_session_update_404_returns_false() -> None:
    rec = _Recorder([_ok(404, {"detail": "not found"})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    result = await store.update(
        "missing", cost_data={"input_tokens": 1},
    )
    assert result is False
    assert rec.requests[0].method == "PATCH"
    await store.close()


@pytest.mark.asyncio
async def test_session_update_none_cost_data_delegates_to_exists() -> None:
    """When cost_data is None, update() delegates to exists() (HEAD probe)."""
    rec = _Recorder([_ok(200), _ok(404)])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    assert await store.update("sess_1", cost_data=None) is True
    assert await store.update("missing", cost_data=None) is False
    # Both calls should have been HEAD, not PATCH.
    assert rec.requests[0].method == "HEAD"
    assert rec.requests[1].method == "HEAD"
    await store.close()


@pytest.mark.asyncio
async def test_session_update_5xx_raises() -> None:
    rec = _Recorder([httpx.Response(500, json={"detail": "boom"})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    with pytest.raises(PlatformError) as exc_info:
        await store.update("sess_1", cost_data={"input_tokens": 1})
    assert exc_info.value.status_code == 500
    assert "500" in str(exc_info.value)
    await store.close()


# ---------------------------------------------------------------------------
# HttpTraceStore
# ---------------------------------------------------------------------------


def _trace(trace_id: str = "trace_x") -> Trace:
    return Trace(
        trace_id=trace_id,
        started_at="2026-04-27T12:00:00+00:00",
        ended_at="2026-04-27T12:00:01+00:00",
        model="test-model",
        session_id="sess_1",
        status="ok",
        spans=[
            Span(
                trace_id=trace_id,
                span_id="s1",
                name="request",
                start_time=0.0,
                end_time=1.0,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_trace_save_posts_full_payload() -> None:
    rec = _Recorder([_ok(201, {"trace_id": "trace_x", "saved": True})])
    store = HttpTraceStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    await store.save_trace(_trace())
    assert rec.requests[0].method == "POST"
    assert rec.requests[0].url.path == "/v1/traces"
    body = json.loads(rec.requests[0].content)
    assert body["trace_id"] == "trace_x"
    assert body["model"] == "test-model"
    assert len(body["spans"]) == 1
    assert body["spans"][0]["span_id"] == "s1"
    await store.close()


@pytest.mark.asyncio
async def test_trace_get_returns_trace_with_spans() -> None:
    payload = {
        "trace_id": "trace_x",
        "started_at": "2026-04-27T12:00:00+00:00",
        "ended_at": "2026-04-27T12:00:01+00:00",
        "model": "test-model",
        "session_id": "sess_1",
        "status": "ok",
        "spans": [
            {
                "trace_id": "trace_x", "span_id": "s1", "parent_span_id": None,
                "name": "request", "start_time": 0.0, "end_time": 1.0,
                "status": "ok", "attributes": {}, "events": [],
            },
        ],
    }
    rec = _Recorder([_ok(200, payload)])
    store = HttpTraceStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    trace = await store.get_trace("trace_x")
    assert trace is not None
    assert trace.trace_id == "trace_x"
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "request"
    await store.close()


@pytest.mark.asyncio
async def test_trace_get_404_returns_none() -> None:
    rec = _Recorder([_ok(404, {"detail": "not found"})])
    store = HttpTraceStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    assert await store.get_trace("missing") is None
    await store.close()


@pytest.mark.asyncio
async def test_trace_list_passes_pagination_params() -> None:
    rec = _Recorder([_ok(200, [])])
    store = HttpTraceStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    await store.list_traces(limit=25, offset=50)
    url = rec.requests[0].url
    assert url.path == "/v1/traces"
    assert url.params["limit"] == "25"
    assert url.params["offset"] == "50"
    await store.close()


# ---------------------------------------------------------------------------
# HttpFeedbackStore
# ---------------------------------------------------------------------------


def _record(feedback_id: str = "fb_local") -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=feedback_id,
        trace_id="trace_x",
        session_id="sess_1",
        rating=1,
        comment="nice",
        correction=None,
        model_id="test-model",
        latency_ms=12.5,
        turn_index=0,
        agent_type="calculus",
        created_at="2026-04-27T12:00:00+00:00",
        user_id="anonymous",
    )


@pytest.mark.asyncio
async def test_feedback_add_returns_platform_id_not_local() -> None:
    """Platform regenerates feedback_id; local id is intentionally discarded."""
    rec = _Recorder([_ok(201, {"feedback_id": "fb_remote"})])
    store = HttpFeedbackStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    returned = await store.add(_record(feedback_id="fb_local"))
    assert returned == "fb_remote"
    body = json.loads(rec.requests[0].content)
    # Local feedback_id and created_at are NOT sent — the platform owns them.
    assert "feedback_id" not in body
    assert "created_at" not in body
    assert body["rating"] == 1
    assert body["trace_id"] == "trace_x"
    assert body["agent_type"] == "calculus"
    await store.close()


@pytest.mark.asyncio
async def test_feedback_get_returns_record() -> None:
    payload = {
        "feedback_id": "fb_1", "trace_id": "trace_x", "session_id": "sess_1",
        "rating": -1, "comment": "no", "correction": None,
        "model_id": None, "latency_ms": None, "turn_index": None,
        "agent_type": None, "created_at": "2026-04-27T12:00:00+00:00",
        "user_id": "user-7",
    }
    rec = _Recorder([_ok(200, payload)])
    store = HttpFeedbackStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    record = await store.get("fb_1")
    assert record is not None
    assert record.user_id == "user-7"
    assert record.rating == -1
    await store.close()


@pytest.mark.asyncio
async def test_feedback_query_passes_filters() -> None:
    rec = _Recorder([_ok(200, [])])
    store = HttpFeedbackStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    since = datetime(2026, 4, 1, tzinfo=timezone.utc)
    await store.query(
        trace_id="trace_x", user_id="user-7",
        since=since, limit=10, offset=20,
    )
    params = rec.requests[0].url.params
    assert params["trace_id"] == "trace_x"
    assert params["user_id"] == "user-7"
    assert params["since"] == since.isoformat()
    assert params["limit"] == "10"
    assert params["offset"] == "20"
    await store.close()


@pytest.mark.asyncio
async def test_feedback_stats_passes_window() -> None:
    rec = _Recorder([_ok(200, [
        {
            "window_start": "2026-04-27T00:00:00+00:00",
            "window_end": "2026-04-28T00:00:00+00:00",
            "agent_type": "calculus",
            "thumbs_up": 3, "thumbs_down": 1, "total": 4,
        },
    ])])
    store = HttpFeedbackStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    stats = await store.stats(window="day", agent_type="calculus")
    assert len(stats) == 1
    assert stats[0].thumbs_up == 3
    assert rec.requests[0].url.params["window"] == "day"
    await store.close()


@pytest.mark.asyncio
async def test_feedback_update_omits_unchanged_fields() -> None:
    payload = {
        "feedback_id": "fb_1", "trace_id": "trace_x", "session_id": None,
        "rating": -1, "comment": "updated", "correction": None,
        "model_id": None, "latency_ms": None, "turn_index": None,
        "agent_type": None, "created_at": "2026-04-27T12:00:00+00:00",
        "user_id": "anonymous",
    }
    rec = _Recorder([_ok(200, payload)])
    store = HttpFeedbackStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    record = await store.update("fb_1", comment="updated")
    assert record is not None
    body = json.loads(rec.requests[0].content)
    assert body == {"comment": "updated"}  # rating, correction omitted
    await store.close()


@pytest.mark.asyncio
async def test_feedback_update_404_returns_none() -> None:
    rec = _Recorder([_ok(404, {"detail": "not found"})])
    store = HttpFeedbackStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    assert await store.update("missing", rating=1) is None
    await store.close()


# ---------------------------------------------------------------------------
# Auth + traceparent forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_request_authorization_overrides_static_token() -> None:
    rec = _Recorder([_ok(201, {"session_id": "s"})])
    store = HttpSessionStore(
        "http://platform.test",
        static_token="static-secret",
        transport=httpx.MockTransport(rec),
    )
    tokens = set_request_context(
        authorization="Bearer per-request-jwt",
        traceparent="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    )
    try:
        await store.create("s")
    finally:
        reset_request_context(tokens)
    assert rec.requests[0].headers["authorization"] == "Bearer per-request-jwt"
    assert rec.requests[0].headers["traceparent"].startswith("00-")
    await store.close()


@pytest.mark.asyncio
async def test_static_token_used_when_no_request_context() -> None:
    rec = _Recorder([_ok(201, {"session_id": "s"})])
    store = HttpSessionStore(
        "http://platform.test",
        static_token="static-secret",
        transport=httpx.MockTransport(rec),
    )
    await store.create("s")
    assert rec.requests[0].headers["authorization"] == "Bearer static-secret"
    assert "traceparent" not in rec.requests[0].headers
    await store.close()


@pytest.mark.asyncio
async def test_no_auth_header_when_unconfigured() -> None:
    rec = _Recorder([_ok(201, {"session_id": "s"})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    await store.create("s")
    assert "authorization" not in rec.requests[0].headers
    await store.close()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5xx_raises_platform_error_with_status() -> None:
    rec = _Recorder([httpx.Response(503, json={"detail": "down"})])
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    with pytest.raises(PlatformError) as exc_info:
        await store.create("s")
    assert exc_info.value.status_code == 503
    assert "503" in str(exc_info.value)
    await store.close()


@pytest.mark.asyncio
async def test_transport_error_raises_platform_error() -> None:
    def _explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")
    store = HttpSessionStore(
        "http://platform.test", transport=httpx.MockTransport(_explode),
    )
    with pytest.raises(PlatformError) as exc_info:
        await store.create("s")
    assert exc_info.value.status_code is None
    assert "unreachable" in str(exc_info.value).lower()
    await store.close()


@pytest.mark.asyncio
async def test_4xx_other_than_404_raises() -> None:
    rec = _Recorder([httpx.Response(401, json={"detail": "unauthorized"})])
    store = HttpFeedbackStore(
        "http://platform.test", transport=httpx.MockTransport(rec),
    )
    with pytest.raises(PlatformError) as exc_info:
        await store.add(_record())
    assert exc_info.value.status_code == 401
    await store.close()
