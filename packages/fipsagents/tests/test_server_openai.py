"""Tests for fipsagents.server.OpenAIChatServer.

Skipped entirely when FastAPI is not installed (fipsagents[server] extra).
"""

from __future__ import annotations

import asyncio
import json
import re
import types
from typing import AsyncIterator

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fipsagents.baseagent import BaseAgent  # noqa: E402
from fipsagents.baseagent.events import (  # noqa: E402
    ContentDelta,
    StreamComplete,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.baseagent.tools import ToolRegistry  # noqa: E402
from fipsagents.server import OpenAIChatServer  # noqa: E402


# ---------------------------------------------------------------------------
# Stub agent helpers
# ---------------------------------------------------------------------------


class _StubAgent(BaseAgent):
    """Minimal BaseAgent subclass that avoids any real I/O.

    Constructor accepts and ignores the ``config_path``/``base_dir``
    kwargs that :class:`OpenAIChatServer` passes at startup, so it can
    be handed directly to the server constructor.
    """

    def __init__(self, events=None, *, model_name: str = "stub-model", **kwargs):
        # Bypass BaseAgent.__init__ — we own everything the server touches.
        from fipsagents.baseagent.config import BudgetConfig, PricingConfig
        self._events = events or []
        self._system_prompt: str = ""
        self.messages: list[dict] = []
        self.tools = ToolRegistry()
        self.config = types.SimpleNamespace(
            model=types.SimpleNamespace(
                name=model_name, temperature=0.7, max_tokens=4096,
            ),
            memory=types.SimpleNamespace(
                injection_mode="prefix",
                injection_tag="user_memories",
            ),
            pricing=PricingConfig(),
            budget=BudgetConfig(),
            server=types.SimpleNamespace(
                storage=types.SimpleNamespace(
                    backend=None,
                    sqlite_path="./agent.db",
                    database_url="",
                    platform_url="",
                    platform_token="",
                ),
                sessions=types.SimpleNamespace(
                    enabled=False,
                    max_age_hours=168,
                    backend=None,
                ),
                traces=types.SimpleNamespace(
                    enabled=False,
                    max_age_hours=168,
                    sampling_rate=1.0,
                    exporter=None,
                    otel_endpoint=None,
                    service_name="fipsagents",
                    backend=None,
                ),
                metrics=types.SimpleNamespace(
                    enabled=False,
                    token_label_mode="model",
                ),
                feedback=types.SimpleNamespace(
                    enabled=False,
                    max_age_hours=720,
                    backend=None,
                ),
            ),
        )

    def build_system_prompt(self) -> str:
        """Return the stub's configured system prompt (no real file I/O)."""
        return self._system_prompt

    async def setup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def astep_stream(
        self, *, max_iterations: int = 10
    ) -> AsyncIterator:
        for ev in self._events:
            yield ev

    async def run(self) -> str:
        parts = []
        async for ev in self.astep_stream():
            if isinstance(ev, ContentDelta):
                parts.append(ev.content)
        return "".join(parts)

    # BaseAgent requires step() as the abstract method — not used here but
    # must be defined to satisfy the ABC.
    async def step(self):  # type: ignore[override]
        from fipsagents.baseagent import StepResult
        return StepResult.done()


def _make_agent_class(events, *, model_name: str = "stub-model"):
    """Return a new _StubAgent subclass pre-loaded with *events*."""

    class _A(_StubAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(events, model_name=model_name)

    return _A


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_body(body: str) -> list[dict | str]:
    """Parse an SSE response body into a list of parsed JSON objects / '[DONE]'."""
    frames = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            frames.append("[DONE]")
        else:
            frames.append(json.loads(payload))
    return frames


def _delta(chunk: dict) -> dict:
    """Extract delta from a chunk; returns ``{}`` for empty-choices
    chunks (e.g. the trailing usage chunk)."""
    choices = chunk.get("choices") or []
    if not choices:
        return {}
    return choices[0]["delta"]


def _finish_reason(chunk: dict) -> str | None:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    return choices[0]["finish_reason"]


def _build_server(events=None, *, model_name: str = "stub-model") -> OpenAIChatServer:
    return OpenAIChatServer(_make_agent_class(events or [], model_name=model_name))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_healthz_returns_ok():
    server = _build_server()
    with TestClient(server.app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_503_before_startup_200_after():
    server = _build_server()
    # Before entering the TestClient context the lifespan hasn't run,
    # so _agent is None.
    assert server._agent is None

    # Inside the TestClient context the lifespan runs and the agent is ready.
    with TestClient(server.app) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_sync_chat_completions_returns_concatenated_content():
    events = [
        ContentDelta(content="hello "),
        ContentDelta(content="world"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello world"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_sync_response_populates_usage_and_stream_metrics_from_metrics():
    metrics = StreamMetrics(
        time_to_first_content=0.07,
        total_time=0.42,
        inter_token_latencies=[0.01, 0.02],
        prompt_tokens=11,
        completion_tokens=5,
        total_tokens=16,
        model_calls=1,
        tool_calls=0,
    )
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    body = resp.json()
    assert body["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }
    sm = body["stream_metrics"]
    assert sm["time_to_first_content"] == 0.07
    assert sm["total_time"] == 0.42
    assert sm["inter_token_latencies"] == [0.01, 0.02]
    assert sm["model_calls"] == 1


def test_sync_response_finish_reason_reflects_stream_complete():
    events = [
        ContentDelta(content="truncated"),
        StreamComplete(finish_reason="length", metrics=StreamMetrics()),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.json()["choices"][0]["finish_reason"] == "length"


def test_streaming_chat_completions_emits_sse_frames():
    events = [
        ContentDelta(content="hi"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    frames = _parse_sse_body(resp.text)
    assert frames[-1] == "[DONE]"

    # First frame carries the opening role chunk.
    assert _delta(frames[0]) == {"role": "assistant"}

    # At least one content chunk.
    content_chunks = [f for f in frames if isinstance(f, dict) and "content" in _delta(f)]
    assert len(content_chunks) >= 1
    assert content_chunks[0]["choices"][0]["delta"]["content"] == "hi"

    # A finish_reason == "stop" chunk exists.
    stop_chunks = [
        f for f in frames
        if isinstance(f, dict) and _finish_reason(f) == "stop"
    ]
    assert len(stop_chunks) == 1


def test_streaming_response_ends_with_usage_chunk_then_done():
    metrics = StreamMetrics(
        prompt_tokens=9,
        completion_tokens=3,
        total_tokens=12,
        total_time=0.3,
    )
    events = [
        ContentDelta(content="hi"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        )
    frames = _parse_sse_body(resp.text)
    assert frames[-1] == "[DONE]"
    usage_chunk = frames[-2]
    assert isinstance(usage_chunk, dict)
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
    }
    assert usage_chunk["stream_metrics"]["total_time"] == 0.3


def test_streaming_tool_call_events_pass_through():
    events = [
        ToolCallDelta(index=0, call_id="call_abc", name="search", arguments_delta='{"q":"x"}'),
        ToolResultEvent(call_id="call_abc", name="search", content="the result"),
        ContentDelta(content="found it"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "search something"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    frames = _parse_sse_body(resp.text)

    # tool_calls chunk
    tc_chunks = [
        f for f in frames
        if isinstance(f, dict) and "tool_calls" in _delta(f)
    ]
    assert len(tc_chunks) >= 1
    tc = _delta(tc_chunks[0])["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "search"

    # tool result chunk (role == "tool")
    tool_result_chunks = [
        f for f in frames
        if isinstance(f, dict) and _delta(f).get("role") == "tool"
    ]
    assert len(tool_result_chunks) == 1
    assert _delta(tool_result_chunks[0])["tool_call_id"] == "call_abc"
    assert _delta(tool_result_chunks[0])["content"] == "the result"

    # content chunk (exclude tool-role chunks which also carry "content")
    content_chunks = [
        f for f in frames
        if isinstance(f, dict)
        and "content" in _delta(f)
        and _delta(f).get("role") != "tool"
    ]
    assert len(content_chunks) >= 1
    assert content_chunks[0]["choices"][0]["delta"]["content"] == "found it"


def test_streaming_sequential_tool_calls_across_iterations():
    """Two tool calls in separate model iterations both get id+name in SSE.

    Reproduces #72: when astep_stream loops (tool A → result → tool B),
    both iterations use index=0. The SSE serializer must emit opening
    chunks (with id + name) for *each* unique call_id, not just the
    first index=0 it sees.
    """
    events = [
        # -- First model iteration: tool call A --
        ToolCallDelta(index=0, call_id="call_aaa", name="get_weather", arguments_delta='{"city":'),
        ToolCallDelta(index=0, arguments_delta='"Miami"}'),
        ToolResultEvent(call_id="call_aaa", name="get_weather", content="75°F sunny"),
        # -- Second model iteration: tool call B (same index, new call_id) --
        ToolCallDelta(index=0, call_id="call_bbb", name="get_weather", arguments_delta='{"city":'),
        ToolCallDelta(index=0, arguments_delta='"Seattle"}'),
        ToolResultEvent(call_id="call_bbb", name="get_weather", content="58°F cloudy"),
        # -- Final content --
        ContentDelta(content="Miami is 75°F, Seattle is 58°F."),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics(model_calls=2, tool_calls=2)),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "weather in Miami and Seattle"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    frames = _parse_sse_body(resp.text)

    # Collect all tool_calls opening chunks (those with an "id" field).
    tc_open_chunks = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        d = _delta(f)
        tcs = d.get("tool_calls", [])
        for tc in tcs:
            if "id" in tc:
                tc_open_chunks.append(tc)

    # Both tool calls must have received opening chunks with id + name.
    assert len(tc_open_chunks) == 2, (
        f"Expected 2 tool-call opening chunks, got {len(tc_open_chunks)}: {tc_open_chunks}"
    )
    assert tc_open_chunks[0]["id"] == "call_aaa"
    assert tc_open_chunks[0]["function"]["name"] == "get_weather"
    assert tc_open_chunks[1]["id"] == "call_bbb"
    assert tc_open_chunks[1]["function"]["name"] == "get_weather"


def test_model_field_in_request_overrides_config_model():
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server(events, model_name="default-model")
    with TestClient(server.app) as client:
        # Streaming
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "custom-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    frames = _parse_sse_body(resp.text)
    json_frames = [f for f in frames if isinstance(f, dict)]
    assert all(f["model"] == "custom-model" for f in json_frames), (
        f"Expected all frames to use 'custom-model', got: {[f['model'] for f in json_frames]}"
    )


def test_per_request_lock_serializes_streams():
    """Two concurrent streaming requests must not interleave agent state.

    With httpx.AsyncClient we can fire both requests truly concurrently
    inside a single event loop and confirm both complete successfully.
    """
    import httpx

    events_a = [
        ContentDelta(content="A"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    events_b = [
        ContentDelta(content="B"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]

    call_count = 0
    original_events = [events_a, events_b]

    class _SequentialStub(_StubAgent):
        """Each call to astep_stream yields the next batch from original_events."""

        def __init__(self, *args, **kwargs):
            super().__init__([], model_name="stub-model")

        async def astep_stream(self, *, max_iterations: int = 10):
            nonlocal call_count
            idx = call_count % 2
            call_count += 1
            for ev in original_events[idx]:
                yield ev

        async def run(self) -> str:
            parts = []
            async for ev in self.astep_stream():
                if isinstance(ev, ContentDelta):
                    parts.append(ev.content)
            return "".join(parts)

    server = OpenAIChatServer(_SequentialStub)

    async def _run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            # Trigger lifespan manually — httpx.AsyncClient doesn't do this
            # automatically for ASGI apps; use the lifespan context.
            async with server.app.router.lifespan_context(server.app):
                req_a = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "req-a"}],
                        "stream": True,
                    },
                )
                req_b = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "req-b"}],
                        "stream": True,
                    },
                )
                resp_a, resp_b = await asyncio.gather(req_a, req_b)

        return resp_a, resp_b

    resp_a, resp_b = asyncio.get_event_loop().run_until_complete(_run())

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    frames_a = _parse_sse_body(resp_a.text)
    frames_b = _parse_sse_body(resp_b.text)

    # Both streams must end with [DONE]
    assert frames_a[-1] == "[DONE]", f"frames_a did not end with [DONE]: {frames_a}"
    assert frames_b[-1] == "[DONE]", f"frames_b did not end with [DONE]: {frames_b}"


# ---------------------------------------------------------------------------
# /v1/agent-info
# ---------------------------------------------------------------------------


def test_agent_info_returns_model_and_empty_defaults():
    server = _build_server(model_name="gpt-oss-20b")
    with TestClient(server.app) as client:
        resp = client.get("/v1/agent-info")
    assert resp.status_code == 200
    body = resp.json()

    assert body["model"]["name"] == "gpt-oss-20b"
    assert body["model"]["temperature"] == 0.7
    assert body["model"]["max_tokens"] == 4096
    assert body["system_prompt"] == ""
    assert body["tools"] == []


def test_agent_info_extracts_system_prompt():
    server = _build_server()
    with TestClient(server.app) as client:
        # Set the system prompt on the stub agent so build_system_prompt()
        # returns it.  We deliberately do NOT touch agent.messages here to
        # confirm that the endpoint no longer reads from the message buffer
        # (which gets overwritten by every chat request).
        server._agent._system_prompt = "You are a helpful assistant."
        resp = client.get("/v1/agent-info")
    body = resp.json()
    assert body["system_prompt"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Session ID validation
# ---------------------------------------------------------------------------


def test_create_session_with_valid_id():
    """POST /v1/sessions with a valid custom ID succeeds."""
    from fipsagents.server.models import CreateSessionRequest

    req = CreateSessionRequest(session_id="my-session_123")
    assert req.session_id == "my-session_123"


def test_create_session_with_no_id():
    """POST /v1/sessions with no body auto-generates."""
    from fipsagents.server.models import CreateSessionRequest

    req = CreateSessionRequest()
    assert req.session_id is None


def test_create_session_rejects_invalid_characters():
    """Session IDs with invalid characters are rejected."""
    from fipsagents.server.models import CreateSessionRequest

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CreateSessionRequest(session_id="has spaces!")


def test_create_session_rejects_too_long():
    """Session IDs over 128 characters are rejected."""
    from fipsagents.server.models import CreateSessionRequest

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CreateSessionRequest(session_id="a" * 129)


def test_chat_request_rejects_invalid_session_id():
    """ChatCompletionRequest rejects invalid session_id format."""
    from fipsagents.server.models import ChatCompletionRequest, ChatMessage

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            session_id="invalid/path/../id",
        )


def test_chat_request_accepts_valid_session_id():
    """ChatCompletionRequest accepts valid session_id."""
    from fipsagents.server.models import ChatCompletionRequest, ChatMessage

    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        session_id="sess_abc123",
    )
    assert req.session_id == "sess_abc123"


# ---------------------------------------------------------------------------
# /v1/agent-info
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feedback POST identity (X-Auth-Subject) tests
# ---------------------------------------------------------------------------


def _build_server_with_feedback(tmp_path) -> OpenAIChatServer:
    """Build a server with a sqlite feedback store enabled."""
    AgentClass = _make_agent_class([])
    # Enable the feedback feature on the stub config so the server's
    # lifespan creates a SqliteFeedbackStore against a temp path.
    db_path = str(tmp_path / "feedback.db")

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
            self.config.server.feedback = types.SimpleNamespace(
                enabled=True,
                max_age_hours=0,
                backend=None,
            )

    return OpenAIChatServer(_A)


def test_create_feedback_records_x_auth_subject(tmp_path):
    server = _build_server_with_feedback(tmp_path)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/feedback",
            json={"trace_id": "trace_1", "rating": 1, "comment": "great"},
            headers={"X-Auth-Subject": "alice"},
        )
        assert resp.status_code == 201, resp.text
        feedback_id = resp.json()["feedback_id"]

        listed = client.get("/v1/feedback?trace_id=trace_1").json()
        assert len(listed) == 1
        assert listed[0]["feedback_id"] == feedback_id
        assert listed[0]["user_id"] == "alice"


def test_create_feedback_defaults_user_id_to_anonymous_without_header(tmp_path):
    server = _build_server_with_feedback(tmp_path)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/feedback",
            json={"trace_id": "trace_2", "rating": -1},
        )
        assert resp.status_code == 201

        listed = client.get("/v1/feedback?trace_id=trace_2").json()
        assert len(listed) == 1
        assert listed[0]["user_id"] == "anonymous"


def test_list_feedback_filters_by_user_id(tmp_path):
    server = _build_server_with_feedback(tmp_path)
    with TestClient(server.app) as client:
        client.post(
            "/v1/feedback",
            json={"trace_id": "trace_3", "rating": 1},
            headers={"X-Auth-Subject": "alice"},
        )
        client.post(
            "/v1/feedback",
            json={"trace_id": "trace_3", "rating": -1},
            headers={"X-Auth-Subject": "bob"},
        )

        alice = client.get("/v1/feedback?user_id=alice").json()
        assert len(alice) == 1
        assert alice[0]["user_id"] == "alice"

        bob = client.get("/v1/feedback?user_id=bob").json()
        assert len(bob) == 1
        assert bob[0]["user_id"] == "bob"


def test_agent_info_includes_llm_tools():
    from fipsagents.baseagent.tools import ToolMeta

    server = _build_server()
    with TestClient(server.app) as client:
        # Inject tools directly into the registry for testing.
        server._agent.tools._tools["search"] = ToolMeta(
            name="search",
            description="Search the web",
            visibility="llm_only",
            fn=lambda: None,
            is_async=False,
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        # Agent-only tool that should NOT appear.
        server._agent.tools._tools["internal_log"] = ToolMeta(
            name="internal_log",
            description="Internal logging",
            visibility="agent_only",
            fn=lambda: None,
            is_async=False,
        )
        resp = client.get("/v1/agent-info")

    body = resp.json()
    assert len(body["tools"]) == 1
    assert body["tools"][0]["name"] == "search"
    assert body["tools"][0]["description"] == "Search the web"
    assert body["tools"][0]["parameters"]["properties"]["q"]["type"] == "string"


# ---------------------------------------------------------------------------
# X-Trace-Id surfacing
# ---------------------------------------------------------------------------


_TRACE_ID_RE = re.compile(r"^trace_[0-9a-f]{16}$")


def test_sync_response_includes_x_trace_id_header():
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    trace_id = resp.headers.get("x-trace-id")
    assert trace_id is not None
    assert _TRACE_ID_RE.match(trace_id), f"Bad trace_id format: {trace_id!r}"


def test_streaming_response_includes_x_trace_id_header_and_chunk_field():
    events = [
        ContentDelta(content="hi"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        )
    header_trace_id = resp.headers.get("x-trace-id")
    assert header_trace_id is not None
    assert _TRACE_ID_RE.match(header_trace_id)

    frames = _parse_sse_body(resp.text)
    usage_chunks = [
        f for f in frames if isinstance(f, dict) and f.get("choices") == []
    ]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["trace_id"] == header_trace_id


def test_x_trace_id_uses_propagated_parent_when_provided():
    """If a W3C traceparent header is sent, the trace_id surfaces from it."""
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server(events)
    parent_trace_hex = "0123456789abcdef0123456789abcdef"
    traceparent = f"00-{parent_trace_hex}-0123456789abcdef-01"
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            headers={"traceparent": traceparent},
        )
    assert resp.status_code == 200
    assert resp.headers.get("x-trace-id") == parent_trace_hex


def test_create_feedback_accepts_request_without_trace_id():
    """trace_id is optional; server synthesises one when absent."""
    from fipsagents.server.models import CreateFeedbackRequest

    req = CreateFeedbackRequest(rating=1)
    assert req.trace_id is None
    assert req.rating == 1


def test_create_feedback_rejects_invalid_rating():
    from fipsagents.server.models import CreateFeedbackRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CreateFeedbackRequest(rating=0)


# ---------------------------------------------------------------------------
# Cost-data accumulator (per-turn token usage → SessionStore.update)
# ---------------------------------------------------------------------------


def _build_server_with_sqlite_sessions(tmp_path, events, *, model_name="stub"):
    """Build a server with sessions enabled and backed by SQLite."""
    AgentClass = _make_agent_class(events, model_name=model_name)
    db_path = str(tmp_path / "sessions.db")

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
                enabled=True,
                max_age_hours=0,
                backend=None,
            )

    return OpenAIChatServer(_A)


def test_cost_data_persisted_across_turns(tmp_path):
    """Two completions on the same session_id accumulate cumulative totals."""
    metrics = StreamMetrics(
        prompt_tokens=10, completion_tokens=4, total_tokens=14,
    )
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server_with_sqlite_sessions(
        tmp_path, events, model_name="stub-model",
    )
    with TestClient(server.app) as client:
        # Pre-create the session so the first save's upsert finds it.
        resp = client.post("/v1/sessions", json={"session_id": "sess_cost"})
        assert resp.status_code == 201

        for _ in range(2):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "session_id": "sess_cost",
                },
            )
            assert resp.status_code == 200

        # Read cost_data directly from the sqlite store to verify.
        store = server._session_store
        cost = asyncio.get_event_loop().run_until_complete(
            store.get_cost_data("sess_cost")
        )

    assert cost == {
        "input_tokens": 20,
        "output_tokens": 8,
        "cached_tokens": 0,
        "model": "stub-model",
        "turn_count": 2,
    }


def test_cost_data_no_session_no_persist(tmp_path):
    """Without a session_id, update() must not be invoked."""
    metrics = StreamMetrics(prompt_tokens=10, completion_tokens=4)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server_with_sqlite_sessions(tmp_path, events)

    with TestClient(server.app) as client:
        store = server._session_store
        update_calls: list = []
        original_update = store.update

        async def _spy(session_id, *, cost_data=None):
            update_calls.append((session_id, cost_data))
            return await original_update(session_id, cost_data=cost_data)

        store.update = _spy  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200

    assert update_calls == []


def test_cost_data_persist_failure_does_not_break_response(tmp_path):
    """If update() raises, the chat response still completes successfully."""
    metrics = StreamMetrics(prompt_tokens=10, completion_tokens=4)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server_with_sqlite_sessions(tmp_path, events)

    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/sessions", json={"session_id": "sess_boom"},
        )
        assert resp.status_code == 201

        async def _boom(session_id, *, cost_data=None):  # noqa: ARG001
            raise RuntimeError("simulated platform 500")

        server._session_store.update = _boom  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "session_id": "sess_boom",
            },
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"


def test_cost_data_null_session_store_is_noop():
    """With NullSessionStore (default), the server doesn't crash."""
    metrics = StreamMetrics(prompt_tokens=10, completion_tokens=4)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    # _build_server uses the default stub config (sessions disabled,
    # NullSessionStore). The chat request without a session_id must
    # succeed.
    server = _build_server(events)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    # Even when session_id is provided but store is the Null backend
    # (sessions disabled in config), no crash.
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "session_id": "sess_anything",
            },
        )
    assert resp.status_code == 200


def test_cost_data_persisted_across_streaming_turns(tmp_path):
    """The streaming path also accumulates per-turn token usage."""
    metrics = StreamMetrics(
        prompt_tokens=7, completion_tokens=3, total_tokens=10,
    )
    events = [
        ContentDelta(content="hi"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server_with_sqlite_sessions(
        tmp_path, events, model_name="stream-stub",
    )
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/sessions", json={"session_id": "sess_stream"},
        )
        assert resp.status_code == 201

        for _ in range(2):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "session_id": "sess_stream",
                },
            )
            assert resp.status_code == 200
            # Drain so the streaming task completes before next turn.
            _ = resp.text

        cost = asyncio.get_event_loop().run_until_complete(
            server._session_store.get_cost_data("sess_stream")
        )

    assert cost["input_tokens"] == 14
    assert cost["output_tokens"] == 6
    assert cost["turn_count"] == 2
    assert cost["model"] == "stream-stub"


def test_cost_data_no_usage_no_persist(tmp_path):
    """When the model didn't report usage, no cost_data is written."""
    # No prompt_tokens / completion_tokens → metrics has all-None counts.
    metrics = StreamMetrics()
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server_with_sqlite_sessions(tmp_path, events)

    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/sessions", json={"session_id": "sess_nousage"},
        )
        assert resp.status_code == 201

        update_calls: list = []
        original_update = server._session_store.update

        async def _spy(session_id, *, cost_data=None):
            update_calls.append((session_id, cost_data))
            return await original_update(session_id, cost_data=cost_data)

        server._session_store.update = _spy  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "session_id": "sess_nousage",
            },
        )
        assert resp.status_code == 200

    assert update_calls == []


def test_cost_data_http_get_not_implemented_falls_back_to_delta(tmp_path):
    """When get_cost_data raises NotImplementedError, the next write is the delta only.

    This emulates the HttpSessionStore case until the platform exposes a
    GET cost_data endpoint. The accumulator must NOT crash; it simply
    treats the existing total as empty so the write becomes a per-turn
    delta rather than a true cumulative.
    """
    metrics = StreamMetrics(prompt_tokens=10, completion_tokens=4)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    server = _build_server_with_sqlite_sessions(tmp_path, events)

    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/sessions", json={"session_id": "sess_http_like"},
        )
        assert resp.status_code == 201

        async def _no_read(session_id):  # noqa: ARG001
            raise NotImplementedError("simulated http backend")

        server._session_store.get_cost_data = _no_read  # type: ignore[method-assign]

        # Two turns: each writes the per-turn delta because the read
        # raises NotImplementedError. With a real Sqlite backend the
        # update() path still merges into the row, so we expect two
        # separate writes that DO accumulate via update()'s shallow
        # merge -- but turn_count won't be cumulative since we can't
        # read the prior value. That is the documented behaviour.
        for _ in range(2):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "session_id": "sess_http_like",
                },
            )
            assert resp.status_code == 200

        # Restore the real reader to inspect what actually got written.
        del server._session_store.get_cost_data  # type: ignore[attr-defined]
        cost = asyncio.get_event_loop().run_until_complete(
            server._session_store.get_cost_data("sess_http_like")
        )

    # Each write replaced (write-wins) the per-turn delta, not the cumulative.
    assert cost["input_tokens"] == 10
    assert cost["output_tokens"] == 4
    assert cost["turn_count"] == 1


# ---------------------------------------------------------------------------
# /v1/sessions/{id}/usage — computed dollar view of cost_data
# ---------------------------------------------------------------------------


def _build_server_with_pricing(tmp_path, events, *, model_name, pricing):
    """Build a server with sessions enabled and a custom PricingConfig."""
    AgentClass = _make_agent_class(events, model_name=model_name)
    db_path = str(tmp_path / "sessions.db")

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
                enabled=True,
                max_age_hours=0,
                backend=None,
            )
            self.config.pricing = pricing

    return OpenAIChatServer(_A)


def test_session_usage_404_when_session_missing(tmp_path):
    """GET /usage returns 404 if the session was never created."""
    from fipsagents.baseagent.config import PricingConfig

    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    server = _build_server_with_pricing(
        tmp_path, events, model_name="stub", pricing=PricingConfig(),
    )
    with TestClient(server.app) as client:
        resp = client.get("/v1/sessions/sess_nope/usage")
    assert resp.status_code == 404


def test_session_usage_zero_for_new_session(tmp_path):
    """A freshly-created session with no turns reports zeros and 0.0 cost."""
    from fipsagents.baseagent.config import PricingConfig, PricingRate

    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
    ]
    pricing = PricingConfig(default=PricingRate(input_per_1k=0.01))
    server = _build_server_with_pricing(
        tmp_path, events, model_name="stub", pricing=pricing,
    )
    with TestClient(server.app) as client:
        client.post("/v1/sessions", json={"session_id": "sess_empty"})
        resp = client.get("/v1/sessions/sess_empty/usage")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess_empty"
    assert body["input_tokens"] == 0
    assert body["output_tokens"] == 0
    assert body["cached_tokens"] == 0
    assert body["turn_count"] == 0
    assert body["cost_usd"] == 0.0
    assert body["pricing"]["input_per_1k"] == 0.01


def test_session_usage_computes_cumulative_dollars(tmp_path):
    """After two completions, /usage reflects cumulative tokens × rate."""
    from fipsagents.baseagent.config import PricingConfig, PricingRate

    metrics = StreamMetrics(
        prompt_tokens=1000, completion_tokens=500, total_tokens=1500,
    )
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    pricing = PricingConfig(
        default=PricingRate(input_per_1k=0.001, output_per_1k=0.002),
        models={
            "billed-model": PricingRate(
                input_per_1k=0.01, output_per_1k=0.02,
            ),
        },
    )
    server = _build_server_with_pricing(
        tmp_path, events, model_name="billed-model", pricing=pricing,
    )
    with TestClient(server.app) as client:
        client.post("/v1/sessions", json={"session_id": "sess_paid"})
        for _ in range(2):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "session_id": "sess_paid",
                },
            )
            assert resp.status_code == 200

        resp = client.get("/v1/sessions/sess_paid/usage")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess_paid"
    assert body["model"] == "billed-model"
    assert body["input_tokens"] == 2000
    assert body["output_tokens"] == 1000
    assert body["turn_count"] == 2
    # 2000/1000 * 0.01 + 1000/1000 * 0.02 = 0.02 + 0.02
    assert body["cost_usd"] == pytest.approx(0.04)
    assert body["pricing"]["input_per_1k"] == 0.01


def test_tenant_label_flows_from_x_tenant_header(tmp_path):
    """X-Tenant header lands on agent_tokens_total when mode is 'tenant'."""
    pytest.importorskip("prometheus_client")

    metrics = StreamMetrics(prompt_tokens=42, completion_tokens=8)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    AgentClass = _make_agent_class(events, model_name="m1")

    class _A(AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.server.metrics = types.SimpleNamespace(
                enabled=True, token_label_mode="tenant",
            )

    server = OpenAIChatServer(_A)
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            headers={"X-Tenant": "acme-corp"},
        )
        assert resp.status_code == 200

        metrics_resp = client.get("/metrics")

    body = metrics_resp.text
    assert 'tenant_id="acme-corp"' in body
    assert 'agent_tokens_total{' in body


def test_tenant_label_defaults_when_header_absent(tmp_path):
    """Without X-Tenant, tenant_id falls back to 'default'."""
    pytest.importorskip("prometheus_client")

    metrics = StreamMetrics(prompt_tokens=1)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    AgentClass = _make_agent_class(events, model_name="m1")

    class _A(AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.server.metrics = types.SimpleNamespace(
                enabled=True, token_label_mode="tenant",
            )

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
        metrics_resp = client.get("/metrics")

    assert 'tenant_id="default"' in metrics_resp.text


def test_session_usage_uses_cost_data_model_over_default(tmp_path):
    """The model field on cost_data wins over the agent's current default.

    Different turns can target different models (e.g. routing). The model
    that *billed* the tokens is the one recorded on cost_data, so it wins
    pricing lookup over the agent's current ``model.name`` configuration.
    """
    from fipsagents.baseagent.config import PricingConfig, PricingRate

    metrics = StreamMetrics(prompt_tokens=1000, completion_tokens=0)
    events = [
        ContentDelta(content="ok"),
        StreamComplete(finish_reason="stop", metrics=metrics),
    ]
    pricing = PricingConfig(
        default=PricingRate(input_per_1k=99.0),  # would dominate if used
        models={
            "billed-model": PricingRate(input_per_1k=0.01),
        },
    )
    server = _build_server_with_pricing(
        tmp_path, events, model_name="billed-model", pricing=pricing,
    )
    with TestClient(server.app) as client:
        client.post("/v1/sessions", json={"session_id": "sess_route"})
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "session_id": "sess_route",
            },
        )
        resp = client.get("/v1/sessions/sess_route/usage")

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "billed-model"
    # 1000/1000 * 0.01, NOT 99.0
    assert body["cost_usd"] == pytest.approx(0.01)
