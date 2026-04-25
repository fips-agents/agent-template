"""Tests for fipsagents.server.OpenAIChatServer.

Skipped entirely when FastAPI is not installed (fipsagents[server] extra).
"""

from __future__ import annotations

import asyncio
import json
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
            server=types.SimpleNamespace(
                storage=types.SimpleNamespace(
                    backend=None,
                    sqlite_path="./agent.db",
                    database_url="",
                ),
                sessions=types.SimpleNamespace(
                    enabled=False,
                    max_age_hours=168,
                ),
                traces=types.SimpleNamespace(
                    enabled=False,
                    max_age_hours=168,
                    sampling_rate=1.0,
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
