"""End-to-end tests for the remote subagent delegation path.

Uses httpx.ASGITransport to wire a parent agent's RemoteSubagentTransport to a
child OpenAIChatServer ASGI app without any real network calls.  The child
server's lifespan is bypassed by pre-injecting the agent instance, which is the
same technique used in the existing server integration tests.
"""

from __future__ import annotations

import json
import types
from typing import Any, AsyncIterator

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")

from fipsagents.baseagent.config import (  # noqa: E402
    RemoteTransportConfig,
    SubagentConfig,
)
from fipsagents.baseagent.events import (  # noqa: E402
    ContentDelta,
    StreamComplete,
    StreamMetrics,
    SubagentCompleted,
    SubagentInvoked,
)
from fipsagents.baseagent.tools.delegate import make_delegate_tool  # noqa: E402
from fipsagents.baseagent.tools import ToolRegistry  # noqa: E402
from fipsagents.server import OpenAIChatServer  # noqa: E402
from fipsagents.subagents.transport import RemoteSubagentTransport  # noqa: E402


# ---------------------------------------------------------------------------
# Child agent stub — used inside the child OpenAIChatServer
# ---------------------------------------------------------------------------


def _make_child_stub(model_name: str = "child-model"):
    """Return a pre-configured child agent stub that OpenAIChatServer accepts."""
    from fipsagents.baseagent.config import BudgetConfig, PricingConfig

    events = [
        ContentDelta(content="child response"),
        StreamComplete(
            finish_reason="stop",
            metrics=StreamMetrics(prompt_tokens=8, completion_tokens=12),
        ),
    ]

    class _ChildStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._events = list(events)
            self.messages: list[dict] = []
            self.tools = ToolRegistry()
            self.config = types.SimpleNamespace(
                model=types.SimpleNamespace(
                    name=model_name,
                    temperature=0.7,
                    max_tokens=4096,
                ),
                memory=types.SimpleNamespace(
                    injection_mode="prefix",
                    injection_tag="user_memories",
                ),
                pricing=PricingConfig(),
                budget=BudgetConfig(),
                server=types.SimpleNamespace(
                    storage=types.SimpleNamespace(
                        backend=None, sqlite_path="./agent.db", database_url="",
                        platform_url="", platform_token="",
                    ),
                    sessions=types.SimpleNamespace(
                        enabled=False, max_age_hours=168, backend=None,
                    ),
                    traces=types.SimpleNamespace(
                        enabled=False, max_age_hours=168, sampling_rate=1.0,
                        exporter=None, otel_endpoint=None,
                        service_name="fipsagents", backend=None,
                    ),
                    metrics=types.SimpleNamespace(
                        enabled=False, token_label_mode="model",
                    ),
                    feedback=types.SimpleNamespace(
                        enabled=False, max_age_hours=720, backend=None,
                    ),
                    files=types.SimpleNamespace(
                        enabled=False,
                        max_file_size_bytes=50 * 1024 * 1024,
                        bytes_dir="./files",
                        sqlite_path="",
                        allowed_mime_types=[],
                        max_age_hours=720,
                        backend=None,
                        bytes_backend=types.SimpleNamespace(
                            type="local_fs", bucket="", endpoint="",
                            region="us-east-1", access_key="", secret_key="",
                            prefix="", path_style=False,
                        ),
                        scanner=types.SimpleNamespace(
                            url="", timeout_seconds=30.0, fail_mode="open",
                        ),
                        parser=types.SimpleNamespace(
                            pdf=types.SimpleNamespace(
                                do_ocr=False, do_table_structure=True,
                            ),
                        ),
                        chunking=types.SimpleNamespace(
                            enabled=False, backend="null", database_url="",
                            embedding_url="", embedding_model="all-MiniLM-L6-v2",
                            embedding_dimension=768, table_name="file_chunks",
                            budget=None, chunk_size_tokens=600,
                            chunk_overlap_tokens=100,
                            small_file_threshold_tokens=4000,
                            retrieval_top_k=5, retrieval_min_score=0.0,
                        ),
                    ),
                ),
            )
            # Subagent contract attrs — not functional on the child but
            # the server code reads them defensively.
            self.subagents: dict = {}
            self._subagent_events: list = []
            self._subagent_token_usage: list = []
            self._delegation_depth: int = 0
            self._inbound_auth_header: str | None = None

        async def setup(self) -> None:
            pass

        async def shutdown(self) -> None:
            pass

        async def astep_stream(
            self, *, max_iterations: int = 10, **_kwargs: Any
        ) -> AsyncIterator:
            for ev in self._events:
                yield ev

        async def run(self) -> str:
            parts = []
            async for ev in self.astep_stream():
                if isinstance(ev, ContentDelta):
                    parts.append(ev.content)
            return "".join(parts)

        def build_system_prompt(self) -> str:
            return ""

        async def step(self):
            from fipsagents.baseagent import StepResult
            return StepResult.done()

    return _ChildStub


# ---------------------------------------------------------------------------
# Recording ASGI middleware — captures requests for test inspection
# ---------------------------------------------------------------------------


class _RequestRecorder:
    """ASGI wrapper that records each HTTP request before delegating."""

    def __init__(self, inner_app: Any, *, records: list[dict]) -> None:
        self._inner = inner_app
        self._records = records

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {
                k.decode("latin-1"): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            # Buffer the body so we can record it and re-play it.
            body_parts: list[bytes] = []
            while True:
                msg = await receive()
                body_parts.append(msg.get("body", b""))
                if not msg.get("more_body"):
                    break
            full_body = b"".join(body_parts)
            try:
                body_json = json.loads(full_body.decode("utf-8"))
            except Exception:
                body_json = None

            self._records.append({
                "headers": headers,
                "path": scope.get("path", ""),
                "body": body_json,
            })

            body_iter = iter([full_body])

            async def _replay() -> dict:
                chunk = next(body_iter, b"")
                return {"type": "http.request", "body": chunk, "more_body": False}

            await self._inner(scope, _replay, send)
        else:
            await self._inner(scope, receive, send)


# ---------------------------------------------------------------------------
# Fixture factory
# ---------------------------------------------------------------------------


def _build_child_server_app(model_name: str = "child-model"):
    """Return (wrapped_app, request_records) with the lifespan bypassed."""
    records: list[dict] = []
    child_class = _make_child_stub(model_name)
    server = OpenAIChatServer(child_class)

    # Bypass the lifespan: inject the agent instance directly so the 503
    # guard passes.  This mirrors what Starlette's TestClient does under
    # the hood via the lifespan context manager.
    server._agent = child_class()

    wrapped = _RequestRecorder(server.app, records=records)
    return wrapped, records


def _build_parent_stub(child_app, *, base_url: str = "http://child.test"):
    """Return (parent_ns, tool_fn, http_client) wired to child_app."""
    cfg = SubagentConfig(
        name="child",
        description="Remote child agent.",
        when_to_use="Use for delegation.",
        transport=RemoteTransportConfig(
            type="remote",
            url=base_url,
            timeout_seconds=30.0,
        ),
        max_depth=3,
        identity="inherit",
        permission_scope=None,
    )

    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=child_app),
        base_url=base_url,
    )

    def transport_factory(name: str, config: SubagentConfig) -> RemoteSubagentTransport:
        return RemoteSubagentTransport(name, config.transport, http_client=http_client)

    parent = types.SimpleNamespace(
        subagents={"child": cfg},
        _subagent_events=[],
        _subagent_token_usage=[],
        _delegation_depth=0,
        _inbound_auth_header=None,
    )

    tool_fn = make_delegate_tool(parent, transport_factory=transport_factory)
    return parent, tool_fn, http_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemoteSubagentEndToEnd:
    @pytest.mark.asyncio
    async def test_child_server_receives_request(self) -> None:
        """Parent's tool call reaches the child ASGI server."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        async with client:
            await tool_fn(agent_name="child", task="hello from parent")

        assert len(records) >= 1, (
            f"Child server received no requests; records: {records}"
        )

    @pytest.mark.asyncio
    async def test_child_request_hits_chat_completions_path(self) -> None:
        """The remote transport posts to /v1/chat/completions."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        async with client:
            await tool_fn(agent_name="child", task="path check")

        assert records
        path = records[0]["path"]
        assert path == "/v1/chat/completions", (
            f"Expected /v1/chat/completions, got: {path!r}"
        )

    @pytest.mark.asyncio
    async def test_child_request_includes_depth_header(self) -> None:
        """The outgoing request carries x-subagent-depth: 1 (depth propagation)."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        async with client:
            await tool_fn(agent_name="child", task="depth check")

        assert records
        headers = records[0]["headers"]
        assert "x-subagent-depth" in headers, (
            f"x-subagent-depth not found in child request headers: {headers}"
        )
        assert headers["x-subagent-depth"] == "1", (
            f"Expected depth '1', got: {headers.get('x-subagent-depth')!r}"
        )

    @pytest.mark.asyncio
    async def test_child_request_includes_traceparent(self) -> None:
        """The outgoing request carries a W3C-compliant traceparent header."""
        from fipsagents.server.propagation import extract_trace_context

        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        async with client:
            await tool_fn(agent_name="child", task="trace check")

        assert records
        headers = records[0]["headers"]
        assert "traceparent" in headers, (
            f"traceparent not found in child request headers: {headers}"
        )
        ctx = extract_trace_context(headers)
        assert ctx is not None, (
            f"traceparent header is malformed: {headers.get('traceparent')!r}"
        )
        assert len(ctx.trace_id) == 32 and len(ctx.parent_span_id) == 16

    @pytest.mark.asyncio
    async def test_result_roundtrips_as_subagent_result(self) -> None:
        """The JSON returned by the tool parses to SubagentResult shape."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        async with client:
            raw = await tool_fn(agent_name="child", task="parse me")

        parsed = json.loads(raw)
        assert parsed["agent_name"] == "child"
        assert parsed["content"] == "child response", (
            f"Unexpected content: {parsed['content']!r}"
        )
        for key in ("tokens_used", "finish_reason", "span_id", "cost_usd"):
            assert key in parsed, f"Missing key in SubagentResult JSON: {key}"

    @pytest.mark.asyncio
    async def test_events_emitted_invoked_then_completed(self) -> None:
        """SubagentInvoked is emitted before SubagentCompleted on the parent."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        async with client:
            await tool_fn(agent_name="child", task="events check")

        events = parent._subagent_events
        assert len(events) == 2, (
            f"Expected 2 events (Invoked + Completed), got: {events}"
        )
        assert isinstance(events[0], SubagentInvoked), (
            f"First event should be SubagentInvoked, got: {type(events[0])}"
        )
        assert isinstance(events[1], SubagentCompleted), (
            f"Second event should be SubagentCompleted, got: {type(events[1])}"
        )

    @pytest.mark.asyncio
    async def test_auth_header_forwarded_when_inbound_set(self) -> None:
        """When identity=inherit and _inbound_auth_header is set, it is forwarded."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        parent._inbound_auth_header = "Bearer test-token-xyz"

        async with client:
            await tool_fn(agent_name="child", task="auth check")

        assert records
        headers = records[0]["headers"]
        assert "authorization" in headers, (
            f"Expected authorization header; headers: {headers}"
        )
        assert headers["authorization"] == "Bearer test-token-xyz", (
            f"Unexpected authorization value: {headers.get('authorization')!r}"
        )

    @pytest.mark.asyncio
    async def test_no_auth_header_when_inbound_is_none(self) -> None:
        """When _inbound_auth_header is None, no authorization header is forwarded."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)
        parent._inbound_auth_header = None

        async with client:
            await tool_fn(agent_name="child", task="no auth check")

        assert records
        headers = records[0]["headers"]
        auth_val = headers.get("authorization", "")
        assert not auth_val, (
            f"Expected no authorization header when inbound is None, got: {auth_val!r}"
        )

    @pytest.mark.asyncio
    async def test_token_usage_appended_to_buffer(self) -> None:
        """After a successful delegation, token usage is appended to the parent buffer."""
        child_app, records = _build_child_server_app()
        parent, tool_fn, client = _build_parent_stub(child_app)

        async with client:
            await tool_fn(agent_name="child", task="token check")

        assert len(parent._subagent_token_usage) == 1, (
            f"Expected 1 token usage entry; got: {parent._subagent_token_usage}"
        )
        usage = parent._subagent_token_usage[0]
        assert "input" in usage and "output" in usage and "cached" in usage, (
            f"Unexpected token usage shape: {usage}"
        )
