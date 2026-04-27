"""OpenAIChatServer — FastAPI server wrapping a BaseAgent subclass."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

try:
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover — helpful error path
    raise ImportError(
        "fipsagents.server requires the [server] extra. "
        "Install with: pip install 'fipsagents[server]'"
    ) from exc

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.events import ContentDelta, StreamComplete, StreamMetrics
from fipsagents.serialization.openai_sse import stream_events_as_sse

from .models import (
    ChatCompletionRequest,
    CreateFeedbackRequest,
    CreateSessionRequest,
    UpdateFeedbackRequest,
    _extract_overrides,
    _messages_to_dicts,
    _sync_response,
)
from .collector import TraceCollector
from .metrics import NullMetricsCollector, create_metrics_collector
from .sessions import SessionStore, create_session_store
from .tracing import TraceStore, create_trace_store
from .feedback import (
    FeedbackRecord,
    FeedbackStore,
    _generate_feedback_id,
    _utc_now_iso,
    create_feedback_store,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------


class OpenAIChatServer:
    """FastAPI server exposing OpenAI-compatible chat completions.

    Wraps any :class:`~fipsagents.baseagent.BaseAgent` subclass, owning the
    agent lifecycle from startup to shutdown. The agent class is instantiated
    once at application start — all requests share a single agent instance,
    serialised through ``_agent_lock``.

    Args:
        agent_class: A :class:`BaseAgent` subclass (pass the class, not an
            instance). The server instantiates it with ``config_path`` and
            ``base_dir`` at startup.
        config_path: Path to the agent YAML config file.
        base_dir: Optional base directory for relative paths inside the agent
            config. Defaults to the config file's parent directory.
        title: FastAPI application title. Defaults to ``agent_class.__name__``.
        version: FastAPI application version string.
    """

    def __init__(
        self,
        agent_class: type[BaseAgent],
        config_path: str | Path = "agent.yaml",
        *,
        base_dir: str | Path | None = None,
        title: str | None = None,
        version: str = "0.1.0",
    ) -> None:
        self._agent_class = agent_class
        self._config_path = Path(config_path)
        self._base_dir = Path(base_dir) if base_dir is not None else None

        self._agent: BaseAgent | None = None
        self._agent_lock = asyncio.Lock()
        self._session_store: SessionStore | None = None
        self._trace_store: TraceStore | None = None
        self._feedback_store: FeedbackStore | None = None
        self._metrics_collector: Any = None  # Set in lifespan
        self._housekeeping_task: asyncio.Task | None = None
        self._sqlite_mgr: Any = None

        app_title = title if title is not None else agent_class.__name__
        self.app = FastAPI(
            title=app_title,
            version=version,
            lifespan=self._lifespan,
        )
        self._register_routes()

    # -- Lifespan ------------------------------------------------------------

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):  # noqa: ARG002
        self._agent = self._agent_class(
            config_path=self._config_path,
            base_dir=self._base_dir,
        )
        await self._agent.setup()

        # Initialize session and trace stores from config.
        server_cfg = self._agent.config.server
        sqlite_conn = None

        has_sqlite_feature = (
            server_cfg.storage.backend == "sqlite"
            and (server_cfg.sessions.enabled or server_cfg.traces.enabled or server_cfg.feedback.enabled)
        )
        if has_sqlite_feature:
            from .sqlite import SqliteConnectionManager

            self._sqlite_mgr = SqliteConnectionManager()
            sqlite_conn = await self._sqlite_mgr.acquire(
                server_cfg.storage.sqlite_path,
            )

        self._session_store = create_session_store(
            server_cfg.storage.backend if server_cfg.sessions.enabled else None,
            sqlite_path=server_cfg.storage.sqlite_path,
            database_url=server_cfg.storage.database_url,
            sqlite_connection=sqlite_conn,
        )
        self._trace_store = create_trace_store(
            server_cfg.storage.backend if server_cfg.traces.enabled else None,
            sqlite_path=server_cfg.storage.sqlite_path,
            database_url=server_cfg.storage.database_url,
            sqlite_connection=sqlite_conn,
            exporter=server_cfg.traces.exporter,
            otel_endpoint=server_cfg.traces.otel_endpoint,
            service_name=server_cfg.traces.service_name,
        )
        self._feedback_store = create_feedback_store(
            server_cfg.storage.backend if server_cfg.feedback.enabled else None,
            sqlite_path=server_cfg.storage.sqlite_path,
            database_url=server_cfg.storage.database_url,
            sqlite_connection=sqlite_conn,
        )

        # Initialize metrics collector.
        self._metrics_collector = create_metrics_collector(
            enabled=server_cfg.metrics.enabled,
        )

        # Only run housekeeping if at least one feature has a persistent backend.
        if server_cfg.storage.backend is not None and (
            server_cfg.sessions.enabled or server_cfg.traces.enabled or server_cfg.feedback.enabled
        ):
            self._housekeeping_task = asyncio.create_task(self._run_housekeeping())

        logger.info("OpenAIChatServer: %s ready", self._agent_class.__name__)
        try:
            yield
        finally:
            if self._housekeeping_task:
                self._housekeeping_task.cancel()
                try:
                    await self._housekeeping_task
                except asyncio.CancelledError:
                    pass
            await self._agent.shutdown()
            await self._session_store.close()
            await self._trace_store.close()
            await self._feedback_store.close()
            if self._sqlite_mgr:
                await self._sqlite_mgr.close_all()
            self._agent = None

    # -- Housekeeping --------------------------------------------------------

    async def _run_housekeeping(self, interval_seconds: int = 3600) -> None:
        """Periodically clean up expired sessions and traces."""
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self._do_housekeeping()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("Housekeeping error", exc_info=True)

    async def _do_housekeeping(self) -> None:
        """Run one housekeeping pass."""
        from datetime import datetime, timedelta, timezone

        if self._agent is None:
            return

        server_cfg = self._agent.config.server

        if server_cfg.sessions.max_age_hours > 0 and self._session_store:
            cutoff = datetime.now(timezone.utc) - timedelta(
                hours=server_cfg.sessions.max_age_hours,
            )
            deleted = await self._session_store.delete_before(cutoff)
            if deleted:
                logger.info("Housekeeping: removed %d expired sessions", deleted)

        if server_cfg.traces.max_age_hours > 0 and self._trace_store:
            cutoff = datetime.now(timezone.utc) - timedelta(
                hours=server_cfg.traces.max_age_hours,
            )
            deleted = await self._trace_store.delete_before(cutoff)
            if deleted:
                logger.info("Housekeeping: removed %d expired traces", deleted)

        if server_cfg.feedback.max_age_hours > 0 and self._feedback_store:
            cutoff = datetime.now(timezone.utc) - timedelta(
                hours=server_cfg.feedback.max_age_hours,
            )
            deleted = await self._feedback_store.delete_before(cutoff)
            if deleted:
                logger.info("Housekeeping: removed %d expired feedback records", deleted)

    # -- Route registration --------------------------------------------------

    def _register_routes(self) -> None:
        self.app.api_route("/healthz", methods=["GET", "HEAD"])(self._healthz)
        self.app.api_route("/readyz", methods=["GET", "HEAD"])(self._readyz)
        self.app.get("/v1/agent-info")(self._agent_info)
        self.app.post("/v1/sessions")(self._create_session)
        self.app.get("/v1/sessions/{session_id}")(self._get_session)
        self.app.delete("/v1/sessions/{session_id}")(self._delete_session)
        self.app.get("/v1/traces")(self._list_traces)
        self.app.get("/v1/traces/{trace_id}")(self._get_trace)
        self.app.post("/v1/feedback")(self._create_feedback)
        self.app.patch("/v1/feedback/{feedback_id}")(self._update_feedback)
        self.app.get("/v1/feedback/stats")(self._feedback_stats)
        self.app.get("/v1/feedback")(self._list_feedback)
        self.app.post("/v1/chat/completions")(self._chat_completions)
        self.app.get("/metrics")(self._metrics_endpoint)

    # -- Endpoint handlers ---------------------------------------------------

    async def _healthz(self) -> dict[str, str]:
        return {"status": "ok"}

    async def _readyz(self):
        if self._agent is None:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return {"status": "ready"}

    async def _agent_info(self):
        if self._agent is None:
            raise HTTPException(status_code=503, detail="Agent not ready")

        agent = self._agent

        # Always read from the prompt files, not agent.messages, which gets
        # overwritten by _collect_sync / _stream on every chat request.
        system_prompt = agent.build_system_prompt()

        info: dict[str, Any] = {}

        # Include agent identity if available in config.
        if (
            agent.config is not None
            and hasattr(agent.config, "agent")
        ):
            info["agent"] = {
                "name": agent.config.agent.name,
                "description": agent.config.agent.description,
                "version": agent.config.agent.version,
            }

        info["model"] = {
            "name": agent.config.model.name,
            "temperature": agent.config.model.temperature,
            "max_tokens": agent.config.model.max_tokens,
        }
        info["system_prompt"] = system_prompt
        info["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in agent.tools.get_llm_tools()
        ]

        return JSONResponse(info)

    async def _create_session(self, body: CreateSessionRequest = Body(default_factory=CreateSessionRequest)):
        if self._session_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        sid = await self._session_store.create(body.session_id)
        return JSONResponse({"session_id": sid}, status_code=201)

    async def _get_session(self, session_id: str):
        if self._session_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        messages = await self._session_store.load(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return JSONResponse({"session_id": session_id, "messages": messages})

    async def _delete_session(self, session_id: str):
        if self._session_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        existed = await self._session_store.delete(session_id)
        if not existed:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return JSONResponse({"deleted": True})

    async def _list_traces(self, limit: int = 50, offset: int = 0):
        if self._trace_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        from dataclasses import asdict
        summaries = await self._trace_store.list_traces(limit=limit, offset=offset)
        return JSONResponse([asdict(s) for s in summaries])

    async def _get_trace(self, trace_id: str):
        if self._trace_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        from dataclasses import asdict
        trace = await self._trace_store.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        return JSONResponse(asdict(trace))

    async def _create_feedback(self, body: CreateFeedbackRequest):
        if self._feedback_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        record = FeedbackRecord(
            feedback_id=_generate_feedback_id(),
            trace_id=body.trace_id,
            session_id=body.session_id,
            rating=body.rating,
            comment=body.comment,
            correction=body.correction,
            model_id=body.model_id,
            latency_ms=body.latency_ms,
            turn_index=body.turn_index,
            agent_type=body.agent_type,
            created_at=_utc_now_iso(),
        )
        feedback_id = await self._feedback_store.add(record)
        return JSONResponse({"feedback_id": feedback_id}, status_code=201)

    async def _update_feedback(self, feedback_id: str, body: UpdateFeedbackRequest):
        if self._feedback_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        record = await self._feedback_store.update(
            feedback_id,
            rating=body.rating,
            comment=body.comment,
            correction=body.correction,
        )
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Feedback {feedback_id} not found",
            )
        from dataclasses import asdict
        return JSONResponse(asdict(record))

    async def _list_feedback(
        self,
        trace_id: str | None = None,
        session_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        if self._feedback_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        from dataclasses import asdict
        from datetime import datetime
        since_dt = datetime.fromisoformat(since) if since else None
        until_dt = datetime.fromisoformat(until) if until else None
        records = await self._feedback_store.query(
            trace_id=trace_id,
            session_id=session_id,
            since=since_dt,
            until=until_dt,
            limit=min(limit, 1000),
            offset=max(offset, 0),
        )
        return JSONResponse([asdict(r) for r in records])

    async def _feedback_stats(
        self,
        window: str = "day",
        agent_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ):
        if self._feedback_store is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        from dataclasses import asdict
        from datetime import datetime
        if window not in ("hour", "day", "week"):
            raise HTTPException(status_code=400, detail="window must be 'hour', 'day', or 'week'")
        since_dt = datetime.fromisoformat(since) if since else None
        until_dt = datetime.fromisoformat(until) if until else None
        results = await self._feedback_store.stats(
            window=window,
            agent_type=agent_type,
            since=since_dt,
            until=until_dt,
        )
        return JSONResponse([asdict(r) for r in results])

    def _should_trace(self) -> bool:
        """Decide whether to trace this request based on sampling rate."""
        if self._trace_store is None or self._agent is None:
            return False
        rate = self._agent.config.server.traces.sampling_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        return random.random() < rate

    async def _metrics_endpoint(self):
        if self._metrics_collector is None or isinstance(
            self._metrics_collector, NullMetricsCollector
        ):
            raise HTTPException(status_code=404, detail="Metrics not enabled")
        from fastapi.responses import Response

        return Response(
            content=self._metrics_collector.generate_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _chat_completions(self, request: Request, req: ChatCompletionRequest):
        if self._agent is None:
            raise HTTPException(status_code=503, detail="Agent not ready")

        agent = self._agent
        model_name = req.model or agent.config.model.name
        incoming = _messages_to_dicts(req.messages)
        overrides = _extract_overrides(req)

        # Session: load prior messages if session_id provided.
        if req.session_id and self._session_store:
            stored = await self._session_store.load(req.session_id)
            if stored:
                incoming = stored + incoming
            else:
                logger.info("Session %s not found; will auto-create on save", req.session_id)

        # Tracing: create collector if sampling says yes.
        collector: TraceCollector | None = None
        if self._should_trace():
            from .propagation import extract_trace_context
            parent_ctx = extract_trace_context(request.headers)

            collector = TraceCollector(
                self._trace_store,
                session_id=req.session_id,
                model=model_name,
                parent_trace_id=parent_ctx.trace_id if parent_ctx else None,
                parent_span_id=parent_ctx.parent_span_id if parent_ctx else None,
            )
            collector.begin_request({
                "model": model_name,
                "stream": req.stream,
                "session_id": req.session_id,
            })

        # Metrics: start timing.
        metrics_start: float | None = None
        if self._metrics_collector is not None:
            metrics_start = self._metrics_collector.record_request_start()

        if not req.stream:
            content, metrics, finish_reason = await self._collect_sync(
                agent, incoming, model_name=model_name,
                overrides=overrides, collector=collector,
            )
            # Session: save after sync response.
            if req.session_id and self._session_store:
                await self._session_store.save(req.session_id, agent.messages)
            if collector:
                await collector.end_request()
            if self._metrics_collector and metrics_start is not None:
                self._metrics_collector.record_request_end(
                    model_name, False, "ok", metrics_start,
                )
            return JSONResponse(
                _sync_response(
                    model_name,
                    content,
                    metrics=metrics,
                    finish_reason=finish_reason,
                )
            )

        return StreamingResponse(
            self._stream(
                incoming, model_name, overrides=overrides,
                session_id=req.session_id, collector=collector,
                metrics_start=metrics_start,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -- Sync ----------------------------------------------------------------

    async def _collect_sync(
        self,
        agent: BaseAgent,
        incoming: list[dict[str, Any]],
        *,
        model_name: str = "",
        overrides: dict[str, Any] | None = None,
        collector: TraceCollector | None = None,
    ) -> tuple[str, StreamMetrics | None, str]:
        """Drive ``astep_stream`` for a non-streaming response.

        Fully drains the iterator so any post-``StreamComplete`` hooks
        in the subclass (e.g. memory writes) run to completion.

        If no ``ContentDelta`` events are emitted (e.g. the agent executed
        tools and the final content was appended directly to
        ``agent.messages`` without streaming deltas), fall back to the
        last assistant message in the conversation history.
        """
        parts: list[str] = []
        metrics: StreamMetrics | None = None
        finish_reason = "stop"
        async with self._agent_lock:
            agent.messages = list(incoming)
            events = agent.astep_stream(max_iterations=10, **(overrides or {}))
            if self._metrics_collector is not None:
                events = self._metrics_collector.observe(
                    events, model=model_name,
                )
            if collector:
                events = collector.observe(events)
            async for event in events:
                if isinstance(event, ContentDelta):
                    parts.append(event.content)
                elif isinstance(event, StreamComplete):
                    metrics = event.metrics
                    finish_reason = event.finish_reason

        content = "".join(parts)

        # Strip echoed memory injection tags from the response. When
        # injection_mode is "user_turn" the framework wraps memories in
        # <injection_tag>...</injection_tag> before sending them to the model.
        # Some models echo those tags back verbatim; strip them defensively.
        if agent.config.memory.injection_mode == "user_turn":
            tag = re.escape(agent.config.memory.injection_tag)
            content = re.sub(
                rf"<{tag}>.*?</{tag}>", "", content, flags=re.DOTALL
            ).strip()

        # Fallback: if no ContentDelta events were yielded but the agent
        # appended an assistant message (common after tool execution in
        # subclasses that override astep_stream), use that content.
        if not content and agent.messages:
            for msg in reversed(agent.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    break

        return content, metrics, finish_reason

    # -- Streaming -----------------------------------------------------------

    async def _stream(
        self,
        incoming: list[dict[str, Any]],
        model_name: str,
        *,
        overrides: dict[str, Any] | None = None,
        session_id: str | None = None,
        collector: TraceCollector | None = None,
        metrics_start: float | None = None,
    ) -> AsyncIterator[str]:
        """Drive the agent's event stream, serialising to OpenAI SSE chunks.

        NOTE: Memory injection tags are stripped in ``_collect_sync`` for
        non-streaming responses. For streaming, tag echoing is rare and
        cross-chunk stripping would add latency/complexity. If needed, a
        post-processing filter can be added later.
        """
        async with self._agent_lock:
            assert self._agent is not None
            self._agent.messages = list(incoming)

            stream_status = "ok"
            try:
                events = self._agent.astep_stream(max_iterations=10, **(overrides or {}))
                if self._metrics_collector is not None:
                    events = self._metrics_collector.observe(
                        events, model=model_name,
                    )
                if collector:
                    events = collector.observe(events)
                async for chunk in stream_events_as_sse(events, model_name):
                    yield chunk
            except Exception:
                logger.exception("Stream errored")
                stream_status = "error"

            # Tracing: finalize after streaming completes.
            if collector:
                await collector.end_request()

            # Metrics: record request end.
            if self._metrics_collector and metrics_start is not None:
                self._metrics_collector.record_request_end(
                    model_name, True, stream_status, metrics_start,
                )

            # Session: save after streaming completes.
            if session_id and self._session_store:
                await self._session_store.save(session_id, self._agent.messages)

    # -- Run -----------------------------------------------------------------

    def run(self, *, host: str = "0.0.0.0", port: int = 8080, **uvicorn_kwargs) -> None:
        """Start the server with uvicorn.

        Requires the ``[server]`` extra (uvicorn is included).

        Args:
            host: Bind address.
            port: Bind port.
            **uvicorn_kwargs: Additional keyword arguments forwarded to
                ``uvicorn.run``.
        """
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError(
                "fipsagents.server requires the [server] extra. "
                "Install with: pip install 'fipsagents[server]'"
            ) from exc

        uvicorn.run(self.app, host=host, port=port, **uvicorn_kwargs)
