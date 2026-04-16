"""Opt-in FastAPI server for OpenAI-compatible chat completions.

Requires the ``fipsagents[server]`` extra (FastAPI + uvicorn).

Example usage::

    from fipsagents.server import OpenAIChatServer
    from myagent import MyAgent

    server = OpenAIChatServer(MyAgent, config_path="agent.yaml")

    if __name__ == "__main__":
        server.run()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover — helpful error path
    raise ImportError(
        "fipsagents.server requires the [server] extra. "
        "Install with: pip install 'fipsagents[server]'"
    ) from exc

from fipsagents.baseagent import BaseAgent
from fipsagents.baseagent.events import ContentDelta, StreamComplete, StreamMetrics
from fipsagents.serialization.openai_sse import stream_events_as_sse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schema
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert incoming Pydantic messages back to OpenAI-shaped dicts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _sync_response(
    model_name: str,
    content: str,
    *,
    metrics: StreamMetrics | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    m = metrics or StreamMetrics()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens,
        },
        "stream_metrics": {
            "time_to_first_reasoning": m.time_to_first_reasoning,
            "time_to_first_content": m.time_to_first_content,
            "total_time": m.total_time,
            "inter_token_latencies": m.inter_token_latencies,
            "model_calls": m.model_calls,
            "tool_calls": m.tool_calls,
        },
    }


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
        logger.info("OpenAIChatServer: %s ready", self._agent_class.__name__)
        try:
            yield
        finally:
            await self._agent.shutdown()
            self._agent = None

    # -- Route registration --------------------------------------------------

    def _register_routes(self) -> None:
        self.app.get("/healthz")(self._healthz)
        self.app.get("/readyz")(self._readyz)
        self.app.post("/v1/chat/completions")(self._chat_completions)

    # -- Endpoint handlers ---------------------------------------------------

    async def _healthz(self) -> dict[str, str]:
        return {"status": "ok"}

    async def _readyz(self):
        if self._agent is None:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return {"status": "ready"}

    async def _chat_completions(self, req: ChatCompletionRequest):
        if self._agent is None:
            raise HTTPException(status_code=503, detail="Agent not ready")

        agent = self._agent
        model_name = req.model or agent.config.model.name
        incoming = _messages_to_dicts(req.messages)

        if not req.stream:
            content, metrics, finish_reason = await self._collect_sync(
                agent, incoming
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
            self._stream(incoming, model_name),
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
    ) -> tuple[str, StreamMetrics | None, str]:
        """Drive ``astep_stream`` for a non-streaming response.

        Fully drains the iterator so any post-``StreamComplete`` hooks
        in the subclass (e.g. memory writes) run to completion.
        """
        parts: list[str] = []
        metrics: StreamMetrics | None = None
        finish_reason = "stop"
        async with self._agent_lock:
            agent.messages = list(incoming)
            async for event in agent.astep_stream(max_iterations=10):
                if isinstance(event, ContentDelta):
                    parts.append(event.content)
                elif isinstance(event, StreamComplete):
                    metrics = event.metrics
                    finish_reason = event.finish_reason
        return "".join(parts), metrics, finish_reason

    # -- Streaming -----------------------------------------------------------

    async def _stream(
        self,
        incoming: list[dict[str, Any]],
        model_name: str,
    ) -> AsyncIterator[str]:
        """Drive the agent's event stream, serialising to OpenAI SSE chunks."""
        async with self._agent_lock:
            assert self._agent is not None
            self._agent.messages = list(incoming)

            try:
                events = self._agent.astep_stream(max_iterations=10)
                async for chunk in stream_events_as_sse(events, model_name):
                    yield chunk
            except Exception:
                logger.exception("Stream errored")
                # stream_events_as_sse already emits an error chunk on
                # exception from the inner iterator; this outer guard handles
                # unexpected errors from setup code before the generator runs.

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
