"""LLM Adapter — FastAPI application.

Entry point for the sidecar that translates OpenAI-compatible requests to
provider-native APIs.  Run with ``python -m llm_adapter.app``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from llm_adapter.config import AdapterConfig
from llm_adapter.models import ChatCompletionRequest
from llm_adapter.providers import get_provider
from llm_adapter.providers.base import BaseProvider

logger = logging.getLogger("llm_adapter")

# Module-level provider instance, set during lifespan.
_provider: BaseProvider | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _provider
    config = AdapterConfig.from_env()
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _provider = get_provider(config.provider)
    await _provider.setup()
    logger.info("LLM Adapter ready (provider=%s, port=%d)", config.provider, config.port)
    yield
    await _provider.shutdown()
    _provider = None


app = FastAPI(title="LLM Adapter", version="0.1.0", lifespan=_lifespan)


@app.get("/healthz")
@app.head("/healthz")
async def healthz():
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Translate and forward a chat completion request."""
    if _provider is None:
        raise HTTPException(status_code=503, detail="Adapter not ready")

    if request.stream:
        return StreamingResponse(
            _provider.chat_completion_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = await _provider.chat_completion(request)
    return JSONResponse(content=response.model_dump())


def main() -> None:
    """CLI entry point."""
    import uvicorn

    config = AdapterConfig.from_env()
    uvicorn.run(app, host="0.0.0.0", port=config.port)


if __name__ == "__main__":
    main()
