"""HTTP server for the Code Sandbox Agent.

Wraps CodeSandboxAgent with FastAPI endpoints for OpenShift deployment.
The Helm chart expects /healthz and /readyz on port 8080.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agent import CodeSandboxAgent

logger = logging.getLogger(__name__)

_agent: CodeSandboxAgent | None = None
_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    app_dir = Path(__file__).parent
    _agent = CodeSandboxAgent(
        config_path=app_dir / "agent.yaml",
        base_dir=app_dir,
    )
    await _agent.setup()
    logger.info("Code Sandbox Agent ready")
    yield
    await _agent.shutdown()


app = FastAPI(title="Code Sandbox Agent", version="0.4.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if _agent is None:
        return JSONResponse({"status": "not ready"}, status_code=503)
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    async with _lock:
        _agent.clear_messages()
        _agent.add_message("user", req.message)
        result = await _agent.run()
        return ChatResponse(response=str(result))
