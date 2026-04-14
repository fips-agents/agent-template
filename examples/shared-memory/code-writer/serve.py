"""HTTP server for the Code Writer agent."""

import json
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agent import CodeWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Code Writer Agent")
agent: CodeWriter | None = None


@app.on_event("startup")
async def startup():
    global agent
    agent = CodeWriter()
    await agent.setup()
    logger.info("Code Writer agent ready")


@app.on_event("shutdown")
async def shutdown():
    if agent:
        await agent.shutdown()


class ChatRequest(BaseModel):
    message: str


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "agent": "code-writer"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not agent:
        return JSONResponse(status_code=503, content={"error": "Agent not ready"})

    # Each request is a fresh generation (stateless workflow pattern)
    agent.clear_messages()
    agent.add_message("user", req.message)
    result = await agent.step()

    try:
        data = json.loads(result.result) if result.result else {}
    except (json.JSONDecodeError, TypeError):
        data = {"response": result.result or "", "code": "", "validation_passed": False}

    return data
