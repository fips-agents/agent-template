"""HTTP server for the Problem Solver agent."""

import json
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agent import ProblemSolver, PROJECT_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Problem Solver Agent")
agent: ProblemSolver | None = None


@app.on_event("startup")
async def startup():
    global agent
    agent = ProblemSolver()
    await agent.setup()
    logger.info("Problem Solver agent ready")


@app.on_event("shutdown")
async def shutdown():
    if agent:
        await agent.shutdown()


class ChatRequest(BaseModel):
    message: str


class NewSessionRequest(BaseModel):
    topic: str = ""


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "agent": "problem-solver"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not agent:
        return JSONResponse(status_code=503, content={"error": "Agent not ready"})

    agent.add_message("user", req.message)
    result = await agent.step()

    try:
        data = json.loads(result.result) if result.result else {}
    except (json.JSONDecodeError, TypeError):
        data = {"response": result.result or "", "memories_written": []}

    return data


@app.post("/new-session")
async def new_session(req: NewSessionRequest):
    """Start a new conversation session. Clears messages and loads relevant memories."""
    if not agent:
        return JSONResponse(status_code=503, content={"error": "Agent not ready"})

    agent.clear_messages()
    loaded_memories = []

    if agent.memory and req.topic:
        memories = await agent.memory.search(
            query=req.topic,
            owner_id="",  # search across all owners for shared memories
            max_results=5,
        )
        for mem in memories:
            content = mem.get("content", "")
            if content:
                loaded_memories.append(content)

    return {
        "status": "new_session",
        "topic": req.topic,
        "memories_loaded": loaded_memories,
    }
