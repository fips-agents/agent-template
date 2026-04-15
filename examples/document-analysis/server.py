"""HTTP server for the Document Analysis workflow.

Wraps the workflow with FastAPI endpoints for OpenShift deployment.
The Helm chart expects /healthz and /readyz on port 8080.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agent import DocumentState, WorkflowRunner, build_graph

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent

app = FastAPI(title="Document Analysis Agent", version="0.1.0")


class AnalyzeRequest(BaseModel):
    document: str


class AnalyzeResponse(BaseModel):
    document_type: str
    report: str
    validation_errors: list[str] = []


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    return {"status": "ready"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Run the document analysis workflow on the submitted document."""
    graph = build_graph()
    runner = WorkflowRunner(graph, max_steps=10)
    result = await runner.start(DocumentState(document=req.document))
    return AnalyzeResponse(
        document_type=result.document_type,
        report=result.report,
        validation_errors=result.validation_errors,
    )


# Keep /chat as an alias for compatibility with the standard agent pattern.
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Compatibility endpoint — treats the message as a document to analyze."""
    graph = build_graph()
    runner = WorkflowRunner(graph, max_steps=10)
    result = await runner.start(DocumentState(document=req.message))
    return ChatResponse(response=result.report)
