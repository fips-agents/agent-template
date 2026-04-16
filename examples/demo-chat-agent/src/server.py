"""FastAPI entry point for the Demo Chat Agent.

All HTTP/SSE glue lives in :class:`fipsagents.server.OpenAIChatServer`.
This module exists only to wire the agent class to the server and
expose ``app`` for uvicorn.
"""

from __future__ import annotations

from pathlib import Path

from fipsagents.server import OpenAIChatServer

from src.agent import DemoChatAgent

APP_DIR = Path(__file__).parent.parent

server = OpenAIChatServer(
    DemoChatAgent,
    config_path=APP_DIR / "agent.yaml",
    base_dir=APP_DIR,
    title="Demo Chat Agent",
    version="0.3.0",
)
app = server.app


if __name__ == "__main__":
    server.run()
