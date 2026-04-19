"""FastAPI entry point for the Gemma 4 Tool Calling example.

All HTTP/SSE glue lives in :class:`fipsagents.server.OpenAIChatServer`.
This module exists only to wire the agent class to the server and
expose ``app`` for uvicorn.
"""

from __future__ import annotations

from pathlib import Path

from fipsagents.server import OpenAIChatServer

from src.agent import GemmaToolAgent

APP_DIR = Path(__file__).parent.parent

server = OpenAIChatServer(
    GemmaToolAgent,
    config_path=APP_DIR / "agent.yaml",
    base_dir=APP_DIR,
    title="Gemma 4 Tool Calling",
    version="0.1.0",
)
app = server.app


if __name__ == "__main__":
    server.run()
