#!/usr/bin/env python3
"""Interactive REPL for the Code Sandbox Agent.

Start the sandbox sidecar first:
    cd sandbox && uvicorn sandbox.app:app --port 8000

Then run this:
    cd examples/code-sandbox-agent && python run.py
"""

import asyncio
import sys
from pathlib import Path

from src.agent import CodeSandboxAgent


async def main() -> None:
    agent = CodeSandboxAgent(
        config_path=Path("agent.yaml"),
        base_dir=Path("."),
    )
    await agent.setup()

    print("Code Sandbox Agent ready. Ask a question (Ctrl-D to quit).\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                break

            if not question:
                continue

            agent.clear_messages()
            agent.add_message("user", question)

            result = await agent.run()
            print(f"\nAgent: {result}\n")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
