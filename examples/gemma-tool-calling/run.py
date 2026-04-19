#!/usr/bin/env python3
"""Interactive REPL for the Gemma 4 Tool Calling example.

Usage:
    cd examples/gemma-tool-calling && python run.py
"""

import asyncio
from pathlib import Path

from src.agent import GemmaToolAgent


async def main() -> None:
    agent = GemmaToolAgent(
        config_path=Path("agent.yaml"),
        base_dir=Path("."),
    )
    await agent.setup()

    print("Gemma 4 Tool Calling Agent ready.")
    print("Try: 'What are the latest CDC guidelines on COVID vaccines?'")
    print("Ctrl-D to quit.\n")

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
