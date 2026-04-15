"""Run the document analysis workflow example."""

import asyncio
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import main

if __name__ == "__main__":
    asyncio.run(main())
