#!/bin/bash
# Set up the shared-memory demo environment.
# Run from demos/shared-memory/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run the demo:"
echo "  1. Start sandbox:        cd ../../sandbox && source .venv/bin/activate && uvicorn sandbox.app:app --port 8000"
echo "  2. Start Problem Solver: cd problem-solver && source ../.venv/bin/activate && uvicorn serve:app --port 8001"
echo "  3. Start Code Writer:    cd code-writer && source ../.venv/bin/activate && uvicorn serve:app --port 8002"
echo ""
echo "Then drive conversations via:"
echo "  curl -X POST http://localhost:8001/chat -H 'Content-Type: application/json' -d '{\"message\": \"Hello\"}'"
