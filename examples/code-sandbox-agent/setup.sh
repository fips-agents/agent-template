#!/usr/bin/env bash
# Bootstrap the code-sandbox-agent example.
# Run from the examples/code-sandbox-agent/ directory.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

cat <<'DONE'

Setup complete.  Run the demo in two terminals:

  Terminal 1 — Start the sandbox sidecar:
    cd sandbox
    ../.venv/bin/uvicorn sandbox.app:app --port 8000

  Terminal 2 — Start the agent REPL:
    cd examples/code-sandbox-agent
    .venv/bin/python run.py

  Environment variables:
    MODEL_ENDPOINT  — LLM API endpoint (default: http://localhost:8321/v1)
    MODEL_NAME      — Model identifier (default: openai/RedHatAI/granite-3.3-8b-instruct)
    SANDBOX_URL     — Sandbox sidecar URL (default: http://localhost:8000)

  Try asking:
    "What is the standard deviation of [23, 45, 12, 67, 34, 89, 56]?"
    "Find all prime numbers between 100 and 200"
    "I have a 20x30 foot yard. How many pallets of 1x2 foot sod do I need
     if each pallet has 80 pieces, and how much will it cost at $450/pallet?"

DONE
