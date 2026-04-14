# Next Session: Code Execution Sandbox Sidecar (v1)

## What to build

Issue #25 — a UBI-based sidecar container that executes LLM-generated Python code safely, plus a `code_executor` tool for agents/workflows.

## Architecture

```
┌─────────────────────┐     HTTP POST      ┌──────────────────────┐
│  Agent / Workflow    │  ──────────────>   │  Sandbox Sidecar     │
│                      │                    │  (FastAPI, UBI)      │
│  code_executor tool  │  <──────────────   │                      │
│  @tool(llm_only)     │   stdout/stderr    │  - import allowlist  │
│                      │                    │  - AST scan          │
└─────────────────────┘                    │  - timeout           │
                                           │  - no network egress │
                                           └──────────────────────┘
```

## Components to create

1. **Sidecar service** (`sandbox/`)
   - `app.py` — FastAPI with `POST /execute` endpoint
   - `guardrails.py` — import allowlist + AST pattern scanner
   - `executor.py` — subprocess-based code runner with timeout
   - `Containerfile` — Red Hat UBI, minimal deps (fastapi, uvicorn)
   - Tests

2. **Agent tool** (`tools/code_executor.py`)
   - `@tool(visibility="llm_only")` that POSTs to the sidecar
   - Configurable sidecar URL via `agent.yaml`

3. **Helm chart update**
   - Sidecar container in the Deployment spec
   - NetworkPolicy: sidecar has zero egress

4. **Example workflow**
   - A node that asks the LLM to solve a problem by writing code
   - Demonstrates the code_executor tool in a workflow context

## Pre-execution guardrails (v1)

Import allowlist (safe modules only):
```python
ALLOWED_IMPORTS = {
    "math", "statistics", "itertools", "functools",
    "re", "datetime", "collections", "json", "csv",
    "string", "textwrap", "decimal", "fractions",
    "random", "operator", "typing",
}
```

AST patterns to block:
- `eval()`, `exec()`, `compile()`
- `subprocess.*`, `os.system`, `os.popen`
- `__import__()`, `importlib`
- `open()` with write mode
- `socket.*`
- Attribute access to `__subclasses__`, `__globals__`, `__builtins__`

## Runtime guardrails (v1)

- Timeout: configurable, default 10s
- Memory: pod resource limit (e.g., 256Mi)
- CPU: pod resource limit (e.g., 500m)
- Network: OpenShift NetworkPolicy denying all egress from the sidecar

## Granite endpoint

The Granite 3.3 8b endpoint is available with tool calling enabled:
- URL: `https://granite-3-3-8b-instruct-granite-model.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com/v1`
- Model: `openai/RedHatAI/granite-3.3-8b-instruct`
- Tool calling: enabled (`--enable-auto-tool-choice --tool-call-parser granite`)

## What was completed this session

- Designed and implemented the workflow template (`templates/workflow/`)
- Extracted BaseAgent into shared `fipsagents` package, published to PyPI
- Added `fips-agents create workflow` to the CLI
- Smoke tested all 6 workflow patterns against real Granite endpoint (6/6 pass)
- Identified MemoryHub SDK drift (api_key → OAuth2 client credentials) — team is fixing
- Researched NVIDIA OpenShell, Cisco DefenseClaw, smolagents sandbox patterns
- Filed #25 (v1 sandbox) and #26 (v2 hardening)

## Key constraints

- Red Hat UBI base images
- FIPS-compatible (no cloud sandbox services like E2B)
- The sidecar is a tool, not a framework feature — agents opt in by adding the tool
- litellm model names need `openai/` prefix for custom OpenAI-compatible endpoints
- Template default in `agent.yaml` should document this
