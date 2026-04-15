# Code Sandbox Agent

A computational assistant that answers questions by writing and executing
Python code in an isolated sandbox, rather than attempting mental math.

## Why code execution?

LLMs are unreliable at multi-step arithmetic. Given a word problem like:

> I have a yard that is 20 x 30 feet, and I want to buy 1' x 2' sod to
> put grass on it. How many pallets do I need if a pallet has 80 pieces,
> and how much do I need to spend if a pallet is $450?

An LLM doing mental math might get the area right (600 sq ft) but fumble
the division, rounding, or unit conversion. A code execution agent
translates this to Python:

```python
import math

yard_area = 20 * 30           # 600 sq ft
sod_area = 1 * 2              # 2 sq ft per piece
pieces_needed = math.ceil(yard_area / sod_area)  # 300 pieces
pallets_needed = math.ceil(pieces_needed / 80)    # 4 pallets
total_cost = pallets_needed * 450                 # $1,800

print(f"Pieces needed: {pieces_needed}")
print(f"Pallets needed: {pallets_needed}")
print(f"Total cost: ${total_cost:,}")
```

The answer is exact, shows its work, and handles the ceiling division
correctly (you can't buy a partial pallet).

## Architecture

```
┌─────────────────────────────┐
│  User asks a question       │
│  "How many pallets..."      │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│  CodeSandboxAgent           │
│  (BaseAgent subclass)       │
│                             │
│  1. LLM writes Python code  │
│  2. Calls code_executor     │──────────┐
│  3. Gets stdout back        │          │
│  4. LLM interprets results  │          │
│  5. Returns answer to user  │          │
└─────────────────────────────┘          │
                                         │
                              ┌──────────▼──────────┐
                              │  Sandbox Sidecar     │
                              │  (FastAPI, port 8000)│
                              │                      │
                              │  - AST guardrails    │
                              │  - CodeGuard patterns│
                              │  - Landlock (Linux)  │
                              │  - python3 -I        │
                              │  - 30s timeout       │
                              │  - 50 KB output cap  │
                              └──────────────────────┘
```

The sandbox sidecar validates code with AST guardrails (import allowlist,
blocked calls, credential detection, SQL injection patterns) before
executing it in an isolated subprocess. On Linux with RHEL 9.6+ kernels,
Landlock LSM further restricts filesystem access.

## Quick start

```bash
cd examples/code-sandbox-agent
./setup.sh
```

Then in two terminals:

```bash
# Terminal 1: sandbox sidecar
cd sandbox
../examples/code-sandbox-agent/.venv/bin/uvicorn sandbox.app:app --port 8000

# Terminal 2: agent REPL
cd examples/code-sandbox-agent
SANDBOX_URL=http://localhost:8000 .venv/bin/python run.py
```

## Example session

```
You: What is the standard deviation of [23, 45, 12, 67, 34, 89, 56]?

Agent: The standard deviation is approximately 25.28.

  Code executed:
    import statistics
    data = [23, 45, 12, 67, 34, 89, 56]
    print(f"Mean: {statistics.mean(data):.2f}")
    print(f"Std dev: {statistics.stdev(data):.2f}")

You: Find all prime numbers between 100 and 200

Agent: There are 21 primes between 100 and 200:
  101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
  151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ENDPOINT` | `http://localhost:8321/v1` | LLM API endpoint |
| `MODEL_NAME` | `openai/RedHatAI/granite-3.3-8b-instruct` | Model to use |
| `SANDBOX_URL` | `http://localhost:8000` | Sandbox sidecar URL |
| `SECURITY_MODE` | `enforce` | `enforce` blocks suspicious tool args; `observe` logs only |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Deploying to OpenShift

This example deploys using the agent-loop Helm chart with the sandbox
sidecar enabled:

```yaml
# values.yaml override
sandbox:
  enabled: true
  image:
    repository: quay.io/yourorg/code-sandbox
    tag: "0.4.0"
  seccomp:
    enabled: true   # Requires Security Profiles Operator
```

The sandbox runs as a sidecar container in the same pod, accessible at
`localhost:8000`. See the agent-loop chart documentation for full
deployment instructions.

## Security layers

The sandbox enforces defense-in-depth:

1. **AST guardrails** — import allowlist (18 modules), blocked calls
   (eval, exec, open, etc.), blocked dunders
2. **CodeGuard patterns** — credential detection, unsafe deserialization,
   SQL injection, weak crypto, path traversal
3. **Tool call inspection** — scans tool arguments for secrets, C2
   patterns, prompt injection before execution
4. **Process isolation** — `python3 -I` flag, temp file cleanup, 50 KB
   output cap, 30s timeout
5. **Container hardening** — drop ALL capabilities, read-only root
   filesystem, non-root user, emptyDir for /tmp
6. **Seccomp profile** (optional) — blocks networking syscalls, ptrace,
   io_uring, namespace manipulation at the kernel level
7. **Landlock** (Linux, RHEL 9.6+) — filesystem allowlist restricting
   read/write paths at the kernel level
