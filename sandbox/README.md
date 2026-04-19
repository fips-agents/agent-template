# Code Execution Sandbox

A FastAPI sidecar that validates and executes LLM-generated Python code under
layered isolation: static analysis, an isolated subprocess, Landlock filesystem
restrictions, plus cluster-level NetworkPolicy and Seccomp when deployed on
OpenShift.

Designed to run under OpenShift's `restricted-v2` Security Context Constraint
(no root, no `SYS_ADMIN`, all capabilities dropped) and on FIPS-enabled
clusters.

## HTTP API

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Liveness/readiness probe |
| `GET /profile` | Active profile introspection (allowed imports, blocklist size, scan stages) |
| `POST /execute` | Validate and run Python code; returns stdout/stderr/exit_code or violations |

The `/execute` request shape is `{"code": "...", "timeout": 10.0}`. On a
validation failure the response is `400` with `{"error": "...", "violations": [...]}`;
on success it is `200` with captured output.

## File Map

| File | Responsibility |
|------|----------------|
| `app.py` | FastAPI wiring. Applies Landlock at import time, loads active profile, exposes the three endpoints. |
| `pipeline.py` | Orchestrates pre-execution scans → execute → post-execution review. Short-circuits on the first violation. |
| `guardrails.py` | AST-based `validate_code()` (import allowlist, blocked calls) and regex-based `blocklist_audit()` (credentials, SQL injection, weak crypto, path traversal). |
| `executor.py` | Subprocess runner: `python3 -I`, temp file in `/tmp`, wall-clock timeout, 50 KB per-stream output cap. |
| `landlock.py` | `ctypes` wrapper for Landlock LSM syscalls. Degrades gracefully when the kernel ABI is unavailable. |
| `profiles.py` + `profiles/*.yaml` | Profile schema and bundled profiles (`minimal`, `data-science`). A profile chooses allowed imports, blocklist entries, and which scan stages run. |
| `Containerfile` | Red Hat UBI 9 image, non-root user, read-only root filesystem. |

## Layered Isolation

1. **Static analysis** (`guardrails.py`) — AST scan blocks `eval`, `exec`,
   `os.system`, `subprocess.*`, socket operations, and non-allowlisted imports.
   Regex scan catches secrets, unsafe deserialization, SQL string formatting,
   weak crypto, and path traversal.
2. **Isolated subprocess** (`executor.py`) — `python3 -I` disables user
   site-packages and `PYTHON*` environment variable pickup. Wall-clock
   timeout and output capping bound resource use.
3. **Landlock LSM** (`landlock.py`) — Filesystem read-only except `/tmp`
   (read-write). Applied at FastAPI startup so subprocesses inherit the
   restriction. No capabilities required — works via `no_new_privs`, which
   `restricted-v2` sets automatically.
4. **Network isolation** (cluster level) — NetworkPolicy with `egress: []`
   enforced at the OVN-Kubernetes hypervisor layer. A process inside the
   container cannot bypass it.
5. **Container hardening** (cluster level) — `runAsNonRoot: true`,
   `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`,
   `capabilities.drop: ALL`, custom Seccomp profile shipped via the
   Security Profiles Operator.

## Local Development

From this directory:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run tests
pytest

# Run the service
uvicorn sandbox.app:app --reload
```

The test suite covers guardrails, profiles, the executor, the pipeline, the
Landlock wrapper (skipped on non-Linux), and end-to-end integration through
the FastAPI app.

## Container Build

```bash
# Local x86_64 build (Mac users add --platform linux/amd64)
podman build -t sandbox:latest -f Containerfile .
```

On OpenShift, prefer a BuildConfig over local builds — it runs natively on
x86_64 in-cluster and pushes directly to the internal registry.

## Further Reading

The rationale for each layer, the alternatives evaluated, and the FIPS-cluster
test results are documented in `sandbox/docs/`:

- [`sandbox-alternatives-evaluation.md`](docs/sandbox-alternatives-evaluation.md) — Why NVIDIA OpenShell and Cisco DefenseClaw weren't adopted, what patterns we borrowed from each.
- [`sandbox-hardening-v2.md`](docs/sandbox-hardening-v2.md) — Full hardening research including FIPS-mode test results.
- [`landlock-openshift-feasibility.md`](docs/landlock-openshift-feasibility.md) — Why Landlock works under `restricted-v2` SCC without elevated privileges.
- [`sandbox-egress-networkpolicy-vs-opa.md`](docs/sandbox-egress-networkpolicy-vs-opa.md) — Why NetworkPolicy is sufficient for zero-egress instead of an OPA/Rego proxy.
