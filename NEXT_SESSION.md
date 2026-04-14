# Next Session: Sandbox Hardening v2 (#26)

## Goal

Harden the code execution sandbox sidecar with kernel-level isolation
and application-level scanning. The v1 sidecar (#25, closed) has AST-based
guardrails, process isolation, and timeouts. V2 adds defense-in-depth:
kernel restrictions (Landlock, seccomp) and static analysis beyond basic
AST patterns.

Tracking issue: #26

## Priority 1: Research spike — what works on OpenShift?

This is research-first, not build-first. The kernel-level isolation tools
(Landlock, seccomp) have constraints on OpenShift that need investigation
before any implementation.

### Open questions to resolve

1. **Can Landlock run inside an OpenShift pod?**
   Landlock requires `CAP_SYS_ADMIN` to configure but then restricts the
   calling process. On OpenShift, the restricted SCC drops most capabilities.
   Research: does the restricted-v2 SCC allow Landlock self-restriction?
   If not, does a custom SCC work? Or does Landlock need to be applied at
   container build time?

2. **Seccomp profile deployment on OpenShift.**
   OpenShift supports seccomp profiles via the Security Profiles Operator.
   Research: can we ship a custom seccomp profile with the Helm chart, or
   does it need cluster-admin installation? What's the workflow for
   updating profiles?

3. **Network policy vs OPA/Rego proxy.**
   The v1 sandbox relies on OpenShift NetworkPolicy for zero-egress. Is an
   OPA/Rego proxy layer worthwhile, or does NetworkPolicy already cover the
   use case? The proxy adds observability (logging denied requests) but
   also complexity.

4. **OpenShell viability.**
   NVIDIA OpenShell is relatively new. Research: is it production-ready for
   our use case? Does it run on OpenShift? What does it provide beyond
   Landlock + seccomp that we'd configure ourselves?

### Research deliverable

A research document at `research/sandbox-hardening-v2.md` with findings
for each question, a recommended approach, and a revised scope for
implementation. The implementation priorities below are provisional —
update them based on findings.

## Priority 2: DefenseClaw-style static analysis (CodeGuard)

This is the most implementation-ready area. The v1 AST guardrails are
effective but narrow. DefenseClaw's CodeGuard patterns add:

### What to build

1. Extend `sandbox/guardrails.py` with additional AST visitors for:
   - Credential patterns (regex scan for API keys, tokens, passwords in
     string literals)
   - Unsafe deserialization (`pickle.loads`, `yaml.unsafe_load`, `marshal`)
   - SQL injection patterns (string formatting into SQL-like strings)
   - Weak crypto (`md5`, `sha1` for security-sensitive use)
   - Path traversal (`../` in string literals passed to file operations)

2. New module `sandbox/codeGuard.py` (or extend guardrails.py — decide
   based on size) for non-AST scanning:
   - Regex-based secret detection across code strings
   - Pattern library that can be extended via config

3. Tests for each new pattern in `sandbox/tests/test_guardrails.py`

### Key constraint

The v1 guardrails return ALL violations at once so the LLM can fix them
in one retry. New patterns must follow this convention — no fail-fast.

## Priority 3: Tool call inspection

Add a scanning layer for tool call arguments, not just code execution.
This catches prompt injection and secret exfiltration via tool arguments.

### What to build

1. New module: `packages/fipsagents/src/fipsagents/baseagent/tool_inspector.py`
   - Scans tool call arguments before execution
   - Secret detection (same regex patterns as CodeGuard)
   - C2 pattern detection (URLs to known-bad domains, base64-encoded
     payloads, unusual encoding patterns)
   - Prompt injection heuristics (instruction-like text in data fields)

2. Wire into `BaseAgent.use_tool()` and `ToolRegistry.execute()` as a
   pre-execution hook

3. Configuration in `agent.yaml`:
   ```yaml
   security:
     tool_inspection:
       enabled: true
       mode: enforce  # or "observe" (log-only)
   ```

### Design question

Should tool inspection be a BaseAgent concern or a separate middleware?
Recommendation: BaseAgent concern — it's a cross-cutting security policy
that should be hard to accidentally bypass. But keep the inspector itself
as a standalone module for testability.

## Priority 4: Audit trail and enforce/observe mode

### What to build

1. Structured audit logging for all security decisions:
   - Code validation: allowed/denied, which rules triggered
   - Tool call inspection: allowed/denied, which patterns matched
   - Sandbox execution: started/completed/timed_out/killed

2. Enforce/observe mode toggle per security layer:
   ```yaml
   security:
     mode: enforce  # global default
     guardrails:
       mode: enforce
     tool_inspection:
       mode: observe  # log-only for new rules while tuning
   ```

3. SIEM forwarding is deferred — structured logs to stdout is sufficient
   for now. OpenShift log aggregation handles collection.

## Key files

### Sandbox sidecar (v1 — modify)
- `sandbox/app.py` -- FastAPI app, `/execute` and `/healthz` endpoints
- `sandbox/guardrails.py` -- AST-based code validation (allowlist + blocklist)
- `sandbox/executor.py` -- Subprocess execution with timeouts, output capping
- `sandbox/Containerfile` -- UBI9 Python 3.11 image
- `sandbox/tests/` -- 4 test files covering app, guardrails, executor, integration

### BaseAgent framework (modify for tool inspection)
- `packages/fipsagents/src/fipsagents/baseagent/tools.py` -- ToolRegistry
- `packages/fipsagents/src/fipsagents/baseagent/agent.py` -- BaseAgent.use_tool()
- `packages/fipsagents/src/fipsagents/baseagent/config.py` -- AgentConfig

### Templates (update config)
- `templates/*/agent.yaml` -- add security config section

## Existing security boundaries (v1)

Understanding what's already in place avoids duplicating effort:

- **Import allowlist**: 18 modules (math, json, csv, etc.). Everything else blocked.
- **Blocked calls**: eval, exec, compile, __import__, open, getattr/setattr/delattr, breakpoint, input
- **Blocked modules**: subprocess, socket, importlib (attribute access blocked)
- **Blocked dunders**: __subclasses__, __globals__, __builtins__
- **Process isolation**: `python3 -I` flag, temp file cleanup
- **Output capping**: 50 KB max per stream
- **Timeout**: configurable, max 30s, process killed on expiry

## Prior session context

- Workflow deployment topology shipped (v0.3.0): RemoteNode, NodeConfig,
  runner auto-wrap, brownfield integration guide. Issues #24, #32 closed.
- Pluggable memory backends shipped (v0.2.0): memoryhub, sqlite, pgvector.
  Issues #27-31 closed.
- Full test suite: 337 tests passing across fipsagents package.
- Sandbox v1 (#25, closed): FastAPI sidecar with AST guardrails, process
  isolation, timeouts. Comprehensive test suite in sandbox/tests/.
