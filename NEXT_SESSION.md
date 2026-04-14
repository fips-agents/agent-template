# Next Session: Sandbox Hardening v2 (#26)

## Goal

Harden the code execution sandbox sidecar with kernel-level isolation
(OpenShell) and application-level scanning (DefenseClaw patterns). Depends
on #25 (v1 sidecar) being complete.

Tracking issue: #26

## Scope

### Kernel-level isolation (OpenShell)

- Landlock LSM: filesystem allowlisting (read-only for stdlib, deny else)
- Seccomp BPF: syscall filtering (block ptrace, mount, raw sockets, kernel
  module loading)
- OPA/Rego network proxy: declarative egress policy (default deny)
- Evaluate running the sidecar inside an OpenShell sandbox on OpenShift

### Application-level scanning (DefenseClaw patterns)

- CodeGuard: regex + AST scanning for credentials, dangerous execution
  patterns, unsafe deserialization, SQL injection, weak crypto, path traversal
- Tool call inspection: secret detection, C2 pattern detection, prompt
  injection in tool arguments
- Audit trail: structured logging of all allow/deny decisions
- SIEM forwarding (Splunk HEC, webhook)

### Additional guardrails

- Persistent scratch filesystem between executions (optional, for multi-step
  computations)
- Output validation / data exfiltration detection
- Enforce/observe mode toggle (log-only vs block)

## Research references

- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)
- [Cisco DefenseClaw](https://github.com/cisco-ai-defense/defenseclaw)
- [smolagents secure code execution](https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution)

## Key files

- `sandbox/` -- existing v1 sidecar (FastAPI, UBI)
- `packages/fipsagents/src/fipsagents/baseagent/` -- BaseAgent framework
- `templates/*/agent.yaml` -- sandbox config section

## Prior session context

- Workflow deployment topology shipped (v0.3.0): RemoteNode, NodeConfig,
  runner auto-wrap, brownfield integration guide. Issue #24.
- Pluggable memory backends shipped (v0.2.0): memoryhub, sqlite, pgvector.
  Issues #27-31 closed.
- Full test suite (333 tests) in place.
