# Code Execution Sandbox: Alternatives Evaluation

Date: 2026-04-15

## Context

The agent-template project needed a code execution sandbox pattern — a way for agents to execute LLM-generated Python code safely. The sandbox runs as a sidecar container on OpenShift, subject to the `restricted-v2` Security Context Constraint. Our requirements:

1. Must run under `restricted-v2` SCC (no root, no SYS_ADMIN, all capabilities dropped, allowPrivilegeEscalation: false)
2. Must work on FIPS-enabled OpenShift clusters
3. Zero network egress from the sandbox container
4. Static analysis of generated code before execution
5. Immutable container images (all code, tools, and config baked in)
6. Minimal operational surface — no additional operators or control planes beyond what OpenShift provides

## Alternatives Evaluated

### NVIDIA OpenShell

OpenShell is NVIDIA's agent execution runtime. It provides Landlock LSM-based filesystem restrictions, seccomp profiles, network namespace isolation, and an embedded OPA/Rego policy proxy. It's a comprehensive approach to sandboxing agent-generated code.

**Why it wasn't the right fit for our use case:**

- OpenShell's architecture uses a privileged supervisor process that requires `SYS_ADMIN`, `NET_ADMIN`, `SYS_PTRACE`, and `SYSLOG` capabilities, and runs as root. OpenShift's `restricted-v2` SCC drops all capabilities and enforces `runAsNonRoot: true`, making this architecture incompatible without a custom SCC that would weaken the cluster's security posture.
- The control plane runs as a K3s cluster inside a container. OpenShift does not support nested container runtimes under its standard security model.
- At the time of evaluation (April 2026, v0.0.29), OpenShell was in active early development with frequent breaking changes, which made it a moving target for production integration.

**What we learned from it:**

OpenShell's design validated several patterns we adopted independently:

1. Two-phase Landlock application: apply filesystem rules first, then call `landlock_restrict_self()`. We adopted this ordering for our deferred Landlock wrapper.
2. Seccomp socket domain denylist: blocking `AF_NETLINK`, `AF_PACKET`, `AF_BLUETOOTH`, `AF_VSOCK` in addition to standard syscall filtering.
3. OCSF-structured audit logging format for security events.

### Cisco DefenseClaw

DefenseClaw's CodeGuard component provides regex-based static analysis of code before execution, scanning across 10 rule categories including credential detection, dangerous exec patterns, unsafe deserialization, SQL injection, and weak cryptography.

**Why it wasn't the right fit for our use case:**

- DefenseClaw is designed as a standalone security layer with its own deployment model. Our sandbox needed static analysis integrated directly into the execution pipeline — guardrails run in the same process as the executor, with results returned to the agent in the same response.
- The regex-based scanning approach is sound, but we needed tighter coupling with our AST-based blocked-call detection (which was already in place for v1 guardrails). Adopting DefenseClaw would have meant maintaining two parallel static analysis systems.
- Our deployment model (immutable container image with everything baked in) favored a small, focused guardrails module over an external dependency with its own release cycle.

**What we learned from it:**

DefenseClaw's CodeGuard rule taxonomy was directly useful for gap analysis. Comparing their 10 rule categories against our v1 guardrails identified 6 gaps we subsequently closed:

| Gap | DefenseClaw Rule | Resolution |
|-----|-----------------|------------|
| Hardcoded API keys and generic secrets | CG-CRED-001 | Added regex patterns for generic API keys and high-entropy strings |
| AWS access key IDs | CG-CRED-002 | Added `AKIA`-prefix pattern detection |
| Embedded private keys | CG-CRED-003 | Added PEM block header/footer detection |
| Unsafe deserialization | CG-DESER-001 | Extended blocked calls to cover `pickle.loads`, `yaml.unsafe_load`, `marshal` |
| SQL injection | CG-SQL-001 | Added detection of string formatting in SQL-like statements |
| Weak cryptography | CG-CRYPTO-001 | Added `md5`/`sha1` detection (also validated on FIPS clusters) |
| Path traversal | CG-PATH-001 | Added `../` detection in file operation arguments |

## What We Built

Rather than adopting either project as a dependency, we built a layered sandbox from components that each operate within OpenShift's security model:

### Layer 1: Static Analysis (application level)

- AST-based blocked-call detection (v1, already existed): blocks `eval()`, `exec()`, `os.system()`, `subprocess.*`, `importlib.*`, socket operations
- Regex-based CodeGuard patterns (v2, informed by DefenseClaw gap analysis): credential detection, unsafe deserialization, SQL injection, weak crypto, path traversal
- All violations collected in a single pass before code reaches the executor

### Layer 2: Isolated Execution (application level)

- `python3 -I` subprocess: isolated mode disables user site-packages, ignores `PYTHON*` environment variables
- Temp file in `/tmp`, cleaned up unconditionally
- Wall-clock timeout with process kill
- Output capped at 50 KB per stream

### Layer 3: Network Isolation (cluster level)

- NetworkPolicy with `egress: []` — blocks all outbound traffic at the OVN-Kubernetes hypervisor level
- This enforcement is outside the pod's trust boundary; a process inside the container cannot bypass it
- DNS (port 53) is also blocked, which is correct for a fully isolated sandbox

### Layer 4: Container Hardening (cluster level)

- SecurityContext: `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`, `capabilities.drop: ALL`
- Seccomp profile shipped as `SeccompProfile` CRD via Security Profiles Operator
- Custom allowlist: only the syscalls the Python subprocess legitimately needs

### Layer 5: Tool Call Inspection (BaseAgent level)

- `ToolInspector` scans tool call arguments for secrets, C2 patterns, and prompt injection before execution
- Same regex patterns as CodeGuard layer, applied to tool arguments rather than generated code
- Configurable enforce/observe mode per layer

### Future: Landlock LSM (deferred to OCP 4.18+)

- Python entrypoint wrapper using `ctypes` to call Landlock syscalls directly
- Filesystem read-only for stdlib, read-write for `/tmp`, deny everything else
- TCP port filtering via ABI 4+
- Works under `restricted-v2` SCC via `no_new_privs` path (no capabilities needed)
- Gated on RHEL 9.6+ kernel availability (Landlock enabled by default in that release)

## Decision Rationale

The driving constraint was OpenShift's `restricted-v2` SCC. Any solution requiring elevated privileges, root access, or nested container runtimes was architecturally incompatible with our deployment target. Both OpenShell and DefenseClaw are designed for environments with more flexibility in security policy — they solve a broader problem than ours.

By building from composable, OpenShift-native primitives (NetworkPolicy, SecurityContext, SeccompProfile, Landlock), each layer operates at its own trust boundary and can be independently verified. The static analysis layer borrows directly from DefenseClaw's rule taxonomy, and the Landlock design borrows from OpenShell's two-phase pattern — both projects contributed meaningfully to the final design even though neither was adopted as a runtime dependency.

## References

- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)
- [Cisco DefenseClaw](https://github.com/cisco-ai-defense/defenseclaw)
- [HuggingFace smolagents secure execution](https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution)
- `research/sandbox-hardening-v2.md` — Full hardening research
- `research/landlock-openshift-feasibility.md` — Landlock feasibility study
- `research/sandbox-egress-networkpolicy-vs-opa.md` — NetworkPolicy vs OPA analysis
