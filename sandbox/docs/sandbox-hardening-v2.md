# Sandbox Hardening v2: Research Findings

**Date:** 2026-04-15
**Tracking issue:** #26
**Status:** Research complete, implementation priorities revised

## Executive Summary

Four areas were investigated for hardening the v1 code execution sandbox:
Landlock LSM, seccomp profiles, network egress policy, and NVIDIA OpenShell.

**Key findings:**

- **Landlock works on OpenShift** under restricted-v2 SCC with no capability
  changes. Gated on RHEL 9.6+ / OCP 4.18+ (kernel must have Landlock enabled
  by default).
- **Seccomp custom profiles** can ship in the Helm chart as `SeccompProfile`
  CRDs via the Security Profiles Operator (GA since OCP 4.12). Cluster-admin
  must install SPO once; namespace admins deploy profiles thereafter.
- **NetworkPolicy is sufficient** for zero-egress. OPA/Rego proxy is
  overengineered for "deny everything" -- skip it.
- **Do not adopt OpenShell.** Alpha-stage (v0.0.29, 832 open issues), requires
  `SYS_ADMIN` + root (incompatible with OpenShift SCCs). Use as inspiration
  for Landlock ordering and seccomp socket denylist.

## Revised Implementation Priorities

Based on research, the original priorities from NEXT_SESSION.md are adjusted:

| Priority | Item | Status | Notes |
|----------|------|--------|-------|
| 1 | CodeGuard static analysis | **Ready to implement** | No kernel/cluster deps. Highest immediate value. |
| 2 | Tool call inspection | **Ready to implement** | BaseAgent concern, no external deps. |
| 3 | Audit trail + enforce/observe | **Ready to implement** | Structured logging, config changes. |
| 4 | Seccomp profile | **Ship in Helm chart** | SeccompProfile CRD, document SPO prerequisite. |
| 5 | Landlock wrapper | **Deferred** | Gated on OCP 4.18+ adoption. Implement as optional entrypoint wrapper with runtime ABI detection. |
| -- | OPA/Rego proxy | **Dropped** | NetworkPolicy sufficient for zero-egress. |
| -- | OpenShell integration | **Dropped** | Incompatible with OpenShift security model. |

---

## Finding 1: Landlock on OpenShift

### Can Landlock run inside an OpenShift pod?

**Yes**, under restricted-v2 SCC with no modifications.

`landlock_restrict_self()` requires one of:
- `CAP_SYS_ADMIN` in the calling namespace, **OR**
- The `no_new_privs` bit set via `prctl(PR_SET_NO_NEW_PRIVS, 1, ...)`

The `no_new_privs` path requires **zero capabilities**. OpenShift's
restricted-v2 SCC sets `allowPrivilegeEscalation: false`, which causes CRI-O
to set `no_new_privs` automatically. The Landlock syscalls
(`landlock_create_ruleset`, `landlock_add_rule`, `landlock_restrict_self`) are
in CRI-O's default seccomp allowlist.

No custom SCC is needed.

### Kernel requirement

**RHEL 9.6+** (kernel >= 5.14.0-568.el9) is required. Earlier RHEL 9 kernels
have Landlock compiled in but not in the default LSM boot list -- enabling it
requires a MachineConfig kernel parameter change (`lsm=...,landlock`).

RHEL 9.6 ships Landlock enabled by default with ABI 5 backported (filesystem +
TCP port filtering). OCP 4.18+ uses RHCOS based on RHEL 9.6.

### Practical limitations

Landlock restricts:
- Filesystem paths (read, write, execute, create, delete, truncate)
- TCP bind/connect by port number (ABI 4+)

Landlock does **not** restrict:
- UDP, ICMP, raw sockets
- IP addresses or remote hosts (port numbers only)
- Already-open file descriptors (stdin/stdout/stderr)
- `chdir`, `stat`, `chmod`, `chown`, `access`
- Shared memory, process memory
- Maximum 16 stacked rulesets

### Implementation approach (when ready)

Apply Landlock from a Python entrypoint wrapper using `ctypes` to call the
three syscalls. The two-phase pattern from OpenShell is correct: apply
filesystem rules first, then drop privileges. Use `best_effort` compatibility
-- degrade gracefully if the kernel ABI is lower than requested.

```python
# Pseudocode for the wrapper
import ctypes

def apply_landlock():
    abi = landlock_create_ruleset(None, 0, VERSION_FLAG)
    if abi < 1:
        return  # Landlock not available, degrade gracefully

    ruleset_fd = landlock_create_ruleset(attr, size, 0)
    # Add read-only rules for stdlib, /tmp read-write
    landlock_add_rule(ruleset_fd, PATH_BENEATH, rule)
    landlock_restrict_self(ruleset_fd, 0)
```

### Sources

- [Landlock kernel documentation](https://docs.kernel.org/userspace-api/landlock.html)
- [RHEL 9.6 release notes](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/)
- OpenShift restricted-v2 SCC documentation
- CRI-O default seccomp profile (`containers/common/pkg/seccomp/seccomp.json`)
- See also: `research/landlock-openshift-feasibility.md` for full details

---

## Finding 2: Seccomp Profile Deployment

### How OpenShift handles seccomp

The restricted-v2 SCC mandates `RuntimeDefault` as the seccomp profile. For
pods without an explicit `seccompProfile`, the SCC admission controller injects
`RuntimeDefault` automatically. This profile blocks ~44 syscalls including
`ptrace`, `mount`, `init_module`, `io_uring_*`, and namespace manipulation.

### Security Profiles Operator (SPO)

**GA since OCP 4.12.** Installed from OperatorHub (cluster-admin operation).
Once installed, namespace-scoped `SeccompProfile` CRD objects can be created by
namespace admins. SPO's DaemonSet syncs profile JSON to every worker node
within seconds.

### Shipping custom profiles in Helm chart

The Helm chart ships a `SeccompProfile` CRD manifest. SPO must be a documented
prerequisite. The deployment flow:

1. Cluster-admin installs SPO once from OperatorHub
2. Helm chart includes `SeccompProfile` CRD in `templates/`
3. SPO auto-distributes to nodes (no reboots)
4. Pod spec references: `localhostProfile: operator/<ns>/sandbox-python.json`

Without SPO, the fallback is MachineConfig (cluster-admin, triggers rolling
node reboots). Document both paths.

### Custom SCC requirement

restricted-v2 only allows `RuntimeDefault` seccomp type. Using a custom
`Localhost` profile requires either:
- A custom SCC listing `localhost/<profile-name>` in `seccompProfiles`
- Or a wildcard `*` in `seccompProfiles`

This is a cluster-admin operation. Document in the Helm chart prerequisites.

### Recommended syscall profile for sandbox

**Default action:** `SCMP_ACT_ERRNO` (deny by default, allowlist approach)

**Must allow** (Python subprocess needs these):
- Process: `fork`, `clone`, `execve`, `wait4`, `waitid`, `exit`, `exit_group`
- I/O: `read`, `write`, `close`, `openat`, `lseek`, `dup2`, `pipe`, `pipe2`
- Memory: `mmap`, `mprotect`, `munmap`, `brk`
- Filesystem: `stat`, `fstat`, `lstat`, `access`, `readlink`, `getcwd`,
  `getdents64`
- Signals: `rt_sigaction`, `rt_sigprocmask`, `rt_sigreturn`
- Misc: `futex`, `set_tid_address`, `arch_prctl`, `getrandom`,
  `clock_gettime`, `prctl`, `prlimit64`, `uname`, `fcntl`

**Block unconditionally:**
- `ptrace`, `process_vm_readv`, `process_vm_writev` (process injection)
- `init_module`, `finit_module`, `delete_module` (kernel modules)
- `mount`, `umount2`, `pivot_root` (filesystem root)
- `socket`, `bind`, `connect`, `listen`, `accept` (networking -- if zero-net)
- `io_uring_*` (container escape vectors)
- `bpf` (eBPF loading)
- `unshare`, `setns` (namespace manipulation)
- `kexec_load`, `kexec_file_load`, `reboot` (system admin)

**Recommended approach:** Start with `SCMP_ACT_LOG` default action in dev/test
to audit actual syscall usage, then lock down to `SCMP_ACT_ERRNO` allowlist.

### Sources

- [SPO on OCP 4.16 docs](https://docs.redhat.com/en/documentation/openshift_container_platform/4.16/html/security_and_compliance/security-profiles-operator)
- [Seccomp defaults in Red Hat OpenShift](https://www.redhat.com/en/blog/seccomp-defaults-in-red-hat-openshift-container-platform)
- [Kubernetes seccomp reference](https://kubernetes.io/docs/reference/node/seccomp/)

---

## Finding 3: Network Egress Policy

### NetworkPolicy is sufficient for zero-egress

An OpenShift NetworkPolicy with `egress: []` blocks all outbound traffic,
enforced by OVN-Kubernetes at the hypervisor level via OVN ACLs. This
enforcement is **outside the pod's trust boundary** -- a process inside the
container cannot bypass it.

```yaml
spec:
  podSelector:
    matchLabels:
      role: code-executor
  policyTypes:
    - Egress
  egress: []
```

This also blocks DNS (port 53), which is correct for a fully isolated sandbox.

### Why OPA/Rego proxy is not needed

OPA proxy enforces policy **inside** the pod's trust boundary (iptables within
the network namespace). For "deny everything," hypervisor-level enforcement is
strictly stronger.

OPA adds value for **selective** egress (URL filtering, header inspection,
per-service rules). When the answer is uniformly "no," it is redundant
overhead.

The one genuine value OPA adds: audit logging of denied connection attempts.
This can be achieved instead via the **OVN Network Observability operator**
(eBPF-based, no pod changes).

### Recommendation

- Ship NetworkPolicy with `egress: []` in the Helm chart
- Add pod security context: `drop: ALL` capabilities, `readOnlyRootFilesystem`,
  `allowPrivilegeEscalation: false`, `runAsNonRoot: true`
- If audit logging of denied connections is needed, use OVN Network
  Observability (cluster-level, no per-pod overhead)
- Do not build an OPA proxy

### Sources

- OpenShift NetworkPolicy documentation
- OVN-Kubernetes enforcement model
- See also: `research/sandbox-egress-networkpolicy-vs-opa.md` for full analysis

---

## Finding 4: NVIDIA OpenShell

### Assessment: Do not adopt

OpenShell is NVIDIA's agent execution runtime. It uses Landlock, seccomp,
network namespaces, and an embedded OPA/Rego proxy. It is technically
interesting but fundamentally incompatible with OpenShift.

**Blockers:**
- Requires `SYS_ADMIN`, `NET_ADMIN`, `SYS_PTRACE`, `SYSLOG` capabilities
- Runs sandbox supervisor as root (`runAsUser: 0`)
- Control plane is a K3s cluster inside a Docker container (nested container
  runtimes unsupported on OpenShift)
- Alpha-stage: v0.0.29, 832 open issues, daily releases, known security design
  trade-offs (privileged supervisor, issue #579)

**Patterns worth borrowing:**
1. Two-phase Landlock: apply filesystem rules first, then restrict self
2. Seccomp socket domain denylist: block `AF_NETLINK`, `AF_PACKET`,
   `AF_BLUETOOTH`, `AF_VSOCK`
3. OCSF-structured audit logging format

### Cisco DefenseClaw CodeGuard patterns

DefenseClaw's CodeGuard component does regex scanning across 10 rule categories.
Comparing to our v1 guardrails:

| Rule | DefenseClaw | Our v1 | Gap? |
|------|------------|--------|------|
| Hardcoded API keys/secrets | CG-CRED-001 | No | **Yes** |
| AWS access key IDs | CG-CRED-002 | No | **Yes** |
| Embedded private keys | CG-CRED-003 | No | **Yes** |
| Dangerous exec (eval, os.system) | CG-EXEC-001 | Yes (blocked calls) | No |
| subprocess shell=True | CG-EXEC-002 | Yes (subprocess blocked) | No |
| Outbound HTTP to variable URLs | CG-NET-001 | Yes (socket blocked) | No |
| Unsafe deserialization | CG-DESER-001 | Partial (importlib blocked) | **Yes** |
| String-formatted SQL | CG-SQL-001 | No | **Yes** |
| MD5/SHA1 usage | CG-CRYPTO-001 | No | **Yes** |
| Path traversal (../) | CG-PATH-001 | No | **Yes** |

Six gaps to close in CodeGuard implementation.

### HuggingFace smolagents approach

Smolagents uses two tiers: a custom AST interpreter (not CPython) for basic
isolation, and remote execution (E2B/Modal/Docker) for stronger isolation. The
Docker security flags are worth mirroring in our pod spec: `cap_drop: ALL`,
`no-new-privileges`, `mem_limit: 512m`, `pids_limit: 100`.

### If stronger kernel isolation is needed later

**gVisor** is the right path for OpenShift. It provides syscall interception
without `SYS_ADMIN`, works with OpenShift's security model via RuntimeClass,
and is supported through the OpenShift sandboxed containers operator.

### Sources

- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)
- [Cisco DefenseClaw](https://github.com/cisco-ai-defense/defenseclaw)
- [smolagents secure code execution](https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution)

---

## Finding 5: FIPS-Mode Cluster Implications

### How FIPS enforcement works on OpenShift

On a FIPS-enabled RHEL/RHCOS node, the kernel boots with `fips=1`. This
switches OpenSSL into FIPS mode system-wide — every process on the node
inherits it, including all containers. There is no per-pod opt-out. The
enforcement is at the OpenSSL provider level: non-approved algorithms
raise hard errors rather than returning results.

The scope of FIPS enforcement varies by operation type:

- **Hashing**: MD5 is blocked for security use. SHA-1 is *not* blocked
  (allowed for both signing and non-security use).
- **Signing**: SHA-1 RSA signing is blocked — `openssl req -sha1` fails
  with `invalid digest`. SHA-256+ is required for certificate generation.
- **TLS cipher suites**: Only AEAD ciphers (AES-GCM, AES-CCM) with
  128/256-bit keys are available. TLSv1.2 and TLSv1.3 only (21 ciphers
  total). No CBC, ChaCha20, RC4, or DES.
- **TLS certificate verification**: SHA-1 signed certificates from
  external sources *can* be verified — FIPS blocks signing, not
  verification of existing signatures.

### Test results (2026-04-15, OCP 4.20.17, OpenSSL 3.5.1, FIPS-enabled)

Validated on cluster `cluster-l78nk.l78nk.sandbox1834.opentlc.com` with
the sandbox sidecar deployed via Helm chart with SPO seccomp profile.

#### Sandbox behavior

| Test | Expected | Actual | Notes |
|------|----------|--------|-------|
| `python3 -I` inherits FIPS | Yes | **Yes** | `/proc/sys/crypto/fips_enabled` = 1 inside subprocess |
| `import hashlib` | Works | **Works** | Module loads fine |
| `hashlib.md5(b"x")` | ValueError | **UnsupportedDigestmodError** | `[digital envelope routines] unsupported` |
| `hashlib.new("md5", b"x")` | ValueError | **ValueError** | `unsupported hash type md5(in FIPS mode)` |
| `hashlib.md5(b"x", usedforsecurity=False)` | Works | **Works** | Returns correct digest |
| `hashlib.sha1(b"x")` | ValueError | **Works** | SHA-1 is NOT blocked by FIPS for hashing |
| `hashlib.sha256(b"x")` | Works | **Works** | FIPS-approved |
| Guardrails fire before FIPS | Yes | **Yes** | Guardrails return `"call to 'hashlib.md5()' uses weak cryptography"` before code reaches executor |

Key corrections from pre-test hypothesis:

1. **SHA-1 hashing is allowed in FIPS mode.** Only MD5 is blocked. Our
   guardrails are stricter than FIPS here (flagging SHA-1 as weak crypto),
   which is the correct security posture.

2. **Two different error types for MD5.** `hashlib.md5()` raises
   `UnsupportedDigestmodError` while `hashlib.new("md5")` raises
   `ValueError`. Both include enough context for an LLM to self-correct.

3. **Guardrails provide better errors than FIPS.** The guardrail message
   `"call to 'hashlib.md5()' uses weak cryptography"` is actionable. The
   FIPS error `[digital envelope routines] unsupported` is not.

#### Agent-to-service TLS connectivity

| Test | Expected | Actual | Notes |
|------|----------|--------|-------|
| Generate SHA-1 cert on FIPS cluster | Fail | **Fail** | `invalid digest` — cannot create SHA-1 signed certs |
| Generate SHA-256 cert on FIPS cluster | Works | **Works** | `sha256WithRSAEncryption` |
| Connect to SHA-1 server, `verify=False` | Fail | **Works** | FIPS does NOT block SHA-1 cert verification |
| Connect to SHA-1 server, `verify=<cert>` | Fail | **Works** | SHA-1 signature verification succeeds |
| Connect to SHA-256 server, `verify=False` | Works | **Works** | Expected |
| Connect to SHA-256 server, `verify=<cert>` | Works | **Works** | Expected |
| Available TLS cipher count | — | **21** | All AEAD (AES-GCM/CCM), TLSv1.2/1.3 only |
| SHA-1 digest ciphers | — | **0** | No cipher suites use SHA-1 as HMAC digest |

Key corrections from pre-test hypothesis:

1. **SHA-1 signed certificates work for TLS connections in FIPS mode.**
   The original hypothesis that `verify=False` cannot bypass FIPS rejection
   for SHA-1 certs was incorrect. FIPS blocks *creating* SHA-1 signatures
   but allows *verifying* existing ones. This is a meaningful distinction:
   agents on FIPS clusters CAN connect to legacy endpoints with SHA-1 certs.

2. **The real TLS constraint is cipher suites, not cert signatures.** Only
   21 AEAD ciphers are available. Endpoints that require CBC-mode ciphers,
   RC4, or other non-AEAD suites will fail to negotiate. This is the actual
   connectivity risk for legacy endpoints.

3. **Error surfacing is clear.** TLS failures produce standard Python
   `ssl.SSLError` or `httpx.ConnectError` with descriptive OpenSSL messages.
   No silent hangs observed — errors propagate immediately.

### Impact on the sandbox sidecar

The sandbox runs Python code in a subprocess on a UBI 9 image. On a FIPS
cluster:

1. **hashlib**: Only `hashlib.md5()` is blocked by FIPS (without
   `usedforsecurity=False`). `hashlib.sha1()` works. However, our
   guardrails catch both `md5()` and `sha1()` before execution, giving
   the LLM clear actionable errors regardless of FIPS mode. This is the
   desired behavior — guardrails are stricter than FIPS.

2. **Allowed modules**: The sandbox's 18 allowed modules (`math`, `json`,
   `csv`, `re`, `datetime`, etc.) do not use OpenSSL. No FIPS impact.

3. **random**: Uses Mersenne Twister, not OpenSSL. No FIPS impact.

4. **Error clarity**: FIPS errors are adequate (`UnsupportedDigestmodError`
   with traceback pointing to exact line), but our guardrails provide
   better messages. No FIPS-specific guardrail needed — the existing
   weak crypto guardrails already cover the relevant cases.

### Impact on agents and MCP servers

Less severe than originally hypothesized. Agents on FIPS clusters:

- **Self-signed SHA-1 certs**: Work for connectivity with `verify=False`
  or when explicitly trusted. No immediate re-issue required, though
  SHA-256+ remains the recommended long-term posture.
- **Self-signed SHA-256 certs**: Work fine (expected).
- **TLS cipher negotiation**: The real constraint. Only 21 AEAD ciphers
  available. Endpoints requiring legacy cipher suites (CBC, RC4) cannot
  negotiate a connection. This failure presents as a clear TLS handshake
  error, not a silent hang.
- **MCP servers proxying to legacy APIs**: Only a concern if the legacy
  endpoint requires non-AEAD cipher suites. SHA-1 certs alone are not
  a blocker.

### What we do today

- **Weak crypto guardrails** catch `hashlib.md5()` and `hashlib.sha1()` at
  the application level, providing clear error messages before FIPS-mode
  OpenSSL could reject them. Guardrails are stricter than FIPS (blocking
  SHA-1 which FIPS allows), which is the right security posture.
- **UBI base images** ship FIPS-aware OpenSSL (3.5.1 on OCP 4.20). No
  additional configuration needed — they respect the host kernel's FIPS
  mode automatically.
- **Seccomp and Landlock** are kernel mechanisms, not crypto. They work
  identically on FIPS and non-FIPS clusters.

### Deployment notes

- **Landlock**: Not available on the tested FIPS cluster (kernel
  5.14.0-570.99.1.el9_6, LSM not in boot list). Landlock availability
  depends on kernel config, not FIPS mode. Seccomp profile provides the
  primary syscall restriction.
- **SPO**: Security Profiles Operator v0.10.0 works correctly on FIPS
  clusters. SeccompProfile CRD installs and the profile is applied to
  the sandbox container.
- **No FIPS-specific configuration needed** for the sandbox sidecar or
  agent. The UBI base image handles FIPS transparency.

### Recommendations (updated post-testing)

- **Document FIPS cipher suite constraints** in agent deployment guide.
  The 21-cipher AEAD-only suite is the primary operational constraint,
  not certificate signature algorithms.
- **No FIPS-specific guardrail needed.** Existing weak crypto guardrails
  cover MD5 and SHA-1 adequately. Adding `usedforsecurity=False` guidance
  is a nice-to-have but not critical since `hashlib` is not in the sandbox
  allowlist anyway.
- **MCP deployment guidance**: MCP servers proxying to legacy endpoints
  only need special handling if those endpoints require non-AEAD cipher
  suites. SHA-1 certificates alone are not a blocker on FIPS clusters.
- **Add FIPS validation to CI**: Include a FIPS test job that validates
  sandbox hashlib behavior and TLS connectivity against a controlled
  endpoint.

### Sources

- [RHEL 9 FIPS mode documentation](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/security_hardening/using-the-system-wide-cryptographic-policies_security-hardening)
- [Python hashlib usedforsecurity parameter](https://docs.python.org/3/library/hashlib.html)
- [OpenShift FIPS support](https://docs.openshift.com/container-platform/latest/installing/installing-fips.html)
- Tested 2026-04-15 on OCP 4.20.17, OpenSSL 3.5.1, RHEL 9.6 kernel 5.14.0-570.99.1

---

## Implementation Plan

### Phase 1: CodeGuard static analysis (this session)

Extend `sandbox/guardrails.py` with six new pattern categories:
1. Credential patterns (regex for API keys, tokens, passwords in string literals)
2. Unsafe deserialization (`pickle.loads`, `yaml.unsafe_load`, `marshal`)
3. SQL injection (string formatting into SQL-like strings)
4. Weak crypto (`md5`, `sha1` for security-sensitive use)
5. Path traversal (`../` in string literals passed to file operations)
6. Regex-based secret detection (AWS keys, PEM blocks, generic high-entropy strings)

All violations collected in one pass (existing convention). Full test coverage.

### Phase 2: Tool call inspection (this session)

New module `packages/fipsagents/src/fipsagents/baseagent/tool_inspector.py`:
- Secret detection in tool arguments (same patterns as CodeGuard)
- C2 pattern detection (suspicious URLs, base64 payloads)
- Prompt injection heuristics (instruction-like text in data fields)
- Wired into `ToolRegistry.execute()` as pre-execution check

### Phase 3: Audit trail + enforce/observe (this session)

- Structured audit logging for all security decisions
- Per-layer enforce/observe mode in `agent.yaml`
- Log to stdout (OpenShift log aggregation handles collection)

### Phase 4: Seccomp profile (future session)

- Ship `SeccompProfile` CRD in Helm chart
- Document SPO prerequisite
- Custom SCC for Localhost seccomp type
- Start with SCMP_ACT_LOG, iterate to SCMP_ACT_ERRNO allowlist

### Phase 5: Landlock wrapper (future session, OCP 4.18+)

- Python entrypoint wrapper with ctypes
- Runtime ABI detection, graceful degradation
- Read-only stdlib, read-write /tmp, deny everything else
