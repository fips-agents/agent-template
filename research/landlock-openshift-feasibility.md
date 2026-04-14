# Landlock LSM in OpenShift Pods: Feasibility Research

**Date:** 2026-04-14  
**Status:** Complete

---

## Summary

Landlock LSM **can run inside an OpenShift pod** under `restricted-v2` SCC without any special capabilities. Because `restricted-v2` sets `allowPrivilegeEscalation: false`, the container runtime sets the `no_new_privs` bit on the process — which is precisely the prerequisite Landlock needs in place of `CAP_SYS_ADMIN`. The Landlock syscalls are present in the `containers/common` default seccomp profile used by CRI-O. The main practical gating factor is whether Landlock is enabled in the node kernel: RHEL 9.6+ enables it by default (ABI 5 backported), while earlier RHEL 9.x releases do not.

---

## Q1: Kernel version and ABI history

Landlock was introduced in **Linux 5.13** (June 2021). Each ABI version adds new access-control surfaces:

| ABI | Kernel | What was added |
|-----|--------|----------------|
| 1 | 5.13 | Filesystem access rights (read, write, execute, mkdir, etc.) |
| 2 | 5.19 | `LANDLOCK_ACCESS_FS_REFER` — controlled file reparenting across directories |
| 3 | 6.2  | `LANDLOCK_ACCESS_FS_TRUNCATE` — restrict file truncation |
| 4 | 6.7  | TCP network: `LANDLOCK_ACCESS_NET_BIND_TCP`, `LANDLOCK_ACCESS_NET_CONNECT_TCP` |
| 5 | 6.10 | `LANDLOCK_ACCESS_FS_IOCTL_DEV` — restrict ioctl on character/block devices |
| 6 | 6.12 | IPC scoping: `LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET`, `LANDLOCK_SCOPE_SIGNAL` |
| 7 | 6.15 | Audit logging for denied accesses (`LANDLOCK_RESTRICT_SELF_LOG_*` flags) |
| 8 | 6.x  | `LANDLOCK_RESTRICT_SELF_TSYNC` for multi-threaded enforcement |

Programs query the running kernel's ABI by passing `LANDLOCK_CREATE_RULESET_VERSION` to `landlock_create_ruleset(2)`. The current highest ABI in mainline as of April 2026 is **ABI 8** (kernel 6.15+).

Sources: [kernel.org Landlock userspace-api](https://docs.kernel.org/userspace-api/landlock.html), [Rust landlock crate ABI enum](https://docs.rs/landlock/latest/landlock/enum.ABI.html), [Landlock news #5](https://seclists.org/oss-sec/2025/q2/167)

---

## Q2: Capability requirements — no CAP_SYS_ADMIN needed

Landlock is explicitly designed as **unprivileged self-restriction**. Calling `landlock_restrict_self(2)` requires the thread to satisfy **one** of:

1. Hold `CAP_SYS_ADMIN` in its user namespace, **OR**
2. Have the `no_new_privs` bit set (via `prctl(PR_SET_NO_NEW_PRIVS, 1, ...)`)

The `no_new_privs` path is the standard unprivileged path. No Linux capabilities are consumed — the process just locks itself down. The kernel docs state: *"Landlock empowers any process, including unprivileged ones, to securely restrict themselves."*

This design prevents an unprivileged sandboxed process from breaking out by spawning a privileged child (set-UID binaries won't gain privileges when `no_new_privs` is set).

Sources: [Landlock userspace-api docs](https://docs.kernel.org/userspace-api/landlock.html), [landlock_restrict_self(2) man page](https://man7.org/linux/man-pages/man2/landlock_restrict_self.2.html)

---

## Q3: Does restricted-v2 SCC allow Landlock self-restriction?

**Yes — restricted-v2 is fully compatible with Landlock.** Here is why:

`restricted-v2` (introduced in OpenShift 4.11) enforces:

- `requiredDropCapabilities: ALL` — all Linux capabilities are dropped
- `allowPrivilegeEscalation: false` — **this is the key**: the container runtime translates this to setting the `no_new_privs` bit on the container process at startup
- `seccompProfile: RuntimeDefault` — the CRI-O/containerd default seccomp profile
- `runAsNonRoot: true`

Because `allowPrivilegeEscalation: false` → `no_new_privs` is set, the container process already satisfies Landlock's prerequisite. No capabilities are needed. The application calls `prctl(PR_SET_NO_NEW_PRIVS, 1, ...)` in its own code (redundant but harmless if the runtime already set it), then calls the three Landlock syscalls.

The `RuntimeDefault` seccomp profile used by CRI-O is derived from `containers/common/pkg/seccomp/seccomp.json`, which explicitly allows all three Landlock syscalls: `landlock_create_ruleset`, `landlock_add_rule`, `landlock_restrict_self`. (Confirmed by direct inspection of the `containers/common` seccomp.json and the `containerd` default profile.)

Sources: [Red Hat blog: Pod Admission and SCCs v2](https://www.redhat.com/en/blog/pod-admission-and-sccs-version-2-in-openshift), [Kubernetes: allowPrivilegeEscalation sets no_new_privs](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/), [containers/common seccomp.json](https://github.com/containers/common/blob/main/pkg/seccomp/seccomp.json), [moby PR #43199 adding Landlock syscalls](https://github.com/moby/moby/pull/43199)

---

## Q4: Custom SCC — when and why?

A custom SCC is **not required** for Landlock to work. `restricted-v2` is sufficient.

A custom SCC would only be warranted in edge cases:

- Your workload needs `allowPrivilegeEscalation: true` (which disables `no_new_privs` and therefore breaks Landlock unless the process holds `CAP_SYS_ADMIN`). This is unusual and generally a security regression.
- You need to add the `NET_BIND_SERVICE` capability (allowed by `restricted-v2`) but want explicit documentation of the Landlock intent.

If you did need a custom SCC for some orthogonal reason and wanted to be explicit, it would look like this:

```yaml
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraints
metadata:
  name: landlock-enabled
allowPrivilegeEscalation: false   # ensures no_new_privs is set
requiredDropCapabilities:
  - ALL
allowedCapabilities: []           # no caps needed for Landlock
runAsUser:
  type: MustRunAsNonRoot
seccompProfiles:
  - runtime/default               # allows landlock_* syscalls
fsGroup:
  type: MustRunAs
seLinuxContext:
  type: MustRunAs
```

This is essentially identical to `restricted-v2` — the point being that no additional permissions are needed.

---

## Q5: Build time vs. runtime application

Landlock **must be applied at process runtime** — it cannot be baked into an image at build time. The restriction is enforced via kernel syscalls at process startup:

1. Process starts
2. Application code calls `prctl(PR_SET_NO_NEW_PRIVS, 1, ...)` (usually already done by the container runtime via `allowPrivilegeEscalation: false`)
3. Application calls `landlock_create_ruleset()`, populates rules with `landlock_add_rule()`, then calls `landlock_restrict_self()` to lock itself

This code runs in the application's `main()` or a wrapper script. It can be "baked in" in the sense that the application binary itself contains the Landlock setup calls — that binary lives in the image. But there is no image-layer-level mechanism to pre-apply the restrictions; the kernel only honors the syscalls at process execution time.

A practical pattern is a thin entrypoint wrapper (shell script or Go/Rust binary) that establishes the Landlock policy and then `exec`s the real application. Tools like [`landrun`](https://github.com/Zouuup/landrun) implement this pattern: `landrun --ro /usr --ro /lib -- myapp`.

Sources: [Landlock userspace-api: example code](https://docs.kernel.org/userspace-api/landlock.html), [landrun](https://github.com/Zouuup/landrun), [NVIDIA NemoClaw sandbox hardening](https://docs.nvidia.com/nemoclaw/latest/deployment/sandbox-hardening.html)

---

## Q6: Practical limitations

**What Landlock can restrict:**
- Filesystem: read, write, execute, directory traversal, make/remove dirs, create special files, link/rename (ABI 1–3, 5)
- TCP network by port number: bind and connect (ABI 4)
- ioctl on device files (ABI 5)
- Abstract Unix socket connections (ABI 6)
- Sending signals to processes outside the domain (ABI 6)

**What Landlock cannot restrict:**
- **UDP, ICMP, raw sockets** — only TCP is controlled (ABI 4+); other protocols are unaffected
- **IP addresses / remote hosts** — TCP rules control port numbers only, not destination IPs
- **IPv6** is subject to TCP port rules in the same manner as IPv4, but there is no finer-grained IPv6 control
- **Pipes and kernel-internal sockets** — cannot be directly restricted (only accessible via `/proc/<pid>/fd/` indirectly)
- **Mount/unmount, pivot_root** — cannot restrict filesystem topology changes
- **`chdir`, `stat`, `chmod`, `chown`, `access`** — these operations are not controllable by Landlock
- **Shared memory, process memory** — out of scope entirely
- **Already-open file descriptors** — Landlock only applies to newly opened files; inherited fds (like stdin/stdout/stderr) are unaffected
- **Ruleset depth** — maximum 16 stacked Landlock rulesets (deep nesting of sandboxed processes hits this limit)

Landlock is complementary to seccomp (syscall filtering) and SELinux/AppArmor (MAC). It fills the filesystem and TCP access control niche for unprivileged self-restriction.

Sources: [Landlock limitations section](https://docs.kernel.org/userspace-api/landlock.html), [Landlock news #5](https://seclists.org/oss-sec/2025/q2/167)

---

## Q7: RHEL 9 / RHCOS kernel status — important nuances

### RHEL 9.0–9.5: Landlock NOT enabled by default

The RHEL 9 kernel series is based on upstream 5.14 with extensive backports. Landlock was not enabled in the default `CONFIG_LSM` list in RHEL 9.0–9.5. The kernel was compiled with `CONFIG_SECURITY_LANDLOCK=y` but Landlock was not included in the boot-time LSM chain by default, so syscalls would fail with `EOPNOTSUPP`.

### RHEL 9.6: Landlock ENABLED by default (ABI 5 backported)

Starting with **kernel-5.14.0-568.el9** (RHEL 9.6.0), Red Hat enabled Landlock in the default LSM boot configuration and backported features through **ABI 5**. This was tracked as [RHEL-8810](https://issues.redhat.com/browse/RHEL-8810). RHEL 9.6 also appears in the Landlock project's list of distributions shipping Landlock enabled by default.

### Known bug: kernel-5.14.0-611.el9 (incomplete ABI 6 backport)

A subsequent kernel release (`5.14.0-611.el9`) contained an **incomplete backport of ABI 6** — the scope flags were partially added but a necessary patch was missing. This caused tools using Landlock (e.g., `xz`) to fail with "Failed to enable the sandbox." The fix was applied upstream; libraries using Landlock defensively (with `best_effort` compatibility mode) are unaffected. Applications that hard-require ABI 6 should check the ABI version at runtime.

### OpenShift node implications

OpenShift worker nodes run **RHCOS**, which tracks the RHEL kernel. Nodes running OpenShift 4.18+ (based on RHEL 9.6) should have Landlock enabled. Earlier OpenShift releases (< 4.18) would require a `MachineConfig` to add `lsm=lockdown,integrity,selinux,landlock` (or similar) to the kernel command line to enable it — but this is a cluster-admin operation.

To verify Landlock is active on a node:
```bash
dmesg | grep landlock
# Expected: "landlock: Up and running."
```

Or from within a pod (check ABI version):
```c
int abi = landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
// Returns ABI version (≥1) if enabled, -1 with errno=EOPNOTSUPP if not enabled
```

Sources: [Landlock news #5 / RHEL 9.6.0 mention](https://seclists.org/oss-sec/2025/q2/167), [Debian bug: RHEL 9 kernel Landlock workaround](https://www.mail-archive.com/debian-bugs-dist@lists.debian.org/msg2069518.html), [RHEL 9.6 release notes](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/9.6_release_notes/index)

---

## Decision Summary

| Question | Answer |
|----------|--------|
| No-CAP usage | Yes — `no_new_privs` is sufficient |
| Works under `restricted-v2` | Yes — `allowPrivilegeEscalation: false` sets `no_new_privs` |
| Custom SCC needed | No |
| Syscalls allowed by default seccomp | Yes (containers/common profile includes all three) |
| RHEL 9 kernel support | RHEL 9.6+ only (kernel ≥5.14.0-568.el9) |
| OpenShift version | OCP 4.18+ (RHCOS based on RHEL 9.6) |
| Apply at build time | No — runtime only, but entrypoint wrapper pattern works |
| Network restriction | TCP ports only (no UDP, no IP filtering) |
| Known RHEL bug | Incomplete ABI 6 backport in 5.14.0-611.el9; fixed in later builds |

---

## Sources

- [Linux kernel Landlock userspace-api documentation](https://docs.kernel.org/userspace-api/landlock.html)
- [Landlock admin guide (system-wide)](https://docs.kernel.org/next/admin-guide/LSM/landlock.html)
- [landlock_restrict_self(2) man page](https://man7.org/linux/man-pages/man2/landlock_restrict_self.2.html)
- [landlock(7) man page](https://www.man7.org/linux/man-pages/man7/landlock.7.html)
- [Rust landlock crate ABI enum](https://docs.rs/landlock/latest/landlock/enum.ABI.html)
- [Landlock news #5 (oss-sec, 2025)](https://seclists.org/oss-sec/2025/q2/167)
- [Red Hat blog: Pod Admission and SCCs v2](https://www.redhat.com/en/blog/pod-admission-and-sccs-version-2-in-openshift)
- [containers/common seccomp.json](https://github.com/containers/common/blob/main/pkg/seccomp/seccomp.json)
- [containerd default seccomp profile](https://github.com/containerd/containerd/blob/main/contrib/seccomp/seccomp_default.go)
- [moby/moby PR #43199 — add Landlock syscalls to default seccomp](https://github.com/moby/moby/pull/43199)
- [Kubernetes: configure security context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/)
- [Debian bug: RHEL 9 incomplete ABI 6 backport](https://www.mail-archive.com/debian-bugs-dist@lists.debian.org/msg2069518.html)
- [landrun — Landlock wrapper tool](https://github.com/Zouuup/landrun)
- [NVIDIA NemoClaw sandbox hardening with Landlock](https://docs.nvidia.com/nemoclaw/latest/deployment/sandbox-hardening.html)
