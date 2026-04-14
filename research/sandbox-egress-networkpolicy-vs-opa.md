# Sandbox Egress: NetworkPolicy vs OPA/Rego Proxy

**Question:** Is OpenShift NetworkPolicy sufficient for a zero-egress code execution sandbox,
or does an OPA/Rego proxy layer add meaningful value?

**Date:** 2026-04-14

---

## Summary and Recommendation

**NetworkPolicy is sufficient for a zero-egress sandbox. OPA proxy is not worth the complexity
in this specific use case.**

The core insight: if the policy is "allow nothing outbound," there is no URL-level filtering
to do, no dynamic rule updates needed, and no header inspection required. OPA proxy adds value
when you need *selective* egress — allow some URLs, inspect request headers, enforce fine-grained
rules per service. For a sandbox where the answer is always "no," NetworkPolicy enforces that at
the kernel/OVN ACL level with zero runtime overhead and zero additional attack surface.

The more productive investment is defense-in-depth at the container/runtime layer: seccomp
profiles, capability drops, and read-only filesystems. These address threat vectors that
NetworkPolicy does not cover, unlike OPA proxy which duplicates NetworkPolicy's blocking
without adding a materially different layer.

---

## What NetworkPolicy Can Do for Zero-Egress

A NetworkPolicy that blocks all outbound traffic from a specific pod is straightforward. The
`egress: []` pattern creates an empty allowlist, which means all outbound connections are
dropped:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sandbox-deny-all-egress
  namespace: sandbox
spec:
  podSelector:
    matchLabels:
      role: code-executor
  policyTypes:
    - Egress
  egress: []
```

This is not theoretical. On OpenShift with OVN-Kubernetes (the default CNI since OCP 4.12),
enforcement happens via Open Virtual Network Access Control Lists applied at the hypervisor
layer — not iptables — which means:

- Rules take effect before packets leave the host, not just at pod boundaries.
- There is no iptables-based bypass that a container process can exploit.
- Pods that use `hostNetwork: true` are explicitly excluded from NetworkPolicy (but the sandbox
  sidecar should never use host networking).

The critical DNS nuance: blocking all egress also blocks DNS (port 53 UDP/TCP). For a
code execution sandbox that is intentionally isolated, this is the correct behavior. If the
sandbox container does not need to resolve any hostnames, don't open port 53. If you add
a DNS exception, you need to be aware that DNS tunneling becomes a viable exfiltration channel.

### OpenShift-Specific Tools

Beyond standard NetworkPolicy, OpenShift provides two additional controls:

**EgressFirewall** (OVN-Kubernetes, `k8s.ovn.org/v1`): A namespace-scoped resource that can
filter by CIDR, FQDN, port, and protocol. Supports FQDN-based rules (e.g., `allow docs.openshift.com`),
but FQDN matching has known limitations — DNS resolution is polled on a 30-minute TTL by default,
creating race conditions where a pod may resolve an updated IP before the firewall catches up.
Not recommended for deny rules or security-critical FQDN-based control.

**AdminNetworkPolicy** (cluster-scoped, OCP 4.14+): A newer cluster-admin-controlled resource
that evaluates before namespace NetworkPolicies. Useful for enforcing baseline isolation that
project owners cannot override.

---

## NetworkPolicy Limitations

| Capability | NetworkPolicy | Notes |
|---|---|---|
| Block all outbound | Yes | `egress: []` |
| IP/CIDR filtering | Yes | Standard |
| Port/protocol filtering | Yes | TCP, UDP |
| FQDN/URL filtering | No | Requires EgressFirewall or Calico/Cilium extensions |
| HTTP path/header inspection | No | L7 only; needs a proxy or service mesh |
| DNS traffic distinction | Partial | Can allow/deny port 53 but cannot inspect DNS content |
| Logging denied connections | No | Requires CNI observability add-ons (e.g., OVN observability, Calico flow logs) |
| Dynamic rule updates without pod restart | Yes | NetworkPolicy changes apply immediately |

### Key Limitations in Detail

**No L7 awareness.** NetworkPolicy operates at L3/L4 (IP address, port, protocol). It cannot
distinguish `GET /api/data` from `POST /api/exfiltrate` on the same port. For a zero-egress
sandbox this does not matter because there are no allowed connections.

**No URL-level filtering.** If you needed to allow `api.openai.com` but block `evil.com`,
NetworkPolicy cannot do this directly without resolving FQDN to IP (fragile). EgressFirewall
attempts FQDN support but with the race condition caveats noted above.

**No per-connection logging.** By default, denied packets are silently dropped. If you want
to know *what* the sandbox attempted to call, you need either the OVN Network Observability
operator, a CNI that supports flow logs (Calico Enterprise, Cilium), or a proxy. This is
probably the most legitimate gap for a sandbox: audit trails showing "container attempted
outbound connection to 1.2.3.4:443" are operationally useful.

**DNS tunneling.** If you allow DNS (port 53 UDP) to enable internal name resolution, a
compromised process can exfiltrate data by encoding it in DNS queries. For a fully zero-egress
sandbox, the correct answer is: don't allow DNS at all. For a sandbox that needs internal
cluster DNS, this is a real concern.

**hostNetwork bypass.** Pods with `hostNetwork: true` are completely outside NetworkPolicy scope.
Ensure the sandbox sidecar spec does not set this.

---

## What OPA Proxy Would Add

OPA in a proxy role (typically via the Envoy ext_authz gRPC API, or OPA-Envoy plugin) sits in
the traffic path and evaluates every request against Rego policy before it is allowed to proceed.

### Deployment Patterns

**Pattern 1: Envoy + OPA sidecar (most common)**
- Envoy is injected as a sidecar or deployed as a DaemonSet proxy.
- Traffic is redirected to Envoy via iptables rules (similar to Istio's approach).
- Envoy calls OPA via the External Authorization (ext_authz) gRPC API for each request.
- OPA evaluates Rego policy and returns allow/deny.
- Adds 2-3 containers per pod (Envoy, OPA-Envoy, kube-mgmt for policy sync).

**Pattern 2: Standalone OPA with iptables redirect**
- iptables REDIRECT rules force all outbound traffic through OPA running on a local port.
- OPA evaluates and proxies or blocks.
- Simpler than full Envoy stack but OPA alone is not a high-performance proxy; better suited
  for HTTP/HTTPS than arbitrary TCP.

**Pattern 3: OPA as Admission Controller (Gatekeeper)**
- OPA Gatekeeper operates as a Kubernetes admission webhook, not a traffic proxy.
- It validates NetworkPolicy manifests at creation time (e.g., "every namespace must have a
  default-deny policy").
- This is a completely different use case — enforcement of *configuration* correctness, not
  runtime traffic interception.

### What OPA Proxy Adds vs. NetworkPolicy (for zero-egress)

| Value Add | Relevant for Zero-Egress? |
|---|---|
| URL-path-level filtering | No — no traffic is allowed, so no paths to filter |
| HTTP header inspection | No — same reason |
| Dynamic rule updates without restart | Marginal — NetworkPolicy itself updates live |
| Per-request audit logging | Yes — this is real value |
| Fine-grained observability | Yes — you can log *what* was attempted |
| Rego policy expressiveness | No — the policy is trivially "deny all" |

For a zero-egress sandbox, OPA proxy's primary surviving benefit is **observability**: you get
structured logs of attempted outbound connections with request context (method, URL, headers,
source pod identity). NetworkPolicy drops silently.

---

## Complexity vs. Value Tradeoff

### OPA Proxy Complexity Cost

Adding Envoy + OPA sidecar to the sandbox pod introduces:

- 2-3 additional containers per sandbox pod
- iptables redirect rules that must be set up in an init container (requires `NET_ADMIN`
  capability — the opposite of what you want in a sandbox)
- A Rego policy that must be loaded, validated, and kept in sync via ConfigMap or OPA bundle server
- Latency added to every connection (typically 1-5ms per request via Unix domain socket,
  more over TCP)
- A new failure mode: if OPA crashes, does Envoy fail open or closed? (This requires
  deliberate configuration.)
- Significantly more operational surface: OPA version upgrades, Envoy version upgrades,
  debug complexity when something is blocked unexpectedly

### Attack Vectors OPA Covers That NetworkPolicy Does Not

Honest assessment: for a true zero-egress sandbox, essentially none.

NetworkPolicy blocks at the OVN ACL level (hypervisor). A process inside the sandbox cannot
bypass this from userspace — there is no iptables manipulation or raw socket trick that works
against OVN ACLs. The only bypass vectors are:

1. `hostNetwork: true` — mitigated by never setting this in the pod spec
2. Privilege escalation to root + namespace escape — mitigated by security context, not OPA proxy
3. Exploiting a bug in OVN itself — an OPA sidecar does not help here
4. Node-level compromise — no container-level control addresses this

OPA proxy enforces policy *inside* the same trust boundary as the application (same node, same
network namespace for the iptables redirect). OVN ACLs enforce policy *outside* that boundary,
at the hypervisor. For blocking, the hypervisor-level control is strictly stronger.

---

## Simpler Alternatives Worth Considering

For a code execution sandbox, the defense-in-depth investments with the highest
security-per-complexity ratio are:

### 1. NetworkPolicy (zero-egress) + SecurityContext hardening

This should be the baseline. In addition to `egress: []`:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 65534
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
      - ALL
```

Dropping `NET_RAW` (included in `drop: ALL`) removes the ability to create raw sockets, which
eliminates one class of NetworkPolicy bypass on older CNIs. On OVN-Kubernetes it's defense-in-depth
rather than required, but still correct practice.

### 2. Seccomp Profile (RuntimeDefault or custom)

`RuntimeDefault` blocks ~300 high-risk syscalls. A custom profile can further restrict to
only the syscalls the sandbox legitimately needs. This addresses kernel attack surface that
NetworkPolicy does not touch at all.

### 3. OVN Network Observability (for audit logging without OPA)

If the primary OPA value you want is **audit logging of denied connections**, OpenShift's
Network Observability operator (based on eBPF + OVN flow export) provides this at the CNI
level without touching the pod at all. You get per-flow records including source pod,
destination IP/port, and allow/deny decision. This is a much simpler path than OPA proxy.

### 4. gVisor / Kata Containers (for stronger isolation)

If the sandbox executes genuinely untrusted code (arbitrary user-submitted scripts), the
upgrade from container to microVM (Kata Containers, which OpenShift AI supports) or gVisor
provides a dedicated kernel boundary. NetworkPolicy still applies on top of this. This
addresses threat vectors that neither NetworkPolicy nor OPA proxy covers.

---

## Decision Matrix

| Scenario | Recommended Approach |
|---|---|
| Zero-egress sandbox, trusted workload | NetworkPolicy `egress: []` + SecurityContext hardening |
| Zero-egress sandbox, want audit logs | NetworkPolicy + OVN Network Observability operator |
| Zero-egress sandbox, untrusted code | NetworkPolicy + gVisor/Kata Containers + seccomp |
| Selective egress (allow some URLs) | EgressFirewall (CIDR) or Calico/Cilium FQDN policies |
| URL/header-level enforcement needed | Envoy + OPA sidecar (justified complexity) |
| Policy governance (correct NP config) | OPA Gatekeeper (admission control, not traffic proxy) |

---

## Final Recommendation

For the agent-template sandbox sidecar:

**Use NetworkPolicy with `egress: []` and full SecurityContext hardening. Skip OPA proxy.**

The OPA proxy pattern is valuable when you have selective egress requirements — when some traffic
must be allowed and you need L7-level control over what gets through. For a sandbox where the
answer is always "block everything," OPA proxy adds operational complexity without a
corresponding security gain, because NetworkPolicy is already enforcing at a stronger trust
boundary (OVN ACLs at the hypervisor, not iptables inside the pod).

If audit logging of denied connections is needed, use the OVN Network Observability operator
rather than OPA. If stronger compute isolation is needed, invest in Kata Containers rather than
an L7 proxy.

---

## Sources

- [OpenShift NetworkPolicy documentation (OCP 4.10)](https://docs.redhat.com/en/documentation/openshift_container_platform/4.10/html/networking/network-policy)
- [OpenShift Egress Firewall API reference (OCP 4.19)](https://docs.redhat.com/en/documentation/openshift_container_platform/4.19/html/network_security/egress-firewall)
- [OVN-Kubernetes EgressFirewall](https://ovn-kubernetes.io/features/network-security-controls/egress-firewall/)
- [OVN-Kubernetes NetworkPolicy enforcement](https://ovn-kubernetes.io/features/network-security-controls/network-policy/)
- [Kubernetes NetworkPolicy recipes — deny all egress](https://github.com/ahmetb/kubernetes-network-policy-recipes/blob/master/11-deny-egress-traffic-from-an-application.md)
- [CNCF: Guide to Kubernetes egress network policies](https://www.cncf.io/blog/2020/02/10/guide-to-kubernetes-egress-network-policies/)
- [Red Hat: Guide to Kubernetes Egress Network Policies](https://www.redhat.com/en/blog/guide-to-kubernetes-egress-network-policies)
- [OPA-Envoy Plugin](https://www.openpolicyagent.org/docs/envoy)
- [OPA-Envoy GitHub](https://github.com/open-policy-agent/opa-envoy-plugin)
- [OPA sidecar with Unix Domain Socket (Envoy Gateway)](https://gateway.envoyproxy.io/docs/tasks/extensibility/opa-sidecar-unix-domain-socket/)
- [Deploying OPA on Kubernetes](https://www.openpolicyagent.org/docs/deploy/k8s)
- [CNCF: How to enforce Kubernetes network security policies using OPA](https://www.cncf.io/blog/2020/09/09/how-to-enforce-kubernetes-network-security-policies-using-opa/)
- [Egress Firewall in OpenShift with OVN Kubernetes (rcarrata.com)](https://rcarrata.com/openshift/egress-firewall/)
- [SUSE: Enforce DNS-based Egress Container Security Policies in Kubernetes and OpenShift](https://www.suse.com/c/enforce-egress-container-security-policies-kubernetes-openshift/)
- [Sandbox isolation discussion (shayon.dev)](https://www.shayon.dev/post/2026/52/lets-discuss-sandbox-isolation/)
- [Northflank: How to sandbox AI agents in 2026](https://northflank.com/blog/how-to-sandbox-ai-agents)
- [Kubernetes SIG: agent-sandbox project](https://github.com/kubernetes-sigs/agent-sandbox)
- [AdminNetworkPolicy / BaselineAdminNetworkPolicy (stderr.at)](https://blog.stderr.at/openshift-platform/security/network-policies/2024-11-06-using-adminnetworkpolicies-and-baselineadminnetworkpolicies/)
- [Kubernetes Linux kernel security constraints](https://kubernetes.io/docs/concepts/security/linux-kernel-security-constraints/)
- [NPEP-133: FQDN Selector for Egress Traffic (upstream proposal)](https://network-policy-api.sigs.k8s.io/npeps/npep-133/)
