# Next Session: OpenShift Deployment & FIPS Cluster Testing

## Goal

Deploy the code-sandbox-agent example to OpenShift and validate it works
end-to-end on a real cluster. Then repeat on a FIPS-enabled cluster to
close out #33. Every example should be proven on-cluster before we
consider it done.

## Priority 1: Deploy code-sandbox-agent to OpenShift

The demo works locally (tested against GPT-OSS-20B on the RHPDS cluster).
Now deploy it as a real pod with the sandbox sidecar.

### Steps

1. **Build the sandbox container image.** Use the remote builder (Mac →
   x86_64). Push to quay.io.
   ```
   cd sandbox
   # Remote build via /build-remote or remote-builder agent
   ```

2. **Scaffold the agent from the template.** Either use `fips-agents create
   agent-loop code-sandbox-agent` or manually adapt the example into a
   deployable structure with the Helm chart.

3. **Build the agent container image.** The agent needs the code_executor
   tool, system prompt, and agent.yaml baked in. Push to quay.io.

4. **Deploy via Helm** with `sandbox.enabled: true`:
   ```yaml
   sandbox:
     enabled: true
     image:
       repository: quay.io/yourorg/code-sandbox
       tag: "0.4.0"
   ```

5. **Test on-cluster.** Curl the agent's /chat endpoint with:
   - Simple computation: "What is the standard deviation of [23, 45, 12, 67, 34, 89, 56]?"
   - Word problem: "I have a 20x30 foot yard, 1x2 foot sod, 80 pieces/pallet, $450/pallet — how many pallets and total cost?"
   - Guardrail test: code that tries `import os` or `eval()`
   - Verify the sandbox sidecar health endpoint: `oc exec <pod> -c sandbox -- curl localhost:8000/healthz`

### Open questions to resolve on-cluster

- **File permissions.** The sandbox Containerfile uses `COPY --chmod=644`
  but the agent Containerfile may not. Verify the sandbox container can
  read all its Python files when running as non-root (UID 1001 on UBI).
  If not, fix permissions in the Containerfile.

- **Read-only root filesystem.** The sandbox container has
  `readOnlyRootFilesystem: true` with an emptyDir at `/tmp`. Verify
  Python can write temp files and the executor works. Check if Python
  needs to write `.pyc` files (it shouldn't with `-I` flag, but verify).

- **Landlock on RHCOS.** The RHPDS cluster may or may not be on RHEL 9.6+.
  Check `oc debug node/<node> -- chroot /host dmesg | grep landlock` to
  see if Landlock is active. If yes, verify the sandbox applies it and
  the executor still works. If not, verify graceful degradation.

- **Resource limits.** The default sandbox limits are 500m CPU / 256Mi
  memory. Are these sufficient for the Python subprocess? Monitor with
  `oc adm top pod`.

- **Seccomp profile.** If SPO is installed on the cluster, try deploying
  with `sandbox.seccomp.enabled: true`. If SPO is not installed, skip
  and document the prerequisite.

## Priority 2: FIPS cluster testing (#33)

Once the FIPS cluster is provisioned (being set up week of 2026-04-14):

1. Deploy the code-sandbox-agent to the FIPS cluster.
2. Run through the test matrix in issue #33:
   - `hashlib.md5()` behavior in the sandbox
   - `hashlib.md5(b"x", usedforsecurity=False)` behavior
   - Agent-to-MCP TLS with self-signed certs
   - Error message clarity
3. Update `research/sandbox-hardening-v2.md` Finding 5 with results.
4. Add FIPS deployment guidance to the Helm chart docs.

## Priority 3: Harden remaining examples

Apply the same on-cluster testing rigor to the shared-memory example:
- Deploy both agents (problem-solver, code-writer) to OpenShift
- Verify inter-agent memory sharing via MemoryHub
- Verify sandbox sidecar works from both agents

## Key files

### Code sandbox agent (deploy)
- `examples/code-sandbox-agent/` — the demo agent
- `sandbox/` — the sidecar (build as separate container)
- `templates/agent-loop/chart/` — Helm chart with sandbox support

### FIPS testing
- `research/sandbox-hardening-v2.md` — Finding 5 (FIPS implications)
- Issue #33 — test matrix

## Prior session context

- v0.4.0 released to PyPI. Sandbox hardening v2 complete:
  CodeGuard patterns, ToolInspector, SecurityConfig, SeccompProfile CRD,
  Landlock wrapper.
- Full test suite: 499 tests passing (1 skipped — Linux-only Landlock).
- Code sandbox agent demo tested locally against GPT-OSS-20B on RHPDS
  cluster. Tool calling works with strengthened system prompt.
- Key lesson: LLMs skip tool calling for "easy" math unless the system
  prompt is explicit that tool use is mandatory.
- FIPS research done but untested. Issue #33 tracks validation.
