# Next Session: Sandbox Profiles & Code Refactoring Pipeline

## What was completed

- Code-sandbox-agent deployed to OpenShift and validated end-to-end
  - Built both images (agent + sandbox sidecar) via OpenShift BuildConfig
  - Created FastAPI server wrapper (server.py) with /chat, /healthz, /readyz
  - Created agent Containerfile (UBI 9 Python 3.11)
  - Deployed via Helm with sandbox sidecar enabled
  - Full test matrix passed: computations, word problems, guardrails, health
- On-cluster findings documented:
  - Landlock active (ABI v5) on RHCOS 9.6 (kernel 5.14.0)
  - Read-only root filesystem confirmed working with /tmp emptyDir
  - File permissions fine with COPY --chmod=644
  - Resources well within limits (agent: 2m/146Mi, sandbox: 7m/34Mi)
  - SPO not installed on RHPDS cluster (seccomp skipped)
  - litellm requires OPENAI_API_KEY even for unauthenticated vLLM (set dummy value)
- Planning doc written: `planning/code-execution-pipeline.md`

## Priority 1: Sandbox profiles (implement)

See `planning/code-execution-pipeline.md` Section 1 for full design.

### Steps

1. Create `sandbox/profiles/` directory with profile YAML configs:
   - `minimal.yaml` (codifies the current hardcoded allowlist)
   - `data-science.yaml` (+ numpy, pandas, scipy with blocklists)

2. Refactor `guardrails.py` to load allowlist from profile YAML instead
   of the hardcoded `ALLOWED_IMPORTS` set. Add blocklist audit stage.

3. Implement pipeline runner in `app.py` (replaces inline
   validate_code/execute_code). See Section 2 of the planning doc.

4. Build and test both profile image variants:
   - `code-sandbox:0.5.0-minimal`
   - `code-sandbox:0.5.0-data-science`

5. Add Helm values for profile selection (`sandbox.profile`).

6. Deploy data-science profile to RHPDS cluster and test with numpy/pandas
   workloads to validate the blocklist catches dangerous attribute access.

## Priority 2: FIPS cluster testing (#33)

Still waiting on FIPS cluster provisioning (week of 2026-04-14).

1. Deploy code-sandbox-agent to the FIPS cluster
2. Run test matrix from issue #33:
   - hashlib.md5() behavior in the sandbox
   - hashlib.md5(b"x", usedforsecurity=False)
   - Agent-to-MCP TLS with self-signed certs
   - Error message clarity
3. Update `research/sandbox-hardening-v2.md` Finding 5 with results
4. Add FIPS deployment guidance to Helm chart docs

## Priority 3: Code refactoring agent MCP servers

See `planning/code-execution-pipeline.md` Section 3 for full design.

### Steps

1. Deploy greploom as MCP server (has built-in MCP support)
2. Deploy sanicode as MCP server (has API server mode)
3. Deploy veripak as MCP server (has built-in MCP support)
4. Scaffold code-refactoring-agent using fips-agents template
5. Wire agent to MCP servers, implement pipeline orchestration
6. Integrate stigcode when it reaches stable release

## Key files

- `planning/code-execution-pipeline.md` -- Authoritative design doc
- `examples/code-sandbox-agent/` -- Deployed example (server.py, Containerfile, values-deploy.yaml)
- `sandbox/` -- Sidecar (to be extended with profiles/)
- Issue #33 -- FIPS test matrix

## Prior session context

- v0.4.0 on PyPI. 499 tests passing.
- Code-sandbox-agent deployed on RHPDS cluster (namespace: code-sandbox-agent)
  - Agent image: internal registry, built via BuildConfig
  - Sandbox image: internal registry, built via BuildConfig
  - Route: code-sandbox-agent-agent-template-code-sandbox-agent.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com
  - Model: GPT-OSS-20B (RedHatAI/gpt-oss-20b) via external route
- ec2-dev-2 was unreachable this session; used BuildConfig instead
- Tool ecosystem identified for refactoring pipeline: treeloom, greploom,
  sanicode, veripak, stigcode. All deploy as MCP servers.
