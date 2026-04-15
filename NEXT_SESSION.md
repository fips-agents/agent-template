# Next Session: Sandbox Profiles & Code Refactoring Pipeline

## What was completed this session

- **FIPS cluster testing (issue #33)**: Full test matrix on OCP 4.20.17 FIPS
  cluster (cluster-l78nk). Key findings: MD5 blocked, SHA-1 allowed by FIPS
  (guardrails stricter, correct). SHA-1 cert TLS verification works (original
  hypothesis wrong). 21 AEAD-only ciphers. SPO + seccomp work on FIPS.
  Updated Finding 5 in sandbox-hardening-v2.md, added FIPS notes to
  values.yaml, commented results on issue #33.

## What was completed prior session

- **Workflow extraction**: Moved 8 workflow modules into fipsagents package
- **Test suite**: 256 tests covering all baseagent + workflow modules
- **Document-analysis example**: 6-node pipeline with 28 tests + live LLM validation

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

## Priority 2: Code refactoring agent MCP servers

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
- `examples/document-analysis/` -- Workflow example (conditional routing, mixed nodes, live LLM)
- `sandbox/` -- Sidecar (to be extended with profiles/)
- Issue #33 -- FIPS test matrix

## Prior session context

- 284 tests passing (256 in fipsagents package + 28 in document-analysis example).
- Code-sandbox-agent deployed on RHPDS cluster (namespace: code-sandbox-agent)
  - Agent image: internal registry, built via BuildConfig
  - Sandbox image: internal registry, built via BuildConfig
  - Route: code-sandbox-agent-agent-template-code-sandbox-agent.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com
  - Model: GPT-OSS-20B (RedHatAI/gpt-oss-20b) via external route
- ec2-dev-2 was unreachable last session; used BuildConfig instead
- Tool ecosystem identified for refactoring pipeline: treeloom, greploom,
  sanicode, veripak, stigcode. All deploy as MCP servers.
