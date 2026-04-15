# Code Execution Pipeline

Design for configurable sandbox profiles, a multi-stage execution pipeline with risk-proportional scan depth, and an agentic code refactoring pipeline backed by MCP servers. This document covers three related concerns that share infrastructure but have different implementation timelines.

## 1. Sandbox Profiles

The sandbox currently has a single security configuration: a fixed import allowlist in `guardrails.py`, a single AST validation pass, and uniform resource limits. This is sufficient for the minimal profile but blocks adoption of the sandbox for workloads that need numpy, pandas, or code analysis libraries.

Profiles make the sandbox configurable without making it permissive. Each profile widens the import allowlist but compensates with additional guardrail stages and targeted blocklists for dangerous attributes on the newly-allowed modules.

### Profile Definitions

| Profile | Added packages | New risk vectors | Scan level | Memory |
|---------|---------------|------------------|------------|--------|
| minimal (current) | math, statistics, collections, itertools, functools, re, datetime, json, csv, string, textwrap, decimal, fractions, random, operator, typing | None | AST only | 256Mi |
| data-science | + numpy, pandas, scipy | `np.load()`, `np.ctypeslib`, `pd.read_*`, `scipy.weave`, C extensions | AST + blocklist audit | 512Mi |
| financial | + numpy, pandas, decimal (already present), openpyxl | Same as data-science + openpyxl file I/O | AST + blocklist audit | 512Mi |
| code-analysis | + treeloom, ast, tokenize, inspect | Full CPG build capability, tree-sitter FFI | AST + self-referential check | 512Mi |

### Configuration Format

Each profile is a YAML file in `sandbox/profiles/`. The runner loads the profile named in the `SANDBOX_PROFILE` environment variable (default: `minimal`).

```yaml
# sandbox/profiles/data-science.yaml
name: data-science
description: Adds numpy, pandas, scipy for numerical computing

imports:
  # Inherits from minimal (all profiles include the minimal set)
  extends: minimal
  additional:
    - numpy
    - pandas
    - scipy

blocklist:
  # (module, attribute) pairs blocked even though the module is allowed
  - [numpy, ctypeslib]
  - [numpy, core.multiarray._reconstruct]  # pickle reconstruction
  - [pandas, read_pickle]
  - [pandas, read_hdf]
  - [pandas, io.sql]
  - [scipy, weave]

resources:
  memory: 512Mi
  cpu: "500m"
  timeout_max: 30.0

scan_stages:
  - ast_scan
  - blocklist_audit
```

### Blocklist Audit

For profiles that allow modules with dangerous subcomponents (data-science, financial), the AST allowlist check alone is insufficient. A module can be on the allowlist while specific attributes of that module remain dangerous.

The blocklist audit is a second AST pass that runs after the import allowlist check passes. It walks attribute access nodes and checks them against the profile's blocklist. Implementation: extend `_GuardrailVisitor` with a profile-aware blocklist, or run a separate lightweight visitor. The blocklist entries come from the profile YAML, not from hardcoded sets in Python.

This is a targeted extension of the current guardrails -- the same AST walking pattern, applied to a per-profile blocklist rather than the existing global blocklists.

### Container Image Strategy

Each profile produces a separate image tag. The minimal image does not carry numpy/pandas/scipy, keeping it small and reducing attack surface.

```
code-sandbox:0.4.0-minimal        (base, stdlib only)
code-sandbox:0.4.0-data-science   (+ numpy, pandas, scipy)
code-sandbox:0.4.0-financial      (+ numpy, pandas, openpyxl)
code-sandbox:0.4.0-code-analysis  (+ treeloom, tree-sitter grammars)
```

Implemented via a multi-stage Containerfile with a build arg:

```dockerfile
ARG PROFILE=minimal
COPY sandbox/profiles/${PROFILE}.yaml /app/profile.yaml
COPY sandbox/profiles/${PROFILE}-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
```

Each profile has a corresponding `<profile>-requirements.txt` in `sandbox/profiles/`.

### Helm Integration

Profile selection is a Helm value:

```yaml
sandbox:
  profile: data-science
  image:
    tag: "0.4.0-data-science"
```

The chart sets `SANDBOX_PROFILE` as an environment variable on the sidecar container and selects the matching image tag. Profile and image tag must agree -- the chart should validate this or derive one from the other.

## 2. Multi-Stage Execution Pipeline

The current execution flow in `app.py` is `validate_code() -> execute_code()`. The pipeline generalizes this to an ordered list of stages that vary by profile.

### Tiers

```
Tier 1 (minimal):       AST scan -> execute -> return
Tier 2 (data-sci/fin):  AST scan -> blocklist audit -> execute -> return
Tier 3 (code-analysis): AST scan -> treeloom CPG taint check -> execute -> output review -> return
```

### Stage Interface

Every validation stage is a callable with the same signature:

```python
def stage(code: str) -> list[str]
```

Returns a list of violation strings. Empty list means the code passed. The pipeline runner calls stages in the order defined by the profile's `scan_stages` list, short-circuiting on the first stage that returns a non-empty list.

Post-execution stages (output review) take the `ExecutionResult` instead of source code. These are a separate category with their own signature:

```python
def post_stage(code: str, result: ExecutionResult) -> list[str]
```

### Stage Descriptions

**AST scan** (existing, `guardrails.validate_code`). Fast pattern matching against the import allowlist, blocked calls, blocked dunders, credential patterns, SQL injection patterns, and path traversal. Runs in under 5ms. Present in all tiers.

**Blocklist audit** (new). Checks attribute access on allowed modules against the profile's blocklist YAML. Same AST walking pattern as the existing guardrails. Under 10ms. Present in Tier 2 and above.

**CPG taint analysis** (new, Tier 3 only). Uses treeloom to build a Code Property Graph from the submitted source and runs field-sensitive taint analysis. Catches data-flow attacks that the AST walker misses -- for example, building dangerous calls through string concatenation or aliasing blocked functions through variable assignment. Adds 100-500ms overhead. This stage is only relevant for the code-analysis profile where the submitted code has access to powerful introspection tools.

**Execute** (existing, `executor.execute_code`). Subprocess with `python3 -I`, timeout, output capping. Present in all tiers.

**Output review** (new, Tier 3 only). Post-execution check on stdout/stderr for sensitive data: API keys, tokens, internal filesystem paths, stack traces that leak deployment details. Uses the same regex patterns from the existing string literal checker in `guardrails.py`, applied to execution output rather than source code. Runs after execution because the code-analysis profile's introspection capabilities could extract sensitive information that is not visible in the source.

### Pipeline Runner

The pipeline runner replaces the inline `validate_code() -> execute_code()` sequence in `app.py`:

```python
async def run_pipeline(code: str, profile: Profile, timeout: float) -> PipelineResult:
    for stage in profile.pre_stages:
        violations = stage(code)
        if violations:
            return PipelineResult(rejected=True, violations=violations)

    result = await execute_code(code, timeout)

    for stage in profile.post_stages:
        violations = stage(code, result)
        if violations:
            return PipelineResult(rejected=True, violations=violations, partial_result=result)

    return PipelineResult(rejected=False, result=result)
```

The `/execute` endpoint delegates to `run_pipeline` instead of calling `validate_code` and `execute_code` directly.

## 3. Code Refactoring Agent Pipeline

This is a separate concern from the sandbox. It is an agent (BaseAgent subclass) that refactors code using MCP servers for analysis, execution, and compliance checking. The sandbox is one of those MCP servers.

### Tool Ecosystem

Five packages form a coherent stack for code analysis, security scanning, and compliance:

**treeloom** (v0.8.1). Language-agnostic Code Property Graph library. Parses source into unified graphs covering AST, control flow, data flow, and call graph edges. Supports Python, JS, TS, Go, Java, C, C++, and Rust. 148KB with 3 core dependencies. Foundation for sanicode and greploom.

**greploom** (v0.4.0). Semantic code search built on treeloom CPGs. Hybrid vector + BM25 search with graph-aware context retrieval and token budget management. SQLite-based, no external servers required. 26KB. Has MCP server support built in.

**sanicode** (v0.12.2). SAST scanner with 705 rules covering 109 CWEs and the MITRE Top 25. Uses treeloom CPGs for field-sensitive taint analysis. Outputs SARIF v2.1.0. Supports 23 languages. 1MB with heavyweight dependencies (FastAPI, litellm, tree-sitter grammars). Has an API server mode.

**veripak** (v0.6.2). Package health auditor. Checks version staleness, EOL status, and CVE exposure across 7 ecosystems (PyPI, npm, Maven, Go, NuGet, MetaCPAN, Packagist). Has an MCP server with deterministic-only mode. 86KB.

**stigcode** (v0.0.1, early stage). SARIF-to-compliance bridge. Consumes SARIF from any SAST tool and produces DISA STIG .ckl files, ATO evidence reports, and NIST 800-53 control matrices. Pure Python, no dependencies. Conditional stage for regulated environments (FedRAMP, DISA STIG).

### MCP Server Deployment Topology

Each tool deploys as its own MCP server. Agents connect via streamable-http (SSE is deprecated per project conventions). This keeps agents thin and tools independently scalable and updatable.

| MCP Server | Key tools exposed | Pipeline stage |
|-----------|------------------|----------------|
| greploom | `search_code`, `get_node_context`, `index_code` | Before refactoring -- understand codebase |
| code-sandbox | `execute_code` | After refactoring -- verify behavior |
| sanicode | `scan_code`, `scan_file` | After refactoring -- security check |
| veripak | `audit_package`, `audit_requirements` | After refactoring -- supply chain check |
| stigcode | `generate_checklist`, `generate_ato_evidence` | After scan -- compliance artifacts (conditional) |

### Agent Pipeline Flow

```
1. greploom    -- index codebase, retrieve context for refactoring target
2. agent       -- plan and generate refactored code
3. [parallel]
   a. sandbox  -- execute tests to verify behavior preserved
   b. sanicode -- scan refactored code for introduced vulnerabilities
   c. veripak  -- audit any new/changed dependencies
4. stigcode    -- produce compliance artifacts (conditional, regulated envs only)
5. agent       -- commit if all stages pass, report failures otherwise
```

Steps 3a-3c run in parallel since they are independent checks on the same output. The agent orchestrates these using `asyncio.gather` through its MCP client connections.

### Deployment Considerations

greploom and sanicode both depend on treeloom. A shared base image or shared CPG index could reduce redundant parsing, but this introduces coupling between their deployment lifecycles. Start with independent deployments; optimize later if the redundant parsing becomes a measurable bottleneck.

sanicode is the heaviest deployment due to tree-sitter grammars and optional LLM integration. It needs its own resource allocation and should not share a pod with lighter services.

veripak requires network access to query package registries (PyPI, npm, etc.). This is a different security posture than the sandbox, which must have no network access. They cannot share a network policy.

stigcode is pure Python, tiny, and stateless. It could be a sidecar, a library call, or a full MCP server. The MCP server approach is consistent with the rest of the stack but may be overengineered for a tool this small. Decide when stigcode reaches a stable release.

### Relationship to Agent Skills

These tools were originally considered as an agent skill. The MCP server approach is more flexible:

- Skills are baked into the agent image, creating tight coupling between the tool and the agent's build/deploy cycle.
- MCP servers are independently deployed and discoverable. Multiple agents can share the same server instances, and servers can be updated without rebuilding the agent image.
- A skill could still exist as a thin orchestration layer that knows which MCP servers to call and in what order, but the actual tools live in the servers.

## 4. Implementation Priorities

**Now (sandbox profiles).** Add profile YAML configs to `sandbox/profiles/`. Extend `guardrails.py` with blocklist audit capability that loads blocklist entries from the active profile. Add Helm values for profile selection. Build and test with the minimal and data-science profiles first.

**Soon (pipeline runner).** Refactor `app.py` to use the pipeline runner instead of inline `validate_code` / `execute_code` calls. This is a prerequisite for adding new stages but can be done with only the existing AST scan stage initially.

**Later (treeloom integration).** Add treeloom as the optional deep-scan stage for the code-analysis profile. treeloom is lightweight enough (148KB) for the sandbox image, but the tree-sitter grammars it depends on add weight.

**Later (MCP servers).** Deploy sanicode, greploom, and veripak as standalone MCP servers using the fips-agents MCP server template. Build the code refactoring agent as a BaseAgent subclass that orchestrates them through its MCP client.

**When ready (compliance).** Integrate stigcode into the pipeline for regulated environments. Depends on stigcode reaching a stable release with at minimum DISA STIG .ckl generation.

## 5. Open Questions

**Per-request vs. per-deployment profile selection.** Should profiles be selectable per-request (via the `/execute` API) or fixed at deployment time (via Helm)? Per-request is more flexible but means the sandbox image must include all profile dependencies, eliminating the attack surface reduction benefit of separate images. The current leaning is per-deployment.

**treeloom placement.** Should treeloom CPG analysis run inside the sandbox sidecar or as a separate pre-execution service? Inside is simpler (single container, no network hop) but adds weight to the sidecar image and ties treeloom's version to the sandbox's release cycle.

**Shared CPG index.** For the refactoring agent, should sanicode share a CPG index with greploom to avoid re-parsing the same codebase? This would require a shared storage layer (PGVector or shared SQLite volume) and introduces coupling between their deployment lifecycles.

**stigcode minimum viable scope.** What is the minimum feature set needed before integrating stigcode? DISA STIG .ckl generation alone, or does it need NIST 800-53 control matrices as well? This depends on which compliance frameworks are needed first by downstream consumers.
