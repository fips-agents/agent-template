# Retrospective: File Upload Track Shipped (#100)

**Date:** 2026-04-29
**Effort:** Two-session arc (2026-04-28 + 2026-04-29) — design, implement, merge, release, and cluster-smoke the file-upload track end-to-end across five repos.
**Issues:** #100 (umbrella), #126/#127/#128/#129/#130 (agent-template), gateway #29/#32/#33/#34, ui #24, examples #24, fips-agents-cli #19
**Commits:** `e641fc9..fd2c413` on agent-template main; `1e984ae..ccf12f6` on gateway-template main; plus PR #130 on `feat/files-bytes-pvc-and-docs` (commits `da7cc9b`, `97d820d`)
**Releases:** `fipsagents v0.16.0` (PyPI), `gateway-template v0.6.0`, GH Release pages backfilled for v0.14.0 / v0.14.1 / v0.14.2 / v0.15.0

## What We Set Out To Do

Issue #100 framed the file-upload track as: `POST /v1/files` + Docling parsing + pluggable `FileStore` (MinIO bytes / Postgres metadata) + ChatCompletion `file_ids` injection + security (MIME, ClamAV, UUID storage) + chunking/pgvector for documents that exceed the model context window. Companion work in gateway, UI, examples, and CLI.

The first session opened six PRs across five repos (no merges). The second session merged everything, cut releases, validated end-to-end on RHPDS, and bundled the smoke-surfaced gaps into a single follow-up PR.

## What Changed

| Change | Type | Rationale |
|---|---|---|
| S3-compatible bytes backend deferred → ADR-0001 | Good pivot | Two-option design (BytesStore ABC vs variant classes) deserved a written commit before code. ADR landed in PR #129 as docs-only — Module 9 already documents MinIO as future state, so the ADR closes the design loop without forcing implementation onto the critical path. |
| Chunking + pgvector deferred entirely | Scope deferral | Issue called this out as "design first" explicitly. Cluster has a deployed embedding model; pgvector backend already works for memory; integration is "another consumer of the same pgvector deployment." Gets its own ADR next. |
| Chart `files.persistence` PVC missing for `bytes_dir` | Missed requirement | Chart wired a PVC for ClamAV's signature DB but not for the agent's own bytes directory. Surfaced by today's cluster smoke when uploads vanished after pod restart would have happened. Fixed in PR #130. |
| `agent.yaml` `bytes_dir` was hard-coded | Missed requirement | Should have been `${FILES_BYTES_DIR:-./files}` from the start so the chart's PVC wiring takes effect without a rebuild. Fixed in PR #130 alongside the persistence block. |
| `oc new-build --binary --strategy=docker` defaults to `Dockerfile` | Friction surfaced | The build context ships `Containerfile` (Podman convention). Required a `dockerfilePath` patch on the BuildConfig before the build would start. Documented in PR #130's README for ad-hoc binary builds. |
| `[files]` extra adds ~5–6 GB (Docling pulls torch + transformers) | Acknowledged cost | Documented in README image-size table. Operators that only ingest text/markdown/JSON can skip the extra — `PlaintextParser` ships in core. |
| gateway-template#33 needed rebase after #32 merged | Minor friction | Both PRs added an `envDurationDefault` helper with different validation semantics (`> 0` vs `>= 0`). Resolved by renaming one to `envDurationDefaultAllowZero`. |
| HttpFileStore deferred to platform readiness | Scope deferral | Currently raises `NotImplementedError`. Requires `/v1/files` surface on `fipsagents-platform`. Two-repo PR. Defer until platform routing is genuinely needed. |

## What Went Well

- **Defer-until-designed discipline held.** Both deferred tracks (S3 BytesStore, chunking + pgvector) got design space rather than half-implementations. ADR-0001 commits to option (a) — `BytesStore` ABC composition — with a backward-compat shim that keeps 0.16.0 deployments working unchanged. The implementation PR will be small and focused because the design conversation already happened.
- **Cluster smoke caught what unit tests didn't.** The `bytes_dir` PVC gap, the `dockerfilePath: Containerfile` BuildConfig default, the `agent_class` constructor signature mismatch in the smoke build's `src/agent.py` — all surfaced only on real OpenShift, not in 1005 passing unit tests. Reinforces the project preference for cluster smokes over local-only validation.
- **NEXT_SESSION.md handoff was load-bearing.** Yesterday's session left a precise punch list — six PR numbers, branch names, exact follow-up scope, recommended next move. Today started in motion: zero discovery overhead before merging.
- **Cross-repo coordination via PR sequencing.** Six PRs across five repos merged cleanly in 30 minutes including the rebase. Most PRs were independent; the one rebase (gw#33 after gw#32) was a small naming conflict, not a design conflict.
- **Release hygiene caught up.** Four GH Release pages backfilled for prior tags + v0.16.0 created with rich notes. `publish.yml` ran clean (5m38s). PyPI propagated within ~30s of the workflow finishing.
- **`/session-close` caught real things.** Pre-existing local-env test brittleness (`parse_status: skipped` vs `failed`), missing chart bump on PR #130, stale `NEXT_SESSION.md` — all flagged before context was lost.

## Gaps Identified

| Gap | Severity | Resolution |
|---|---|---|
| Chart was missing `bytes_dir` PVC | Fixed in PR #130 | New `files.persistence` block, mirrors ClamAV pattern, sets `FILES_BYTES_DIR` env. Chart bumped to 0.7.0. |
| `agent.yaml` `bytes_dir` hard-coded | Fixed in PR #130 | Now uses `${FILES_BYTES_DIR:-./files}`. |
| Image size cost not in scaffolded README | Fixed in PR #130 | README image-size table; CLAUDE.md File Uploads note. |
| `oc new-build --binary` `Containerfile` gotcha undocumented | Fixed in PR #130 | README section for ad-hoc binary builds. |
| `test_inject_unparsed_file_emits_stub` brittle to local Docling state | Fixed in PR #130 (3rd commit) | Assertion accepts `parse_status: skipped` OR `failed`. CI without `[files]` extra → skipped; dev with extra → failed; same stub-emission contract verified either way. |
| `.gitleaks.toml` allowlist appears not to apply when run from subdir | Fixed in PR #130 (3rd commit) | Header comment documents that gitleaks must be invoked from repo root or with `--config`. False alarm during `/session-close` confirmed the config works correctly when invoked properly. |
| Smoke ran with `scanner.enabled=false` | Follow-up | ClamAV sidecar end-to-end path (HTTP shim + `{infected, viruses}` contract) untested in real cluster. Tracked in NEXT_SESSION's "Other tracks". |
| HttpFileStore raises NotImplementedError | Accept | Platform-routed backend deferred until platform `/v1/files` exists. Documented in 0.16.0 release notes. |
| ClamAV reference image lacks `{infected, viruses}` JSON shim | Follow-up | `clamav/clamav:stable` exposes clamd on TCP 3310 only. Operators need to wrap with FastAPI shim or swap in their org image. Sidecar Containerfile under `agent-template/sidecars/clamav-shim/` would close the gap. |
| 1006 → 1005 test count delta from prior session | Accepted (env-only) | Local-env Docling brittleness now tolerated; CI count unchanged. |

## Action Items

- [ ] Merge PR #130 once reviewed
- [ ] Re-smoke `fipsagents-files-smoke` namespace with `--set files.persistence.enabled=true`, force pod restart, verify uploaded bytes survive
- [ ] File implementation tracking issue: "feat(server): S3-compatible BytesStore (one bullet of #100)" — design is in ADR-0001
- [ ] Draft chunking + pgvector retrieval ADR (the third design-first bullet on #100)
- [ ] Follow-up: deploy a real ClamAV sidecar in the smoke namespace to exercise the virus-scan path
- [ ] Follow-up: contribute the FastAPI/clamd shim back as a sibling reference image (or `agent-template/sidecars/clamav-shim/`)

## Patterns

**Start:**
- When adding a new persistence concern (file bytes, etc.), check the chart for a corresponding PVC in the same PR. The chart had ClamAV's PVC but not the agent's `bytes_dir` — the asymmetry was the bug. Symmetric audit ("if it persists, where does it persist on the chart?") would have caught it before smoke.
- When the build context uses `Containerfile`, set `dockerfilePath` on every `oc new-build --strategy=docker` BuildConfig immediately rather than discovering the failure on first build. README now documents this for the ad-hoc case; the chart's BuildConfig template already handles it.

**Stop:** Nothing new.

**Continue:**
- **Defer-until-designed discipline.** Both S3 BytesStore and chunking+pgvector were correctly held back behind ADRs. ADR-0001 closes one design loop with a concrete backward-compat shim and a half-page implementation sketch — the eventual PR will be smaller and faster because the design conversation already happened. Project-stated preference; this session validated it again.
- **Cluster smoke as default.** The `bytes_dir` PVC gap, the `dockerfilePath` build gotcha, and the `agent_class` constructor mismatch all surfaced only on real OpenShift. Project-stated preference; this session validated it again.
- **NEXT_SESSION.md with PR numbers + branch names + recommended next move.** Eleven sessions and counting; zero wasted discovery time at session start.
- **`/session-close` checklist.** Caught the env-tolerant test gap, missing chart bump, stale NEXT_SESSION.md, and the gitleaks-cwd false alarm — all before context was lost.
- **Release PRs as a separate commit.** v0.16.0 went through PR #128 (release-only) rather than direct push to main. Clean history, traceable bump, separate review surface from feature work.
