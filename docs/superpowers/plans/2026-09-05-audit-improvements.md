# Core Audit Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Harden the headless/core project paths found during the non-GUI audit while preserving numerical, checkpoint, export, and deterministic replay contracts.

**Architecture:** Keep `xrr_fitter.api` as the public boundary. Add narrow error adapters at CLI and filesystem boundaries, then incrementally replace untyped cross-layer values with existing domain types or small local protocols. Performance changes must be benchmark-led and must not remove required full-data evidence evaluations.

**Tech Stack:** Python 3.12, NumPy/SciPy, pytest, existing repository verification tools, GitHub Actions YAML.

## Global Constraints

- Do not modify the GUI implementation or the active GUI worktree.
- Keep all work in `/Users/dala/Desktop/XRR-Fitter/audit-improvements`.
- Preserve project schema, deterministic seed lineage, checkpoints/resume, cancellation, export atomicity, and public API behavior.
- Do not add runtime dependencies, compatibility shims, silent fallbacks, or broad exception swallowing.
- Every production behavior change follows RED -> GREEN -> focused verification.
- Run relevant verification through `tools/verify.py`; keep generated environments/caches outside the repository.

### Task 1: Normalize CLI persistence failures

**Files:**
- Modify: `src/xrr_fitter/cli/commands.py:54-99`
- Test: `tests/unit/cli/test_dispatch.py`

**Interfaces:** Add a private `_save_project_or_error(project, path)` helper that calls `api.save_project()` and raises `CommandError` with `INVALID_INPUT` for the same persistence exceptions already handled by `_load()`.

- [x] Add tests proving `fit` and `mcmc` persistence failures return the input-error code and never print a traceback.
- [x] Run the focused tests and observe RED.
- [x] Implement the helper and route both commands through it.
- [x] Run CLI tests and the `unit`/`integration` gates that own CLI coverage (there is no standalone `cli` verifier mode).

### Task 2: Make export directory fsync portable

**Files:**
- Modify: `src/xrr_fitter/io/export_run.py:238-243`
- Test: `tests/unit/io/test_export_run.py`

**Interfaces:** `_sync_directory()` may suppress `EACCES`/`EPERM` only on Windows, matching `project_codec._fsync_directory()`; all other directory and all regular-file I/O errors remain fatal.

- [x] Add a parametrized test for Windows `EACCES` and `EPERM` during directory open/sync.
- [x] Run the focused test and observe RED.
- [x] Implement the narrow platform-specific handling, preferably sharing a small internal predicate without changing public API.
- [x] Run export durability tests and integration export verification.

### Task 3: Define and test domain failure boundaries

**Files:**
- Modify: `src/xrr_fitter/services/datasets.py`, `src/xrr_fitter/services/fitting_phases/operations.py`, `src/xrr_fitter/services/fitting_phases/joint_execution.py`, and only the affected batch helpers.
- Test: corresponding files under `tests/unit/services/`.

**Interfaces:** Introduce no global exception hierarchy. Read-only preflight catches only documented input/domain exceptions; cancellation and programmer errors propagate there. Batch import, isolated retry, and worker execution intentionally retain per-row/per-dataset failure isolation and must keep recording their exception identity.

- [x] Add regression tests using a sentinel `RuntimeError` to prove preflight programmer errors propagate, while existing row-level failure tests continue to prove batch isolation.
- [x] Run those tests and observe RED.
- [x] Replace only the preflight broad catches with explicit `(OSError, ValueError, TypeError, KeyError, EvaluationConstraintError)` tuples; retain documented batch/retry catches and `BaseException` worker/cleanup paths where isolation or cancellation semantics require them.
- [x] Run all services unit tests and integration workflows.

### Task 4: Tighten high-value type contracts without new dependencies

**Files:**
- Modify: `src/xrr_fitter/services/batch.py`, `src/xrr_fitter/services/fitting.py`, `src/xrr_fitter/analysis/mcmc.py`, `src/xrr_fitter/evaluation_parameters.py`, and nearby model imports.
- Test: existing unit tests plus a new focused type-contract test only where runtime validation is needed.

**Interfaces:** Replace `object` at public/internal seams with existing dataclasses/protocols and precise mappings; do not alter serialized schemas or add a runtime type-checking dependency.

- [x] Inventory the first seam in each module, retain existing invalid-input tests, and characterize the narrowed type contracts.
- [x] Run the focused tests and observe RED where the current contract is ambiguous.
- [x] Narrow annotations and validation incrementally, keeping runtime behavior unchanged for valid inputs.
- [x] Run quality and all non-GUI unit tests.

### Task 5: Add benchmark coverage before performance changes

**Files:**
- Modify: `tools/benchmark_automatic_fit.py`, `tools/verify_registry.py`, and benchmark tests if needed.
- Test: `tests/unit/tools/test_benchmark_automatic_fit.py` or a new non-GUI benchmark contract test.

**Interfaces:** Record single-dataset, multi-dataset/adaptive, wall time, evaluation count, and peak memory where the platform supports it. Existing deterministic benchmark output remains valid.

- [x] Add deterministic benchmark cases and assertions for output schema and replay identity.
- [x] Run the benchmark contract tests and observe RED.
- [x] Implement only the measurement additions first.
- [x] Profile MCMC mapping and stage publication; optimize only if a measured hotspot justifies it.
- [x] Re-run numerical/regression suites and compare benchmark baselines.

### Task 6: Reduce maintainability risk by ownership-based extraction

**Files:**
- Modify one bounded slice at a time from `fit/stages.py`, `services/batch.py`, and `analysis/profiles.py`.
- Test: migrate the directly owned tests with each slice.

**Interfaces:** Keep existing import paths internal and preserve `xrr_fitter.api`; no forwarding-only compatibility modules. Extract by data flow/ownership, not arbitrary line counts.

- [x] Select one cohesive helper cluster with a dependency map and characterization tests.
- [x] Run the characterization tests before extraction.
- [x] Extract the cluster, update imports, and remove duplicate definitions.
- [x] Run Radon and the affected unit/integration tests before selecting the next cluster.

### Task 7: Improve CI and release supply-chain evidence

**Files:**
- Modify: `.github/workflows/verify.yml`, `.github/workflows/pr-verify.yml`, `pyproject.toml`, lock-generation/release tools, and documentation as required.
- Test: `tests/architecture/` and `tests/unit/tools/` for workflow and manifest contracts.

**Interfaces:** Keep platform-specific lock files and runner contracts. Add coverage/type/dependency evidence as reporting or incremental gates only after tool availability is explicit; do not silently broaden dependency resolution.

- [x] Add tests for shared setup and existing artifact identity invariants; defer shared cache until a separate trust/rollout decision.
- [x] Run architecture tests and observe RED.
- [x] Refactor duplicated setup through a local composite action, preserve existing artifact hashes, add deterministic offline lock SBOM generation, and document platform support and evidence limits.
- [x] Run clean-tree quality/tools verification and inspect generated release evidence.

## Final Verification

- [x] Run focused CLI/export tests.
- [x] Run non-GUI unit, integration, spawn, regression, tools, quality, and Radon gates from a clean external-cache environment; CLI is covered by unit/integration.
- [x] Run the automatic benchmark and compare deterministic output.
- [x] Confirm no files under `src/xrr_fitter/gui` changed and the active GUI worktree is untouched.
- [x] Review `git diff --check`, worktree status, and generated-artifact hygiene.

## Execution record — 2026-09-07

The completed checklist covers the no-new-dependency audit slice, not a release
approval or completion of the separately deferred rollouts below. Work remains
isolated on `audit-improvements`; no merge, push, tag, or release was performed.

**Overall status: incomplete.** Implemented audit changes are verified, but the
open verification and approval-bound items below are not counted as completed.

| Slice | Implementation commits | Outcome |
| --- | --- | --- |
| CLI persistence / Windows directory sync | `b3475dc`, `59a5324` | Narrow error adapters; regular-file I/O errors remain fatal. |
| Preflight failure boundaries | `bd9d4fd`, `9a27c28` | Expected `EvaluationConstraintError` remains a domain failure; programmer errors propagate. |
| Domain type seams | `8d88c66`, `fa5d33c` | MCMC/batch/evaluation contracts; private frozen sharing view; no serialized schema change. |
| Measurement / ownership extraction | `3dfa806`, `c1b57fc`, `791ca93`, `ad86e4a` | Process peak RSS, profile selection, batch routing and stage schedule ownership; existing bindings retained. |
| CI / dependency evidence | `0cf7cd5`, `951ad8d`, `76e4606`, `2ea1552` | Shared locked setup, fail-fast pip metadata check, registered action tests, offline CycloneDX inventory. |

The type-only slices preserve existing runtime validation instead of adding a
new validator. Benchmark/profile measurements did not justify another numerical
implementation, cache, or reduced full-data publication evidence. Benchmark
schema is now `xrr-automatic-benchmark-v2`; external consumers must allow the new
process high-watermark field. It is not per-case allocated memory.

### Verification evidence

- Clean-checkout implementation `76e4606`: quality **199**, unit **1704**,
  integration **14**, spawn **4**, regression **50** tests passed; quality also
  passed the unchanged Radon policy. GUI and the full 220-case statistical
  release corpus were not run in that initial verification slice.
- The tools gate exposed an omitted second exact-registry expectation:
  **1 failed / 467 passed**. `2ea1552` fixes that test expectation without changing
  production code; focused verifier tests passed **25**, and the full clean
  tools gate passed **468** tests at `2ea1552`.
- Single, batch-of-four and adaptive benchmarks each repeated twice: every
  deterministic per-run field matches both `c369ed0` and `ad86e4a`. Concurrent
  gate timings are not evidence of a speedup; adaptive outlier `review` remains
  an expected outcome.
- macOS **45** and Windows **38** locked components passed the official
  CycloneDX 1.6 JSON schema, cross-checkout byte replay and LF-only serialization
  checks. Installed environment `pip check` passed.
- `tools/verify_distribution.py` independently passed reproducible wheel/sdist
  builds and installed-artifact smoke on `76e4606`, retaining its existing
  two-artifact hash manifest. This is not a full distribution/release gate pass.
- Core and SBOM reviews found no blocking defects. All reports, profiles,
  generated inventories and build artifacts remain outside the repository.
  Tool-owned temporary benchmark/build environments are automatically removed.

### Toolchain follow-up — 2026-09-08

`c5b3ea0` upgrades the existing pytest/setuptools/wheel tools after the advisory
baseline. `276135a` tracks the new wheel license path with exact member and byte
checks; `a6f74a0` rebinds the test collection to that source (3095 nodes). The
application source, numerical inputs and verification gate count are unchanged.

At `a6f74a0`, quality **200**, tools **472**, unit **1704**, integration **14**,
spawn **4** and regression **50** passed (**2444 tests**). Full `distribution`
and local `identity` gates passed for the same artifact bundle. These records
bind that exact commit, not subsequent documentation-only commits. The
continuation revalidated installed lock versions, `pip check`, artifact identity
and clean delivery-checkout hygiene. A fresh `quality` rerun passed all 200 tests
and Radon at `a6f74a0`.

The complete statistical gate passed on fixed `c626d62`: **2 tests, 220 cases**,
**9472.87 seconds (2:37:52)**, verifier exit code **0**. Its application source,
corpus/support code, verifier/outcome code and macOS lock match `a6f74a0` exactly.
The previous run's incomplete supervisor record is retained separately and is
not used as evidence of a successful verifier exit.

Coverage, static-type and advisory measurements are recorded in
`docs/architecture/audit-reporting-baseline.md`. They are reporting-only evidence,
not permanent tooling dependencies or additional CI gates. See
`2026-09-08-audit-toolchain-security.md` for the migration and statistical status.

### Open verification and approval-bound work

- [x] **Resolve the failing distribution gate:** `a6a3e26` refreshes the existing
  `fonttools 4.63.0 -> 4.64.0`, `kiwisolver 1.5.0 -> 1.5.1` and Windows-only
  `pyinstaller-hooks-contrib 2026.6 -> 2026.7` pins. The shared-package contract
  requires the first two updates in both platform locks. Package sets,
  `pyproject.toml`, resolver policy and gates remain unchanged. The complete
  `distribution` gate passed on `5a47af1`, including fresh Windows resolution,
  reproducible wheel/sdist builds and installed-artifact smoke.
- [x] Rebind and verify local release identity. `5a47af1` refreshes the collected
  test manifest (3090 nodes, source `a6a3e26`) and lock binding; the existing
  release-spec generator changes only `lock_sha256`. Test collection reproduced
  identical bytes from two checkouts. The `identity` gate built and validated
  the same `5a47af1` distribution bundle. This is not tag/release approval and
  test collection does not establish test execution success.
- [x] Run the complete 220-case statistical corpus at the audit implementation
  revision and retain its result. The fixed `c626d62` rerun passed both tests in
  9472.87 seconds, with verifier exit code 0. Its source/lock binding and actual
  tool exit result are retained in the continuation evidence directory below.
- [ ] Decide the additional tooling scope and implement it after approval.
  Shared cache, coverage/static typing/vulnerability-scanner dependencies or
  gates, complete native/artifact SBOM and extra release assets remain deferred
  pending approval. External reporting baselines and the three existing tool
  security upgrades are complete; this does not enable permanent gates or fix
  the remaining type/coverage debt. See `docs/architecture/dependency-evidence.md`
  for tradeoffs.
- [ ] Complete target-platform verification. Actual GitHub Actions execution,
  Windows runtime/executable validation and annotated-tag release acceptance
  remain unverified. Local YAML/bash tests do not replace target-runner validation.
- [x] Review overlapping changes before integration. Read-only comparison against
  GUI HEAD `6e44db4` and its active changes found 13 shared non-GUI paths. The
  relevant contracts are MCMC/progress callback arity, stage skip/resume behavior,
  `ParameterFreedom` constructors, profile metadata and export format selection.
  Keep these GUI-side contracts together with the audit's type annotations,
  error boundaries and extracted owners; selecting either entire file is unsafe.
  No merge or GUI edit was performed. The combined tree still needs a separate
  integration run after reconciliation.

Continuation evidence is retained at
`/private/tmp/xrr-audit-resume-20260908.5ZafEx`: `statistical-result.json`,
`statistical-invocation.json`, `statistical-exec-results.jsonl`,
`fresh-validation.json`, `quality-result.json` and `quality.log`.
The old incomplete statistical evidence remains at
`/private/tmp/xrr-audit-secure-statistical-20260908`; neither record was overwritten.
