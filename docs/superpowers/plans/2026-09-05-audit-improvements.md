# Core Audit Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

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

- [ ] Add tests proving `fit` and `mcmc` persistence failures return the input-error code and never print a traceback.
- [ ] Run the focused tests and observe RED.
- [ ] Implement the helper and route both commands through it.
- [ ] Run CLI tests and the non-GUI CLI verification.

### Task 2: Make export directory fsync portable

**Files:**
- Modify: `src/xrr_fitter/io/export_run.py:238-243`
- Test: `tests/unit/io/test_export_run.py`

**Interfaces:** `_sync_directory()` may suppress `EACCES`/`EPERM` only on Windows, matching `project_codec._fsync_directory()`; all other directory and all regular-file I/O errors remain fatal.

- [ ] Add a parametrized test for Windows `EACCES` and `EPERM` during directory open/sync.
- [ ] Run the focused test and observe RED.
- [ ] Implement the narrow platform-specific handling, preferably sharing a small internal predicate without changing public API.
- [ ] Run export durability tests and integration export verification.

### Task 3: Define and test domain failure boundaries

**Files:**
- Modify: `src/xrr_fitter/services/datasets.py`, `src/xrr_fitter/services/fitting_phases/operations.py`, `src/xrr_fitter/services/fitting_phases/joint_execution.py`, and only the affected batch helpers.
- Test: corresponding files under `tests/unit/services/`.

**Interfaces:** Introduce no global exception hierarchy. Read-only preflight catches only documented input/domain exceptions; cancellation and programmer errors propagate there. Batch import, isolated retry, and worker execution intentionally retain per-row/per-dataset failure isolation and must keep recording their exception identity.

- [ ] Add regression tests using a sentinel `RuntimeError` to prove preflight programmer errors propagate, while existing row-level failure tests continue to prove batch isolation.
- [ ] Run those tests and observe RED.
- [ ] Replace only the preflight broad catches with explicit `(OSError, ValueError, TypeError, KeyError)` tuples; retain documented batch/retry catches and `BaseException` worker/cleanup paths where isolation or cancellation semantics require them.
- [ ] Run all services unit tests and integration workflows.

### Task 4: Tighten high-value type contracts without new dependencies

**Files:**
- Modify: `src/xrr_fitter/services/batch.py`, `src/xrr_fitter/services/fitting.py`, `src/xrr_fitter/analysis/mcmc.py`, `src/xrr_fitter/evaluation_parameters.py`, and nearby model imports.
- Test: existing unit tests plus a new focused type-contract test only where runtime validation is needed.

**Interfaces:** Replace `object` at public/internal seams with existing dataclasses/protocols and precise mappings; do not alter serialized schemas or add a runtime type-checking dependency.

- [ ] Inventory the first seam in each module and write tests for invalid values at that seam.
- [ ] Run the focused tests and observe RED where the current contract is ambiguous.
- [ ] Narrow annotations and validation incrementally, keeping runtime behavior unchanged for valid inputs.
- [ ] Run quality and all non-GUI unit tests.

### Task 5: Add benchmark coverage before performance changes

**Files:**
- Modify: `tools/benchmark_automatic_fit.py`, `tools/verify_registry.py`, and benchmark tests if needed.
- Test: `tests/unit/tools/test_benchmark_automatic_fit.py` or a new non-GUI benchmark contract test.

**Interfaces:** Record single-dataset, multi-dataset/adaptive, wall time, evaluation count, and peak memory where the platform supports it. Existing deterministic benchmark output remains valid.

- [ ] Add deterministic benchmark cases and assertions for output schema and replay identity.
- [ ] Run the benchmark contract tests and observe RED.
- [ ] Implement only the measurement additions first.
- [ ] Profile MCMC mapping and stage publication; optimize only if a measured hotspot justifies it.
- [ ] Re-run numerical/regression suites and compare benchmark baselines.

### Task 6: Reduce maintainability risk by ownership-based extraction

**Files:**
- Modify one bounded slice at a time from `fit/stages.py`, `services/batch.py`, and `analysis/profiles.py`.
- Test: migrate the directly owned tests with each slice.

**Interfaces:** Keep existing import paths internal and preserve `xrr_fitter.api`; no forwarding-only compatibility modules. Extract by data flow/ownership, not arbitrary line counts.

- [ ] Select one cohesive helper cluster with a dependency map and characterization tests.
- [ ] Run the characterization tests before extraction.
- [ ] Extract the cluster, update imports, and remove duplicate definitions.
- [ ] Run Radon and the affected unit/integration tests before selecting the next cluster.

### Task 7: Improve CI and release supply-chain evidence

**Files:**
- Modify: `.github/workflows/verify.yml`, `.github/workflows/pr-verify.yml`, `pyproject.toml`, lock-generation/release tools, and documentation as required.
- Test: `tests/architecture/` and `tests/unit/tools/` for workflow and manifest contracts.

**Interfaces:** Keep platform-specific lock files and runner contracts. Add coverage/type/dependency evidence as reporting or incremental gates only after tool availability is explicit; do not silently broaden dependency resolution.

- [ ] Add tests for shared setup/cache invariants and artifact identity requirements.
- [ ] Run architecture tests and observe RED.
- [ ] Refactor duplicated setup through supported workflow mechanisms, add reproducible artifact hashes/SBOM generation, and document platform support.
- [ ] Run clean-tree quality/tools verification and inspect generated release evidence.

## Final Verification

- [ ] Run focused CLI/export tests.
- [ ] Run non-GUI unit, integration, regression, CLI, quality, and Radon gates from a clean external-cache environment.
- [ ] Run the automatic benchmark and compare deterministic output.
- [ ] Confirm no files under `src/xrr_fitter/gui` changed and the active GUI worktree is untouched.
- [ ] Review `git diff --check`, worktree status, and generated-artifact hygiene.
