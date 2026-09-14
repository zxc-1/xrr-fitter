# Fitting v2 / Current Main Integration Plan

> **For agentic workers:** Use `executing-plans` / `subagent-driven-development`; honor the user's same-model/same-effort requirement. Only the controller changes Git state. The checkboxes below describe this integration, not the already completed scientific experiments.

**Goal:** Integrate PR #28 with current `main`, retain both branches' behavior, and merge only after every PR check succeeds.

**Architecture:** Keep `main`'s current GUI/native lifecycle, parameter freedom, stage controls and verification infrastructure. Carry fitting v2's numerical/inference contracts into those entry points; share one model representation and explicit keyword arguments at changed interfaces. Resolve textual conflicts before diagnosing semantic failures, without choosing one side wholesale.

**Tech Stack:** Python 3.12, NumPy/SciPy, Qt/PySide6 6.11.2 with the repository Cocoa patch, pytest, Ruff, Radon and repository verification commands.

**Pre-commit checkpoint:** Implementation and task-scoped reviews are complete, including the three GUI review fixes. The remaining unchecked items are post-commit verification and publication gates; record their actual exact-head results in PR #28 and the external integration receipts rather than predicting success in this snapshot.

## Global Constraints

- Feature baseline: `a49c43af5644cedcb72c9cb53ab52081e476dfc7`; integration base: `66d0f2ccca1a431a57e0c82b697f280b93ca343a`.
- Schema 5 / `xrr-fit-v2-poisson-5`, objective string `"2"`, diagnostic v4 / `poisson_refit_null_rms_v4`, refit policy `declared_sobol4_lbfgsb_trf_v3` remain in force.
- Keep B999, alpha .01, four starts, each path's budget 80, bootstrap success >= 200 and failure rate <= .20. Do not weaken statistical, quality, type, coverage, or CI gates.
- Keep joint bootstrap owner/content seals and joint-only `poisson_joint_content_tolerance_v1` (.95 nominal, .96 content, .95 MC assurance); no memberwise single-bootstrap substitutes.
- Existing sealed scientific experiments and audits are immutable. They are not rerun or represented as new integrated-source validation.
- `xrr_fitter.api` remains the only supported API; follow `docs/architecture/r23-clean-break.md`. No compatibility layers, duplicate profile classes or silent fallbacks.
- Preserve all upstream test/CI/dependency improvements, without inventing new production dependencies or changing deployment defaults.
- Work only on the feature worktree. Do not force push, bypass checks, alter branch protections, delete branches/worktrees/evidence, tag, or release.
- Every behavioral repair needs a reproduced failing test, a minimal change and a passing regression. Preserve original failures in external integration evidence.

## Task 1: Record baseline and cross-branch RED tests

**Files:** `tests/unit/services/test_main_integration.py`; external publication `integration-001/` evidence.

- [x] Verify clean feature HEAD, fetch main and record exact commits/merge preflight.
- [x] Add four concrete public-contract tests: theta with Poisson zero counts; RANGE_ONLY not locked; profile threshold with total-likelihood delta; actual bootstrap attempts including a failed refit.
- [x] Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../venvs/repo/bin/python -m pytest -o addopts= --import-mode=importlib -p no:cacheprovider -q tests/unit/services/test_main_integration.py`. Record four assertion failures before production edits (`integration-001/public-contract-red-001.log`).
- [x] Merge with `git merge --no-commit --no-ff origin/main`, retain both parents and the conflict snapshot. Do not commit until the combined tree is tested.

## Task 2: Model and import boundary (controller)

**Files:** `src/xrr_fitter/model/{analysis,data,fitting,profile,parameters,provenance}.py`, `src/xrr_fitter/io/{xy,project_codec,codec_results,codec_inference,codec_declarations}.py`, `src/xrr_fitter/services/{datasets,projects,bootstrap_ownership}.py`, `src/xrr_fitter/api.py`; corresponding model/io/services tests.

**Interfaces:** Retain `ParameterFreedom` from main. Import functions expose both `angle_convention="two_theta"` and `noise_model="robust_log"`; all internal calls pass them by name so neither can occupy the other's positional slot. `read_xy_bytes` keeps both keyword-only. Carry both in source rereads and saved prepared data.

```python
data = api.import_data(path, api.BeamSpec("monochromatic"), angle_convention="theta", noise_model="poisson")
assert data.angle_convention == "theta"
assert data.intensity_raw[0] == 0
setting = api.ParameterSetting("instrument.scale", 1.0, 0.5, 2.0, api.ParameterFreedom.RANGE_ONLY)
assert not setting.locked
```

- [x] Retain skip normalization, `skipped_stages` and `terminated_early` across checkpoint/search/result and provenance.
- [x] Keep one `ParameterProfile` in `model/profile.py`; append finite optional `objective_threshold` while retaining `delta_total`, `objective_point_count`, LR/support/confidence metadata.
- [x] Derive report `bootstrap_sample_count` from `bootstrap_evidence.attempted_count` (zero without evidence), not a configured budget or separately asserted counter.
- [x] Merge angle/noise preparation and persisted source identity; update current `ParameterSetting` call sites from booleans to the enum without conflating RANGE_ONLY with FIXED.
- [x] Run the four new contract tests and focused io/model tests; then document the immutable interfaces to the numerical and GUI tasks. Core review: `integration-001/task-core-review-report-001.md`, spec/quality PASS.

## Task 3: Numerical stage controls (bounded worker)

**Files:** `src/xrr_fitter/fit/` and `tests/unit/fit/`; no model, analysis, service, GUI, Git or dependency edits by this worker.

**Interfaces:** Main local callback `(value, iteration, nfev, step)` and global callback `(xk, objective, generation, nfev)`; model fields from Task 2. `stage_schedule.committed_parent_summary` selects actual committed parents after legal skips.

- [x] Preserve feature adaptive grids, declared-baseline continuation, budgets, diagnostic refit and Poisson local `ftol=None` / `fit_only=True`.
- [x] Preserve main progress metrics, pause/skip cancellation probes and typed stage scheduling.
- [x] Reproduce and repair stage-skip, resume, budget and callback regressions. A/B/E skips terminate incomplete searches; C/D skips may continue to the final ensemble.
- [x] Run `tests/unit/fit/test_stage_skip.py`, `test_stage_skip_contracts.py`, `test_checkpoint.py`, adaptive/search evidence and Poisson solver tests. Supply the exact diff and RED/GREEN report for independent review. Targeted 454 passed; test-only scheduling relocation then 29 passed and 110-file scoped Radon PASS. Independent review: `integration-001/task-fit-review-report-001.md`, spec/quality PASS; 46 tests and five focused probes passed against the same 110-file snapshot.

## Task 4: Analysis and service orchestration (controller, after core interfaces)

**Files:** `src/xrr_fitter/analysis/{profile_tasks,profile_selection,profiles,report}.py`, `src/xrr_fitter/services/fitting.py`, `services/fitting_phases/`, `services/{batch,batch_routing,batch_publication,exports}.py`; corresponding analysis/services tests.

- [x] Retain main's typed profile selection and stage control integration, plus feature rescue solver and calibrated residual owners.
- [x] Publish the actual threshold used for scan closure, not an estimate reconstructed from sampled minima. Preserve total-objective threshold scaling.
- [x] Keep terminated searches UNTRUSTED, without uncertainty or automatic retry; preserve cooperative cancellation through every diagnostic and joint-bootstrap operation.
- [x] Keep the batch module split and export contract changes from main. Carry current source imports, hash checks and noise model explicitly through batch rereads.
- [x] Run focused profile, joint bootstrap ownership/persistence, automatic-skip and service cancellation tests; repair each reproduced failure without weakening assertions. Core review spec/quality PASS; 17 real export integration cases and 18 profile semantic cases passed.

## Task 5: GUI and current evidence adapters (bounded worker after numerical worker stops)

**Files:** `src/xrr_fitter/gui/`, `tests/gui/` and plot fixtures; no changes to core numerical/model contracts.

- [x] Keep main's navigation, inspector, native lifecycle, plot scheduling and pause/skip/stop controls.
- [x] Carry feature noise selector, raw-count display, unavailable/rejected calibrated diagnostics and interval metadata into current layouts.
- [x] Use the actual profile threshold/bootstrap attempt interfaces, retaining both count and angle selections on import.
- [x] Update test fixtures to current valid bootstrap evidence and enum declarations; keep both branch regression assertions.
- [x] Repair all three independent GUI findings with genuine RED→GREEN: unavailable covariance is withheld across views; absent Bootstrap is not an observed 0% failure rate; an open dialog follows current expert depth while preserving live cancellation. Full GUI 1603 passed; independent fix review spec/quality PASS, 58 tests and original-input probes passed. Evidence: `integration-001/task-gui-fix-report-001.md` and `integration-001/task-gui-fix-review-report-001.md`.
- [ ] Run `python tools/verify.py gui` from a clean normal clone of the committed integration.

## Task 6: Integrated validation and review

**Files:** `tools/verify_registry.py`, corresponding registry tests, current example files only if canonical regeneration is needed. Reports and environments stay outside the worktree.

- [x] Preserve both branches' integration/regression registry entries, with each test owned once. Keep main's type policies and CI definitions.
- [x] Use main's official owned macOS environment setup and existing locked audit dependencies; no bare Qt replacement or lock changes to obtain green results.
- [x] Run targeted tests first, Ruff without writing a checkout cache and complete Radon. Final pre-commit AST/Ruff/format/Radon cover 819 Python files with no source drift; four GUI integration workflows add nine passing cases. Evidence: `integration-001/merged-final-precommit-001/` and `integration-001/gui-integration-current-002.execution.json`.
- [ ] Run the seven modes in a clean ordinary clone with exact committed source-byte binding.
- [ ] Run coverage/typing and the repository's existing audit workflows. Do not treat stale release test manifests as an authorization to loosen ordinary PR checks or claim a release.
- [x] Independently review task diffs against both parents and fix all material findings. Core, fit, core quality-only supplement and GUI fix reviews pass; source hashes still match their reviewed snapshots.
- [ ] Independently review the combined committed whole-branch delta against both parents; fix any new material findings and reverify changed paths.

## Task 7: Publish and merge

- [ ] Commit the reviewed integration with both parents, push normally to PR #28, update its body with exact validation revisions.
- [ ] Wait for all 12 PR checks: seven modes + checkpoint, coverage, typing, advisories and Windows headless. Empty or skipped required results are not green.
- [ ] Recheck exact head/base and mergeability, then run `gh pr merge 28 --repo zxc-1/xrr-fitter --merge --match-head-commit "$HEAD"` only after green. No admin override or branch deletion.
- [ ] Confirm GitHub MERGED state and merge SHA. Retain worktree/branches/evidence and report results, cleanup and remaining limitations.
