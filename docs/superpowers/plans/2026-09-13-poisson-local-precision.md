# Poisson Local Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for this tightly coupled numeric change; request independent read-only review afterward.

**Goal:** Remove relative-total-cost early stopping from Poisson local refinement without weakening the existing absolute observed/product agreement limits.

**Architecture:** Keep the current analytic bounded TRF estimator, starts, loss and per-path budgets. Disable only `ftol` for Poisson-containing objectives; preserve `xtol=1e-10`, `gtol=1e-10`, failure and cancellation semantics. This partial repair does not claim to resolve the original low-count Gauss-Newton slow convergence or ACF power.

**Tech Stack:** Existing Python 3.12 / NumPy / SciPy / pytest; no new dependencies.

## Global Constraints

- Preserve all existing uncommitted work; no commit, merge, push or release.
- Do not change likelihood/deviance, bounds, budget, `B=999`, `alpha=.01`, or agreement limits `(1e-4,1e-6,1e-6)`.
- Do not adopt the failed L-BFGS-B, TNC or trust-constr probes, or turn unsuccessful optimizer results into success.
- Development seed37 is not independent acceptance; the old 4400-case FAIL remains.
- Evidence root: `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913`.
- This is an intermediate uncommitted numerical revision; no new scientific registration or version freeze until all approved changes and strict version binding are complete. No old-file compatibility path.

## Evidence / bounded design

`development/footprint-37-ftol-none/report.json` and `development/surface-37-ftol-none/report.json` use the original v1 implementation with only relative-cost stopping disabled in an isolated process. Both complete all999 null refits and target p=.001. Large irreducible Q (about1.83e8/4.98e6) caused relative ftol to stop at points whose tiny parameter differences nevertheless violated the unchanged absolute Q/KL limits. Four full L-BFGS-B combined probes failed on other null rows; do not substitute that algorithm based on the two original bad-path successes.

### Task N1: Product single/joint absolute precision

Files: `src/xrr_fitter/fit/local_search.py`, `src/xrr_fitter/fit/joint_solvers.py`, new `tests/regression/test_poisson_local_precision.py`.

- [x] Build real existing non-holdout seed37 footprint/surface contexts through the current generator; run all four declaration/Sobol starts with budget80. Exercise single and a shared-parameter two-member joint context. Require all final pairs to agree in unit/Q/KL with `(1e-4,1e-6,1e-6)` and verify the irreducible total Q is above1e6.
- [x] Record actual RED using `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -q -p no:cacheprovider tests/regression/test_poisson_local_precision.py`.
- [x] Change only the solver keyword: single `ftol=None if problem.config.noise_model == "poisson" else 1e-10`; joint `ftol=None if any(member.config.noise_model == "poisson" for member in problem.problems) else 1e-10`. Keep every other keyword and publication check.
- [x] Record GREEN and related local/joint/cancellation tests. Add explicit Poisson/non-Poisson/mixed-mode keyword controls; mixed diagnostic calibration remains unsupported.

### Task N2: Same precision in the diagnostic estimator

Files: `src/xrr_fitter/fit/diagnostic_refit.py`, `tests/regression/test_poisson_local_precision.py`, `tests/unit/fit/test_diagnostic_refit.py`.

- [x] Add a real strong-mismatch regression requiring the diagnostic refit to match the polished product. Observe a new RED after N1, not just a mocked keyword difference.
- [x] Change the diagnostic solver keyword to `ftol=None if any(member.config.noise_model == "poisson" for member in problem.members) else 1e-10`; keep all declared paths, same80 total each, all-success requirement, losses and cancellation boundaries.
- [x] Update the explicit current-estimator contract assertion from relative `ftol=1e-10` to `None` for Poisson; retain both existing `xtol/gtol` assertions.
- [x] Record GREEN for the fresh real regressions and all existing diagnostic tests. Re-run unchanged baseline failures119/148 only as development evidence, with unavailable still counted.

### Task N3: Bind the changed numerical rule to a strict current identity

Files: `model/project.py`, `model/fitting.py`, `model/diagnostic_calibration.py`; exact-version model/fit/codec/GUI/export tests, declaration-only examples.

- [x] Observe RED for the new defaults and rejection of the former defaults: `algorithm_version="xrr-fit-v2-poisson-2"`, `diagnostic_version="poisson-refit-null-v2"`, `method="poisson_refit_null_v2"`, `refit_policy="declared_sobol4_xtol_v2"`. Keep schema4 and objective_version string"2" because neither data layout nor likelihood changes.
- [x] Replace the one supported identity in constructors and validators; reject the previous identities rather than adding a compatibility branch. Update current expectations and examples, which contain no saved fit/checkpoint.
- [x] Keep the old acceptance protocol's expected versions and review identity unchanged. Its registration must reject the revised runtime until a separately reviewed new-seed protocol replaces it; add a real version-preflight test. No new registration is authorized by this numerical revision.
- [x] Run fresh model/codec/provenance/config/export and exact version tests, then broader engineering modes.

### Task N4: Review and status

- [x] Produce a scoped diff against persistent baseline-source; independently review numerical scope, non-Poisson controls, actual RED/GREEN and residual risks.
- [x] Run fresh relevant repository regression/quality checks; do not reuse lost historic reports as fresh evidence.
- [x] Update the existing progress ledger with actual outcomes and exact artifacts. Keep version freeze/new holdout pending the remaining estimator/statistical design; preserve every failed probe. No release or legacy compatibility work.

## 2026-09-13 execution closeout — precision subplan only

N1–N4 complete. All seven relevant official MODEs ran on the same `validation-precision-003/source` snapshot (`e903df38072364cbf1b18d0aec84207891ad5a764ff72060689eb27e8370cec1`): quality189, tools587, unit2594, integration39, gui711, spawn4, regression65; total4189 passed, full Radon PASS. The existing hidden-canvas GUI layout warning remains. Both independent scoped reviews are Approved with0 open findings; the source and review hashes are checked in `precision-engineering-final.json`.

N1/N2 had real mismatch RED→GREEN; N3 strict identity tests had real rejection RED→GREEN. The two stale checkpoint expectations were independently derived from a sole `diagnostic_version` change, with complete non-Poisson frozen numerical/progress equality; RED2→GREEN14. Review added six pairwise checks and four real solver success flags; stronger baseline RED4→current GREEN14. Two tools contract expectations were then corrected without changing the protocol or registry, RED2→GREEN4; `validation-precision-001/unit.log` and `validation-precision-002/tools.log` remain failed historical attempts, not retroactive passes.

Current-API development replays: footprint/surface37 available at999/999, target p=.001; replay119/148 remain null-refit unavailable. These are development evidence, not independent acceptance or a new coverage-rate estimate. No scalar optimizer, ACF scale change, threshold/budget relaxation, old-format compatibility, new registration or repeated corpus was adopted.

**Remaining outside this subplan:** low-count GN slow convergence; ACF scale/power design; approval of the next estimator/statistical design; new unexposed-seed preregistration and full scientific acceptance. The old4400-case FAIL remains. The proposed fixed two-stage numerical direction has been presented for a decision, not assumed approved. Preserve the branch, worktree, all uncommitted work and persistent evidence; no commit/merge/push/release. See `2026-09-12-poisson-diagnostic-calibration-progress.md` for the canonical current ledger. Final ledger-only changes are separately bound by `status-update-003/` and `precision-final-handoff.json`.
