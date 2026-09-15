# Poisson two-stage refit implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use executing-plans task-by-task; the main agent owns the numerical critical path, existing same-effort agents may perform independent read-only review. Track steps below; no automatic commits or release.

**Goal:** Repair the exposed low-count null-refit convergence failures within the unchanged per-path budget using the approved fixed localization → TRF design.

**Architecture:** A bounded numerical kernel owns phase transitions, accepted-point handoff and real request counting. The existing diagnostic refitter owns full-axis validity and all-start selection. The coordinator aggregates immutable stage/work evidence into the strictly versioned saved calibration.

**Tech Stack:** Python 3.12, existing NumPy/SciPy, pytest, repository verify/Radon scripts; no new dependencies.

## Current closeout — frozen 003 verified; T1–T4 engineering complete

- Fixed per-start L-BFGS-B → charged full handoff → TRF is implemented and reviewed; starts, M=80 development budget, native success and all-path success remain unchanged. Immutable work evidence, strict codec and the sole v3 estimator identity are complete.
- Final ordinary clone: `validation-two-stage-003/source`, 690 files; snapshot SHA256 `807292b7470be0069fd7c707300f83abeac1d5f9b827d3e64b60a41230c0396b`. All seven MODEs ran on this same implementation: quality 189, tools 587, unit 2740, integration 39, GUI 711, spawn 4, regression 70: **4340 passed**, all exit 0. Full Radon: **597 files, 0 issues, PASS**; one existing GUI small-canvas warning is retained.
- All four public API B999 development replays are PASS: replay119/replay148/footprint37/surface37 each completed 999/999 nulls, 1000 refits and 4000 paths. Actual maximum path nfev is 48/46/42/41 respectively, within 80; family p is .573/.544/.001/.001. Both injected target adjusted p values are .001. Replay families legitimately have three columns, injected families four; the recorder verifies the exact ordered family axis.
- `two-stage-final-code-review-002.md` remains Changes required with its original R6 finding. The successful unit/work dimension mismatch has RED 4 failed / 28 passed → GREEN 430 passed and is independently Closed in `refit-dimension-review-003.md`. Final 003 verification covers this guard, not merely the earlier 002 implementation.
- `two-stage-offline-audit-003.json` binds the complete matrices and 16000 paths, including 20 rejecting negative controls. `development-evidence-review-002.md` must be read with `development-seed-erratum-002.md`; its rounded seed table is not usable for replay. `final-validation-review-003.md` independently approves all final receipts and exact numerical-artifact equivalence to the independently audited 002 runs; no new findings.
- Final engineering summary: `two-stage-engineering-final.json`, SHA256 `b03f674f1db4a263041400ce967b5ad7127a14c7d8e9f8f29a2f816f4f98cbbb`. All relative evidence paths in this section use the evidence root below. Subsequent six-document synchronization is recorded separately in `status-update-two-stage-003/`; it is not retroactively part of the frozen 003 source.
- This closes only the approved numerical engineering subplan. ACF scale competition is not repaired; old 81% coverage and the historical 4400 + 200 experiment's 4/8 FAIL are not new v3 results. No corpus/holdout rerun or overall scientific PASS is claimed. Preserve evidence, branch and uncommitted work; a new unexposed-seed scientific protocol still requires separate approval.

## Global constraints

- Work in `/Users/dala/Desktop/XRR-Fitter/fit-algorithm-v2`, branch `feat/fitting-algorithm-v2`; preserve all prior uncommitted work and evidence.
- Follow `../specs/2026-09-13-poisson-two-stage-refit-design.md`. Preserve budgets, starts, likelihood, MATCH_LIMITS, alpha, B and all-path success.
- No ACF/RMS, compatibility, repeated old holdout/corpus, dependencies, commit/merge/push/release.
- Evidence: `/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/two-stage-001`; failed attempts get unique filenames.
- Every behavior change needs observed RED → GREEN; tests use `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, external basetemp/MPLCONFIGDIR. Formal verification uses `tools/verify.py MODE`.

## T1 — bounded kernel and exact bad-path regressions

Files: create `src/xrr_fitter/fit/diagnostic_solver.py`, `tests/unit/fit/test_diagnostic_solver.py`, `tests/regression/test_poisson_two_stage_refit.py`; modify `src/xrr_fitter/fit/diagnostic_refit.py` and its unit tests.

Interface: `TwoStageSolver(system, loss, validate, *, budget, poisson, cancelled=None)` exposes `solve(start) -> (unit_or_none, failure_or_none)`, actual `nfev` and ordered stage exits. `validate(unit)` performs the original full-axis evaluation and returns its existing failure string or None. Kernel numerical exceptions use the same existing NUMERICAL_ERRORS; unexpected errors/cancellation propagate. No publication evaluation is substituted for validation.

- [x] Preserve the current 678-file baseline and Git status before changes (`baseline-manifest.json`).
- [x] Add original development null regressions, using current generator/compile code and fixed original product-source coordinates, not saved old-format input. Assert `failure_reason is None`, four attempts and `nfev <= 4*M`; first observe the original 80-request failures.
- [x] Add kernel contract tests before implementation. For M=9 assert localization cap4, handoff1, refinement at most4 (or more if localization stops early). Exercise real callbacks under returned-success, ABNORMAL and hard cap, with a rejected last trial; handoff must be the last accepted point. Test Q/g against the actual residual/loss chain, not a fabricated normalized objective.
- [x] Implement the fixed phase kernel and wire diagnostic paths without changing product local/joint solvers. Core budget formula is `limit = (budget - 1) // 2`; TRF max_nfev is `budget - nfev` after charged handoff. Preserve raw statuses/messages and actual work on expected exceptions.
- [x] Migrate old solver fakes to issue the work they report. Add nonfinite/invalid handoff, callback mutation, tiny-budget, zero-dimensional, cancellation and programming-error cases; retain all prior multi-dimensional/shared/prior tests.
- [x] Run fresh focused GREEN and compare each original failure path's stage counts/statuses. Do not choose a new fraction or thresholds based on these outcomes without documenting a design revision.

Focused command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/two-stage-001/mpl-cache /Users/dala/Desktop/XRR-Fitter/venvs/repo/bin/python -m pytest -p no:cacheprovider --basetemp=/Users/dala/Desktop/XRR-Fitter/.codex-artifacts/poisson-diagnostic-20260913/two-stage-001/pytest-t1 tests/unit/fit/test_diagnostic_solver.py tests/unit/fit/test_diagnostic_refit.py tests/regression/test_poisson_two_stage_refit.py tests/regression/test_poisson_local_precision.py -q`.

## T2 — immutable stage/work evidence and coordinator

Files: `src/xrr_fitter/model/diagnostic_work.py`, `src/xrr_fitter/model/diagnostic_calibration.py`, `src/xrr_fitter/analysis/residual_calibration.py`, `src/xrr_fitter/io/codec_diagnostic_calibration.py`, provenance/architecture registration as required; matching model/analysis/codec tests.

Interface: immutable `DiagnosticSolverExit`, `DiagnosticPathSummary` and `DiagnosticRefitWork` form a canonical joint path histogram; counts/max/budget are derived from coupled bins. DiagnosticRefit carries per-refit work; DiagnosticCalibration keeps separate observed_work/null_work aggregates. Work nfev equals the existing nfev field; missing/forged/inconsistent current evidence fails explicitly.

- [x] Test immutable/pickle behavior, valid aggregation, count/max/budget corruption, actual observed and null batch accumulation, early failure evidence and strict codec seal/missing keys. Observe RED.
- [x] Implement only this work record, aggregation and strict codec; retain existing probability/failure/index and qualification semantics.
- [x] Verify exact sum of all stage nfev, count consistency and maximum cap against actual kernel calls, including failures before optimization and locked problems. Run model/analysis/codec/service/export focused GREEN.

## T3 — one current estimator identity and unchanged controls

Files: current algorithm/diagnostic defaults and validators, declaration-only examples, exact-version/fingerprint tests, regression registry and its tests. Do not edit old protocol versions or review hash.

- [x] Add RED for current v3 identity, rejection of former v2 identities and actual old-protocol rejection of v3.
- [x] Change only the approved estimator identities: `xrr-fit-v2-poisson-3`, `poisson-refit-null-v3`, `poisson_refit_null_v3`, `declared_sobol4_lbfgsb_trf_v3`; retain schema4 and objective string2. Update exact current declarations, no compatibility.
- [x] Derive any changed frozen config fingerprints independently from only the diagnostic_version delta; compare full non-Poisson numerical/progress behavior, do not bless arbitrary new hashes.
- [x] Register the new real regression module in the existing MODE and migrate exact inventory expectation with RED → GREEN.

## T4 — development validation, formal checks and independent closeout

Files: new reproducible scripts/reports only under the stage evidence root; update project status docs after evidence is verified.

- [x] Run current public API B999 for replay119, replay148, footprint37, surface37 with unchanged inputs/options/thresholds. Recording wrappers may observe but never replace solver status/options or numeric output. Verify strict save/load, phase/work sums and original target detections. Preserve failed jobs.
- [x] Freeze a normal Git validation clone of the final tree; run `python tools/verify.py quality`, `tools`, `unit`, `integration`, `gui`, `spawn`, `regression` and complete Radon with receipts bound to the same source. Do not rerun the 220-case corpus here.
- [x] Independent review of spec/quality, actual RED/GREEN, cap/handoff/failure/identity evidence and residual risks. Resolve any findings with new tests and fresh verification.
- [x] Update progress with actual outcomes, exact evidence paths, cleanup and remaining scientific risk. Preserve historical precision/holdout reports. No overall scientific PASS until a separate new-seed protocol is approved and completed.

## Historical execution checkpoint — implementation done, T2 re-review / T4 pending

- T1: original failures retained in `regression-red-001.log`; five exposed null controls now pass. Rejected-trial NaN J and unpaired Jacobian work bypass each have RED/GREEN and independent early review. Actual single/joint multivariate and active-prior Q/g checks now persist in `tests/unit/fit/test_diagnostic_refit_priors.py` (split from the refit test module for the unchanged Radon policy).
- T2: independent `work-evidence-review.md` found five evidence consistency gaps. `native-status-red-001.log` (14 failed), `work-axis-red-001.log` (5 failed), and `work-count-red-001.log` (8 failed) precede their fixes. Related GREEN reached 469 tests; after cohesive work-validation/test splitting, `work-refactor-green-001.log` is 379 passed and `radon-scoped-preflight-002.json` is PASS. New read-only T2 re-review is pending; do not label the first review approved.
- T3: `identity-red-001.log` precedes `identity-green-001.log` (277 passed). Frozen hashes had real 2-test RED; `frozen-identity-derivation-001.json` independently proves the sole config delta and unchanged complete non-Poisson numerical/progress records. `frozen-green-001.log`: 68 passed.
- T4: derived recorder is `verify_two_stage_v3.py` under the evidence root. The old precision driver is untouched. No complete B999 run or formal MODE result is yet claimed at this checkpoint.
- No ACF/RMS, budget, MATCH_LIMITS, alpha, B, success rule, dependency or release change. Ruff is absent from the existing environment and offline cache; its unavailable preflight is retained rather than installing an unapproved tool. Official quality/Radon remain mandatory.

## Historical checkpoint — T1–T3 reviewed; second frozen T4 validation pending

- T2 re-review is now Spec compliant / Quality Approved, with all five findings closed: `work-evidence-recheck-001.md` (SHA256 `cb34f8bf38d3a5f66f3b52543512842cf3a921047670c12b2cc8764145c99c40`), independent 170/170 matrix and 72 focused tests. The initial finding report remains unchanged.
- T3 independent review is Spec compliant / Quality Approved, no open items: `identity-review-001.md` (SHA256 `c40605c465b2070186d544e045920d7157fa7bfe9c4719f480353fcf767c4fa8`), 690/690 source hashes, full config/numerical/progress comparison and 34/34 value-boundary checks.
- `validation-two-stage-001` completed every MODE: quality 189, tools 587, integration 39, spawn 4 and regression 70 passed; unit was 1 failed / 2729 passed and GUI was 25 failed / 686 passed. Original logs/receipts are retained. The unit failure was a provenance test changing total work without its required stage histogram; the corrected fixture has real RED then 147 focused passes. The GUI driver omitted the previously used offscreen environment: the same focus module passes 27/27 with only `QT_QPA_PLATFORM=offscreen`. No product/GUI behavior was changed for either issue.
- All four first B999 development computations completed 999/999 nulls, 1000 refits and 4000 paths, within M=80. The 119/148 recorder reports remain FAIL because it incorrectly required four columns: their geometry-applicable family has three, as did the previous saved declarations. `verify_two_stage_v3_002.py` checks the full ordered family axis and seed instead; its RED was 2 failed / 3 passed, GREEN is 5 passed. It does not add/drop a detector or change statistics.
- Preserve every 001 output and original driver. Derive `run_formal_modes_002.py` with GUI offscreen and recorded environment; freeze a new ordinary Git clone, then rerun all seven MODEs and all four fixed development cases on that same 002 snapshot. Runtime must remain `89d6b106be632f5bd3b08f583dd10445bf35bc40aeee2eb26ea8d716ccd25561`. This is repeatable development validation, not independent coverage or scientific acceptance.

## Historical checkpoint — 002 verified; final review dimension fix requires 003 validation

- Frozen 002 (`933953ce2acd2ece43df06b4caa6b8b3fe00088df2fe4574f1bb2103e9779843`) completed all seven MODEs on the same 690-file source: quality 189, tools 587, unit 2731, integration 39, GUI 711, spawn 4, regression 70: **4331 passed**. Full Radon: 597 files, 0 issues, PASS. GUI has one preserved Matplotlib small-canvas layout warning. `snapshot-validation-review-002.md` independently approves snapshot/fixture/receipt integrity; the older focused 147/27 summaries lack embedded command/environment receipts and are not used to invent those details.
- All four 002 B999 development replays are PASS, each with 999/999 nulls and 4000 paths. `two-stage-offline-audit-002.json` independently recomputes matrix ranks/seals and all 16000 callback/path records, plus 20 rejecting negative controls. Inputs are identical to precision-v2 apart from current estimator identities/output paths; raw solver logs and matrices are byte-identical to 001. Scientific acceptance is still not claimed.
- Offline reader failures are preserved: `offline-audit-002.log` used the wrong previous-report field name; `offline-audit-002b.log` incorrectly equated SciPy's internal nfev to actual residual callbacks. `native-count-explanation-002.json` records 233 same-coordinate TRF cache hits (native 2 versus actual 1, all xtol); the approved budget counts actual callbacks. Corrected `audit_development_002c.py` retains exact callbacks, phase/total caps, native exits and accepted handoff checks.
- Final whole-change review identified another runtime evidence inconsistency: a nonempty successful unit vector could be paired with zero-dimensional work, or the reverse. No such mismatch was observed in the development runs. A minimal successful-result dimension/work cross-check now rejects both directions, including pickle reconstruction; valid zero/nonzero path prefixes and failure work remain accepted. `refit-dimension-red-001.log`: 4 failed / 28 passed; `refit-dimension-green-001.log`: 430 passed. Exact two-file patch, source hashes and original commands are in `refit-dimension-fix-003.patch` and `refit-dimension-validation-003.json`.
- Keep 002 and its PASS results immutable; they do not verify this later guard. Freeze 003, re-run the same seven MODEs and four fixed B999 development cases, then obtain final independent fix/result review before closing T4. No numerical solver, budget, statistic, identity, compatibility or dependency change accompanies the guard.
