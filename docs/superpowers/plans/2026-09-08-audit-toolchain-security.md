# Audit Toolchain Security Implementation Plan

> Execute with `executing-plans` and RED -> GREEN evidence. Keep changes in the
> audit worktree; do not touch the active GUI worktree or publish a release.

**Goal:** Remove the four advisory findings in the pinned Python build/test
toolchain while preserving deterministic packaging and existing verification.

**Architecture:** Upgrade existing tools, not application dependencies or gate
definitions. Keep the exact lock validators, artifact allowlists and identity
checks. Rebuild derived release metadata with the existing generators.

**Tech Stack:** Python 3.12, pytest 9.0.3, setuptools 83.0.0, wheel 0.46.2.

## Constraints

- `xrr_fitter.api` remains the supported public boundary; no GUI implementation,
  numerical configuration, runtime algorithm, checkpoint or schema change.
- Keep the macOS and Windows shared-package versions identical. Review any
  dependency-closure change caused by the three tool upgrades explicitly.
- Keep runtime direct pins, refnx Git commit, pip version, runner contracts,
  CI gate count, release asset count and public entry points unchanged.
- Generate environments, audit/coverage logs and build artifacts externally.
- Do not equate vulnerability database results with complete native/VCS auditing.
- Record actual process completion; interrupted/lost statistical runs are not PASS.

## Steps

### 1. Pin The Security-Fixed Tools

Files: `pyproject.toml`, both platform locks, `tools/build_release_spec.py`,
`tools/distribution_source.py`, `tests/unit/tools/test_build_release_spec.py`,
`tests/architecture/test_distribution.py`.

- [x] Add an exact contract test for build requirements
  `("setuptools==83.0.0", "wheel==0.46.2")`, test dependency `pytest==9.0.3`,
  builder policy constants and both committed locks.
- [x] Update only version-bound build fixtures and observe failures against the
  old implementation with the existing isolated Python environment.
- [x] Set the target build/test pins and matching internal policy constants.
- [x] Install a new external environment from the updated lock; verify `pip check`
  and exact installed package versions. Keep old environments for comparison.
- [x] Exercise the unchanged generated-metadata contract; review any real builder
  differences and retain complete archive-content/byte checks.

### 2. Rebind Delivery Evidence

Files: `verification/release-spec.json`, `verification/r23/tests.json`.

- [x] Generate release spec with `tools/build_release_spec.py` and compare fields;
  allowed changes are tool pins, test pins and lock hash (plus only independently
  verified builder-generated metadata if the new tool actually changes it).
- [x] Run related tool and distribution architecture tests before committing.
- [x] Commit source/tests/locks, then regenerate the test manifest from that full
  source SHA with `tools/collect_test_manifest.py` and compare two checkouts.
- [x] Run full `quality`, `tools`, `unit`, `integration`, `spawn`, `regression`,
  `distribution` and `identity` gates in a clean clone with the new environment.
- [x] Repeat ordinary-pin advisory lookup and official CycloneDX schema checks.
  Keep refnx and native artifact audit limitations explicit.

### 3. Statistical Completion And Reporting

- [x] Run the complete existing `statistical` gate on an immutable validated
  checkout. Retain the invocation, source/lock identity, output and actual exit
  code; do not alter seeds, workload, worker count or thresholds.
- [x] Preserve incomplete prior reports separately and require an exit-zero
  outcome before updating the statistical checklist.
- [x] Update audit evidence with exact commits, results, remaining type/coverage
  debt and the still-unverified live Actions/Windows surface.
- [x] Confirm only intended audit documentation remains pending and validation
  checkouts are clean. Preserve unrelated GUI worktree changes; do not merge,
  push, tag or publish without permission for that separate operation.

## Execution Evidence

- `c5b3ea0`: version migration, new pin contract tests and derived release spec.
  RED: 21 failures against old policy; GREEN: 88 focused tests. No dependency
  package additions; both platform shared pins remain equal.
- The new builder moves wheel LICENSE to `.dist-info/licenses/LICENSE`.
  `276135a` updates the exact path (no old-layout allowance), adds a metadata-set
  test and verifies real built license bytes. RED: old path rejected by the new
  test; GREEN: 80 archive/build tests. The sdist generated-file set is unchanged.
- `a6f74a0`: test manifest bound to `276135a`, 3095 collected nodes with identical
  bytes from two checkouts. Collection is not test execution evidence.
- Ordinary-pin advisory scan: zero findings in 44 macOS and 38 Windows entries;
  VCS/native caveats remain. Full distribution and local identity gates pass at
  `a6f74a0`; quality 200, tools 472, unit 1704, integration 14, spawn 4 and
  regression 50 all passed in the new environment (2444 tests total).
- Statistical verification passed independently on fixed `c626d62`. Compared with
  `a6f74a0`, application source, corpus/support code, verifier/outcome code and
  macOS dependency lock are identical; only license verification and its tests
  and manifest changed. The direct rerun passed both tests covering all 220 cases
  in **9472.87 seconds (2:37:52)**; the verifier returned **exit code 0**.
- The prior supervisor stopped recording at 02:01 CST on 2026-09-08. Its child
  later produced `2 passed in 8602.63s` at 03:44 CST, but no final supervisor or
  verifier exit code was retained. Keep the log and stale `RUNNING` status as
  incomplete whole-gate evidence; neither is silently converted to exit zero.
  The successful direct rerun at the same fixed `c626d62` used unchanged inputs.
- Fresh continuation checks passed: installed exact lock versions, `pip check`,
  artifact identity validation and clean delivery-checkout hygiene. The
  `a6f74a0` quality rerun passed 200 tests and the unchanged Radon policy.
  Evidence is retained at `/private/tmp/xrr-audit-resume-20260908.5ZafEx`, including
  the direct tool result in `statistical-result.json` (session 57719, chunk
  `f725cb`, exit 0). Existing release artifacts remain bound to `a6f74a0`, not
  the subsequent documentation commit.
- All four documentation files are included in the closing change. The main and
  validation checkouts are unchanged. The independently active GUI worktree
  changed while the long test ran; those changes were neither edited nor
  reverted by this audit. No whole-GUI-worktree-unchanged claim is made.
