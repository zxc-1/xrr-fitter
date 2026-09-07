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

- [ ] Add an exact contract test for build requirements
  `("setuptools==83.0.0", "wheel==0.46.2")`, test dependency `pytest==9.0.3`,
  builder policy constants and both committed locks.
- [ ] Update only version-bound build fixtures and observe failures against the
  old implementation with the existing isolated Python environment.
- [ ] Set the target build/test pins and matching internal policy constants.
- [ ] Install a new external environment from the updated lock; verify `pip check`
  and exact installed package versions. Keep old environments for comparison.
- [ ] Exercise the unchanged generated-metadata contract; review any real builder
  differences and retain complete archive-content/byte checks.

### 2. Rebind Delivery Evidence

Files: `verification/release-spec.json`, `verification/r23/tests.json`.

- [ ] Generate release spec with `tools/build_release_spec.py` and compare fields;
  allowed changes are tool pins, test pins and lock hash (plus only independently
  verified builder-generated metadata if the new tool actually changes it).
- [ ] Run related tool and distribution architecture tests before committing.
- [ ] Commit source/tests/locks, then regenerate the test manifest from that full
  source SHA with `tools/collect_test_manifest.py` and compare two checkouts.
- [ ] Run full `quality`, `tools`, `unit`, `integration`, `spawn`, `regression`,
  `distribution` and `identity` gates in a clean clone with the new environment.
- [ ] Repeat ordinary-pin advisory lookup and official CycloneDX schema checks.
  Keep refnx and native artifact audit limitations explicit.

### 3. Statistical Completion And Reporting

- [ ] Run the complete existing `statistical` gate on an immutable validated
  checkout. An external supervisor records start time, heartbeat and exit code;
  it does not alter seeds, workload, worker count or thresholds.
- [ ] Preserve incomplete prior reports separately and require an exit-zero
  outcome before updating the statistical checklist.
- [ ] Update audit evidence with exact commits, results, remaining type/coverage
  debt and the still-unverified live Actions/Windows surface.
- [ ] Confirm clean worktrees and no GUI edits. Do not merge, push, tag or publish
  without explicit permission for that separate operation.
