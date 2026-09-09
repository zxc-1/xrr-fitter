# Original Audit Completion Plan

> **For agentic workers:** Use `executing-plans` or
> `subagent-driven-development`; keep task briefs, reports and the progress
> ledger outside the repository.

**Goal:** Complete the remaining non-GUI recommendations from session
`01a0718b-883e-72e0-aa5e-b5f9ed43268c`, preserving the original audit scope.

**Architecture:** Retain the public `xrr_fitter.api` boundary, domain-owned
numerical evidence, immutable transactions and existing verification modes.
Add development-only audit tooling with explicit inputs and external reports;
introduce strict checks only for concretely repaired type boundaries. Bind
supply-chain reports to actual package/archive bytes, not only version strings.

**Tech Stack:** Python 3.12, existing pytest/verifier, coverage.py 7.16.0,
mypy 2.3.1, pip-audit 2.10.1, SciPy stubs matching SciPy 1.18.0.

## Global Constraints

- Work only in the existing sibling `audit-improvements` worktree.
- Do not edit GUI source or the active GUI worktree, or run GUI test selections.
- Preserve numerical outputs, ordering, seed lineage, nfev, cancellation,
  checkpoint/resume, export atomicity and full-data candidate evidence.
- Do not add runtime dependencies, blanket diagnostic suppression, synthetic
  success, reduced checks or unreviewed deployment changes.
- Use RED -> GREEN for changed behavior/type contracts; use deterministic
  generation and contract tests for lock/config metadata.
- Keep reports/environments/caches outside the repository; never weaken hygiene
  to accommodate a linked worktree. Final gates run in a clean ordinary clone.
- Local development/CI audit configuration was confirmed by the user's
  continuation. Remote push/dispatch, runner account/permission changes,
  merge/tag/release are separate operations, not implied permission.

## Acceptance Map

### 1. Type and Ownership Boundaries

Files: `services/batch.py`, `services/batch_routing.py`, `fit/stages.py`,
`analysis/profile_selection.py`, `analysis/profile_tasks.py` and owned tests.

- [x] Routing inputs: material, structure component, dataset and acquisition
  models; preserve signature bytes and field order.
- [x] Profile selection/task graph: `UncertaintyReport`, `ModelEvaluation`,
  generic plans/results and the existing ordered task runner.
- [x] Stage orchestration: `FitEvaluationContext`, local/global result types
  and typed progress closures; retain full-data reevaluation.
- [x] Batch source/result/checkpoint containers and callback contracts.
- [x] Extract cohesive publication responsibilities where the data flow
  demonstrates separate ownership; no mechanical file-size target.
- [x] Confirm a strict checked module set and record remaining project-wide
  diagnostics separately from this incremental acceptance.

Type acceptance command uses mypy `--follow-imports=silent`, not `skip`: imported
types are analyzed, while unrelated modules are not claimed to be repaired.
Each selected module must pass `--check-untyped-defs --disallow-untyped-defs`.

### 2. Reproducible Engineering Checks

Files: `tools/audit_reports.py`, `tools/lock_audit_tools.py`, `tools/verify_registry.py`,
`tools/audit-requirements.lock`, `pyproject.toml`, `.github/workflows/audit.yml`
and owned tool/workflow tests.

- [x] Resolve a separate hash-pinned development tool closure constrained by
  existing application pins. Do not inject an extra into wheel `Requires-Dist`.
- [x] Coverage uses exactly the existing unit/integration/spawn/regression
  selections, including outcome enforcement, and captures subprocesses.
- [x] Write JSON/XML/HTML coverage evidence outside the checkout. Test or tool
  failure remains failure; no retries are merged into an all-green claim.
- [x] Type checking reports diagnostics and exits nonzero for any selected
  module error, tool crash or missing tool.
- [x] Advisory scanning validates both locks, queries exact ordinary pins,
  records VCS/native exclusions and fails on findings or database/tool failure.
- [x] Wire independent read-only hosted CI jobs and retain reports even when a
  check fails. Preserve existing standard/release mode semantics.
- [x] Tests cover exact command selections, failure propagation, missing tools,
  external outputs, pinned tools and workflow permissions.

### 3. Package Bytes and SBOM

Files: existing lock validators/SBOM/distribution tools, new package-byte
verification helpers, platform manifests and their tests.

- [x] Resolve target-platform wheel/sdist identities and record SHA-256 from
  trusted resolution; reject byte mismatch before installation and cache reuse.
- [x] Treat the pinned `refnx` Git source separately; do not invent a PyPI
  version or imply Git commit identity is a downloaded wheel hash.
- [x] Inventory installed/archive components, dependency relationships,
  licenses and bundled native binaries with byte identities.
- [ ] Bind SBOM and scan evidence to the distribution/executable it describes;
  enforce declared completeness, explicitly fail or retain incomplete status
  when native composition cannot be established.
- [x] Integrate reviewable evidence assets without weakening existing exact
  release identity and artifact-manifest checks.

### 4. Cache and Runner Isolation

- [ ] Separate PR and trusted build cache keys by trust domain, platform,
  Python/pip and lock/package-manifest hashes; never cache a venv or credentials.
- [ ] Verify cached package bytes before use, reject broad restore keys, and
  exercise cold/warm/mismatch/failure-cleanup paths.
- [ ] Check unique job workspaces, external environments, process/permission
  boundaries and bounded cleanup against the active runner configuration.
- [x] Retain read-only runner evidence; obtain explicit approval for any host
  permission/account/service changes before applying them.

### 5. End-to-End Verification

- [ ] Focused tests and independent reviews for each completed slice.
- [x] Clean-checkout quality/tools/unit/integration/spawn/regression and Radon.
- [ ] Fresh coverage, selected types, both-platform advisories and package-byte
  checks; reproducible artifact/SBOM and distribution/identity verification.
- [x] Repeat complete statistical validation when application source changes
  invalidate the previous exact-source binding.
- [ ] Exact-candidate GitHub Actions and real Windows CLI/export/executable
  evidence, without substituting macOS simulations for Windows runtime.
- [ ] Compare original recommendations against results, preserve failed and
  superseded evidence, and keep overall status incomplete until every required
  implementation and target-platform validation is accounted for.

GUI reconciliation, merging the branch, publishing a tag and public release
are not additional completion requirements invented by this audit.

Ordinary wheel identities, source archives and archive inventories are verified.
The wheel SBOM explicitly remains incomplete; final artifact binding and native
relationships, the refnx build closure and wheel, shared verified caching and
actual runner account isolation remain open. Disabling pip caching does not
complete shared-cache acceptance. Read-only runner inspection found the same
account as local development; no permission or account changes were authorized.

The wheel-cache tool now binds exact keys to trust domains, target, tool
versions and input hashes; every restore verifies all bytes and rejects
unrelated files. Both platform archives have real local cold/warm evidence.
The Windows audit integrates exact-key restore/save, but hosted cache behavior
and migration of existing macOS/refnx build caches remain unverified.

Candidate `acf5769` passed clean-checkout quality (200), tools (595), unit (1732),
integration (14), spawn (4), regression (50), coverage, selected types and both
platform ordinary-pin advisory checks. Git push was authorized but failed because
the local GitHub CLI token is invalid. Exact remote/Windows execution and a
successful independent review remain pending.

## Current Evidence

The execution ledger is
`/private/tmp/xrr-audit-completion-20260908.xoI3FB/progress.md`.
The original plan remains `2026-09-05-audit-improvements.md`; this document
expands its deferred acceptance work rather than replacing the original scope.
