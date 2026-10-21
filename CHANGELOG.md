# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The source tree prepares **0.2.5**. No 0.2.5 tag or release has been published.
The existing 0.2.4 draft retains its original artifacts and source identity.

### Added
- Folder import and filename-driven structure review, with clearer fitting,
  checkpoint, progress, and result feedback throughout the GUI.
- An opt-in Windows native-platform GUI check for rendering, Qt events, project
  save/reopen, and widget teardown.
- Parsing of short COFF import records in Windows dependency inventories,
  preserving explicit unresolved native and runtime dependencies.

### Fixed
- MCMC walker initialization now accounts for local parameter scales and
  constraints, rejects coincident or infeasible walkers, and preserves the
  configured sampling budget and seed behavior.
- Closed windows and their plot menus release owned widgets; GUI verification
  no longer accumulates those widgets across tests.
- Application quit asks about unsaved changes, and cancelling preserves the
  project and recovery draft. Scrolling past unfocused editors preserves values.
- Failed candidates are excluded from the GUI's "same solution" labels.
- File-dialog and export-busy-dialog test handling, splitter input boundaries,
  and stalled-test stack reporting avoid the observed verification deadlocks.
- README scientific identities and release status now match the implemented
  schema and the distinction between tagged drafts and published releases.
- Online help uses a published, immutable guide snapshot so candidate builds
  open the manual before their version tag exists.

### Security
- Update the pinned bootstrap, build and audit installer to pip 26.2.2 for
  CVE-2026-13346, preserving exact wheel hashes and installation-byte checks.
- Update locked Pillow inputs from 12.3.0 to 13.0.0 to resolve the advisory
  failures, with verified wheel bytes and refreshed environment evidence.

## [0.2.4] — draft candidate, tagged 2026-09-16

As checked on 2026-10-21, this candidate is still unpublished. The annotated tag
and draft assets bind commit `7c93d06119d38ab169a2e698c8b7612aea5810fb`.

### Added
- Explicit robust-log, Gaussian, and Poisson fitting objectives, project schema
  5, calibrated Poisson diagnostics, and joint bootstrap provenance and intervals.
- GUI workspace redesign and native Qt Cocoa accessibility lifecycle fixes.
- Verified dependency inputs, installed/frozen artifact inventories, hosted
  macOS verification, and explicit statistical-computation authorization.

### Fixed
- Pending automatic results and plot refresh, Windows export publication,
  and clean-environment distribution smoke checks.

## [0.2.3] — 2026-08-22

The following entries were already present in the source tagged 0.2.3; they are
retained here rather than presented as new 0.2.5 changes.

### Added
- Ordered task runners that parallelize stage restarts, independent batch
  fits, and profile/bootstrap analysis while preserving deterministic order.
- Substrate backing editor dialog in the structure editor.
- `emit_warning` control so the optimizer inner loop can suppress repeated
  Gauss-Hermite convergence warnings while retaining the structured diagnostic.
- `.editorconfig` and `.gitattributes` for consistent formatting and line endings.

### Changed
- Extracted the differentiable geometry engine from `evaluation` into
  `physics/geometry`.
- Renamed the default integration branch to `main`.
- The 12-hour statistical verification gate now runs only on release tags.

### Removed
- The R22 reference-replay verification subsystem, its release-spec oracle, and
  the R22 migration ledger. The full subsystem is preserved on the
  `archive/r22-reference` branch.

### Fixed
- Filename-driven batch imports now order layers from backing to surface.

## Earlier published releases

- [0.2.2](https://github.com/zxc-1/xrr-fitter/releases/tag/v0.2.2) — 2026-07-31.
- [0.2.1](https://github.com/zxc-1/xrr-fitter/releases/tag/v0.2.1) — 2026-07-30.

[Unreleased]: https://github.com/zxc-1/xrr-fitter/compare/v0.2.4...main
[0.2.4]: https://github.com/zxc-1/xrr-fitter/tree/v0.2.4
[0.2.3]: https://github.com/zxc-1/xrr-fitter/releases/tag/v0.2.3
