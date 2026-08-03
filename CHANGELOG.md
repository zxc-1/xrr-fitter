# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## Released

Prior releases are recorded as annotated Git tags (`v0.2.1`, `v0.2.2`).
