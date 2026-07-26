# Repository Guidance

- Use Python 3.12 and the `src/` package layout.
- Treat `xrr_fitter.api` as the only supported Python API and
  `python -m xrr_fitter` / `xrr-fitter` as the only application entry points.
- Follow the dependency graph in `docs/architecture/r23-clean-break.md`.
- Do not add legacy layouts, import shims, compatibility modules, dual
  implementations, or silent fallbacks.
- Migrate each production domain with its tests in the same change.
- Run repository verification with `python tools/verify.py MODE`.
- Run the complete complexity policy with `python tools/check_radon.py`.
