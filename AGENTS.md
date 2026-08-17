# Repository Guidance

- Use Python 3.12 and the `src/` package layout.
- Treat `xrr_fitter.api` as the only supported Python API. The supported
  application entry points are `python -m xrr_fitter` / `xrr-fitter` for the
  GUI and `python -m xrr_fitter.cli.main` / `xrr-fitter-cli` for headless CLI
  workflows.
- Follow the dependency graph in `docs/architecture/r23-clean-break.md`.
- Do not add legacy layouts, import shims, compatibility modules, dual
  implementations, or silent fallbacks.
- Migrate each production domain with its tests in the same change.
- Run repository verification with `python tools/verify.py MODE`.
- Run the complete complexity policy with `python tools/check_radon.py`.
