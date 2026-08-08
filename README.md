# XRR Fitter

XRR Fitter is a desktop application for fitting X-ray reflectivity (XRR)
measurements to layered thin-film structure models. It provides an interactive
PySide6 GUI with a filename-driven automatic fitting path, plus expert tools for
building structures, running global-then-local optimization, and quantifying
parameter uncertainty.

![XRR Fitter GUI](docs/images/gui-light-1280x760.png)

## Features

- Automatic structure construction and fitting from strict filename layer stacks.
- Single-point fitting or same-import, same-physics joint refinement with
  point-local thickness and batch uniformity summaries.
- Guided four-step workflow for routine fitting, plus a dockable expert
  workspace whose panel layout is saved with the project.
- Interactive expert structure editor for layers, periodic stacks, gradients,
  and the substrate backing, with reflectivity and SLD-profile plots shown
  side by side.
- Global screening followed by local least-squares refinement with checkpointed,
  resumable fits.
- Uncertainty analysis: bootstrap resampling, MCMC sampling, and parameter
  correlation diagnostics.
- Joint fitting across multiple datasets with shared parameters.
- Filename-driven batch fitting and deterministic project/export round-trips.

## Requirements

- Python 3.12 (`>=3.12,<3.13`)
- macOS on Apple Silicon (arm64)

Runtime dependencies (numpy, scipy, periodictable, pandas, xlsxwriter,
matplotlib, PySide6) are declared in `pyproject.toml` and pinned in
`requirements-macos-arm64-py312.lock`.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos-arm64-py312.lock
pip install .
```

## Usage

Launch the desktop shell:

```bash
xrr-fitter            # installed entry point
python -m xrr_fitter  # module entry point
python -m xrr_fitter --help
```

Sample projects and data live in `examples/` (`single-layer`,
`mo-si-periodic`).

### Automatic filename workflow

The final space-separated part of each filename stem declares finite film layers
from the substrate side to the surface side:

```text
<sample-id> <substrate-side-layer>+...+<surface-side-layer>.xy
```

For example, `P1 Zr.xy` declares one Zr film. The substrate defaults to Si. A
stack such as `P1 Si+Zr.xy`, whose substrate-side finite layer is itself Si,
prompts once per matching structure group for the actual substrate. A Si
substrate receives a 10 A SiO2 native-oxide layer unless the adjacent layer is
already exactly SiO2.

Selecting files imports every valid row and starts automatic fitting. A singleton
physical signature runs as a single fit; multiple datasets from the same import
batch with the same signature are prefit and jointly refined. Unknown material
codes use a direct effective-SLD model and report effective SLD/electron density,
but do not invent a mass density in g/cm3.

The standard result view reports per-point status and statistics membership as
well as layer and uniformity values. Leave the guided flow via **View ▸ Guidance
mode**, then enable **Expert mode** to use manual independent/joint fitting,
profile diagnostics, MCMC, and explicit result export; automatic fitting itself
does not export files.

## Public API

`xrr_fitter.api` is the only supported Python API. Everything else under
`xrr_fitter` is internal and may change without notice.

## Development

```bash
pip install -e '.[test]'
python tools/verify.py MODE      # quality | tools | unit | gui | integration | ...
python tools/check_radon.py      # complexity policy
```

`tools/verify.py` runs the repository's verification gates; the same gates run
in CI (`.github/workflows/verify.yml`). Architecture, dependency, and public-API
rules are enforced under `tests/architecture/`. See `AGENTS.md` for repository
conventions.

## Documentation

- `docs/user-guide.md` — end-user workflows
- `docs/algorithm.md` — fitting algorithm and physics
- `docs/architecture/r23-clean-break.md` — architecture and dependency graph

## License

License to be determined. Until a license is added, all rights are reserved by
the authors.
