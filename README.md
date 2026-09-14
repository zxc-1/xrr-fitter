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
- Explicit robust-log, known-sigma Gaussian, and raw-count Poisson objectives.
- Uncertainty analysis: bootstrap resampling, MCMC sampling, and parameter
  correlation diagnostics.
- Joint fitting across multiple datasets with shared parameters.
- Filename-driven batch fitting and deterministic project/export round-trips.

## Requirements

- Python 3.12 (`>=3.12,<3.13`)
- macOS on Apple Silicon (arm64)

Runtime dependencies (numpy, scipy, periodictable, pandas, xlsxwriter,
matplotlib, orsopy, jsonschema, PySide6) are declared in `pyproject.toml` and
pinned in `requirements-macos-arm64-py312.lock`.

## Installation

```bash
python3.12 -m venv "${TMPDIR:-/tmp}/xrr-fitter-venv"
source "${TMPDIR:-/tmp}/xrr-fitter-venv/bin/activate"
pip install -r requirements-macos-arm64-py312.lock
pip install .
```

Keep the environment outside the checkout: the repository hygiene gate rejects
generated `.venv`/`venv` directories even when Git ignores them.

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

### Fitting algorithm V2

Projects use algorithm `xrr-fit-v2`, objective version `2`, and project schema
`3`. Older algorithm, project, and checkpoint versions are rejected rather
than silently migrated. This is an algorithm change, not a release or tag.

Choose the noise model in the fitting panel, through `api.set_fit_config`, or
explicitly on the CLI:

```bash
python -m xrr_fitter.cli.main fit project.xrrproj.json --noise-model gaussian --output fitted.xrrproj.json
```

- `robust_log` (default): exploratory robust log loss; intensity plus the
  stabilizing floor must be positive. Zero intensity can be valid.
- `gaussian`: every fitted row needs a known, finite, positive intensity
  standard deviation in the same units as the observation. Finite negative
  observations are allowed.
- `poisson`: unmerged, nonnegative integer **raw counts**, including zeros;
  normalized intensities, count rates, and background-subtracted values are
  not interchangeable with counts.

Omitting `--noise-model` keeps the project's saved mode. Changing the mode
invalidates old results and checkpoints without re-enabling excluded rows.
CLI mode requirements are written to stderr, leaving JSON progress on stdout.

Reports distinguish statistical covariance from search-start spread and retain
unavailable reasons. Robust loss-support profiles and fast exploratory
bootstrap samples are not labelled as 95% confidence intervals. Formal
percentile bootstrap intervals require at least 200 successful samples;
Gaussian/Poisson likelihood-ratio profiles additionally require the recorded
regularity conditions. Exported CSV, workbooks, JSON, and ORSO use the saved
evidence and residual units rather than recomputing uncertainty.

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
- `docs/acceptance/fitting-algorithm-v2.md` — V2 validation evidence and limits

## License

XRR Fitter is licensed under the MIT License. See [`LICENSE`](LICENSE).
