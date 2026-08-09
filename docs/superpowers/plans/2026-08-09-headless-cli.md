# Headless CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the complete fit, sample, export, and validate workflow runnable unattended from a console entry point, with no Qt event loop and no new business logic.

**Architecture:** A new `xrr_fitter.cli` package sits beside `xrr_fitter.gui` at the same architectural level, reaching domain behavior only through `xrr_fitter.api`. Each subcommand is one module that loads a project, calls existing public operations, renders progress to stderr (or JSON Lines to stdout), and maps the result onto a fixed exit-code contract. Resume is not a separate command: the on-disk checkpoint is a project file written by `save_project`, so `load_project` plus `fit_project` re-enters the resume path inside the service layer automatically.

**Tech Stack:** Python 3.12, standard library `argparse`, existing `xrr_fitter.api` boundary. No new dependency.

**Design source:** `docs/superpowers/specs/2026-08-09-headless-cli-design.md`

## Global Constraints

- Do not import `xrr_fitter.services`, `xrr_fitter.fit`, `xrr_fitter.analysis`, `xrr_fitter.io`, or `xrr_fitter.physics` from `cli`. The only permitted domain edge is `xrr_fitter.api`.
- Do not import PySide6 anywhere under `src/xrr_fitter/cli/`, directly or transitively. The CLI must run with Qt absent.
- Do not add, remove, or change any signature in `api.py`. The single permitted `api.py` edit is re-exporting the existing `ConfidenceClass` type.
- Do not change `services/`, `fit/`, `analysis/`, `physics/`, `io/`, or `model/`.
- Do not use `pytest.skip`, `xfail`, or conditional collection. `tests/outcome_gate.py` fails the entire run on `skipped`/`xfailed`/`xpassed`/`deselected`.
- Every `__init__.py` must be exactly 0 bytes; `tests/architecture/test_naming_rules.py` asserts `path.stat().st_size == 0`.
- Test module stems must not start with their parent directory name. Under `tests/unit/cli/` use `test_fit.py`, not `test_cli_fit.py`.
- `tests/conftest.py` must remain the only `conftest.py` in the repository.
- New code under `src/`, `tests/`, and `tools/` must pass `tools/check_radon.py` (per-block CC ≤ 10, file average CC ≤ 5.0, MI rank A).
- Call `freeze_support()` as the first statement of the CLI `main()`, matching `src/xrr_fitter/__main__.py`.
- Keep user-facing messages in Chinese, matching `io/source.py`.
- Do not stage or modify `.claude/` or root-level probe files.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/xrr_fitter/cli/__init__.py` | Empty package marker (0 bytes). |
| `src/xrr_fitter/cli/exit_codes.py` | The exit-code contract and the result-to-code mapping. |
| `src/xrr_fitter/cli/progress.py` | Render `FitProgress` to stderr text or stdout JSON Lines. |
| `src/xrr_fitter/cli/commands.py` | One handler per subcommand, each calling only `api`. |
| `src/xrr_fitter/cli/main.py` | Argument parser, `freeze_support()`, dispatch, top-level error mapping. |
| `src/xrr_fitter/api.py` | Re-export the existing `ConfidenceClass` so exit codes need no hardcoded label. |
| `pyproject.toml` | Declare `[project.scripts] xrr-fitter-cli`. |
| `tools/build_release_spec.py` | Render `[project.scripts]` alongside `[project.gui-scripts]`. |
| `tools/verify_registry.py` | Register `tests/unit/cli` in the `unit` mode. |
| `tests/architecture/test_dependency_rules.py` | Own the `cli` and `__main__` allowlist rows. |
| `tests/architecture/test_public_api.py` | Keep `PUBLIC_NAMES` exact after the re-export. |
| `tests/architecture/test_distribution.py` | Keep the release-spec entry-point assertion exact. |
| `tests/unit/tools/test_verify_registry.py` | Keep the exact-registry assertion in sync. |
| `tests/unit/cli/test_exit_codes.py` | Prove the exit-code mapping in isolation. |
| `tests/unit/cli/test_progress.py` | Prove both progress renderings. |
| `tests/unit/cli/test_dispatch.py` | Prove parser wiring and `freeze_support()` ordering. |
| `tests/integration/test_cli_workflow.py` | Prove real subprocess runs, Qt absence, and CLI/service equivalence. |
| `docs/user-guide.md` | Document the CLI section. |

---

### Task 1: Open the architecture edge and the api gap

**Files:**
- Modify: `tests/architecture/test_dependency_rules.py:74-85`
- Modify: `tests/architecture/test_public_api.py:9-192`
- Modify: `src/xrr_fitter/api.py:3-10,116-166`
- Create: `src/xrr_fitter/cli/__init__.py`

**Interfaces:**
- Consumes: existing `ConfidenceClass` from `xrr_fitter.model.analysis`.
- Produces: `ALLOWED["cli"] == {"api"}`, `ALLOWED["__main__"] == {"gui", "cli"}`, and `ConfidenceClass` in `api.__all__`.
- Preserves: every existing entry in `ALLOWED`, `PACKAGE_EDGE_EXCEPTIONS`, `PUBLIC_NAMES`, and `SIGNATURES`.
- Removes: nothing.

- [ ] **Step 1: Write the failing architecture expectations**

In `tests/architecture/test_dependency_rules.py`, add a fixture test next to the existing `test_fixture_checker_allows_gui_domain_access_only_through_public_api`:

```python
def test_fixture_checker_allows_cli_domain_access_only_through_public_api() -> None:
    source = """
import xrr_fitter.api as api

def run(path):
    return api.load_project(path)
"""

    assert _fixture_kinds("cli.commands", source, "api") == set()
    assert "package-edge" in _fixture_kinds(
        "cli.commands",
        "from xrr_fitter.services.fitting import fit_project\n",
        "services.fitting",
    )
    assert "package-edge" in _fixture_kinds(
        "cli.commands",
        "from xrr_fitter.fit.resume import validate_resume_checkpoint\n",
        "fit.resume",
    )
```

In `tests/architecture/test_public_api.py`, insert `"ConfidenceClass"` into `PUBLIC_NAMES` in alphabetical position — between `"BeamSpec"` and `"DataColumnMapping"`.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts= --import-mode=importlib -p tests.outcome_gate tests/architecture/test_dependency_rules.py::test_fixture_checker_allows_cli_domain_access_only_through_public_api tests/architecture/test_public_api.py::test_api_exports_only_the_complete_supported_surface -q
```

Expected RED: the dependency fixture fails with `unregistered package owner` or an unexpected `package-edge` for `cli`; the api test fails because `tuple(api.__all__) != PUBLIC_NAMES`.

- [ ] **Step 3: Open the two edges**

In `tests/architecture/test_dependency_rules.py`, change the `ALLOWED` table to add one row and extend one:

```python
    "gui": {"gui", "api"},
    "cli": {"cli", "api"},
    "__main__": {"gui", "cli"},
```

`cli` includes itself because the package has multiple internal modules; `gui` already follows this shape. Do not add `services`, `io`, or `fit` to the `cli` row, and do not add a `PACKAGE_EDGE_EXCEPTIONS` entry — an exception here would require the three-way fixture the comment block above that table demands, and no CLI need justifies one.

In `src/xrr_fitter/api.py`, add `ConfidenceClass` to the existing `from xrr_fitter.model.analysis import (...)` block in alphabetical position and add the matching `__all__` entry.

Create `src/xrr_fitter/cli/__init__.py` as an empty file:

```bash
: > src/xrr_fitter/cli/__init__.py
```

- [ ] **Step 4: Confirm GREEN**

Run the exact command from Step 2.

Expected: `2 passed`.

- [ ] **Step 5: Commit the boundary change**

```bash
git add src/xrr_fitter/api.py src/xrr_fitter/cli/__init__.py tests/architecture/test_dependency_rules.py tests/architecture/test_public_api.py
git commit -m "feat: open cli architecture edge and export ConfidenceClass"
```

---

### Task 2: Build the exit-code contract and progress rendering

**Files:**
- Create: `tests/unit/cli/test_exit_codes.py`
- Create: `tests/unit/cli/test_progress.py`
- Create: `src/xrr_fitter/cli/exit_codes.py`
- Create: `src/xrr_fitter/cli/progress.py`

**Interfaces:**
- Consumes: `api.ConfidenceClass`, `api.ProjectFitResult`, `api.ProjectValidation`, `api.FitProgress`.
- Produces: `SUCCESS`, `NOT_CONVERGED`, `INVALID_INPUT`, `STALE_SOURCE`, `fit_exit_code(result)`, `validation_exit_code(validation)`, `render_text(progress)`, `render_json(progress)`.
- Preserves: `ProjectFitResult.datasets[i].fit_result.confidence` semantics; no re-derivation of confidence.
- Removes: nothing.

- [ ] **Step 1: Write the failing exit-code contract**

Create `tests/unit/cli/test_exit_codes.py`:

```python
"""Prove the CLI exit-code contract maps published results, not labels."""

from __future__ import annotations

import pytest

import xrr_fitter.api as api
from xrr_fitter.cli import exit_codes


def test_codes_are_the_documented_four() -> None:
    assert (
        exit_codes.SUCCESS,
        exit_codes.NOT_CONVERGED,
        exit_codes.INVALID_INPUT,
        exit_codes.STALE_SOURCE,
    ) == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (api.ConfidenceClass.TRUSTED, 0),
        (api.ConfidenceClass.CORRELATED, 0),
        (api.ConfidenceClass.MULTIPLE, 1),
        (api.ConfidenceClass.UNTRUSTED, 1),
    ],
)
def test_confidence_decides_the_fit_exit_code(confidence, expected) -> None:
    assert exit_codes.confidence_exit_code(confidence) == expected


def test_untrusted_dataset_dominates_a_mixed_result() -> None:
    codes = (
        api.ConfidenceClass.TRUSTED,
        api.ConfidenceClass.UNTRUSTED,
        api.ConfidenceClass.TRUSTED,
    )

    assert exit_codes.worst_exit_code(codes) == exit_codes.NOT_CONVERGED


def test_cancelled_result_is_not_reported_as_success() -> None:
    assert exit_codes.cancelled_exit_code() == exit_codes.NOT_CONVERGED


def test_every_confidence_member_has_a_mapping() -> None:
    for member in api.ConfidenceClass:
        assert exit_codes.confidence_exit_code(member) in {0, 1}
```

The last test is the point of exporting the enum: adding a fifth `ConfidenceClass` member later fails here instead of silently defaulting to success.

Create `tests/unit/cli/test_progress.py`:

```python
"""Prove both progress renderings carry the full FitProgress contract."""

from __future__ import annotations

import json

import numpy as np

import xrr_fitter.api as api
from xrr_fitter.cli import progress as progress_module


def _progress() -> api.FitProgress:
    return api.FitProgress(
        dataset_id="P1",
        stage="B",
        completed=3,
        total=5,
        best_objective=1.25,
        message="粗搜索完成",
        preview_qz_a_inv=np.array([0.01, 0.02]),
        preview_model_normalized=np.array([1.0, 0.5]),
    )


def test_text_rendering_is_one_line_without_preview_arrays() -> None:
    line = progress_module.render_text(_progress())

    assert "\n" not in line
    assert "P1" in line and "3/5" in line and "粗搜索完成" in line
    assert "0.01" not in line


def test_json_rendering_is_one_parsable_line_with_scalar_fields_only() -> None:
    payload = json.loads(progress_module.render_json(_progress()))

    assert payload == {
        "dataset_id": "P1",
        "stage": "B",
        "completed": 3,
        "total": 5,
        "best_objective": 1.25,
        "message": "粗搜索完成",
    }


def test_json_rendering_survives_a_missing_dataset_id() -> None:
    bare = api.FitProgress(
        dataset_id=None,
        stage="A",
        completed=0,
        total=1,
        best_objective=float("inf"),
        message="开始",
    )

    payload = json.loads(progress_module.render_json(bare))

    assert payload["dataset_id"] is None
    assert payload["best_objective"] is None
```

`best_objective` starts at `inf` in practice and `json.dumps` would emit bare `Infinity`, which is not valid JSON for strict parsers. Mapping non-finite values to `null` keeps the stream consumable by any JSON Lines reader.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts= --import-mode=importlib -p tests.outcome_gate tests/unit/cli -q
```

Expected RED: `ModuleNotFoundError: No module named 'xrr_fitter.cli.exit_codes'`.

- [ ] **Step 3: Implement the two leaf modules**

Create `src/xrr_fitter/cli/exit_codes.py`. Keep the mapping an explicit dict over enum members so a new member raises `KeyError` instead of defaulting:

```python
"""The CLI exit-code contract for unattended orchestration."""

from __future__ import annotations

from collections.abc import Iterable

import xrr_fitter.api as api


SUCCESS = 0
NOT_CONVERGED = 1
INVALID_INPUT = 2
STALE_SOURCE = 3

_CONFIDENCE_CODES = {
    api.ConfidenceClass.TRUSTED: SUCCESS,
    api.ConfidenceClass.CORRELATED: SUCCESS,
    api.ConfidenceClass.MULTIPLE: NOT_CONVERGED,
    api.ConfidenceClass.UNTRUSTED: NOT_CONVERGED,
}


def confidence_exit_code(confidence: api.ConfidenceClass) -> int:
    """Map one published confidence class onto its exit code."""
    return _CONFIDENCE_CODES[confidence]


def worst_exit_code(values: Iterable[api.ConfidenceClass]) -> int:
    """Return the least favourable exit code across every dataset."""
    return max((confidence_exit_code(item) for item in values), default=SUCCESS)


def cancelled_exit_code() -> int:
    """Report a cancelled run as unconverged rather than successful."""
    return NOT_CONVERGED


def fit_exit_code(result: api.ProjectFitResult) -> int:
    """Derive the process exit code from a published project fit result."""
    if result.cancelled:
        return cancelled_exit_code()
    return worst_exit_code(item.fit_result.confidence for item in result.datasets)
```

Create `src/xrr_fitter/cli/progress.py`. Do not serialize the preview arrays — they are per-iteration curves and would dominate the stream:

```python
"""Human-readable and JSON Lines renderings of fit progress."""

from __future__ import annotations

import json
from math import isfinite

import xrr_fitter.api as api


def _objective(value: float) -> float | None:
    return value if isfinite(value) else None


def render_text(progress: api.FitProgress) -> str:
    """Render one progress event as a single human-readable line."""
    dataset = progress.dataset_id or "-"
    objective = _objective(progress.best_objective)
    best = "-" if objective is None else f"{objective:.6g}"
    return (
        f"[{dataset}] 阶段 {progress.stage} "
        f"{progress.completed}/{progress.total} "
        f"best={best} {progress.message}"
    )


def render_json(progress: api.FitProgress) -> str:
    """Render one progress event as a single JSON Lines record."""
    return json.dumps(
        {
            "dataset_id": progress.dataset_id,
            "stage": progress.stage,
            "completed": progress.completed,
            "total": progress.total,
            "best_objective": _objective(progress.best_objective),
            "message": progress.message,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
```

- [ ] **Step 4: Confirm GREEN**

Run the exact command from Step 2.

Expected: all tests in `tests/unit/cli` pass, and `tests/outcome_gate.py` reports no skipped outcome.

- [ ] **Step 5: Commit the contract modules**

```bash
git add src/xrr_fitter/cli/exit_codes.py src/xrr_fitter/cli/progress.py tests/unit/cli/test_exit_codes.py tests/unit/cli/test_progress.py
git commit -m "feat: add cli exit-code contract and progress rendering"
```

---

### Task 3: Wire the subcommands and the parser

**Files:**
- Create: `tests/unit/cli/test_dispatch.py`
- Create: `src/xrr_fitter/cli/commands.py`
- Create: `src/xrr_fitter/cli/main.py`

**Interfaces:**
- Consumes: `api.load_project`, `api.inspect_sources`, `api.preflight_fit`, `api.fit_project`, `api.fit_automatically`, `api.summarize_automatic_results`, `api.run_mcmc`, `api.export_result`, `api.save_project`.
- Produces: `build_parser()`, `main(argv=None) -> int`, and one `run_*` handler per subcommand.
- Preserves: every consumed signature exactly as locked in `tests/architecture/test_public_api.py`.
- Removes: nothing.

- [ ] **Step 1: Write the failing dispatch contract**

Create `tests/unit/cli/test_dispatch.py`. Monkeypatch at the `api` module the CLI imports, so no fit actually runs:

```python
"""Prove parser wiring, freeze-support ordering, and api-only dispatch."""

from __future__ import annotations

import pytest

from xrr_fitter.cli import main as cli_main


def test_subcommands_are_exactly_the_designed_four() -> None:
    parser = cli_main.build_parser()
    actions = [
        action
        for action in parser._subparsers._group_actions  # noqa: SLF001
        if action.choices
    ]

    assert len(actions) == 1
    assert sorted(actions[0].choices) == ["export", "fit", "mcmc", "validate"]


def test_no_subcommand_is_an_input_error_not_a_crash(capsys) -> None:
    assert cli_main.main([]) == 2

    assert "usage: xrr-fitter-cli" in capsys.readouterr().err


def test_freeze_support_runs_before_any_command(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli_main, "freeze_support", lambda: events.append("freeze"))
    monkeypatch.setattr(
        cli_main,
        "_dispatch",
        lambda arguments: (events.append("dispatch"), 0)[1],
    )

    assert cli_main.main(["fit", "project.json"]) == 0
    assert events == ["freeze", "dispatch"]


def test_missing_project_file_is_an_input_error(tmp_path, capsys) -> None:
    missing = tmp_path / "absent.json"

    assert cli_main.main(["validate", str(missing)]) == 2

    assert str(missing) in capsys.readouterr().err


def test_stale_source_maps_to_its_own_exit_code(monkeypatch, tmp_path) -> None:
    import xrr_fitter.api as api
    from xrr_fitter.cli import commands

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")
    stale = api.ProjectValidation(
        datasets=(),
        issues=(api.ValidationIssue(code="source", message="源文件已变化"),),
    )
    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(commands.api, "inspect_sources", lambda project: stale)

    assert cli_main.main(["validate", str(project_path)]) == 3


def test_cli_package_never_imports_pyside6() -> None:
    import ast
    from pathlib import Path

    root = Path(cli_main.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            assert not any(name.startswith("PySide6") for name in names), path
```

The stale-source test distinguishes exit code `3` from `2`: a project that parses but whose sources moved is a different operational failure than a malformed project, and an orchestrator needs to tell them apart.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts= --import-mode=importlib -p tests.outcome_gate tests/unit/cli/test_dispatch.py -q
```

Expected RED: `ModuleNotFoundError: No module named 'xrr_fitter.cli.main'`.

- [ ] **Step 3: Implement the handlers**

Create `src/xrr_fitter/cli/commands.py`. Import the api module as `api` so tests can monkeypatch `commands.api.<name>`, and keep every handler under the Radon block limit by extracting the shared load-and-validate step:

```python
"""One handler per CLI subcommand, reaching domain behavior only through api."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import xrr_fitter.api as api
from xrr_fitter.cli import exit_codes, progress as progress_module


class CommandError(Exception):
    """A user-facing failure carrying its own exit code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _load(path: str) -> api.XrrProject:
    target = Path(path)
    if not target.is_file():
        raise CommandError(f"工程文件不存在：{target}", exit_codes.INVALID_INPUT)
    try:
        return api.load_project(target)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise CommandError(f"工程文件不合法：{error}", exit_codes.INVALID_INPUT) from error


def _require_fresh_sources(project: api.XrrProject) -> None:
    validation = api.inspect_sources(project)
    if not validation.valid:
        detail = "；".join(item.message for item in validation.issues) or "源文件校验失败"
        raise CommandError(detail, exit_codes.STALE_SOURCE)


def _progress_sink(as_json: bool) -> Callable[[api.FitProgress], None]:
    if as_json:
        return lambda event: print(progress_module.render_json(event), flush=True)
    return lambda event: print(progress_module.render_text(event), file=sys.stderr, flush=True)


def run_validate(arguments) -> int:
    """Check sources and fit readiness without running any optimizer."""
    project = _load(arguments.project)
    _require_fresh_sources(project)
    readiness = api.preflight_fit(project)
    print(readiness.message)
    return exit_codes.SUCCESS if readiness.ready else exit_codes.INVALID_INPUT


def run_fit(arguments) -> int:
    """Run the fit pipeline, resuming automatically from any stored checkpoint."""
    project = _load(arguments.project)
    _require_fresh_sources(project)
    sink = _progress_sink(arguments.json_progress)
    result = _fit_result(project, arguments, sink)
    if arguments.output is not None:
        api.save_project(result.updated_project, arguments.output)
    for warning in result.warnings:
        print(warning, file=sys.stderr)
    return exit_codes.fit_exit_code(result)


def _fit_result(project, arguments, sink) -> api.ProjectFitResult:
    if arguments.auto:
        return api.fit_automatically(project, progress_callback=sink)
    readiness = api.preflight_fit(project)
    if not readiness.ready:
        raise CommandError(readiness.message, exit_codes.INVALID_INPUT)
    return api.fit_project(project, progress_callback=sink)


def run_mcmc(arguments) -> int:
    """Sample the selected candidate and persist the augmented project."""
    project = _load(arguments.project)
    _require_fresh_sources(project)
    config = api.McmcConfig(
        walkers=arguments.walkers,
        burn_in=arguments.burn_in,
        production_steps=arguments.steps,
    )
    updated = api.run_mcmc(
        project,
        arguments.dataset,
        arguments.candidate,
        config,
        progress_callback=_progress_sink(arguments.json_progress),
    )
    api.save_project(updated, arguments.output or arguments.project)
    return exit_codes.SUCCESS


def run_export(arguments) -> int:
    """Publish an existing project's results atomically."""
    project = _load(arguments.project)
    manifest = api.export_result(project, arguments.output_dir)
    print(manifest.directory)
    return exit_codes.SUCCESS
```

Do not call `api.start_fit_job()` from these handlers. That path spawns a worker process and exists for cancellable GUI use; the CLI is already its own process and the synchronous call keeps the exit code a direct function of the returned result.

Create `src/xrr_fitter/cli/main.py`:

```python
"""Console entry point for unattended XRR workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from multiprocessing import freeze_support
import sys

from xrr_fitter.cli import commands, exit_codes


_HANDLERS = {
    "fit": commands.run_fit,
    "mcmc": commands.run_mcmc,
    "export": commands.run_export,
    "validate": commands.run_validate,
}


def _add_progress_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json-progress",
        action="store_true",
        help="把进度写成 stdout 的 JSON Lines，而不是 stderr 的文本行",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="xrr-fitter-cli",
        description="X 射线反射率拟合的无头命令行入口",
    )
    subparsers = parser.add_subparsers(dest="command")

    fit = subparsers.add_parser("fit", help="运行拟合流水线")
    fit.add_argument("project")
    fit.add_argument("--auto", action="store_true", help="走自动批次拟合路径")
    fit.add_argument("--output", help="把更新后的工程写到该路径")
    _add_progress_flag(fit)

    mcmc = subparsers.add_parser("mcmc", help="对选定候选运行 MCMC 采样")
    mcmc.add_argument("project")
    mcmc.add_argument("--dataset", required=True)
    mcmc.add_argument("--candidate", required=True)
    mcmc.add_argument("--walkers", type=int, required=True)
    mcmc.add_argument("--burn-in", type=int, required=True, dest="burn_in")
    mcmc.add_argument("--steps", type=int, required=True)
    mcmc.add_argument("--output")
    _add_progress_flag(mcmc)

    export = subparsers.add_parser("export", help="发布已有结果")
    export.add_argument("project")
    export.add_argument("output_dir")

    validate = subparsers.add_parser("validate", help="只读校验工程与源文件")
    validate.add_argument("project")
    return parser


def _dispatch(arguments: argparse.Namespace) -> int:
    return _HANDLERS[arguments.command](arguments)


def main(argv: Sequence[str] | None = None) -> int:
    """Enable frozen workers, parse arguments, and run one subcommand."""
    freeze_support()
    parser = build_parser()
    arguments = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if arguments.command is None:
        parser.print_usage(sys.stderr)
        return exit_codes.INVALID_INPUT
    try:
        return _dispatch(arguments)
    except commands.CommandError as error:
        print(str(error), file=sys.stderr)
        return error.code


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Confirm GREEN**

Run the exact command from Step 2.

Expected: every dispatch test passes, including the AST check that no CLI module imports PySide6.

- [ ] **Step 5: Register the new unit directory**

In `tools/verify_registry.py`, add `"tests/unit/cli"` to the `unit` mode tuple, keeping the existing order convention with the new entry after `"tests/unit/services"`. Apply the identical edit to `_expected_registry` in `tests/unit/tools/test_verify_registry.py` — `test_registry_is_exact_for_completed_suites` asserts exact equality, so the two must change together.

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py tools
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py unit
```

Expected: both exit 0, and the `unit` run now collects `tests/unit/cli`.

- [ ] **Step 6: Commit the CLI surface**

```bash
git add src/xrr_fitter/cli/commands.py src/xrr_fitter/cli/main.py tests/unit/cli/test_dispatch.py tools/verify_registry.py tests/unit/tools/test_verify_registry.py
git commit -m "feat: add headless cli subcommands"
```

---

### Task 4: Declare the entry point and repair the release renderer

**Files:**
- Modify: `pyproject.toml:25-27`
- Modify: `tools/build_release_spec.py:210-232,292-298`
- Modify: `tests/architecture/test_distribution.py:25-50`

**Interfaces:**
- Consumes: `pyproject.toml`'s `[project]` table.
- Produces: `[project.scripts] xrr-fitter-cli = "xrr_fitter.cli.main:main"`, and a release spec that renders both script tables.
- Preserves: `[project.gui-scripts] xrr-fitter` exactly as it is today.
- Removes: nothing.

- [ ] **Step 1: Write the failing distribution expectation**

In `tests/architecture/test_distribution.py`, extend the entry-point assertion next to the existing `gui-scripts` check:

```python
    assert payload["project"]["scripts"] == {
        "xrr-fitter-cli": "xrr_fitter.cli.main:main",
    }
```

Add a renderer test proving both tables reach the generated metadata:

```python
def test_release_spec_renders_both_script_tables() -> None:
    module = _release_spec_module()
    rendered = module._render_pyproject(  # noqa: SLF001
        {
            "name": "x",
            "version": "0",
            "requires-python": ">=3.12",
            "dependencies": [],
            "optional-dependencies": {"test": []},
            "gui-scripts": {"xrr-fitter": "xrr_fitter.__main__:main"},
            "scripts": {"xrr-fitter-cli": "xrr_fitter.cli.main:main"},
        }
    )

    assert "[project.gui-scripts]" in rendered
    assert "[project.scripts]" in rendered
    assert '"xrr_fitter.cli.main:main"' in rendered
```

Match the real helper name and call convention in `tools/build_release_spec.py` when writing this test; the assertion content is what matters, not the exact private name used here.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts= --import-mode=importlib -p tests.outcome_gate tests/architecture/test_distribution.py -q
```

Expected RED: `KeyError: 'scripts'` or a missing `[project.scripts]` section, because `tools/build_release_spec.py:217` reads only `gui-scripts`.

- [ ] **Step 3: Declare the entry point and generalize the renderer**

In `pyproject.toml`, add the table directly after the existing one:

```toml
[project.scripts]
xrr-fitter-cli = "xrr_fitter.cli.main:main"
```

In `tools/build_release_spec.py`, replace the single `gui-scripts` branch with a loop over both table names so neither can be silently dropped:

```python
    for table in ("gui-scripts", "scripts"):
        scripts = project.get(table)
        if scripts is None:
            continue
        if not isinstance(scripts, dict) or not scripts:
            raise ValueError(f"invalid {table} metadata")
        lines.extend(("", f"[project.{table}]"))
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in sorted(scripts.items()))
```

Update `_expected_generated_metadata` so `entry_points.txt` is expected when either table is present:

```python
    if project.get("gui-scripts") is None and project.get("scripts") is None:
        return GENERATED_METADATA
```

- [ ] **Step 4: Confirm GREEN and regenerate the release spec**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts= --import-mode=importlib -p tests.outcome_gate tests/architecture/test_distribution.py -q
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py distribution
```

Expected: both exit 0. If `tools/build_release_spec.py` writes `verification/release-spec.json`, regenerate it in the same commit so the fixture cannot drift from `pyproject.toml`; inspect the diff and confirm the only change is the added `scripts` table.

- [ ] **Step 5: Confirm the installed console script actually runs**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m xrr_fitter.cli.main --help
```

Expected: exit 0 and `usage: xrr-fitter-cli` on stdout. This is the module path the console script wraps; the packaged-script check happens in `distribution` mode above.

- [ ] **Step 6: Commit the packaging change**

```bash
git add pyproject.toml tools/build_release_spec.py tests/architecture/test_distribution.py verification/release-spec.json
git commit -m "feat: declare xrr-fitter-cli console entry point"
```

---

### Task 5: Prove Qt absence and CLI/service equivalence

**Files:**
- Create: `tests/integration/test_cli_workflow.py`
- Modify: `tools/verify_registry.py`
- Modify: `tests/unit/tools/test_verify_registry.py`
- Modify: `docs/user-guide.md`

**Interfaces:**
- Consumes: the installed module path `xrr_fitter.cli.main`, the example projects already used by `tests/integration/`.
- Produces: subprocess evidence that the CLI runs without PySide6 importable, and that a CLI fit equals a same-process `api.fit_project` fit bit for bit.
- Preserves: `SERVICE_SEED_TREE_VERSION` determinism as an asserted fact rather than a claim.
- Removes: nothing.

- [ ] **Step 1: Write the failing integration contract**

Create `tests/integration/test_cli_workflow.py`, reusing the PySide6 guard idiom from `tests/integration/test_entrypoints.py`:

```python
"""Prove the CLI runs without Qt and agrees with the in-process api."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _guarded_environment(tmp_path: Path) -> dict[str, str]:
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "class Guard:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'PySide6' or fullname.startswith('PySide6.'):\n"
        "            raise RuntimeError('PySide6 imported by the headless CLI')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Guard())\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(guard), str(ROOT / "src")))
    return environment


def _run(arguments: tuple[str, ...], environment: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        (sys.executable, "-m", "xrr_fitter.cli.main", *arguments),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_help_renders_without_importing_pyside6(tmp_path: Path) -> None:
    result = _run(("--help",), _guarded_environment(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage: xrr-fitter-cli" in result.stdout
    assert "PySide6 imported" not in result.stdout + result.stderr


def test_validate_reports_a_clean_example_project(tmp_path: Path) -> None:
    project_path = _write_example_project(tmp_path)

    result = _run(("validate", str(project_path)), _guarded_environment(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr


def test_json_progress_is_line_delimited_json(tmp_path: Path) -> None:
    project_path = _write_example_project(tmp_path)

    result = _run(
        ("fit", str(project_path), "--json-progress", "--output", str(tmp_path / "out.json")),
        _guarded_environment(tmp_path),
    )

    assert result.returncode in {0, 1}, result.stdout + result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert records
    assert all(set(item) == {
        "best_objective",
        "completed",
        "dataset_id",
        "message",
        "stage",
        "total",
    } for item in records)


def test_cli_fit_matches_the_in_process_api_fit(tmp_path: Path) -> None:
    import xrr_fitter.api as api

    project_path = _write_example_project(tmp_path)
    cli_output = tmp_path / "cli.json"

    result = _run(
        ("fit", str(project_path), "--output", str(cli_output)),
        _guarded_environment(tmp_path),
    )
    assert result.returncode in {0, 1}, result.stdout + result.stderr

    direct = api.fit_project(api.load_project(project_path))
    direct_output = tmp_path / "direct.json"
    api.save_project(direct.updated_project, direct_output)

    assert json.loads(cli_output.read_text(encoding="utf-8")) == json.loads(
        direct_output.read_text(encoding="utf-8")
    )
```

Write `_write_example_project` using whichever example-project helper the neighbouring integration tests already use, so this file introduces no new fixture format. If the fastest available example still makes the equivalence test slow, shrink its fit budget through the project's own `FitConfig` rather than adding a marker — every mode in `pr-verify.yml` runs on every PR, and a `skip` would trip `tests/outcome_gate.py`.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts= --import-mode=importlib -p tests.outcome_gate tests/integration/test_cli_workflow.py -q
```

Expected RED: the runs fail before the assertions because `xrr_fitter.cli.main` is not yet reachable under the guarded environment, or the equivalence comparison differs.

- [ ] **Step 3: Register the integration module**

In `tools/verify_registry.py`, add `"tests/integration/test_cli_workflow.py"` to the `integration` mode's explicit file list, and mirror the edit in `_expected_registry` in `tests/unit/tools/test_verify_registry.py`.

- [ ] **Step 4: Confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -o addopts= --import-mode=importlib -p tests.outcome_gate tests/integration/test_cli_workflow.py -q
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py tools
```

Expected: both exit 0. If the equivalence test fails, do not loosen it to a tolerance comparison — a mismatch means the CLI is not going through the same seeded path, which is the defect this test exists to catch.

- [ ] **Step 5: Document the CLI**

Add a new `## 14. 无头命令行` section to `docs/user-guide.md` after section 13. Cover the four subcommands, the exit-code table, and one paragraph on resume that matches the design:

```markdown
续跑不需要单独的子命令。`--output` 或 GUI 写出的 checkpoint 就是一份完整工程文件，
再次对它执行 `xrr-fitter-cli fit` 会自动核对 data hash、structure、instrument、
parameter settings、config、stage graph、candidate order、seed ledger 与 joint
layout fingerprint，一致则从记录的 stage 继续，不一致则显式拒绝并保留原结果。
```

- [ ] **Step 6: Run the full affected gate set**

Run each command separately:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py quality
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py tools
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py unit
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py integration
PYTHONDONTWRITEBYTECODE=1 python tools/verify.py distribution
PYTHONDONTWRITEBYTECODE=1 python tools/check_radon.py
PYTHONDONTWRITEBYTECODE=1 python tools/check_hygiene.py
git diff --check
```

Expected: every command exits 0. `quality` covers the architecture suite including the new `cli` allowlist row and the `PUBLIC_NAMES` change; `check_radon.py` covers the new modules; `check_hygiene.py` confirms no stray root-level artifacts.

- [ ] **Step 7: Commit the verification layer**

```bash
git add tests/integration/test_cli_workflow.py tools/verify_registry.py tests/unit/tools/test_verify_registry.py docs/user-guide.md
git commit -m "test: prove headless cli runs without qt and matches api"
```

---

## 最终验收记录

完成全部 Task 后，在此记录本轮新鲜验证证据，不要复制预期文本：

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 单元 | `python tools/verify.py unit` | |
| 工具 | `python tools/verify.py tools` | |
| 质量与架构 | `python tools/verify.py quality` | |
| 集成 | `python tools/verify.py integration` | |
| 发布 | `python tools/verify.py distribution` | |
| 复杂度 | `python tools/check_radon.py` | |
| 卫生 | `python tools/check_hygiene.py` | |

剩余风险与未验证项也写在这里，包括 Windows 上 `xrr-fitter-cli` 控制台脚本的实际行为——
这一条只能在 `windows-executable.yml` 跑过之后才算验证。
