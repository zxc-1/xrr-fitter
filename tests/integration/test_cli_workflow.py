"""Prove the CLI runs without Qt and agrees with the in-process api."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import xrr_fitter.api as api


ROOT = Path(__file__).resolve().parents[2]


def _fast_single_project() -> api.XrrProject:
    value = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    budget = replace(
        value.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=5,
        local_nfev_per_parameter=1,
        bootstrap_samples=1,
    )
    config = replace(
        api.FitConfig.fast(value.master_seed),
        budget=budget,
        local_workers=1,
        scale_prior_enabled=False,
    )
    value = replace(value, fit_config=config)
    dataset_id = value.datasets[0].dataset_id
    definitions = api.describe_parameters(value, dataset_id)
    free_name = "component.0.thickness_a"
    settings = tuple(
        api.ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name == free_name else definition.initial,
            definition.upper if definition.name == free_name else definition.initial,
            locked=definition.name != free_name,
        )
        for definition in definitions
    )
    return api.set_parameter_settings(value, dataset_id, settings)


def _write_example_project(tmp_path: Path) -> Path:
    project_path = tmp_path / "project.xrrproj.json"
    api.save_project(_fast_single_project(), project_path)
    return project_path


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


def _run(
    arguments: tuple[str, ...], environment: dict[str, str]
) -> subprocess.CompletedProcess:
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
    assert all(
        set(item)
        == {
            "best_objective",
            "completed",
            "dataset_id",
            "message",
            "stage",
            "total",
        }
        for item in records
    )


def test_cli_fit_matches_the_in_process_api_fit(tmp_path: Path) -> None:
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
