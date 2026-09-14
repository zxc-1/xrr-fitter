"""Prove the CLI runs without Qt and agrees with the in-process api."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

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
            freedom=api.ParameterFreedom.from_locked(definition.name != free_name),
        )
        for definition in definitions
    )
    return api.set_parameter_settings(value, dataset_id, settings)


def _write_example_project(tmp_path: Path) -> Path:
    project_path = tmp_path / "project.xrrproj.json"
    api.save_project(_fast_single_project(), project_path)
    return project_path


def _write_fitted_project(tmp_path: Path) -> Path:
    project_path = tmp_path / "fitted.xrrproj.json"
    fitted = api.fit_project(_fast_single_project()).updated_project
    api.save_project(fitted, project_path)
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


def _run(arguments: tuple[str, ...], environment: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        (sys.executable, "-m", "xrr_fitter.cli.main", *arguments),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_mode_display(result: subprocess.CompletedProcess, mode: str) -> None:
    assert result.stderr.startswith(f"噪声模式：{mode}")
    requirement = {
        "robust_log": "强度加稳定下限后大于 0",
        "gaussian": "每个拟合点提供有限、严格为正且与强度同单位的标准差 sigma",
        "poisson": "未合并的原始非负整数计数",
    }[mode]
    assert requirement in result.stderr.splitlines()[0]


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
    _assert_mode_display(result, "robust_log")
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
    _assert_mode_display(result, "robust_log")
    assert result.stdout == ""

    direct = api.fit_project(api.load_project(project_path))
    direct_output = tmp_path / "direct.json"
    api.save_project(direct.updated_project, direct_output)

    assert json.loads(cli_output.read_text(encoding="utf-8")) == json.loads(direct_output.read_text(encoding="utf-8"))


def test_cli_export_prints_verifiable_manifest_evidence(tmp_path: Path) -> None:
    project_path = _write_fitted_project(tmp_path)
    output_directory = tmp_path / "exports"

    result = _run(
        ("export", str(project_path), str(output_directory)),
        _guarded_environment(tmp_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    run_line, manifest_line, digest_line = result.stdout.splitlines()
    run_directory = Path(run_line)
    manifest_path = Path(manifest_line.removeprefix("manifest: "))
    manifest_digest = digest_line.removeprefix("manifest_sha256: ")
    assert manifest_path == run_directory / "export_manifest.json"
    assert manifest_digest == sha256(manifest_path.read_bytes()).hexdigest()


def _mode_project(
    tmp_path: Path, *, saved_mode: str = "robust_log", sigma: bool = True, fractional: bool = False
) -> api.XrrProject:
    template = _fast_single_project()
    dataset = template.datasets[0]
    data = api.import_data(Path(template.base_directory) / dataset.source_path, dataset.beam)
    angles = data.two_theta_deg[::20]
    counts = np.rint(10000 * data.intensity_normalized[::20])
    if fractional:
        counts[2] += 0.5
    columns = (angles, counts, np.sqrt(counts + 1)) if sigma else (angles, counts)
    source = tmp_path / "counts.xy"
    np.savetxt(source, np.column_stack(columns), fmt="%.17g")
    value = api.set_fit_config(api.new_project(), replace(template.fit_config, noise_model=saved_mode))
    value = api.add_dataset(
        value,
        source,
        dataset.instrument,
        beam=dataset.beam,
        column_mapping=api.DataColumnMapping(intensity_sigma=2 if sigma else None),
    )
    value = api.set_structure(value, "counts", dataset.structure)
    value = api.set_parameter_settings(value, "counts", dataset.parameter_settings)
    mask = np.array(value.datasets[0].fit_mask)
    mask[5] = False
    value = api.set_fit_mask(value, "counts", mask)
    automatic = api.DatasetAutomation(import_batch_id="mode-test", role=api.AutomaticRole.UNROUTED)
    return replace(
        value,
        measurement_preset=template.measurement_preset,
        datasets=(replace(value.datasets[0], automation=automatic),),
    )


@pytest.mark.parametrize("mode", ("gaussian", "poisson"))
def test_cli_mode_change_discards_old_checkpoint_and_matches_public_api(tmp_path: Path, mode: str) -> None:
    original = api.fit_project(_mode_project(tmp_path)).updated_project
    assert original.datasets[0].checkpoint is not None
    project_path = tmp_path / "before.json"
    api.save_project(original, project_path)
    original_bytes = project_path.read_bytes()
    output = tmp_path / "cli.json"

    result = _run(
        ("fit", str(project_path), "--noise-model", mode, "--output", str(output)),
        _guarded_environment(tmp_path),
    )

    assert result.returncode in {0, 1}, result.stdout + result.stderr
    _assert_mode_display(result, mode)
    assert result.stdout == ""
    loaded = api.load_project(output)
    assert loaded.fit_config.noise_model == mode
    assert loaded.datasets[0].fit_mask == original.datasets[0].fit_mask
    assert {candidate.noise_model for candidate in loaded.datasets[0].last_valid_result.candidates} == {mode}
    expected = api.set_fit_config(api.load_project(project_path), replace(original.fit_config, noise_model=mode))
    direct = api.fit_project(expected).updated_project
    direct_output = tmp_path / "direct.json"
    api.save_project(direct, direct_output)
    assert json.loads(output.read_text()) == json.loads(direct_output.read_text())
    assert project_path.read_bytes() == original_bytes


def test_cli_without_mode_override_resumes_saved_likelihood_checkpoint(tmp_path: Path) -> None:
    original = api.fit_project(_mode_project(tmp_path, saved_mode="gaussian")).updated_project
    before = original.datasets[0]
    assert before.checkpoint is not None and before.checkpoint.stage == "E"
    resumable = replace(original, datasets=(replace(before, last_valid_result=None),))
    source = tmp_path / "saved-gaussian.json"
    output = tmp_path / "resumed.json"
    api.save_project(resumable, source)

    result = _run(("fit", str(source), "--output", str(output)), _guarded_environment(tmp_path))

    assert result.returncode in {0, 1}, result.stdout + result.stderr
    _assert_mode_display(result, "gaussian")
    assert result.stdout == ""
    loaded = api.load_project(output)
    assert loaded.fit_config.noise_model == "gaussian"
    assert loaded.datasets[0].checkpoint.stage_summaries == before.checkpoint.stage_summaries
    assert loaded.datasets[0].last_valid_result.child_seeds == before.last_valid_result.child_seeds
    assert loaded.datasets[0].fit_mask == before.fit_mask


@pytest.mark.parametrize("mode", ("gaussian", "poisson"))
def test_cli_auto_forwards_the_explicit_mode_to_real_fitting(tmp_path: Path, mode: str) -> None:
    original = _mode_project(tmp_path)
    source = tmp_path / "automatic.json"
    output = tmp_path / "fitted.json"
    api.save_project(original, source)

    result = _run(
        ("fit", str(source), "--auto", "--noise-model", mode, "--output", str(output)),
        _guarded_environment(tmp_path),
    )

    assert result.returncode in {0, 1}, result.stdout + result.stderr
    _assert_mode_display(result, mode)
    assert result.stdout == ""
    loaded = api.load_project(output)
    fitted = loaded.datasets[0].last_valid_result
    assert loaded.fit_config.noise_model == mode
    assert fitted is not None
    assert {candidate.noise_model for candidate in fitted.candidates} == {mode}
    assert loaded.datasets[0].fit_mask == original.datasets[0].fit_mask


@pytest.mark.parametrize("automatic", (False, True))
@pytest.mark.parametrize(
    ("mode", "sigma", "fractional", "message"),
    (("gaussian", False, False, "requires known intensity sigma"), ("poisson", True, True, "integer raw counts")),
)
def test_cli_invalid_mode_input_fails_before_progress_or_output(
    tmp_path: Path, mode: str, sigma: bool, fractional: bool, message: str, automatic: bool
) -> None:
    original = _mode_project(tmp_path, sigma=sigma, fractional=fractional)
    source = tmp_path / "invalid.json"
    output = tmp_path / "must-not-exist.json"
    api.save_project(original, source)
    original_bytes = source.read_bytes()
    auto = ("--auto",) if automatic else ()

    result = _run(
        ("fit", str(source), *auto, "--noise-model", mode, "--json-progress", "--output", str(output)),
        _guarded_environment(tmp_path),
    )

    assert result.returncode == 2, result.stdout + result.stderr
    _assert_mode_display(result, mode)
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert not output.exists()
    assert source.read_bytes() == original_bytes
