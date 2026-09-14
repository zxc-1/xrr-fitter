"""Public locked-model fitting must publish analysis, including an empty axis."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import xrr_fitter.api as api

ROOT = Path(__file__).resolve().parents[2]


def _locked_project(tmp_path: Path, mode: str, joint: bool) -> api.XrrProject:
    template = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    original = template.datasets[0]
    config = replace(
        api.FitConfig.fast(1701), noise_model=mode, local_workers=1, profile_steps=5, scale_prior_enabled=False
    )
    budget = replace(config.budget, short_de_maxiter=0, full_de_maxiter=0, bootstrap_samples=8)
    project = api.set_fit_config(api.new_project(), replace(config, budget=budget))
    source = tmp_path / "locked.xy"
    np.savetxt(source, np.column_stack((np.linspace(0.1, 3.2, 80), np.ones(80), np.full(80, 0.02))), fmt="%.17g")
    instrument = replace(original.instrument, footprint_mode="none")
    for _ in range(2 if joint else 1):
        project = api.add_dataset(
            project, source, instrument, beam=original.beam, column_mapping=api.DataColumnMapping(intensity_sigma=2)
        )
        dataset_id = project.datasets[-1].dataset_id
        project = api.set_structure(project, dataset_id, original.structure)
        settings = tuple(
            api.ParameterSetting(value.name, value.initial, value.initial, value.initial, locked=True)
            for value in api.describe_parameters(project, dataset_id)
        )
        project = api.set_parameter_settings(project, dataset_id, settings)
    return api.set_batch_mode(project, "joint") if joint else project


def _roundtrip_fit(project, target):
    output = api.fit_project(project)
    assert not output.cancelled
    assert all(dataset.last_valid_result is not None for dataset in output.updated_project.datasets), output.warnings
    api.save_project(output.updated_project, target)
    return api.load_project(target)


def _assert_empty_uncertainty_axis(report):
    assert report.correlation_names == ()
    assert report.profiles == ()
    assert report.bootstrap_performed is False
    assert report.correlation_matrix.shape == (0, 0)


def _assert_residual_diagnostics_performed(report):
    assert report.member_residuals
    assert all(member.executed for member in report.member_residuals)


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
@pytest.mark.parametrize("joint", (False, True))
def test_public_locked_fit_publishes_and_roundtrips_empty_parameter_axis(tmp_path, mode, joint) -> None:
    project = _locked_project(tmp_path, mode, joint)
    assert api.preflight_fit(project).ready

    saved = _roundtrip_fit(project, tmp_path / "fitted.xrrproj.json")

    for dataset in saved.datasets:
        result = dataset.last_valid_result
        assert result.best_candidate.valid
        assert result.best_candidate.unit_vector.shape == (0,)
        _assert_empty_uncertainty_axis(result.uncertainty)
        _assert_residual_diagnostics_performed(result.uncertainty)
