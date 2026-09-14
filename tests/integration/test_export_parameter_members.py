"""Real joint exports retain exact member ownership through persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.integration.test_export_workflow import (
    _assert_export_matches_saved_result,
    _mode_project,
    _thickness_only,
)
from tests.support.export_sigma_cases import _assert_saved_sigmas, _assert_thickness_sigma, _automatic_sigma_project

import xrr_fitter.api as api


def _nonshared_project(tmp_path: Path, noise_model: str) -> api.XrrProject:
    value = api.set_sharing_rules(_mode_project(tmp_path, noise_model, True), ())
    second = value.datasets[1]
    observations = np.loadtxt(tmp_path / "observations.xy")
    observations[:, 2 if noise_model == "gaussian" else 1] *= 2
    source = tmp_path / "second-observations.xy"
    np.savetxt(source, observations, fmt="%.17g")
    value = api.remove_dataset(value, second.dataset_id)
    value = api.add_dataset(value, source, second.instrument, beam=second.beam, column_mapping=second.column_mapping)
    dataset_id = value.datasets[-1].dataset_id
    value = api.set_structure(value, dataset_id, second.structure)
    value = _thickness_only(value, dataset_id)
    return api.set_batch_mode(value, "joint")


def _assert_nonshared_member_axis(project: api.XrrProject) -> None:
    report = project.datasets[0].last_valid_result.uncertainty
    assert report.covariance is not None
    assert report.parameter_sigma[0] != report.parameter_sigma[1]
    expected_members = tuple(
        (api.ParameterReference(item.dataset_id, "component.0.thickness_a"),) for item in project.datasets
    )
    assert report.parameter_members == expected_members


@pytest.mark.parametrize("noise_model", ("gaussian", "poisson"))
def test_nonshared_joint_exports_each_members_own_saved_sigma(tmp_path: Path, noise_model: str) -> None:
    value = _nonshared_project(tmp_path, noise_model)
    operation = api.fit_project(value)
    assert all(item.fit_result.best_candidate is not None for item in operation.datasets), operation.warnings
    fitted = operation.updated_project
    project_path = tmp_path / "nonshared.xrrproj.json"
    api.save_project(fitted, project_path)
    loaded = api.load_project(project_path)
    manifest = api.export_result(loaded, tmp_path / "exports", include_ort=True)

    _assert_nonshared_member_axis(loaded)
    for dataset in loaded.datasets:
        _assert_thickness_sigma(manifest, dataset, f"{dataset.dataset_id}:component.0.thickness_a")
        _assert_export_matches_saved_result(manifest, dataset)


def _assert_automatic_member(dataset, members, manifest) -> None:
    assert dataset.automation.role is api.AutomaticRole.JOINT
    assert dataset.automation.status is api.AutomaticStatus.PASSED
    report = dataset.last_valid_result.uncertainty
    assert report.parameter_members == members
    assert report.covariance is not None
    assert report.parameter_sigma[0] > 0.0
    expected = {parameter.name: None for parameter in dataset.last_valid_result.best_candidate.parameters}
    expected["component.0.density_scale"] = float(report.parameter_sigma[0])
    _assert_saved_sigmas(manifest, dataset.dataset_id, expected)
    _assert_export_matches_saved_result(manifest, dataset)


def test_automatic_joint_exports_saved_sigma_without_persisted_project_sharing_rules(tmp_path: Path) -> None:
    value = _automatic_sigma_project(tmp_path)
    fitted = api.fit_automatically(value).updated_project
    project_path = tmp_path / "automatic.xrrproj.json"
    api.save_project(fitted, project_path)
    loaded = api.load_project(project_path)
    manifest = api.export_result(loaded, tmp_path / "exports", include_ort=True)

    assert loaded.batch_mode == "independent"
    assert loaded.sharing_rules == ()
    members = (
        tuple(api.ParameterReference(dataset.dataset_id, "component.0.density_scale") for dataset in loaded.datasets),
    )
    for dataset in loaded.datasets:
        _assert_automatic_member(dataset, members, manifest)
