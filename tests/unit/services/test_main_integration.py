"""Cross-branch public contracts retain Poisson evidence and the newer UI inputs."""

from dataclasses import fields, replace
from inspect import signature

import numpy as np
import pytest

import xrr_fitter.api as api
from xrr_fitter.model.bootstrap import BootstrapResult
from xrr_fitter.services import projects as project_service


def test_theta_poisson_import_keeps_angles_and_unscaled_zero_counts(tmp_path):
    assert {"angle_convention", "noise_model"} <= signature(api.import_data).parameters.keys()
    path = tmp_path / "theta-counts.xy"
    path.write_text("0.1 0\n0.2 7\n0.3 12\n", encoding="utf-8")
    data = api.import_data(path, api.BeamSpec("monochromatic"), angle_convention="theta", noise_model="poisson")
    np.testing.assert_allclose(data.two_theta_deg, [0.2, 0.4, 0.6])
    np.testing.assert_array_equal(data.intensity_raw, [0.0, 7.0, 12.0])
    assert data.angle_convention == "theta"
    assert np.all(data.fit_mask)


def test_current_parameter_freedom_does_not_turn_range_only_into_fixed():
    assert "freedom" in {field.name for field in fields(api.ParameterSetting)}
    setting = api.ParameterSetting("instrument.scale", 1.0, 0.5, 2.0, api.ParameterFreedom.RANGE_ONLY)
    assert not setting.locked
    assert setting.freedom is api.ParameterFreedom.RANGE_ONLY


def test_project_revalidation_retains_theta_and_poisson_source_declarations(tmp_path, monkeypatch):
    source = tmp_path / "theta-counts.xy"
    source.write_text("0.1 0\n0.2 7\n0.3 12\n", encoding="utf-8")
    project = api.new_project()
    project = api.set_fit_config(project, replace(project.fit_config, noise_model="poisson"))
    project = api.add_dataset(project, source, api.InstrumentSpec(), angle_convention="theta")
    reread = project_service.read_xy
    prepared = []

    def record_prepared(*args, **kwargs):
        data = reread(*args, **kwargs)
        prepared.append(data)
        return data

    monkeypatch.setattr(project_service, "read_xy", record_prepared)
    target = tmp_path / "theta.xrrproj.json"
    api.save_project(project, target)
    restored = api.load_project(target)

    assert restored.datasets[0].angle_convention == "theta"
    assert len(prepared) == 2
    assert [data.angle_convention for data in prepared] == ["theta", "theta"]
    for data in prepared:
        np.testing.assert_allclose(data.two_theta_deg, [0.2, 0.4, 0.6])
        np.testing.assert_array_equal(data.intensity_raw, [0.0, 7.0, 12.0])
        assert np.all(data.fit_mask)


def test_profile_display_threshold_coexists_with_total_likelihood_evidence():
    assert "objective_threshold" in signature(api.ParameterProfile).parameters
    profile = api.ParameterProfile(
        "thickness",
        np.array([9.0, 10.0, 11.0]),
        np.array([2.1, 2.0, 2.1]),
        True,
        True,
        interval_kind="likelihood_ratio",
        confidence_level=0.95,
        method="likelihood_ratio",
        delta_total=3.84,
        objective_point_count=100,
        objective_threshold=2.0384,
    )
    assert profile.objective_delta == pytest.approx(0.0384)
    assert profile.objective_threshold == pytest.approx(2.0384)
    assert not profile.objectives.flags.writeable


def test_report_displays_actual_bootstrap_attempts_including_failed_refits():
    assert hasattr(api.UncertaintyReport, "bootstrap_sample_count")
    evidence = BootstrapResult(
        ("x",),
        np.zeros((200, 1)),
        (("x", 0.0, 0.0),),
        1 / 201,
        201,
        ((200, "numerical_failure"),),
        method="gaussian_parametric",
    )
    report = api.UncertaintyReport(
        ("x",),
        np.eye(1),
        (),
        evidence.intervals,
        evidence.failure_rate,
        (),
        (),
        False,
        (),
        bootstrap_performed=True,
        bootstrap_evidence=evidence,
    )
    assert report.bootstrap_sample_count == 201
    assert report.bootstrap_evidence.successful_samples == 200
