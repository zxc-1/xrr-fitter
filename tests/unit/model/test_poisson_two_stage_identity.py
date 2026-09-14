"""A new estimator must not reuse precision-v2 or the old reviewed protocol."""

from dataclasses import replace

import pytest

from xrr_fitter.io.project_codec import project_from_dict, project_to_dict
from xrr_fitter.model.diagnostic_calibration import DiagnosticCalibration
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.project import XrrProject


def test_two_stage_runtime_has_one_strict_identity_without_changing_likelihood():
    project = XrrProject.new((), master_seed=7)
    assert project.schema_version == 5
    assert project.algorithm_version == "xrr-fit-v2-poisson-5"
    assert project.fit_config.objective_version == "2"
    assert project.fit_config.diagnostic_version == "poisson-refit-null-v4"
    calibration = DiagnosticCalibration("unavailable", 999, 7, "a" * 64, unavailable_reason="not_started")
    assert calibration.method == "poisson_refit_null_rms_v4"
    assert calibration.refit_policy == "declared_sobol4_lbfgsb_trf_v3"


def test_previous_refit_configuration_is_rejected_instead_of_reinterpreted():
    with pytest.raises(ValueError, match="diagnostic_version"):
        replace(FitConfig.fast(7), diagnostic_version="poisson-refit-null-v2")


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "poisson_refit_null_v2"},
        {"refit_policy": "declared_sobol4_xtol_v2"},
    ],
)
def test_previous_calibration_estimator_identity_is_rejected(changes):
    with pytest.raises(ValueError, match="unsupported"):
        DiagnosticCalibration("unavailable", 999, 7, "a" * 64, unavailable_reason="not_started", **changes)


def test_previous_project_algorithm_is_rejected_without_a_compatibility_branch():
    payload = project_to_dict(XrrProject.new((), master_seed=7))
    payload["algorithm_version"] = "xrr-fit-v2-poisson-2"
    with pytest.raises(ValueError, match="algorithm_version"):
        project_from_dict(payload)


def test_old_reviewed_protocol_really_rejects_the_new_runtime(load_tool_module, monkeypatch):
    tool = load_tool_module("check_poisson_diagnostics")
    assert tool.protocol.version_identity()["algorithm_version"] == "xrr-fit-v2-poisson-5"
    monkeypatch.setattr(
        tool.protocol,
        "EXPECTED_VERSIONS",
        {
            "schema_version": 4,
            "algorithm_version": "xrr-fit-v2-poisson-1",
            "objective_version": "2",
            "diagnostic_version": "poisson-refit-null-v1",
        },
    )
    with pytest.raises(ValueError, match="integrated version"):
        tool._check_versions()
