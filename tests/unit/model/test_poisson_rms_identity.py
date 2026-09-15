"""A changed statistic gets a new identity without renaming its refitter."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import xrr_fitter.api as api
from xrr_fitter.model.diagnostic_calibration import DiagnosticCalibration


def test_rms_uses_one_current_identity_and_keeps_the_two_stage_refitter():
    project = api.new_project()
    assert project.schema_version == 5
    assert project.algorithm_version == "xrr-fit-v2-poisson-5"
    assert project.fit_config.objective_version == "2"
    assert project.fit_config.diagnostic_version == "poisson-refit-null-v4"
    evidence = DiagnosticCalibration("unavailable", 999, 7, "a" * 64, unavailable_reason="not_started")
    assert evidence.method == "poisson_refit_null_rms_v4"
    assert evidence.refit_policy == "declared_sobol4_lbfgsb_trf_v3"


@pytest.mark.parametrize("version", ["poisson-refit-null-v1", "poisson-refit-null-v2", "poisson-refit-null-v3"])
def test_old_statistical_configuration_is_not_reinterpreted(version):
    with pytest.raises(ValueError, match="diagnostic_version"):
        replace(api.FitConfig.fast(7), diagnostic_version=version)


@pytest.mark.parametrize("method", ["poisson_refit_null_v1", "poisson_refit_null_v2", "poisson_refit_null_v3"])
def test_old_calibration_method_is_rejected(method):
    with pytest.raises(ValueError, match="unsupported"):
        DiagnosticCalibration("unavailable", 999, 7, "a" * 64, method=method, unavailable_reason="not_started")


@pytest.mark.parametrize("filename", ["single-layer.xrrproj.json", "mo-si-periodic.xrrproj.json"])
def test_declaration_examples_use_the_current_statistical_version(filename):
    root = Path(__file__).resolve().parents[3]
    payload = json.loads((root / "examples" / filename).read_text())
    assert payload["schema_version"] == 5
    assert payload["algorithm_version"] == "xrr-fit-v2-poisson-5"
    assert payload["fit_config"]["diagnostic_version"] == "poisson-refit-null-v4"
    assert payload["fit_config"]["objective_version"] == "2"
    assert all(item["last_valid_result"] is None and item["checkpoint"] is None for item in payload["datasets"])
