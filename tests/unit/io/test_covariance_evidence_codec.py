"""V2 inference evidence survives persistence without fabricating calibration."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import joint_scale_searches, scale_candidate, scale_problem

from xrr_fitter.analysis.report import build_uncertainty_report
from xrr_fitter.io.codec_results import _uncertainty_from_dict, _uncertainty_to_dict
from xrr_fitter.model.analysis import CovarianceEvidence, ResidualEvidence
from xrr_fitter.services.fitting import _analyze_joint_searches


@pytest.mark.parametrize("systematic", [False, True])
def test_joint_covariance_method_and_member_diagnostics_roundtrip(systematic: bool) -> None:
    problem, searches = joint_scale_searches(systematic=systematic)
    report = _analyze_joint_searches(problem, searches, ((), ()))[0].uncertainty
    payload = _uncertainty_to_dict(report)
    restored = _uncertainty_from_dict(payload)
    assert restored.covariance_evidence is not None
    assert restored.covariance_evidence.method == report.covariance_evidence.method
    assert restored.covariance_evidence.unavailable_reason == report.covariance_evidence.unavailable_reason
    assert restored.member_residuals == report.member_residuals
    np.testing.assert_array_equal(restored.search_parameter_spread, report.search_parameter_spread)
    if report.covariance is not None:
        np.testing.assert_array_equal(restored.covariance, report.covariance)
        assert restored.covariance.flags.writeable is False
    else:
        assert restored.covariance is None
        assert restored.parameter_sigma is None


def test_covariance_and_residual_evidence_are_immutable_after_pickle() -> None:
    problem = scale_problem("gaussian")
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    copied = pickle.loads(pickle.dumps(report))
    assert copied.covariance.flags.writeable is False
    assert copied.parameter_sigma.flags.writeable is False
    assert copied.member_residuals == report.member_residuals


def test_report_rejects_sigma_inconsistent_with_covariance_evidence() -> None:
    problem = scale_problem("gaussian")
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    with pytest.raises(ValueError, match="sigma.*covariance|covariance.*sigma"):
        replace(report, parameter_sigma=report.parameter_sigma * 10)


def test_report_rejects_contradictory_member_diagnostic_summary() -> None:
    problem = scale_problem(systematic=True)
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    with pytest.raises(ValueError, match="residual.*summary|summary.*residual"):
        replace(report, systematic_residual=False)


def test_unexecuted_residual_evidence_cannot_claim_no_problem() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ResidualEvidence("sample", False, False, False, 80, (), "not_run")


@pytest.mark.parametrize("matrix", [np.zeros((2, 2)), np.array([[1, 2], [2, 1]])])
def test_covariance_matrix_cannot_contradict_its_rank_or_positivity(matrix) -> None:
    with pytest.raises(ValueError, match="covariance"):
        CovarianceEvidence(("a", "b"), matrix, "gaussian_known_sigma", 1)


def test_v2_codec_requires_inference_fields_instead_of_reconstructing_old_reports() -> None:
    problem = scale_problem("gaussian")
    payload = _uncertainty_to_dict(build_uncertainty_report(problem, (scale_candidate(problem),)))
    payload.pop("covariance_evidence", None)
    with pytest.raises(ValueError, match="covariance_evidence"):
        _uncertainty_from_dict(payload)
