"""Statistical uncertainty must not inherit optimization or display scaling."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import joint_scale_searches, scale_candidate, scale_problem

from xrr_fitter.analysis.joint import analyze_joint_ensemble
from xrr_fitter.analysis.report import build_uncertainty_report
from xrr_fitter.evaluation import evaluate_model_jacobian, values_and_jacobians
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.services.fitting import _analyze_joint_searches


def test_known_gaussian_sigma_and_independent_replication_are_calibrated() -> None:
    reports = []
    for repeats in (1, 10):
        problem = scale_problem("gaussian", repeats=repeats)
        candidate = scale_candidate(problem)
        report = build_uncertainty_report(problem, (candidate,))
        _, derivatives = values_and_jacobians(problem, candidate.unit_vector)
        slope = evaluate_model_jacobian(problem, candidate.unit_vector)[:, 0]
        expected = derivatives["instrument.scale"][0] / np.sqrt(slope @ slope)
        assert report.parameter_sigma[0] == pytest.approx(expected, rel=1e-9)
        assert hasattr(report, "covariance_evidence"), "reports must record the statistical method and rank"
        assert report.covariance_evidence.method == "gaussian_known_sigma"
        reports.append(report)
    assert reports[1].parameter_sigma[0] == pytest.approx(reports[0].parameter_sigma[0] / np.sqrt(10), rel=1e-9)


def test_poisson_covariance_uses_expected_count_information_including_zero_counts() -> None:
    problem = scale_problem("poisson")
    candidate = scale_candidate(problem)
    assert np.any(problem.data.intensity_raw == 0)
    higher = scale_candidate(problem, scale=0.6)
    lower = scale_candidate(problem, scale=0.4)
    slope = (higher.model_normalized - lower.model_normalized) / 0.2
    mu = problem.data.normalization * candidate.model_normalized
    expected = 1 / np.sqrt(np.sum((problem.data.normalization * slope) ** 2 / mu))
    report = build_uncertainty_report(problem, (candidate,))
    assert report.parameter_sigma[0] == pytest.approx(expected, rel=1e-9)


def test_robust_covariance_is_score_sandwich_not_inverse_curvature() -> None:
    problem = scale_problem()
    candidate = scale_candidate(problem)
    mask = problem.data.fit_mask
    residual = candidate.residuals[mask]
    jacobian = evaluate_model_jacobian(problem, candidate.unit_vector)[:, 0]
    c, weights = problem.config.c_decades, problem.weights[mask]
    radius = np.hypot(c, residual)
    score = weights**2 * residual / (c * radius)
    curvature = weights**2 * c / radius**3
    bread = np.sum(curvature * jacobian**2)
    meat = np.sum((score * jacobian) ** 2)
    _, derivatives = values_and_jacobians(problem, candidate.unit_vector)
    expected = derivatives["instrument.scale"][0] * np.sqrt(meat) / bread
    report = build_uncertainty_report(problem, (candidate,))
    assert report.parameter_sigma[0] == pytest.approx(expected, rel=1e-9)


def test_correlated_robust_residuals_withhold_iid_covariance() -> None:
    problem = scale_problem(systematic=True)
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    assert report.systematic_residual is True
    assert report.residual_autocorrelation is True
    assert report.parameter_sigma is None
    assert report.covariance is None
    assert "block_bootstrap" in report.covariance_evidence.unavailable_reason


def test_joint_search_spread_is_not_statistical_sigma_and_diagnostics_are_unknown() -> None:
    report, confidence, reasons = analyze_joint_ensemble(
        variable_names=("shared-scale",),
        candidate_ids=("E-0", "E-1", "E-2", "E-3"),
        unit_vectors=np.full((4, 1), 0.5),
        physical_values=np.array([[0.46], [0.49], [0.51], [0.54]]),
        objectives=(1.0,) * 4,
        valid=(True,) * 4,
        diagnostics=((),) * 4,
        thresholds=FitConfig.fast(17).confidence,
    )
    assert report.parameter_sigma is None
    assert report.systematic_residual is None
    assert report.residual_autocorrelation is None
    assert report.search_parameter_spread[0] == pytest.approx(np.std([0.46, 0.49, 0.51, 0.54], ddof=1))
    assert confidence.value != "可信"
    assert "residual_diagnostics_not_executed" in reasons


def test_covariance_evidence_rejects_singular_directions_without_zero_variances() -> None:
    model = import_module("xrr_fitter.model.analysis")
    assert hasattr(model, "CovarianceEvidence"), "covariance needs an immutable availability contract"
    covariance = import_module("xrr_fitter.analysis.covariance")
    evidence = covariance.covariance_from_matrices(
        ("a", "b"),
        np.ones((2, 2)),
        np.zeros((2, 2)),
        np.ones((2, 2)),
        np.eye(2),
        method="gaussian_known_sigma",
    )
    assert evidence.rank == 1
    assert evidence.matrix is None
    assert evidence.unidentifiable_names == ("a", "b")
    assert evidence.unavailable_reason == "rank_deficient"


def test_boundary_solution_does_not_publish_regular_covariance() -> None:
    problem = scale_problem("gaussian")
    bound = problem.parameter_definitions[problem.variables[0].parameter_index].lower
    report = build_uncertainty_report(problem, (scale_candidate(problem, scale=bound),))
    assert report.boundary_hits
    assert report.parameter_sigma is None
    assert report.covariance_evidence.unavailable_reason == "boundary_solution"


def test_zero_mean_poisson_does_not_become_zero_information_or_zero_sigma(monkeypatch) -> None:
    problem = scale_problem("poisson")
    candidate = scale_candidate(problem)
    data = replace(problem.data, intensity_raw=np.zeros(80), intensity_normalized=np.zeros(80))
    problem = replace(problem, data=data)
    boundary = import_module("xrr_fitter.evaluation")
    assert hasattr(boundary, "statistical_information"), "Poisson inference needs a strict numerical boundary"
    statistics = import_module("xrr_fitter.evaluation_inference")
    result = boundary.evaluate_model(problem, candidate.unit_vector)
    monkeypatch.setattr(statistics, "evaluate_model", lambda *_args: replace(result, model_normalized=np.zeros(80)))
    with pytest.raises(statistics.StatisticalUnavailableError, match="zero_mean"):
        statistics.statistical_information(problem, candidate.unit_vector)


def test_real_joint_scale_fit_uses_statistics_not_four_start_convergence_spread() -> None:
    problem, searches = joint_scale_searches()
    results = _analyze_joint_searches(problem, searches, ((), ()))
    report = results[0].uncertainty
    assert report.parameter_sigma is not None
    assert report.parameter_sigma[0] > 1000 * report.search_parameter_spread[0]
    assert report.covariance_evidence.method == "robust_sandwich_iid"
    assert tuple(item.dataset_id for item in report.member_residuals) == problem.dataset_ids
    assert all(item.executed for item in report.member_residuals)
    assert results[1].uncertainty is report


def test_real_joint_systematic_members_cannot_be_reported_as_trusted() -> None:
    problem, searches = joint_scale_searches(systematic=True)
    result = _analyze_joint_searches(problem, searches, ((), ()))[0]
    report = result.uncertainty
    assert len(report.member_residuals) == 2
    assert all(item.systematic and item.autocorrelation for item in report.member_residuals)
    assert report.systematic_residual is True
    assert report.parameter_sigma is None
    assert result.confidence.value != "可信"


def test_joint_gaussian_80_and_800_observation_information_adds() -> None:
    problem, searches = joint_scale_searches("gaussian", repeats=10)
    report = _analyze_joint_searches(problem, searches, ((), ()))[0].uncertainty
    assert report.covariance_evidence.matrix is not None
    local = build_uncertainty_report(problem.problems[0], (searches[0].best_candidate,))
    assert report.parameter_sigma[0] == pytest.approx(local.parameter_sigma[0] / np.sqrt(11), rel=1e-9)
