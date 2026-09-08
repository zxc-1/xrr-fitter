"""Prior, constraint, rank, and global weighting boundaries for covariance."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import joint_scale_searches, scale_candidate, scale_problem
from tests.unit.fit.joint_evaluation_cases import _unequal_roughness_joint

from xrr_fitter.analysis.covariance import covariance_from_matrices
from xrr_fitter.analysis.report import build_uncertainty_report
from xrr_fitter.evaluation import encode_physical_vector, statistical_information, values_and_jacobians
from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector, joint_inference_layout
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.io.codec_results import _uncertainty_from_dict, _uncertainty_to_dict
from xrr_fitter.model.parameters import ConstraintNode, ConstraintRule, ParameterReference, ParameterSetting
from xrr_fitter.services.fitting import _joint_point_evidence


def test_data_derived_prior_changes_bread_without_becoming_an_independent_score() -> None:
    plain = scale_problem("gaussian")
    problem = replace(plain, scale_prior_center=0.5, scale_prior_tau_decades=0.05)
    candidate = scale_candidate(problem)
    data, prior, meat = statistical_information(problem, candidate.unit_vector)
    _, derivative = values_and_jacobians(problem, candidate.unit_vector)
    mapping = derivative["instrument.scale"][0]
    expected = mapping**2 * data[0, 0] / (data[0, 0] + prior[0, 0]) ** 2
    report = build_uncertainty_report(problem, (candidate,))
    np.testing.assert_array_equal(meat, data)
    assert prior[0, 0] > 0
    assert report.covariance[0, 0] == pytest.approx(expected, rel=1e-10)


def test_prior_does_not_hide_data_rank_deficiency() -> None:
    evidence = covariance_from_matrices(
        ("a", "b"), np.ones((2, 2)), np.eye(2), np.ones((2, 2)), np.eye(2), method="gaussian_known_sigma"
    )
    assert evidence.rank == 1
    assert evidence.matrix is None
    assert evidence.unidentifiable_names == ("a", "b")


def test_cross_dataset_scale_constraint_enters_global_covariance() -> None:
    local = (scale_problem("gaussian"), scale_problem("gaussian", seed=18))
    rule = ConstraintRule(
        ParameterReference("right", "instrument.scale"),
        ConstraintNode(
            "mul",
            operands=(
                ConstraintNode("const", value=2.0),
                ConstraintNode("ref", reference=ParameterReference("left", "instrument.scale")),
            ),
        ),
    )
    problem = compile_joint_problem(("left", "right"), local, (), (rule,))
    unit = encode_physical_vector(local[0], {"instrument.scale": 0.5})
    plus = evaluate_joint_vector(problem, unit + 1e-6).residuals
    minus = evaluate_joint_vector(problem, unit - 1e-6).residuals
    slope = (plus - minus) / 2e-6
    mapping = values_and_jacobians(local[0], unit)[1]["instrument.scale"][0]
    covariance, members = _joint_point_evidence(problem, unit)
    assert tuple(item.dataset_id for item in members) == ("left", "right")
    assert covariance.matrix[0, 0] == pytest.approx(mapping**2 / (slope @ slope), rel=1e-7)


def test_joint_robust_member_weights_enter_bread_once_and_meat_twice() -> None:
    problem, searches = joint_scale_searches(repeats=10)
    unit = searches[0].best_candidate.unit_vector
    units, scatters, mapping, alphas = joint_inference_layout(problem, unit)
    assert alphas == pytest.approx((5.5, 0.55))
    bread, meat = 0.0, 0.0
    for local, local_unit, scatter, alpha in zip(problem.problems, units, scatters, alphas, strict=True):
        data, _prior, scores = statistical_information(local, local_unit)
        bread += alpha * (scatter.T @ data @ scatter)[0, 0]
        meat += alpha**2 * (scatter.T @ scores @ scatter)[0, 0]
    covariance, _residuals = _joint_point_evidence(problem, unit)
    assert covariance.matrix[0, 0] == pytest.approx(mapping[0, 0] ** 2 * meat / bread**2, rel=1e-10)


def test_shared_physical_roughness_mapping_matches_finite_difference() -> None:
    problem = _unequal_roughness_joint()
    unit = np.full(len(problem.global_variables), 0.4)
    mapping = joint_inference_layout(problem, unit)[2]
    for column in range(unit.size):
        step = np.eye(unit.size)[column] * 1e-6
        plus, minus = (evaluate_joint_vector(problem, point) for point in (unit + step, unit - step))
        differences = []
        for variable in problem.global_variables:
            reference = variable.members[0]
            index = problem.dataset_ids.index(reference.dataset_id)
            high = {value.name: value.value for value in plus.local_evaluations[index].parameters}
            low = {value.name: value.value for value in minus.local_evaluations[index].parameters}
            differences.append((high[reference.parameter_name] - low[reference.parameter_name]) / 2e-6)
        np.testing.assert_allclose(mapping[:, column], differences, rtol=1e-7, atol=1e-7)


def test_all_locked_parameters_have_empty_not_missing_covariance_and_roundtrip() -> None:
    initial = scale_problem("gaussian")
    settings = tuple(
        ParameterSetting(value.name, value.initial, value.lower, value.upper, locked=True)
        for value in initial.parameter_definitions
    )
    problem = compile_fit_problem(initial.data, initial.structure, initial.instrument, initial.config, settings)
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    assert report.covariance.shape == (0, 0)
    restored = _uncertainty_from_dict(_uncertainty_to_dict(report))
    assert restored.covariance.shape == (0, 0)
    assert restored.parameter_sigma.shape == (0,)


def test_joint_derivative_failure_preserves_executed_residual_evidence(monkeypatch) -> None:
    problem, searches = joint_scale_searches("gaussian")
    fitting = import_module("xrr_fitter.services.fitting")

    def failed_layout(*_args):
        raise FloatingPointError("nonfinite joint constraint Jacobian")

    monkeypatch.setattr(fitting, "joint_inference_layout", failed_layout)
    report = fitting._analyze_joint_searches(problem, searches, ((), ()))[0].uncertainty
    assert report.covariance is None
    assert "nonfinite joint constraint Jacobian" in report.covariance_evidence.unavailable_reason
    assert all(item.executed for item in report.member_residuals)
