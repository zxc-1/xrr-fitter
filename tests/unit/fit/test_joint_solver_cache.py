"""Shared joint solver systems preserve real constraints and thread ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib import import_module
from threading import Barrier

import numpy as np
import pytest
from tests.support.joint_constraint_cases import (
    cross_constraint_chain_joint,
    cross_constraint_joint,
    cross_roughness_constraint_joint,
)
from tests.unit.fit.joint_evaluation_cases import _joint, _unequal_roughness_joint

from xrr_fitter.fit.joint_evaluation import evaluate_joint_jacobian, evaluate_joint_vector
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import initial_joint_vector
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.parameters import ParameterSetting


def _callbacks(problem, cancelled=None):
    module = import_module("xrr_fitter.fit.joint_solvers")
    assert hasattr(module, "cached_joint_least_squares_callbacks"), "joint callbacks must share one numerical system"
    return module.cached_joint_least_squares_callbacks(problem, cancelled)


def _count_systems(monkeypatch):
    module = import_module("xrr_fitter.fit.joint_solvers")
    assert hasattr(module, "joint_least_squares_system"), "joint solver must use the combined local systems"
    original = module.joint_least_squares_system
    calls = []

    def counted(problem, unit):
        calls.append(unit.copy())
        return original(problem, unit)

    monkeypatch.setattr(module, "joint_least_squares_system", counted)
    return calls


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
def test_joint_same_point_callbacks_evaluate_each_member_system_once(monkeypatch, mode) -> None:
    problem = _joint(scale_prior=True)
    members = []
    for member in problem.problems:
        counts = np.rint(member.data.intensity_raw)
        data = replace(
            member.data,
            intensity_raw=counts,
            intensity_normalized=counts / member.data.normalization,
            intensity_sigma_normalized=np.full(counts.size, 0.02),
        )
        members.append(replace(member, data=data, config=replace(member.config, noise_model=mode)))
    problem = compile_joint_problem(problem.dataset_ids, tuple(members), problem.sharing_rules)
    unit = initial_joint_vector(problem)
    expected_residual = evaluate_joint_vector(problem, unit).residuals
    expected_jacobian = evaluate_joint_jacobian(problem, unit)
    module = import_module("xrr_fitter.fit.joint_evaluation")
    assert hasattr(module, "least_squares_system"), "joint solver must combine actual per-member systems"
    original = module.least_squares_system
    members_seen = []

    def counted(member, value):
        members_seen.append(id(member))
        return original(member, value)

    monkeypatch.setattr(module, "least_squares_system", counted)
    calls = _count_systems(monkeypatch)
    residual, jacobian = _callbacks(problem)
    np.testing.assert_allclose(residual(unit), expected_residual, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian(unit.copy()), expected_jacobian, rtol=1e-12, atol=1e-12)
    assert len(calls) == 1
    assert members_seen == [id(member) for member in problem.problems]


@pytest.mark.parametrize(
    "factory", [_unequal_roughness_joint, cross_constraint_chain_joint, cross_roughness_constraint_joint]
)
def test_combined_joint_system_preserves_global_constraint_derivatives(factory) -> None:
    problem = factory()
    unit = np.full(len(problem.global_variables), 0.45)
    residual, jacobian = _callbacks(problem)
    actual = jacobian(unit)
    step = 1e-6
    finite = np.column_stack(
        [
            (residual(unit + direction * step) - residual(unit - direction * step)) / (2 * step)
            for direction in np.eye(unit.size)
        ]
    )
    np.testing.assert_allclose(actual, finite, rtol=5e-5, atol=5e-8)


def test_joint_cache_is_thread_local_and_owns_input_and_output_arrays(monkeypatch) -> None:
    problem = _joint()
    units = (np.array([0.4]), np.array([0.6]))
    expected = tuple(
        (evaluate_joint_vector(problem, unit).residuals, evaluate_joint_jacobian(problem, unit)) for unit in units
    )
    calls = _count_systems(monkeypatch)
    residual, jacobian = _callbacks(problem)
    barrier = Barrier(2)

    def evaluate(index):
        value = units[index].copy()
        observed = residual(value)
        np.testing.assert_allclose(observed, expected[index][0], rtol=1e-12, atol=1e-12)
        observed[:] = -999
        barrier.wait(timeout=10)
        tangent = jacobian(value.copy())
        np.testing.assert_allclose(tangent, expected[index][1], rtol=1e-12, atol=1e-12)
        tangent[:] = -999
        np.testing.assert_allclose(residual(value), expected[index][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jacobian(value), expected[index][1], rtol=1e-12, atol=1e-12)
        value[:] = units[1 - index]
        np.testing.assert_allclose(residual(value), expected[1 - index][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jacobian(value), expected[1 - index][1], rtol=1e-12, atol=1e-12)

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(evaluate, range(2)))
    assert len(calls) == 4


def test_joint_cache_polls_cancellation_even_on_a_cache_hit() -> None:
    problem = _joint()
    cancelled = False
    residual, jacobian = _callbacks(problem, lambda: cancelled)
    unit = initial_joint_vector(problem)
    residual(unit)
    cancelled = True
    with pytest.raises(SearchCancelled):
        jacobian(unit)


def test_joint_solver_uses_the_shared_system_callbacks(monkeypatch) -> None:
    module = import_module("xrr_fitter.fit.joint_solvers")
    problem = _joint()
    calls = _count_systems(monkeypatch)
    solved = module.solve_joint(problem, initial_joint_vector(problem), 5, None)
    assert len(calls) == solved.nfev
    assert all(
        value.model_normalized.size == member.data.fit_mask.size
        for member, value in zip(problem.problems, solved.evaluation.local_evaluations, strict=True)
    )


def test_joint_system_keeps_constraint_failure_sentinels_and_recovers_at_next_point() -> None:
    problem = cross_constraint_joint(multiplier=2.0)
    residual, jacobian = _callbacks(problem)
    invalid = np.array([0.9])
    expected = evaluate_joint_vector(problem, invalid)
    assert expected.valid is False
    np.testing.assert_array_equal(residual(invalid), expected.residuals)
    np.testing.assert_array_equal(jacobian(invalid), np.zeros((expected.residuals.size, 1)))
    valid = np.array([0.01])
    expected = evaluate_joint_vector(problem, valid)
    assert expected.valid is True
    np.testing.assert_allclose(residual(valid), expected.residuals, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian(valid), evaluate_joint_jacobian(problem, valid), rtol=1e-12, atol=1e-12)


def test_all_locked_joint_system_has_the_correct_empty_column_axis() -> None:
    module = import_module("xrr_fitter.fit.joint_solvers")
    original = _joint()
    members = []
    for member in original.problems:
        settings = tuple(
            ParameterSetting(item.name, item.initial, item.lower, item.upper, locked=True)
            for item in member.parameter_definitions
        )
        members.append(compile_fit_problem(member.data, member.structure, member.instrument, member.config, settings))
    problem = compile_joint_problem(original.dataset_ids, tuple(members), ())
    unit = initial_joint_vector(problem)
    assert unit.size == 0
    residual, jacobian = _callbacks(problem)
    expected = evaluate_joint_vector(problem, unit)
    np.testing.assert_array_equal(residual(unit), expected.residuals)
    assert jacobian(unit).shape == (expected.residuals.size, 0)
    solved = module.solve_joint(problem, unit, 5, None)
    assert solved.stop_reason == "no_free_parameters"
    assert solved.evaluation.valid


def test_joint_cache_does_not_hide_unexpected_system_errors(monkeypatch) -> None:
    module = import_module("xrr_fitter.fit.joint_evaluation")
    problem = _joint()
    sentinel = RuntimeError("unexpected system error")

    def broken(*_args):
        raise sentinel

    monkeypatch.setattr(module, "least_squares_system", broken)
    residual, _jacobian = _callbacks(problem)
    with pytest.raises(RuntimeError) as caught:
        residual(initial_joint_vector(problem))
    assert caught.value is sentinel


def test_joint_system_preserves_valid_residuals_when_prior_derivative_overflows() -> None:
    problem = _joint(scale_prior=True)
    members = tuple(replace(member, scale_prior_tau_decades=np.nextafter(0.0, 1.0)) for member in problem.problems)
    problem = replace(problem, problems=members)
    unit = initial_joint_vector(problem)
    expected = evaluate_joint_vector(problem, unit)
    assert expected.valid
    expected_jacobian = evaluate_joint_jacobian(problem, unit)
    np.testing.assert_array_equal(expected_jacobian, np.zeros_like(expected_jacobian))
    residual, jacobian = _callbacks(problem)
    np.testing.assert_array_equal(residual(unit), expected.residuals)
    np.testing.assert_array_equal(jacobian(unit), expected_jacobian)
