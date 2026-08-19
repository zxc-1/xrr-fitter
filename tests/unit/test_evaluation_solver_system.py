from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec


def test_solver_system_derivative_failure_keeps_one_scale_prior_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1804), scale_prior_enabled=False),
    )
    problem = replace(problem, scale_prior_center=1.0)
    unit = np.full(len(problem.variables), 0.55)

    def derivative_failure(_problem, _unit):
        raise FloatingPointError("derivative overflow")

    monkeypatch.setattr(evaluation, "_model_residual_jacobian", derivative_failure)

    residual, jacobian = evaluation.least_squares_system(problem, unit)

    expected_rows = int(np.count_nonzero(problem.data.fit_mask)) + 1
    assert residual.shape == (expected_rows,)
    assert jacobian.shape == (expected_rows, len(problem.variables))
    np.testing.assert_array_equal(
        residual,
        evaluation.least_squares_residual(problem, unit),
    )


def test_scale_prior_residual_rejects_extreme_tau_overflow_without_warning() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1806), scale_prior_enabled=False),
    )
    problem = replace(
        problem,
        scale_prior_center=1.1,
        scale_prior_tau_decades=np.nextafter(0.0, 1.0),
    )
    unit = encode_physical_vector(problem, {"instrument.scale": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        residual = evaluation.least_squares_residual(problem, unit)

    np.testing.assert_array_equal(residual, np.full(residual.shape, 1e6))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_scale_prior_residual_preserves_adjacent_extreme_log_delta() -> None:
    center = 1e308
    scale = np.nextafter(center, np.inf)
    problem = SimpleNamespace(
        scale_prior_center=center,
        scale_prior_tau_decades=1.0,
    )

    observed = evaluation._scale_prior_residual_from_scale(problem, scale)

    expected = np.log1p((scale - center) / center) / np.log(10.0)
    assert observed == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert observed > 0.0


def test_scale_prior_jacobian_rejects_extreme_tau_overflow_without_warning() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1807), scale_prior_enabled=False),
    )
    problem = replace(
        problem,
        scale_prior_center=1.0,
        scale_prior_tau_decades=np.nextafter(0.0, 1.0),
    )
    unit = encode_physical_vector(problem, {"instrument.scale": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="scale prior Jacobian"):
            evaluation.least_squares_residual_jacobian(problem, unit)

    assert not any(item.category is RuntimeWarning for item in caught)


def test_joint_solver_system_matches_separate_residual_and_jacobian_paths() -> None:
    problem = compile_fit_problem(
        prepared_data(size=72),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1801), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.55)

    residual, jacobian = evaluation.least_squares_system(problem, unit)

    np.testing.assert_array_equal(
        residual,
        evaluation.least_squares_residual(problem, unit),
    )
    np.testing.assert_allclose(
        jacobian,
        evaluation.least_squares_residual_jacobian(problem, unit),
        rtol=0.0,
        atol=0.0,
    )


def test_cached_solver_callbacks_evaluate_one_parameter_vector_once() -> None:
    calls: list[np.ndarray] = []

    def system(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(np.array(unit, copy=True))
        return unit + 1.0, np.diag(unit + 2.0)

    residual, jacobian = evaluation.cached_least_squares_callbacks(system)
    first = np.asarray([0.2, 0.4])
    second = np.asarray([0.3, 0.5])

    np.testing.assert_array_equal(residual(first), first + 1.0)
    np.testing.assert_array_equal(jacobian(first.copy()), np.diag(first + 2.0))
    np.testing.assert_array_equal(residual(second), second + 1.0)
    np.testing.assert_array_equal(jacobian(second.copy()), np.diag(second + 2.0))

    assert len(calls) == 2
    np.testing.assert_array_equal(calls, (first, second))


def test_cached_solver_callbacks_isolate_concurrent_optimizer_threads() -> None:
    calls: list[np.ndarray] = []
    barrier = Barrier(2)

    def system(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(np.array(unit, copy=True))
        return unit + 1.0, np.diag(unit + 2.0)

    residual, jacobian = evaluation.cached_least_squares_callbacks(system)

    def evaluate(unit: np.ndarray) -> None:
        np.testing.assert_array_equal(residual(unit), unit + 1.0)
        barrier.wait()
        np.testing.assert_array_equal(jacobian(unit), np.diag(unit + 2.0))

    units = (np.asarray([0.2, 0.4]), np.asarray([0.3, 0.5]))
    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(evaluate, units))

    assert len(calls) == 2


def test_shared_problem_log_probability_uses_the_soft_l1_data_likelihood() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=19), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.45)
    residual = evaluation.least_squares_residual(problem, unit)
    weights = problem.weights[problem.data.fit_mask]
    c = problem.config.c_decades
    expected = -float(np.sum(weights**2 * 2.0 * c**2 * (np.sqrt(1.0 + (residual / c) ** 2) - 1.0))) / (2.0 * c**2)

    assert evaluation.problem_log_probability(problem, unit) == expected
