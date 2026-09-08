from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import EvaluationConstraintError
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec


def test_shared_solver_primitives_match_the_compiled_objective() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=18), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.55)
    residual = evaluation.least_squares_residual(problem, unit)
    jacobian = evaluation.least_squares_residual_jacobian(problem, unit)
    rho = evaluation.least_squares_loss(problem)(residual**2)

    assert jacobian.shape == (residual.size, unit.size)
    optimizer_objective = 0.5 * float(np.sum(rho[0])) / residual.size
    assert optimizer_objective == pytest.approx(
        evaluate_vector(problem, unit).objective,
        rel=1e-12,
        abs=1e-14,
    )


def test_least_squares_loss_handles_extreme_positive_robust_scale_without_underflow() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1808), scale_prior_enabled=False),
    )
    problem = replace(problem, config=replace(problem.config, c_decades=1e-200))
    squared = np.ones(np.count_nonzero(problem.data.fit_mask))
    weights = problem.weights[problem.data.fit_mask]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rho = evaluation.least_squares_loss(problem)(squared)

    radius = np.hypot(problem.config.c_decades, np.sqrt(squared))
    expected = np.vstack(
        (
            4.0 * weights**2 * (radius - problem.config.c_decades) / problem.config.c_decades,
            (2.0 * weights**2 / radius) / problem.config.c_decades,
            -(weights**2 / radius**3) / problem.config.c_decades,
        )
    )
    np.testing.assert_allclose(rho, expected, rtol=1e-15, atol=0.0)
    assert np.all(np.isfinite(rho))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_least_squares_loss_rejects_unrepresentable_dimensionless_curvature() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1810), scale_prior_enabled=False),
    )
    problem = replace(problem, config=replace(problem.config, c_decades=1e-200))
    squared = np.full(np.count_nonzero(problem.data.fit_mask), 1e-320)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="robust loss"):
            evaluation.least_squares_loss(problem)(squared)

    assert not any(item.category is RuntimeWarning for item in caught)


def test_least_squares_loss_preserves_nonzero_quadratic_value_near_zero() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1809), scale_prior_enabled=False),
    )
    squared = np.full(np.count_nonzero(problem.data.fit_mask), 1e-24)

    rho = evaluation.least_squares_loss(problem)(squared)

    assert np.all(rho[0] > 0.0)
    np.testing.assert_allclose(
        rho[0],
        2.0 * problem.weights[problem.data.fit_mask] ** 2 * squared / problem.config.c_decades**2,
        rtol=1e-15,
        atol=0.0,
    )


def test_split_solver_helpers_keep_invalid_scale_prior_row_consistent() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1803), scale_prior_enabled=False),
    )
    problem = replace(problem, scale_prior_center=1.0)
    unit = np.full(len(problem.variables), 0.55)

    class InvalidEvaluation:
        valid = False

    residual = evaluation.least_squares_residual(
        problem,
        unit,
        evaluator=lambda _problem, _unit: InvalidEvaluation(),
    )

    def invalid_jacobian(_problem, _unit):
        raise EvaluationConstraintError("constraint_violation:test")

    jacobian = evaluation.least_squares_residual_jacobian(
        problem,
        unit,
        jacobian_evaluator=invalid_jacobian,
    )

    np.testing.assert_array_equal(residual, np.full(residual.shape, 1e6))
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))
