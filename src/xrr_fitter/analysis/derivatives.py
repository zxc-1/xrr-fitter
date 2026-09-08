"""Objective derivatives and physical covariance helpers."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation import (
    _scale_prior_jacobian,
    _scale_prior_residual,
    data_score_information,
    evaluate_model,
    evaluate_model_jacobian,
    values_by_name,
)
from xrr_fitter.model.fitting import FitEvaluationContext


def _derivative_inputs(problem: FitEvaluationContext, unit_vector: np.ndarray):
    unit = np.asarray(unit_vector, dtype=float)
    if unit.shape != (len(problem.variables),):
        raise ValueError("objective derivative unit vector has the wrong shape")
    evaluation = evaluate_model(problem, unit)
    if not evaluation.valid or not np.isfinite(evaluation.objective):
        raise ValueError("cannot differentiate an invalid objective evaluation")
    jacobian = np.asarray(evaluate_model_jacobian(problem, unit), dtype=float)
    residual = np.asarray(evaluation.fit_residuals, dtype=float)
    if jacobian.shape != (residual.size, unit.size):
        raise ValueError("objective residual Jacobian has the wrong shape")
    return evaluation, residual, jacobian


def objective_gradient(problem: FitEvaluationContext, unit_vector: np.ndarray) -> np.ndarray:
    """Differentiate displayed J=Q/N through the complete physical graph."""
    evaluation, residual, jacobian = _derivative_inputs(problem, unit_vector)
    score, _information = data_score_information(problem, residual)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        gradient = jacobian.T @ score / problem.objective_point_count
    prior = _scale_prior_residual(problem, evaluation)
    if prior:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            gradient += (2.0 * prior / problem.objective_point_count) * _scale_prior_jacobian(problem, unit_vector)
    if np.any(~np.isfinite(gradient)):
        raise FloatingPointError("objective gradient is not finite")
    gradient.setflags(write=False)
    return gradient


def objective_information(problem: FitEvaluationContext, unit_vector: np.ndarray) -> np.ndarray:
    """Return total Q/2 Gauss-Newton curvature, not covariance or mean curvature."""
    _evaluation, residual, jacobian = _derivative_inputs(problem, unit_vector)
    _score, curvature = data_score_information(problem, residual)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        information = jacobian.T @ (curvature[:, None] * jacobian)
    if problem.scale_prior_center is not None:
        prior_jacobian = _scale_prior_jacobian(problem, unit_vector)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            information += np.outer(prior_jacobian, prior_jacobian)
    if np.any(~np.isfinite(information)):
        raise FloatingPointError("objective information is not finite")
    information.setflags(write=False)
    return information


def physical_parameter_jacobian(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> np.ndarray:
    unit = np.asarray(unit_vector, dtype=float)
    count = len(problem.variables)
    if unit.shape != (count,):
        raise ValueError("physical mapping unit vector has the wrong shape")
    names = tuple(variable.name for variable in problem.variables)
    if not names:
        return np.empty((0, 0), dtype=float)
    columns = []
    for index in range(count):
        lower_step = min(1e-5, unit[index])
        upper_step = min(1e-5, 1.0 - unit[index])
        span = lower_step + upper_step
        if span <= 0.0:
            columns.append(np.zeros(count, dtype=float))
            continue
        lower, upper = unit.copy(), unit.copy()
        lower[index] -= lower_step
        upper[index] += upper_step
        lower_values = values_by_name(problem, lower)
        upper_values = values_by_name(problem, upper)
        columns.append(np.asarray([upper_values[name] - lower_values[name] for name in names]) / span)
    result = np.column_stack(columns)
    result.setflags(write=False)
    return result


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be square")
    if np.any(~np.isfinite(values)):
        raise ValueError("covariance must be finite")
    diagonal = np.clip(np.diag(values), 0.0, np.inf)
    scale = np.sqrt(diagonal)
    denominator = scale[:, None] * scale[None, :]
    correlation = np.divide(
        values,
        denominator,
        out=np.zeros_like(values),
        where=denominator > 0.0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    correlation[np.diag_indices_from(correlation)] = np.where(diagonal > 0.0, 1.0, 0.0)
    correlation.setflags(write=False)
    return correlation


def covariance_from_correlation(sigma: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    scale = np.asarray(sigma, dtype=float)
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must be square")
    if scale.shape != (matrix.shape[0],):
        raise ValueError("sigma length must match the correlation dimension")
    if np.any(~np.isfinite(scale)) or np.any(scale < 0.0):
        raise ValueError("sigma must be finite and nonnegative")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("correlation must be finite")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            covariance = scale[:, None] * matrix * scale[None, :]
    except FloatingPointError as error:
        raise FloatingPointError("reconstructed covariance must be finite") from error
    if np.any(~np.isfinite(covariance)):
        raise FloatingPointError("reconstructed covariance must be finite")
    covariance.setflags(write=False)
    return covariance


def strong_parameter_correlations(
    names: tuple[str, ...],
    correlation: np.ndarray,
    threshold: float = 0.95,
) -> tuple[tuple[str, str, float], ...]:
    matrix = np.asarray(correlation, dtype=float)
    if matrix.shape != (len(names), len(names)):
        raise ValueError("correlation matrix shape must match names")
    return tuple(
        (names[first], names[second], float(matrix[first, second]))
        for first in range(len(names))
        for second in range(first + 1, len(names))
        if abs(matrix[first, second]) >= threshold
    )


def thickness_density_pairs(names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    available = set(names)
    pairs = []
    for name in names:
        if not name.endswith(".thickness_a"):
            continue
        density = name.removesuffix(".thickness_a") + ".density_scale"
        if density in available:
            pairs.append((name, density))
    return tuple(pairs)
