"""Objective derivatives and physical covariance helpers."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation import evaluate_model, evaluate_model_jacobian, values_by_name
from xrr_fitter.model.fitting import FitEvaluationContext


def _derivative_inputs(problem: object, unit_vector: np.ndarray):
    unit = np.asarray(unit_vector, dtype=float)
    if unit.shape != (len(problem.variables),):
        raise ValueError("objective derivative unit vector has the wrong shape")
    evaluation = evaluate_model(problem, unit)
    if not evaluation.valid or not np.isfinite(evaluation.objective):
        raise ValueError("cannot differentiate an invalid objective evaluation")
    jacobian = np.asarray(evaluate_model_jacobian(problem, unit), dtype=float)
    residual = np.asarray(evaluation.fit_log_residuals_decades, dtype=float)
    if jacobian.shape != (residual.size, unit.size):
        raise ValueError("objective residual Jacobian has the wrong shape")
    weights = np.asarray(problem.weights[problem.data.fit_mask], dtype=float)
    return evaluation, residual, jacobian, weights


def _scale_prior(problem: object) -> tuple[int, object] | None:
    if problem.scale_prior_center is None:
        return None
    for index, variable in enumerate(problem.variables):
        if variable.name == "instrument.scale":
            return index, problem.parameter_definitions[variable.parameter_index]
    return None


def objective_gradient(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> np.ndarray:
    evaluation, residual, jacobian, weights = _derivative_inputs(problem, unit_vector)
    influence = 2.0 * weights**2 * residual / np.sqrt(1.0 + (residual / problem.config.c_decades) ** 2) / residual.size
    gradient = jacobian.T @ influence
    prior = _scale_prior(problem)
    if prior is not None:
        index, definition = prior
        scale = next(value.value for value in evaluation.parameters if value.name == "instrument.scale")
        delta = np.log10(scale) - np.log10(problem.scale_prior_center)
        derivative = np.log10(definition.upper / definition.lower)
        gradient[index] += 2.0 * delta * derivative / problem.scale_prior_tau_decades**2 / residual.size
    result = np.array(gradient, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def objective_information(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> np.ndarray:
    _evaluation, residual, jacobian, weights = _derivative_inputs(problem, unit_vector)
    curvature = 2.0 * weights**2 / residual.size / (1.0 + (residual / problem.config.c_decades) ** 2) ** 1.5
    information = jacobian.T @ (curvature[:, None] * jacobian)
    prior = _scale_prior(problem)
    if prior is not None:
        index, definition = prior
        derivative = np.log10(definition.upper / definition.lower) / problem.scale_prior_tau_decades
        information[index, index] += 2.0 * derivative**2 / residual.size
    result = np.array(information, dtype=float, copy=True)
    result.setflags(write=False)
    return result


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
    covariance = scale[:, None] * matrix * scale[None, :]
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
