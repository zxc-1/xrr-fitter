"""Objective derivatives and physical covariance helpers."""

from __future__ import annotations

from math import log

import numpy as np

from xrr_fitter.evaluation import evaluate_model, evaluate_model_jacobian, values_by_name
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.parameters import _log10_ratio, _log_interval_width


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


def _log_decades_per_unit(definition: object) -> float:
    """Return the log-transform span without forming an overflowing ratio."""
    return _log_interval_width(definition.lower, definition.upper) / log(10.0)


def _scale_prior_unit_derivative(problem: object, definition: object) -> float:
    """Return the standardized prior tangent or reject an unusable scale."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            derivative = np.divide(
                _log_decades_per_unit(definition),
                problem.scale_prior_tau_decades,
            )
    except FloatingPointError as error:
        raise FloatingPointError("scale prior derivative is not finite") from error
    if not np.isfinite(derivative):
        raise FloatingPointError("scale prior derivative is not finite")
    return float(derivative)


def _robust_influence(residual: np.ndarray, weights: np.ndarray, c_decades: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        scaled = residual / c_decades
        denominator = np.sqrt(1.0 + scaled**2)
        influence = 2.0 * weights**2 * residual / denominator / residual.size
    unstable = ~np.isfinite(denominator) | ~np.isfinite(influence)
    if np.any(unstable):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            normalized = residual[unstable] / np.hypot(c_decades, residual[unstable])
            influence[unstable] = 2.0 * weights[unstable] ** 2 * c_decades * normalized / residual.size
    return influence


def _scale_prior_gradient(
    problem: object,
    evaluation: object,
    prior: tuple[int, object] | None,
    residual_count: int,
) -> tuple[int, float] | None:
    if prior is None:
        return None
    index, definition = prior
    scale = next(value.value for value in evaluation.parameters if value.name == "instrument.scale")
    delta = _log10_ratio(scale, problem.scale_prior_center)
    if delta == 0.0:
        return None
    derivative = _scale_prior_unit_derivative(problem, definition)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            standardized = np.divide(delta, problem.scale_prior_tau_decades)
            contribution = 2.0 * standardized * derivative / residual_count
    except FloatingPointError as error:
        raise FloatingPointError("scale prior gradient is not finite") from error
    if not np.isfinite(contribution):
        raise FloatingPointError("scale prior gradient is not finite")
    return index, float(contribution)


def objective_gradient(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> np.ndarray:
    evaluation, residual, jacobian, weights = _derivative_inputs(problem, unit_vector)
    influence = _robust_influence(residual, weights, problem.config.c_decades)
    gradient = jacobian.T @ influence
    prior_gradient = _scale_prior_gradient(problem, evaluation, _scale_prior(problem), residual.size)
    if prior_gradient is not None:
        index, contribution = prior_gradient
        gradient[index] += contribution
    if np.any(~np.isfinite(gradient)):
        raise FloatingPointError("objective gradient is not finite")
    result = np.array(gradient, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _robust_curvature(residual: np.ndarray, weights: np.ndarray, c_decades: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        scaled = residual / c_decades
        shape = (1.0 + scaled**2) ** 1.5
        curvature = 2.0 * weights**2 / residual.size / shape
    unstable = ~np.isfinite(shape) | ~np.isfinite(curvature)
    if np.any(unstable):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            ratio = c_decades / np.hypot(c_decades, residual[unstable])
            curvature[unstable] = 2.0 * weights[unstable] ** 2 / residual.size * ratio**3
    return curvature


def _scale_prior_information(
    problem: object,
    prior: tuple[int, object] | None,
    residual_count: int,
) -> tuple[int, float] | None:
    if prior is None:
        return None
    index, definition = prior
    derivative = _scale_prior_unit_derivative(problem, definition)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            increment = np.divide(2.0 * np.multiply(derivative, derivative), residual_count)
    except (FloatingPointError, OverflowError) as error:
        raise FloatingPointError("scale prior information is not finite") from error
    if not np.isfinite(increment):
        raise FloatingPointError("scale prior information is not finite")
    return index, float(increment)


def objective_information(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> np.ndarray:
    _evaluation, residual, jacobian, weights = _derivative_inputs(problem, unit_vector)
    curvature = _robust_curvature(residual, weights, problem.config.c_decades)
    information = jacobian.T @ (curvature[:, None] * jacobian)
    prior_information = _scale_prior_information(problem, _scale_prior(problem), residual.size)
    if prior_information is not None:
        index, increment = prior_information
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                updated = np.add(information[index, index], increment)
        except (FloatingPointError, OverflowError) as error:
            raise FloatingPointError("scale prior information is not finite") from error
        if not np.isfinite(updated):
            raise FloatingPointError("scale prior information is not finite")
        information[index, index] = updated
    if np.any(~np.isfinite(information)):
        raise FloatingPointError("objective information is not finite")
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
