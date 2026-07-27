"""Deterministic analytic local least-squares search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log

import numpy as np
from scipy.optimize import least_squares

from xrr_fitter.fit.objective import evaluate_jacobian, evaluate_vector
from xrr_fitter.model.fitting import ModelEvaluation


class SearchCancelled(RuntimeError):
    """Raised when a cooperative search cancellation is observed."""


@dataclass(frozen=True, slots=True)
class LocalSearchResult:
    unit_vector: np.ndarray
    evaluation: ModelEvaluation
    stop_reason: str
    nfev: int

    def __post_init__(self) -> None:
        unit = np.array(self.unit_vector, dtype=float, copy=True)
        unit.setflags(write=False)
        object.__setattr__(self, "unit_vector", unit)


def _validated_unit(problem: object, value: np.ndarray, field: str) -> np.ndarray:
    unit = np.asarray(value, dtype=float)
    valid = (
        unit.ndim == 1
        and unit.shape == (len(problem.variables),)
        and np.all(np.isfinite(unit))
        and np.all((unit >= 0.0) & (unit <= 1.0))
    )
    if not valid:
        raise ValueError(f"{field} must be a finite unit vector with the compiled shape and bounds")
    return np.array(unit, copy=True)


def _scale_prior_residual(problem: object, evaluation: ModelEvaluation) -> float | None:
    if problem.scale_prior_center is None:
        return None
    scale = next(value.value for value in evaluation.parameters if value.name == "instrument.scale")
    return (
        (np.log10(scale) - np.log10(problem.scale_prior_center))
        / problem.scale_prior_tau_decades
    )


def local_residual(problem: object, unit_vector: np.ndarray) -> np.ndarray:
    """Return raw log residual rows followed by the optional scale-prior row."""
    unit = _validated_unit(problem, unit_vector, "unit vector")
    evaluation = evaluate_vector(problem, unit)
    row_count = int(np.count_nonzero(problem.data.fit_mask)) + int(
        problem.scale_prior_center is not None
    )
    if not evaluation.valid:
        return np.full(row_count, 1e6, dtype=float)
    residual = np.array(evaluation.fit_log_residuals_decades, dtype=float, copy=True)
    prior = _scale_prior_residual(problem, evaluation)
    return residual if prior is None else np.concatenate((residual, np.asarray([prior])))


def _scale_prior_jacobian(problem: object) -> np.ndarray:
    row = np.zeros(len(problem.variables), dtype=float)
    if problem.scale_prior_center is None:
        return row
    for index, coordinate in enumerate(problem.variables):
        if coordinate.name != "instrument.scale":
            continue
        definition = problem.parameter_definitions[coordinate.parameter_index]
        if definition.transform == "log":
            derivative = log(definition.upper / definition.lower) / log(10.0)
        else:
            scale = definition.lower + 0.5 * (definition.upper - definition.lower)
            derivative = (definition.upper - definition.lower) / (scale * log(10.0))
        row[index] = derivative / problem.scale_prior_tau_decades
    return row


def local_jacobian(problem: object, unit_vector: np.ndarray) -> np.ndarray:
    """Return the analytic raw-residual and optional prior Jacobian."""
    unit = _validated_unit(problem, unit_vector, "unit vector")
    try:
        jacobian = np.array(evaluate_jacobian(problem, unit), dtype=float, copy=True)
    except FloatingPointError:
        jacobian = np.zeros(
            (np.count_nonzero(problem.data.fit_mask), len(problem.variables)),
            dtype=float,
        )
    except ValueError as error:
        if str(error) != "cannot differentiate nonpositive fitted angle":
            raise
        jacobian = np.zeros(
            (np.count_nonzero(problem.data.fit_mask), len(problem.variables)),
            dtype=float,
        )
    if problem.scale_prior_center is not None:
        jacobian = np.vstack((jacobian, _scale_prior_jacobian(problem)))
    return jacobian


def _least_squares_loss(problem: object) -> Callable[[np.ndarray], np.ndarray]:
    weights = np.asarray(problem.weights[problem.data.fit_mask], dtype=float)
    c_decades = problem.config.c_decades
    data_count = weights.size

    def loss(squared: np.ndarray) -> np.ndarray:
        values = np.asarray(squared, dtype=float)
        rho = np.empty((3, values.size), dtype=float)
        data = values[:data_count]
        scaled = 1.0 + data / c_decades**2
        rho[0, :data_count] = (
            4.0 * weights**2 * c_decades**2 * (np.sqrt(scaled) - 1.0)
        )
        rho[1, :data_count] = 2.0 * weights**2 / np.sqrt(scaled)
        rho[2, :data_count] = -(weights**2 / c_decades**2) * scaled ** (-1.5)
        if values.size > data_count:
            rho[0, data_count:] = 2.0 * values[data_count:]
            rho[1, data_count:] = 2.0
            rho[2, data_count:] = 0.0
        return rho

    return loss


def _poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


def solve_local(
    problem: object,
    start: np.ndarray,
    *,
    max_nfev: int,
    cancelled: Callable[[], bool] | None = None,
) -> LocalSearchResult:
    """Optimize one compiled start using its exact analytic Jacobian."""
    unit = _validated_unit(problem, start, "start")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    _poll(cancelled)
    if unit.size == 0:
        evaluation = evaluate_vector(problem, unit)
        return LocalSearchResult(unit, evaluation, "no_free_parameters", 1)
    start_evaluation = evaluate_vector(problem, unit)

    def residual(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        return local_residual(problem, value)

    def jacobian(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        return local_jacobian(problem, value)

    optimized = least_squares(
        residual,
        unit,
        jac=jacobian,
        bounds=(0.0, 1.0),
        loss=_least_squares_loss(problem),
        max_nfev=max_nfev,
        method="trf",
        x_scale="jac",
    )
    result_unit = _validated_unit(problem, optimized.x, "solver result")
    evaluation = evaluate_vector(problem, result_unit)
    if start_evaluation.valid and (
        not evaluation.valid or evaluation.objective > start_evaluation.objective
    ):
        return LocalSearchResult(
            unit,
            start_evaluation,
            "local_objective_increased",
            int(optimized.nfev),
        )
    return LocalSearchResult(
        result_unit,
        evaluation,
        str(optimized.message),
        int(optimized.nfev),
    )
