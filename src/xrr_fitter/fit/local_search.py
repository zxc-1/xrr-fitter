"""Deterministic analytic local least-squares search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import least_squares

from xrr_fitter.evaluation import (
    cached_least_squares_callbacks,
    least_squares_loss,
    least_squares_residual,
    least_squares_residual_jacobian,
    least_squares_system,
)
from xrr_fitter.fit.objective import evaluate_jacobian, evaluate_vector
from xrr_fitter.model.fitting import ModelEvaluation


class SearchCancelled(RuntimeError):
    """Raised when a cooperative search cancellation is observed."""

    _xrr_cooperative_cancellation = True


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


def local_residual(problem: object, unit_vector: np.ndarray) -> np.ndarray:
    """Delegate the solver residual chain to the shared evaluation boundary."""
    return least_squares_residual(
        problem,
        _validated_unit(problem, unit_vector, "unit vector"),
        evaluator=evaluate_vector,
    )


def local_jacobian(problem: object, unit_vector: np.ndarray) -> np.ndarray:
    """Delegate the residual Jacobian chain to the shared evaluation boundary."""
    return least_squares_residual_jacobian(
        problem,
        _validated_unit(problem, unit_vector, "unit vector"),
        jacobian_evaluator=evaluate_jacobian,
    )


def _least_squares_loss(problem: object) -> Callable[[np.ndarray], np.ndarray]:
    return least_squares_loss(problem)


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
    system_residual, system_jacobian = cached_least_squares_callbacks(partial(least_squares_system, problem))

    def residual(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        return system_residual(value)

    def jacobian(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        return system_jacobian(value)

    optimized = least_squares(
        residual,
        unit,
        jac=jacobian,
        bounds=(0.0, 1.0),
        loss=_least_squares_loss(problem),
        max_nfev=max_nfev,
        method="trf",
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    result_unit = _validated_unit(problem, optimized.x, "solver result")
    evaluation = evaluate_vector(problem, result_unit)
    tolerance = max(1e-12, 1e-8 * start_evaluation.objective)
    if start_evaluation.valid and (
        not evaluation.valid or evaluation.objective > start_evaluation.objective + tolerance
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
