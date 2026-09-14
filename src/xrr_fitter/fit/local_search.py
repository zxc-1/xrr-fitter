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


class StageSkipped(RuntimeError):
    """Raised by a cancellation probe to abandon the current stage only.

    取消探针是求解器每到一个阶段边界都会问的那一句「要停了吗」。跳过复用同一批停车
    点，只是走另一个出口：抛这个异常而不是返回 True，搜索作废当前阶段后接着往下走。
    它刻意不带 ``_xrr_cooperative_cancellation``——那个标记的意思是「整次运行到此为
    止」，而跳过恰恰相反。
    """


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


def _step_length(previous: list[np.ndarray | None], value: np.ndarray) -> float | None:
    """Return the distance actually travelled since the last reported point."""
    last = previous[0]
    previous[0] = np.array(value, dtype=float, copy=True)
    return None if last is None else float(np.linalg.norm(previous[0] - last))


def solve_local(
    problem: object,
    start: np.ndarray,
    *,
    max_nfev: int,
    cancelled: Callable[[], bool] | None = None,
    iteration_callback: Callable[[np.ndarray, int, int, float | None], None] | None = None,
    callback_interval: int = 10,
) -> LocalSearchResult:
    """Optimize one compiled start using its exact analytic Jacobian.

    ``iteration_callback`` receives ``(unit_vector, iteration, nfev, step)``. The
    iteration count is the number of Jacobian evaluations, which is what ``trf``
    spends one of per iteration; ``step`` is the distance from the previously
    reported point, and is ``None`` on the first report because no step has been
    taken yet. SciPy's ``trf`` never exposes its trust-region radius, so that
    travelled distance is the only step length available without inventing one.
    """
    unit = _validated_unit(problem, start, "start")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    _poll(cancelled)
    if unit.size == 0:
        evaluation = evaluate_vector(problem, unit)
        return LocalSearchResult(unit, evaluation, "no_free_parameters", 1)
    start_evaluation = evaluate_vector(problem, unit)
    system_residual, system_jacobian = cached_least_squares_callbacks(partial(least_squares_system, problem))
    nfev_counter = [0]
    njev_counter = [0]
    reported: list[np.ndarray | None] = [None]

    def residual(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        nfev_counter[0] += 1
        if iteration_callback is not None and nfev_counter[0] % callback_interval == 0:
            iteration_callback(value, njev_counter[0], nfev_counter[0], _step_length(reported, value))
        return system_residual(value)

    def jacobian(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        njev_counter[0] += 1
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
