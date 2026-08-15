"""SciPy solver boundaries for global joint fitting coordinates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from xrr_fitter.fit.joint_evaluation import (
    JointEvaluation,
    evaluate_joint_jacobian,
    evaluate_joint_vector,
    joint_least_squares_loss,
)
from xrr_fitter.fit.local_search import SearchCancelled


@dataclass(frozen=True, slots=True)
class SolvedJoint:
    unit_vector: np.ndarray
    evaluation: JointEvaluation
    stop_reason: str
    nfev: int
    objective_increased: bool = False


def poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


def solve_joint(
    problem: object,
    start: np.ndarray,
    max_nfev: int,
    cancelled: Callable[[], bool] | None,
) -> SolvedJoint:
    """Run bounded local least squares in the compiled global layout."""
    unit = np.asarray(start, dtype=float)
    poll(cancelled)
    if unit.size == 0:
        return SolvedJoint(
            unit,
            evaluate_joint_vector(problem, unit),
            "no_free_parameters",
            1,
        )
    initial_evaluation = evaluate_joint_vector(problem, unit)

    def residual(value: np.ndarray) -> np.ndarray:
        poll(cancelled)
        return evaluate_joint_vector(problem, value).residuals

    def jacobian(value: np.ndarray) -> np.ndarray:
        poll(cancelled)
        return evaluate_joint_jacobian(problem, value)

    solved = least_squares(
        residual,
        unit,
        jac=jacobian,
        bounds=(0.0, 1.0),
        loss=joint_least_squares_loss(problem),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        x_scale="jac",
        max_nfev=max_nfev,
        callback=lambda *_args, **_kwargs: poll(cancelled),
    )
    result_unit = np.array(solved.x, dtype=float, copy=True)
    evaluation = evaluate_joint_vector(problem, result_unit)
    tolerance = max(1e-12, 1e-8 * initial_evaluation.objective)
    objective_increased = bool(
        initial_evaluation.valid
        and (not evaluation.valid or evaluation.objective > initial_evaluation.objective + tolerance)
    )
    if objective_increased:
        return SolvedJoint(
            np.array(unit, dtype=float, copy=True),
            initial_evaluation,
            "local_objective_increased",
            int(solved.nfev),
            True,
        )
    return SolvedJoint(
        result_unit,
        evaluation,
        str(solved.message),
        int(solved.nfev),
    )


def solve_joint_global(
    problem: object,
    start: np.ndarray,
    population: np.ndarray,
    *,
    seed: int,
    maxiter: int,
    cancelled: Callable[[], bool] | None,
) -> SolvedJoint:
    """Run differential evolution in the compiled global layout."""
    unit = np.asarray(start, dtype=float)
    poll(cancelled)
    if unit.size == 0:
        return SolvedJoint(
            unit,
            evaluate_joint_vector(problem, unit),
            "no_free_parameters",
            1,
        )

    def objective(value: np.ndarray) -> float:
        poll(cancelled)
        return evaluate_joint_vector(problem, value).objective

    solved = differential_evolution(
        objective,
        [(0.0, 1.0)] * len(problem.global_variables),
        init=np.asarray(population, dtype=float),
        seed=np.random.default_rng(seed),
        maxiter=maxiter,
        updating="deferred",
        polish=False,
        tol=1e-6,
        workers=1,
        callback=lambda *_args, **_kwargs: poll(cancelled),
    )
    result_unit = np.array(solved.x, dtype=float, copy=True)
    return SolvedJoint(
        result_unit,
        evaluate_joint_vector(problem, result_unit),
        str(solved.message),
        int(solved.nfev),
    )


__all__ = ["SolvedJoint", "poll", "solve_joint", "solve_joint_global"]
