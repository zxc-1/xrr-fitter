"""SciPy solver boundaries for global joint fitting coordinates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from xrr_fitter.evaluation import cached_least_squares_callbacks
from xrr_fitter.fit.adaptive_grid import GenerationStagnation
from xrr_fitter.fit.joint_evaluation import (
    JointEvaluation,
    evaluate_joint_vector,
    joint_least_squares_loss,
    joint_least_squares_system,
)
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.model.search import SearchEvidence


@dataclass(frozen=True, slots=True)
class SolvedJoint:
    unit_vector: np.ndarray
    evaluation: JointEvaluation
    stop_reason: str
    nfev: int
    objective_increased: bool = False
    converged: bool = True
    population: np.ndarray | None = None
    population_energies: np.ndarray | None = None
    search_evidence: tuple[SearchEvidence, ...] = ()


def poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


def cached_joint_least_squares_callbacks(problem: object, cancelled: Callable[[], bool] | None = None):
    """Own one thread-local system cache for this optimizer and compiled problem."""
    residual, jacobian = cached_least_squares_callbacks(partial(joint_least_squares_system, problem))

    def evaluate(callback, value):
        poll(cancelled)
        result = callback(value)
        poll(cancelled)
        return result

    return partial(evaluate, residual), partial(evaluate, jacobian)


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

    residual, jacobian = cached_joint_least_squares_callbacks(problem, cancelled)

    solved = least_squares(
        residual,
        unit,
        jac=jacobian,
        bounds=(0.0, 1.0),
        loss=joint_least_squares_loss(problem),
        # One high-count member can dominate the total cost without fixing the shared point.
        ftol=None if any(member.config.noise_model == "poisson" for member in problem.problems) else 1e-10,
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
        converged=bool(solved.success),
    )


def refit_resampled_joint(problem, start, members, *, cancelled=None) -> np.ndarray | str:
    """Refit one complete generated shared problem and return global physical values."""
    generated = compile_joint_problem(problem.dataset_ids, members, problem.sharing_rules, problem.constraint_rules)
    budget = members[0].config.budget
    maximum = max(budget.local_min_nfev, budget.local_nfev_per_parameter * max(1, len(problem.global_variables)))
    solved = solve_joint(generated, start, maximum, cancelled)
    if not solved.evaluation.valid or solved.objective_increased or not solved.converged:
        return f"joint_fit_failed:{solved.stop_reason}"
    values = {
        (dataset_id, parameter.name): parameter.value
        for dataset_id, evaluation in zip(generated.dataset_ids, solved.evaluation.local_evaluations, strict=True)
        for parameter in evaluation.parameters
    }
    return np.asarray(
        [
            values[(variable.members[0].dataset_id, variable.members[0].parameter_name)]
            for variable in generated.global_variables
        ]
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

    members = np.asarray(population, dtype=float)
    stagnation = GenerationStagnation()
    evaluations = 0

    def objective(value: np.ndarray) -> float:
        nonlocal evaluations
        poll(cancelled)
        result = evaluate_joint_vector(problem, value, fit_only=True)
        stagnation.observe(value, result.objective)
        evaluations += 1
        if evaluations == len(members):
            stagnation.start_generations()
        return result.objective

    def generation_finished(_unit: np.ndarray, convergence: float = 0.0) -> bool:
        poll(cancelled)
        return stagnation.finish_generation()

    solved = differential_evolution(
        objective,
        [(0.0, 1.0)] * len(problem.global_variables),
        init=members,
        seed=np.random.default_rng(seed),
        maxiter=maxiter,
        updating="deferred",
        polish=False,
        # Match the single-curve three-generation rule, not energy-spread convergence.
        tol=0.0,
        atol=-1.0,
        workers=1,
        callback=generation_finished,
    )
    result_unit = np.array(solved.x, dtype=float, copy=True)
    return SolvedJoint(
        result_unit,
        evaluate_joint_vector(problem, result_unit),
        "three_generation_stagnation" if stagnation.stopped else str(solved.message),
        int(solved.nfev),
        population=np.array(getattr(solved, "population", population), dtype=float, copy=True),
        population_energies=np.array(getattr(solved, "population_energies", ()), dtype=float, copy=True),
    )


__all__ = ["SolvedJoint", "poll", "solve_joint", "solve_joint_global"]
