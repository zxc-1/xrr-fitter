"""Ordered task graph for batches of independent parameter profiles.

The profile math remains in ``analysis.profiles``. This module owns only the
runtime graph that prepares problem-bound plans, flattens every lower and upper
walk into one phase, then finalizes each profile in a second phase. The caller
supplies the plan operations so the runtime graph never imports its mathematical
owner. Service code may inject an ordered thread runner; direct analysis calls
execute the same task sequence serially.

Direction results carry their authoritative grid positions and runner results
are consumed by declaration index. Completion timing therefore cannot change
profile arrays, progress order, or the requested parameter order. Plans contain
process-local closures and are never serialized into analysis requests,
checkpoints, results, or projects.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from xrr_fitter.evaluation import (
    EvaluationConstraintError,
)
from xrr_fitter.model.analysis import ParameterProfile
from xrr_fitter.model.fitting import FitEvaluationContext


def _problem_profile_plan(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    name: str,
    prepare_plan: Callable[..., object],
    evaluate: Callable[..., object],
    cache_callbacks: Callable[..., tuple[Callable, Callable]],
    least_squares_system: Callable[..., tuple[np.ndarray, np.ndarray]],
    least_squares_loss: Callable[..., object],
    values_by_name: Callable[..., dict[str, float]],
    cancelled: Callable[[], bool] | None = None,
) -> object:
    """Bind one declared parameter to a prepared generic profile plan."""
    from xrr_fitter.analysis.binary_profiles import (
        binary_derived_profiles,
        build_binary_profile,
    )

    derived = {item.name for item in binary_derived_profiles(problem)}
    if name in derived:
        return build_binary_profile(
            problem,
            unit_vector,
            name,
            profile_builder=prepare_plan,
            cancelled=cancelled,
        )
    names = tuple(variable.name for variable in problem.variables)
    if name not in names:
        raise ValueError(f"unknown profile parameter: {name}")
    index = names.index(name)

    def objective(unit: np.ndarray) -> float:
        try:
            evaluation = evaluate(problem, unit)
        except EvaluationConstraintError:
            return np.inf
        return evaluation.objective if evaluation.valid else np.inf

    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * max(1, len(problem.variables)),
    )
    steps = 11 if problem.config.budget.bootstrap_samples < 100 else 41
    residual, jacobian = cache_callbacks(
        partial(least_squares_system, problem)
    )
    return prepare_plan(
        objective,
        unit_vector,
        parameter_index=index,
        name=name,
        value_mapper=lambda unit: values_by_name(problem, unit)[name],
        residual=residual,
        residual_jacobian=jacobian,
        residual_loss=least_squares_loss(problem),
        least_squares_max_nfev=maximum,
        steps=steps,
        cancelled=cancelled,
    )


def _run_profile_tasks(
    tasks: tuple[Callable[[], object], ...],
    task_runner,
) -> tuple[object, ...]:
    """Execute one ordered phase and reject incomplete runner responses."""
    results = (
        tuple(task() for task in tasks)
        if task_runner is None
        else tuple(task_runner(tasks))
    )
    if len(results) != len(tasks):
        raise RuntimeError("task runner returned an unexpected result count")
    return results


def build_problem_profiles(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    names: tuple[str, ...],
    *,
    prepare_plan: Callable[..., object],
    scan_plan_direction: Callable[[object, int], object],
    finish_plan: Callable[[object, tuple[object, object]], object],
    evaluate: Callable[..., object],
    cache_callbacks: Callable[..., tuple[Callable, Callable]],
    least_squares_system: Callable[..., tuple[np.ndarray, np.ndarray]],
    least_squares_loss: Callable[..., object],
    values_by_name: Callable[..., dict[str, float]],
    cancelled: Callable[[], bool] | None = None,
    task_runner=None,
) -> tuple[ParameterProfile, ...]:
    """Build profiles through flattened direction and refinement task batches.

    Phase one declares exactly two tasks per profile in lower/upper order. Phase
    two declares exactly one finalization task per profile. Both runner results
    are indexed rather than observed by completion time.
    """
    plans = tuple(
        _problem_profile_plan(
            problem,
            unit_vector,
            name,
            prepare_plan,
            evaluate,
            cache_callbacks,
            least_squares_system,
            least_squares_loss,
            values_by_name,
            cancelled,
        )
        for name in names
    )
    direction_tasks = tuple(
        partial(scan_plan_direction, plan, direction)
        for plan in plans
        for direction in (-1, 1)
    )
    directional = _run_profile_tasks(direction_tasks, task_runner)
    scans = tuple(
        (directional[2 * index], directional[2 * index + 1])
        for index in range(len(plans))
    )
    finish_tasks = tuple(
        partial(finish_plan, plan, scan)
        for plan, scan in zip(plans, scans, strict=True)
    )
    finished = _run_profile_tasks(finish_tasks, task_runner)
    return tuple(result[0] for result in finished)
