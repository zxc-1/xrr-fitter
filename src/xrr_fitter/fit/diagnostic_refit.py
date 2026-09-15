"""Fixed declared-start numerical estimator for Poisson diagnostic calibration.

Observed and generated problems use the same deterministic starts and the same
bounded analytic solver. No search winner is accepted as a warm start, and one
failed path invalidates the whole refit rather than selecting a successful
subset. Joint members are recompiled into their common constraint layout.

``nfev`` counts actual localization Q/g and refinement residual requests plus
one full-axis handoff validation, not Jacobian calls or the final publication
traversal. Both phases share the original per-path budget. Cooperative cancellation
is propagated through SearchCancelled and never becomes a partial refit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from scipy.stats import qmc

from xrr_fitter.evaluation import (
    encode_physical_vector,
    least_squares_loss,
    least_squares_system,
)
from xrr_fitter.fit import diagnostic_solver
from xrr_fitter.fit.diagnostic_solver import NUMERICAL_ERRORS, numerical_reason
from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector, joint_least_squares_loss, joint_least_squares_system
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import initial_joint_vector
from xrr_fitter.fit.joint_solvers import poll
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.model.diagnostic_calibration import DiagnosticRefit
from xrr_fitter.model.diagnostic_work import (
    PHASES,
    DiagnosticPathSummary,
    DiagnosticRefitWork,
    DiagnosticSolverExit,
    combine_refit_work,
)
from xrr_fitter.model.evaluation import ModelEvaluation
from xrr_fitter.model.fitting import FitEvaluationContext

EMPTY_REFIT_WORK = DiagnosticRefitWork()


def diagnostic_starts(initial: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the declaration and three fixed Sobol interiors, deduplicated in order."""
    unit = np.array(initial, dtype=float, copy=True)
    if unit.ndim != 1 or np.any(~np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("diagnostic initial unit vector must be finite, one-dimensional, and in [0, 1]")
    starts = [unit]
    if unit.size:
        for point in qmc.Sobol(unit.size, scramble=False).random_base2(2)[1:]:
            if not any(np.array_equal(point, previous) for previous in starts):
                starts.append(np.array(point, copy=True))
    return tuple(starts)


@dataclass(frozen=True, slots=True)
class _RefitProblem:
    members: tuple[FitEvaluationContext, ...]
    evaluate: Callable[[np.ndarray], tuple[float, tuple[ModelEvaluation, ...]]]
    system: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    loss: Callable[[np.ndarray], np.ndarray]


def _failed(reason: str, work: DiagnosticRefitWork = EMPTY_REFIT_WORK) -> DiagnosticRefit:
    return DiagnosticRefit(None, (), work.nfev, work.path_count, reason, work)


def _evaluation_axes(member, evaluation):
    full_shape = member.data.qz_a_inv.shape
    fitted_shape = (int(np.count_nonzero(member.data.fit_mask)),)
    shapes = (
        (evaluation.qz_a_inv.shape, full_shape),
        (evaluation.model_normalized.shape, full_shape),
        (evaluation.fit_residuals.shape, fitted_shape),
        (evaluation.fit_weighted_residuals.shape, fitted_shape),
    )
    if any(actual != expected for actual, expected in shapes):
        return "diagnostic_refit_incomplete_evaluation"
    return None


def _evaluation_finiteness(member, evaluation):
    qz, mean = evaluation.qz_a_inv, evaluation.model_normalized
    if np.any(np.isinf(qz)) or np.any(np.isinf(mean)) or not np.array_equal(np.isnan(qz), np.isnan(mean)):
        return "diagnostic_refit_nonfinite_reporting_axes"
    fitted = (
        qz[member.data.fit_mask],
        mean[member.data.fit_mask],
        evaluation.fit_residuals,
        evaluation.fit_weighted_residuals,
    )
    if any(np.any(~np.isfinite(value)) for value in fitted):
        return "diagnostic_refit_nonfinite_fitted_evaluation"
    return None


def _evaluation_failure(member, evaluation):
    if not evaluation.valid:
        return f"diagnostic_refit_invalid_evaluation:{evaluation.reason}"
    if evaluation.noise_model != member.config.noise_model:
        return "diagnostic_refit_noise_mismatch"
    if not np.isfinite(evaluation.objective):
        return "diagnostic_refit_nonfinite_objective"
    return _evaluation_axes(member, evaluation) or _evaluation_finiteness(member, evaluation)


def _evaluations_failure(problem, score, evaluations):
    if len(evaluations) != len(problem.members):
        return "diagnostic_refit_incomplete_members"
    for index, (member, evaluation) in enumerate(zip(problem.members, evaluations, strict=True)):
        reason = _evaluation_failure(member, evaluation)
        if reason is not None:
            return f"{reason}:member={index}"
    if not np.isfinite(score):
        return "diagnostic_refit_nonfinite_objective"
    return None


def _validate_handoff(problem, unit):
    score, evaluations = problem.evaluate(unit)
    return _evaluations_failure(problem, score, evaluations)


def _path_work(solver, declared_paths, zero_dimensional):
    entered = {stage.phase: stage for stage in solver.exits}
    missing_kind = "zero_dimensional" if zero_dimensional else "not_entered"
    stages = tuple(
        entered.get(phase) or DiagnosticSolverExit(phase, None, False, missing_kind, 0, missing_kind)
        for phase in PHASES
    )
    return DiagnosticRefitWork(declared_paths, (DiagnosticPathSummary(solver.budget, stages),))


def _solve_path(problem, start, cancelled, declared_paths):
    budget = problem.members[0].config.budget
    callbacks = diagnostic_solver.TwoStageSolver(
        problem.system,
        problem.loss,
        partial(_validate_handoff, problem),
        budget=max(budget.local_min_nfev, budget.local_nfev_per_parameter * max(1, start.size)),
        poisson=any(member.config.noise_model == "poisson" for member in problem.members),
        cancelled=cancelled,
    )
    try:
        poll(cancelled)
        unit, failure = callbacks.solve(start)
        if failure is not None:
            return float("inf"), _failed(failure, _path_work(callbacks, declared_paths, start.size == 0))
        score, evaluations = problem.evaluate(unit)
        poll(cancelled)
        failure = _evaluations_failure(problem, score, evaluations)
        if failure is not None:
            return float("inf"), _failed(failure, _path_work(callbacks, declared_paths, start.size == 0))
        return score, DiagnosticRefit(
            unit, evaluations, callbacks.nfev, 1, work=_path_work(callbacks, declared_paths, start.size == 0)
        )
    except NUMERICAL_ERRORS as error:
        poll(cancelled)
        return float("inf"), _failed(numerical_reason(error), _path_work(callbacks, declared_paths, start.size == 0))


def _refit(problem: _RefitProblem, initial: np.ndarray, cancelled) -> DiagnosticRefit:
    best, best_score, work = None, float("inf"), DiagnosticRefitWork()
    starts = diagnostic_starts(initial)
    for attempted, start in enumerate(starts, start=1):
        score, result = _solve_path(problem, start, cancelled, len(starts))
        work = combine_refit_work(work, result.work)
        if result.failure_reason is not None:
            return replace(result, nfev=work.nfev, attempted_paths=attempted, work=work)
        if score < best_score:
            best, best_score = result, score
    poll(cancelled)
    return replace(best, nfev=work.nfev, attempted_paths=attempted, work=work)


def _single_evaluation(problem, unit):
    evaluation = evaluate_vector(problem, unit)
    return evaluation.objective, (evaluation,)


def _joint_evaluation(problem, unit):
    evaluation = evaluate_joint_vector(problem, unit)
    return evaluation.objective, evaluation.local_evaluations


def refit_diagnostic_single(
    problem: FitEvaluationContext, *, cancelled: Callable[[], bool] | None = None
) -> DiagnosticRefit:
    """Run every declared path on one compiled observed or resampled context."""
    poll(cancelled)
    try:
        initial = encode_physical_vector(problem, {})
    except NUMERICAL_ERRORS as error:
        poll(cancelled)
        return _failed(numerical_reason(error))
    numerical = _RefitProblem(
        (problem,),
        partial(_single_evaluation, problem),
        partial(least_squares_system, problem),
        least_squares_loss(problem),
    )
    return _refit(numerical, initial, cancelled)


def refit_diagnostic_joint(template, members, *, cancelled: Callable[[], bool] | None = None) -> DiagnosticRefit:
    """Recompile and refit all generated members together, including shared parameters."""
    poll(cancelled)
    try:
        problem = compile_joint_problem(
            template.dataset_ids, members, template.sharing_rules, template.joint_constraint_rules
        )
        poll(cancelled)
        initial = initial_joint_vector(problem)
    except NUMERICAL_ERRORS as error:
        poll(cancelled)
        return _failed(numerical_reason(error))
    numerical = _RefitProblem(
        problem.problems,
        partial(_joint_evaluation, problem),
        partial(joint_least_squares_system, problem),
        joint_least_squares_loss(problem),
    )
    return _refit(numerical, initial, cancelled)


__all__ = ["diagnostic_starts", "refit_diagnostic_single", "refit_diagnostic_joint"]
