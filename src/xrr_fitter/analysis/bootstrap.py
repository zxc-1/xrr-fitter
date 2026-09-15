"""Deterministic local and problem-bound bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from math import isfinite

import numpy as np
from scipy.optimize import least_squares

from xrr_fitter.analysis.bootstrap_generation import (
    BOOTSTRAP_METHODS,
    Recompile,
    bootstrap_source,
    draw_replicates,
    poll_cancelled,
)
from xrr_fitter.analysis.bootstrap_samples import (
    BootstrapFit as BootstrapFit,
)
from xrr_fitter.analysis.bootstrap_samples import (
    BootstrapProgress,
    TaskRunner,
)
from xrr_fitter.analysis.bootstrap_samples import (
    bootstrap_local as bootstrap_local,
)
from xrr_fitter.analysis.bootstrap_samples import (
    bootstrap_result_from_fits as _bootstrap_result,
)
from xrr_fitter.analysis.bootstrap_samples import (
    run_tasks as _run_tasks,
)
from xrr_fitter.analysis.bootstrap_samples import (
    validated_bootstrap_names as _validated_names,
)
from xrr_fitter.analysis.bootstrap_samples import (
    validated_sample_count as _validated_sample_count,
)
from xrr_fitter.analysis.residual_resampling import (
    residual_block_length as residual_block_length,
)
from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    cached_least_squares_callbacks,
    evaluate_model,
    least_squares_loss,
    least_squares_system,
    values_by_name,
)
from xrr_fitter.model.analysis import BootstrapResult
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.provenance import bootstrap_provenance_sha256


def _owned_bootstrap(
    problem: FitEvaluationContext,
    candidate: object,
    result: BootstrapResult,
) -> BootstrapResult:
    candidate_id = getattr(candidate, "candidate_id", None)
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("bootstrap candidate must have a nonempty candidate_id")
    return replace(
        result,
        candidate_id=candidate_id,
        provenance_sha256=bootstrap_provenance_sha256(problem, candidate, result),
    )


def _local_bootstrap_fit(problem: object, start: np.ndarray) -> np.ndarray | str:
    if start.size == 0:
        return np.array(start, copy=True)
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * max(1, len(problem.variables)),
    )
    try:
        initial = evaluate_model(problem, start)
        residual, jacobian = cached_least_squares_callbacks(partial(least_squares_system, problem))
        optimized = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(0.0, 1.0),
            loss=least_squares_loss(problem),
            max_nfev=maximum,
            method="trf",
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        fitted = evaluate_model(problem, np.asarray(optimized.x, dtype=float))
    except (EvaluationConstraintError, FloatingPointError) as error:
        return f"{type(error).__name__}:{error}"
    return _converged_result(initial, optimized, fitted)


def _converged_result(initial, optimized, fitted) -> np.ndarray | str:
    if not optimized.success:
        return f"optimizer_nonconvergence:{optimized.message}"
    tolerance = max(1e-12, 1e-8 * initial.objective)
    if not fitted.valid or not isfinite(fitted.objective):
        return f"invalid_fit:{fitted.reason}"
    if fitted.objective > initial.objective + tolerance:
        return "local_objective_increased"
    return np.asarray(optimized.x, dtype=float)


def _fit_problem_bootstrap_sample(
    problem: object,
    candidate: object,
    names: tuple[str, ...],
    cancelled: Callable[[], bool] | None,
) -> np.ndarray | str | None:
    poll_cancelled(cancelled)
    if isinstance(problem, str):
        return problem
    start = np.asarray(candidate.unit_vector, dtype=float)
    fitted = _local_bootstrap_fit(problem, start)
    poll_cancelled(cancelled)
    if fitted is None or isinstance(fitted, str):
        return fitted
    mapped = values_by_name(problem, fitted)
    return np.asarray([mapped[name] for name in names], dtype=float)


def bootstrap_problem_local(
    problem: FitEvaluationContext,
    candidate: object,
    *,
    sample_count: int,
    child_seed: int,
    recompile: Recompile,
    cancelled: Callable[[], bool] | None = None,
    progress: BootstrapProgress | None = None,
    task_runner: TaskRunner | None = None,
) -> BootstrapResult:
    """Bootstrap one accepted candidate and refit every synthetic curve."""
    if not candidate.valid or candidate.noise_model != problem.config.noise_model:
        raise ValueError("bootstrap requires a valid candidate in the declared noise model")
    names = _validated_names(tuple(variable.name for variable in problem.variables))
    count = _validated_sample_count(sample_count)
    source = bootstrap_source(problem, candidate.model_normalized, candidate.residuals[problem.data.fit_mask])
    rng = np.random.default_rng(child_seed)
    contexts = draw_replicates((source,), count, rng, recompile, cancelled)
    tasks = tuple(
        partial(
            _fit_problem_bootstrap_sample,
            context if isinstance(context, str) else context[0],
            candidate,
            names,
            cancelled,
        )
        for context in contexts
    )
    fitted_values = _run_tasks(tasks, task_runner)
    poll_cancelled(cancelled)
    result = _bootstrap_result(
        names, fitted_values, count, progress, method=BOOTSTRAP_METHODS[problem.config.noise_model]
    )
    return _owned_bootstrap(problem, candidate, result)
