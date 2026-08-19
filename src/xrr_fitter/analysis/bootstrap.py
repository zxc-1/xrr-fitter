"""Deterministic local and problem-bound bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from math import isfinite

import numpy as np
from scipy.optimize import least_squares

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
    moving_block_draw as _moving_block_draw,
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


def _sorted_fit_indices(problem: object) -> np.ndarray:
    fit_indices = np.flatnonzero(problem.data.fit_mask)
    order = np.argsort(problem.data.qz_a_inv[fit_indices], kind="stable")
    return fit_indices[order]


def _candidate_center(problem: object, candidate: object, indices: np.ndarray):
    residuals = np.asarray(candidate.log_residuals_decades, dtype=float)[indices]
    model = np.asarray(candidate.model_normalized, dtype=float)[indices]
    if (
        residuals.shape != model.shape
        or residuals.size < 3
        or np.any(~np.isfinite(residuals))
        or np.any(~np.isfinite(model))
    ):
        raise ValueError("bootstrap candidate has invalid fitted residuals")
    names = tuple(variable.name for variable in problem.variables)
    physical = {parameter.name: parameter.value for parameter in candidate.parameters}
    return names, physical, residuals, model


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


def _synthetic_context(
    problem: object,
    sorted_indices: np.ndarray,
    synthetic_fit: np.ndarray,
):
    normalized = np.array(problem.data.intensity_normalized, copy=True)
    normalized[sorted_indices] = synthetic_fit
    data = replace(
        problem.data,
        intensity_normalized=normalized,
        intensity_raw=normalized * problem.data.normalization,
    )
    return replace(problem, data=data)


def _local_bootstrap_fit(problem: object, start: np.ndarray) -> np.ndarray | None:
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
    except (EvaluationConstraintError, FloatingPointError):
        return None
    tolerance = max(1e-12, 1e-8 * initial.objective)
    if not fitted.valid or not isfinite(fitted.objective):
        return None
    if fitted.objective > initial.objective + tolerance:
        return np.array(start, copy=True)
    return np.asarray(optimized.x, dtype=float)


def _fit_problem_bootstrap_sample(
    problem: object,
    candidate: object,
    names: tuple[str, ...],
    cancelled: Callable[[], bool] | None,
) -> np.ndarray | None:
    if cancelled is not None and cancelled():
        raise InterruptedError("cancelled")
    start = np.asarray(candidate.unit_vector, dtype=float)
    fitted = _local_bootstrap_fit(problem, start)
    if fitted is None:
        return None
    mapped = values_by_name(problem, fitted)
    return np.asarray([mapped[name] for name in names], dtype=float)


def bootstrap_problem_local(
    problem: FitEvaluationContext,
    candidate: object,
    *,
    sample_count: int,
    child_seed: int,
    cancelled: Callable[[], bool] | None = None,
    progress: BootstrapProgress | None = None,
    task_runner: TaskRunner | None = None,
) -> BootstrapResult:
    """Bootstrap one accepted candidate and refit every synthetic curve."""
    sorted_indices = _sorted_fit_indices(problem)
    names, physical, residuals, model = _candidate_center(problem, candidate, sorted_indices)
    names = _validated_names(names)
    count = _validated_sample_count(sample_count)
    sigma = problem.data.intensity_sigma_normalized
    explicit_sigma = None if sigma is None else np.asarray(sigma, dtype=float)[sorted_indices]
    block_length = residual_block_length(residuals)
    rng = np.random.default_rng(child_seed)
    contexts = []
    for _sample_index in range(count):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        if explicit_sigma is not None:
            synthetic_fit = rng.normal(model, explicit_sigma)
        else:
            sampled = _moving_block_draw(residuals, block_length, rng)
            synthetic_fit = (model + problem.data.r_floor) * 10.0 ** (-sampled) - problem.data.r_floor
        synthetic_fit = np.clip(synthetic_fit, problem.data.r_floor, np.inf)
        contexts.append(_synthetic_context(problem, sorted_indices, synthetic_fit))

    del physical
    tasks = tuple(
        partial(
            _fit_problem_bootstrap_sample,
            context,
            candidate,
            names,
            cancelled,
        )
        for context in contexts
    )
    fitted_values = _run_tasks(tasks, task_runner)
    result = _bootstrap_result(names, fitted_values, count, progress)
    return _owned_bootstrap(problem, candidate, result)
