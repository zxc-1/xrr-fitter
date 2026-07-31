"""Deterministic local and problem-bound bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import isfinite

import numpy as np
from scipy.optimize import least_squares

from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    evaluate_model,
    least_squares_loss,
    least_squares_residual,
    least_squares_residual_jacobian,
    values_by_name,
)
from xrr_fitter.model.analysis import BootstrapResult
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.provenance import bootstrap_provenance_sha256


BootstrapFit = Callable[[np.random.Generator, int], np.ndarray | None]
BootstrapProgress = Callable[[int, int], None]


def _validated_names(parameter_names: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(parameter_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("bootstrap parameter_names must be nonempty strings")
    return names


def _validated_sample_count(sample_count: int) -> int:
    valid = (
        not isinstance(sample_count, bool)
        and isinstance(sample_count, (int, np.integer))
        and sample_count >= 1
    )
    if not valid:
        raise ValueError("bootstrap sample_count must be a positive integer")
    return int(sample_count)


def bootstrap_local(
    fit_sample: BootstrapFit,
    parameter_names: tuple[str, ...],
    *,
    sample_count: int,
    child_seed: int,
    progress: BootstrapProgress | None = None,
) -> BootstrapResult:
    """Aggregate callback fits in index order using one deterministic stream."""
    names = _validated_names(parameter_names)
    count = _validated_sample_count(sample_count)
    rng = np.random.default_rng(child_seed)
    samples: list[np.ndarray] = []
    failures = 0
    for sample_index in range(count):
        fitted = fit_sample(rng, sample_index)
        if fitted is None:
            failures += 1
        else:
            vector = np.asarray(fitted, dtype=float)
            if vector.shape != (len(names),) or np.any(~np.isfinite(vector)):
                raise ValueError("bootstrap fit returned an invalid parameter vector")
            samples.append(vector)
        if progress is not None:
            progress(sample_index + 1, count)
    matrix = np.vstack(samples) if samples else np.empty((0, len(names)), dtype=float)
    failure_rate = failures / count
    if failure_rate > 0.20 or matrix.shape[0] == 0:
        intervals: tuple[tuple[str, float, float], ...] = ()
    else:
        lower, upper = np.percentile(matrix, (2.5, 97.5), axis=0)
        intervals = tuple(
            (name, float(lower[index]), float(upper[index]))
            for index, name in enumerate(names)
        )
    return BootstrapResult(names, matrix, intervals, float(failure_rate))


def _first_nonpositive_lag(centered: np.ndarray, denominator: float) -> int | None:
    for lag in range(1, min(25, centered.size - 1) + 1):
        autocorrelation = float(centered[:-lag] @ centered[lag:]) / denominator
        if autocorrelation <= 0.0:
            return lag
    return None


def residual_block_length(residuals: np.ndarray) -> int:
    values = np.asarray(residuals, dtype=float)
    centered = values - np.mean(values)
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        return 3
    crossing = _first_nonpositive_lag(centered, denominator)
    return int(np.clip(25 if crossing is None else crossing, 3, 25))


def _moving_block_draw(
    residuals: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(residuals, dtype=float)
    blocks: list[np.ndarray] = []
    size = 0
    while size < values.size:
        start = int(rng.integers(0, values.size))
        block = values[(start + np.arange(block_length)) % values.size]
        blocks.append(block)
        size += block.size
    return np.concatenate(blocks)[: values.size]


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
        optimized = least_squares(
            lambda value: least_squares_residual(problem, value),
            start,
            jac=lambda value: least_squares_residual_jacobian(problem, value),
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


def bootstrap_problem_local(
    problem: FitEvaluationContext,
    candidate: object,
    *,
    sample_count: int,
    child_seed: int,
    cancelled: Callable[[], bool] | None = None,
    progress: BootstrapProgress | None = None,
) -> BootstrapResult:
    """Bootstrap one accepted candidate and refit every synthetic curve."""
    sorted_indices = _sorted_fit_indices(problem)
    names, physical, residuals, model = _candidate_center(
        problem, candidate, sorted_indices
    )
    sigma = problem.data.intensity_sigma_normalized
    explicit_sigma = None if sigma is None else np.asarray(sigma, dtype=float)[sorted_indices]
    block_length = residual_block_length(residuals)

    def fit_sample(rng: np.random.Generator, _sample_index: int) -> np.ndarray | None:
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        if explicit_sigma is not None:
            synthetic_fit = rng.normal(model, explicit_sigma)
        else:
            sampled = _moving_block_draw(residuals, block_length, rng)
            synthetic_fit = (
                (model + problem.data.r_floor) * 10.0 ** (-sampled)
                - problem.data.r_floor
            )
        synthetic_fit = np.clip(synthetic_fit, problem.data.r_floor, np.inf)
        synthetic_problem = _synthetic_context(problem, sorted_indices, synthetic_fit)
        start = np.asarray(candidate.unit_vector, dtype=float)
        fitted = _local_bootstrap_fit(synthetic_problem, start)
        if fitted is None:
            return None
        mapped = values_by_name(synthetic_problem, fitted)
        return np.asarray([mapped[name] for name in names], dtype=float)

    del physical
    result = bootstrap_local(
        fit_sample,
        names,
        sample_count=sample_count,
        child_seed=child_seed,
        progress=progress,
    )
    return _owned_bootstrap(problem, candidate, result)
