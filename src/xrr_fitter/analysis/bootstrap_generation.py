"""Ordered noise-model draws with unchanged observation identity and masks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from xrr_fitter.analysis.residual_resampling import moving_block_draw, residual_block_length
from xrr_fitter.evaluation import EvaluationConstraintError, validate_noise_data
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitEvaluationContext

Recompile = Callable[[FitEvaluationContext, PreparedData], FitEvaluationContext]
BOOTSTRAP_METHODS = {
    "gaussian": "gaussian_parametric",
    "poisson": "poisson_parametric",
    "robust_log": "robust_log_moving_block",
}


@dataclass(frozen=True, slots=True)
class BootstrapSource:
    problem: FitEvaluationContext
    indices: np.ndarray
    model: np.ndarray
    centered_residuals: np.ndarray
    block_length: int


def bootstrap_source(problem: FitEvaluationContext, model: np.ndarray, fit_residuals: np.ndarray) -> BootstrapSource:
    validate_noise_data(problem.data, problem.config.noise_model)
    indices = np.flatnonzero(problem.data.fit_mask)
    if problem.objective_point_count != indices.size or np.any(problem.sampling_multipliers != 1.0):
        raise ValueError("bootstrap requires the full fitted observation grid")
    order = np.argsort(problem.data.qz_a_inv[indices], kind="stable")
    residuals = np.asarray(fit_residuals, dtype=float)
    prediction = np.asarray(model, dtype=float)
    if residuals.shape != indices.shape or prediction.shape != problem.data.fit_mask.shape:
        raise ValueError("bootstrap source axes must match the full fitted observations")
    model = prediction[indices[order]]
    residuals = residuals[order]
    if indices.size < 3 or np.any(~np.isfinite(residuals)) or np.any(~np.isfinite(model)):
        raise ValueError("bootstrap source must contain finite fitted residuals and predictions")
    centered = residuals.copy()
    block_length = 1
    if problem.config.noise_model == "robust_log":
        scale = float(np.max(np.abs(centered)))
        if scale > 0:
            centered -= float(np.mean(centered / scale)) * scale
        block_length = residual_block_length(centered)
    return BootstrapSource(problem, indices[order], model, centered, block_length)


def _draw(source: BootstrapSource, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    data, mode = source.problem.data, source.problem.config.noise_model
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        if mode == "poisson":
            counts = _poisson_draw(data.normalization * source.model, rng)
            if np.any(counts > 2**53):
                raise FloatingPointError("Poisson draw exceeds exact floating count representation")
            return counts / data.normalization, counts
        if mode == "gaussian":
            normalized = rng.normal(source.model, data.intensity_sigma_normalized[source.indices])
        else:
            sampled = moving_block_draw(source.centered_residuals, source.block_length, rng)
            normalized = (source.model + data.r_floor) * 10.0 ** (-sampled) - data.r_floor
            if np.any(normalized <= -data.r_floor):
                raise FloatingPointError("log bootstrap draw is outside the representable log domain")
        return normalized, normalized * data.normalization


def _poisson_draw(mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    try:
        return rng.poisson(mean)
    except ValueError as error:
        raise FloatingPointError(f"Poisson draw unavailable:{error}") from error


def synthetic_context(source: BootstrapSource, rng: np.random.Generator, recompile: Recompile) -> FitEvaluationContext:
    normalized, raw = _draw(source, rng)
    data = source.problem.data
    new_normalized, new_raw = data.intensity_normalized.copy(), data.intensity_raw.copy()
    new_normalized[source.indices], new_raw[source.indices] = normalized, raw
    generated = replace(data, intensity_normalized=new_normalized, intensity_raw=new_raw)
    return recompile(source.problem, generated)


def poll_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("cancelled")


def draw_replicates(sources: tuple[BootstrapSource, ...], count: int, rng, recompile: Recompile, cancelled):
    draws = []
    for _index in range(count):
        poll_cancelled(cancelled)
        try:
            draws.append(tuple(synthetic_context(source, rng, recompile) for source in sources))
        except (EvaluationConstraintError, FloatingPointError) as error:
            draws.append(f"generation_failed:{type(error).__name__}:{error}")
    poll_cancelled(cancelled)
    return tuple(draws)
