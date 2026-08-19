"""Residual autocorrelation and actionable model diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.stats import spearmanr

from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.instrument import PhysicsDiagnostic


@dataclass(frozen=True, slots=True)
class _OrderedResiduals:
    indices: np.ndarray
    qz: np.ndarray
    theta: np.ndarray
    residual: np.ndarray


def _ordered_data(problem: object, candidate: object) -> _OrderedResiduals:
    qz = np.asarray(problem.data.qz_a_inv, dtype=float)
    fit_mask = np.asarray(problem.data.fit_mask, dtype=bool)
    residual = np.asarray(candidate.log_residuals_decades, dtype=float)
    two_theta_value = getattr(problem.data, "two_theta_deg", None)
    two_theta = qz if two_theta_value is None else np.asarray(two_theta_value, dtype=float)
    if not (qz.shape == two_theta.shape == fit_mask.shape == residual.shape and qz.ndim == 1):
        raise ValueError("residual diagnostic arrays have incompatible shapes")
    selected = np.flatnonzero(fit_mask & np.isfinite(qz) & np.isfinite(two_theta) & np.isfinite(residual))
    order = np.argsort(qz[selected], kind="stable")
    indices = selected[order]
    return _OrderedResiduals(
        indices,
        qz[indices],
        two_theta[indices] / 2.0,
        residual[indices],
    )


def ordered_fit_residuals(
    problem: FitEvaluationContext,
    candidate: object,
) -> np.ndarray:
    """Return finite fitted residuals in stable increasing-q order."""
    result = np.array(_ordered_data(problem, candidate).residual, copy=True)
    result.setflags(write=False)
    return result


def _normalized_centered_residuals(values: np.ndarray) -> tuple[np.ndarray, float] | None:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        centered = values / scale
        centered -= np.mean(centered)
    amplitude = float(np.max(np.abs(centered)))
    if scale <= 1.0:
        below_noise = amplitude * scale <= 1e-8
    else:
        below_noise = amplitude <= 1e-8 / scale
    if below_noise or amplitude == 0.0:
        return None
    centered /= amplitude
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        return None
    return centered, denominator


def _has_significant_autocorrelation(centered: np.ndarray, denominator: float) -> bool:
    significant = 0
    threshold = 3.0 / np.sqrt(centered.size)
    for lag in range(1, min(20, centered.size // 5) + 1):
        autocorrelation = float(centered[:-lag] @ centered[lag:]) / denominator
        if abs(autocorrelation) > threshold:
            significant += 1
            if significant >= 2:
                return True
    return False


def residual_autocorrelation_flag(residuals: np.ndarray) -> bool:
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or values.size < 4 or np.any(~np.isfinite(values)):
        raise ValueError("residuals must be a finite vector with at least four points")
    normalized = _normalized_centered_residuals(values)
    return False if normalized is None else _has_significant_autocorrelation(*normalized)


def _third_median_drop_at_least(values: np.ndarray, threshold: float) -> bool:
    third = max(1, values.size // 3)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        drop = float(np.median(values[:third]) - np.median(values[-third:]))
    if np.isfinite(drop):
        return drop >= threshold
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return False
    normalized = values / scale
    normalized_drop = float(np.median(normalized[:third]) - np.median(normalized[-third:]))
    if normalized_drop <= 0.0:
        return False
    if scale <= 1.0:
        return normalized_drop * scale >= threshold
    return normalized_drop >= threshold / scale


def _trend_detected(
    coordinate: np.ndarray,
    residual: np.ndarray,
    correlation_limit: float,
) -> bool:
    if (
        coordinate.size == 0
        or residual.size == 0
        or np.all(coordinate == coordinate[0])
        or np.all(residual == residual[0])
    ):
        return False
    correlation = float(spearmanr(coordinate, residual).statistic)
    return correlation <= correlation_limit and _third_median_drop_at_least(
        residual,
        0.05,
    )


def _footprint(problem: object, data: _OrderedResiduals) -> PhysicsDiagnostic | None:
    instrument = problem.instrument
    if instrument.footprint_mode != "none" or instrument.footprint_spill_angle_deg != 0.0:
        return None
    count = min(data.indices.size, max(10, int(np.ceil(0.15 * data.indices.size))))
    if not _trend_detected(data.theta[:count], data.residual[:count], -0.75):
        return None
    return PhysicsDiagnostic(
        "suspected_unmodeled_footprint",
        "low-angle residual trend suggests an unmodeled footprint",
        tuple(int(index) for index in data.indices[:count]),
    )


def _background(problem: object, data: _OrderedResiduals) -> PhysicsDiagnostic | None:
    if problem.instrument.background_kind != "constant":
        return None
    count = min(data.indices.size, max(20, int(np.ceil(0.20 * data.indices.size))))
    if not _trend_detected(data.qz[-count:], data.residual[-count:], -0.70):
        return None
    return PhysicsDiagnostic(
        "suspected_diffuse_background",
        "high-q residual trend suggests a nonconstant background",
        tuple(int(index) for index in data.indices[-count:]),
    )


def _surface(data: _OrderedResiduals) -> PhysicsDiagnostic | None:
    start = data.indices.size // 2
    indices = data.indices[start:]
    qz = data.qz[start:]
    residual = data.residual[start:]
    if indices.size < 20 or np.ptp(qz) <= 0.0:
        return None
    uniform_qz = np.linspace(qz[0], qz[-1], qz.size)
    detrended = signal.detrend(np.interp(uniform_qz, qz, residual), type="linear")
    spectrum = np.abs(np.fft.rfft(detrended))
    frequencies = np.fft.rfftfreq(
        uniform_qz.size,
        d=float(uniform_qz[1] - uniform_qz[0]),
    )
    thickness_a = 2.0 * np.pi * frequencies
    eligible = (thickness_a >= 2.0) & (thickness_a <= 50.0)
    eligible[0] = False
    nonzero = spectrum[1:]
    median = float(np.median(nonzero))
    mad = float(np.median(np.abs(nonzero - median)))
    threshold = median + 5.0 * 1.4826 * mad
    if not np.any(eligible) or float(np.max(spectrum[eligible])) <= threshold:
        return None
    return PhysicsDiagnostic(
        "surface_thin_layer_residual",
        "high-q residual spectrum suggests a 2-50 A surface layer",
        tuple(int(index) for index in indices),
    )


def diagnose_residual_patterns(
    problem: FitEvaluationContext,
    candidate: object,
) -> tuple[PhysicsDiagnostic, ...]:
    data = _ordered_data(problem, candidate)
    if data.indices.size < 10:
        return ()
    unique: dict[tuple[str, tuple[int, ...]], PhysicsDiagnostic] = {}
    for diagnostic in (_footprint(problem, data), _background(problem, data), _surface(data)):
        if diagnostic is not None:
            unique.setdefault((diagnostic.code, diagnostic.point_indices), diagnostic)
    return tuple(unique.values())
