"""Thickness and periodicity candidates from fitted reflectivity curves."""

from __future__ import annotations

import numpy as np
from scipy import signal

MIN_THICKNESS_A = 2.0
MAX_THICKNESS_A = 2e5


def scaled_qz4_transform(qz_a_inv: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return a finite-scale equivalent of ``qz**4 * values``."""
    qz = np.asarray(qz_a_inv, dtype=float)
    amplitudes = np.asarray(values, dtype=float)
    q_scale = float(np.max(np.abs(qz)))
    amplitude_scale = float(np.max(np.abs(amplitudes)))
    if q_scale == 0.0 or amplitude_scale == 0.0:
        return np.zeros_like(qz)
    scaled_qz = qz / q_scale
    return np.square(np.square(scaled_qz)) * (amplitudes / amplitude_scale)


def scaled_qz4_with_floor(
    qz_a_inv: np.ndarray,
    values: np.ndarray,
    r_floor: float,
) -> np.ndarray:
    """Scale the ``qz**4 * max(values + floor, floor)`` feature safely."""
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    amplitude_scale = max(float(r_floor), float(np.max(positive)))
    if amplitude_scale == 0.0:
        return np.zeros_like(np.asarray(qz_a_inv, dtype=float))
    amplitudes = positive / amplitude_scale + r_floor / amplitude_scale
    return scaled_qz4_transform(qz_a_inv, amplitudes)


def _validate_max_candidates(max_candidates: int) -> None:
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")


def uniform_feature_view(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a validated curve onto one uniform, detrended q grid."""
    qz = np.asarray(qz_a_inv, dtype=float)
    values = np.asarray(transformed, dtype=float)
    if qz.ndim != 1 or qz.size < 16 or values.shape != qz.shape:
        raise ValueError("feature arrays must be equal vectors with at least 16 points")
    if np.any(~np.isfinite(qz)) or np.any(~np.isfinite(values)):
        raise ValueError("feature arrays must be finite and qz strictly increasing")
    if np.any(np.diff(qz) <= 0.0):
        raise ValueError("feature qz must be strictly increasing")
    uniform_qz = np.linspace(qz[0], qz[-1], qz.size)
    uniform_values = np.interp(uniform_qz, qz, values)
    detrended = signal.detrend(uniform_values)
    return uniform_qz, detrended, float(uniform_qz[1] - uniform_qz[0])


def span_cannot_resolve_bounded_thickness(qz_a_inv: np.ndarray) -> bool:
    return bool(qz_a_inv[-1] - qz_a_inv[0] < 2.0 * np.pi / MAX_THICKNESS_A)


def spectral_thickness_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
    max_candidates: int = 8,
) -> np.ndarray:
    """Rank physical thicknesses by peaks in the Hann-windowed q spectrum."""
    _validate_max_candidates(max_candidates)
    uniform_qz, detrended, delta_q = uniform_feature_view(qz_a_inv, transformed)
    if span_cannot_resolve_bounded_thickness(uniform_qz):
        return np.empty(0, dtype=float)
    windowed = detrended * signal.windows.hann(detrended.size, sym=False)
    spectrum = np.abs(np.fft.rfft(windowed))
    thickness = 2.0 * np.pi * np.fft.rfftfreq(uniform_qz.size, delta_q)
    valid_indices = np.flatnonzero((thickness >= MIN_THICKNESS_A) & (thickness <= MAX_THICKNESS_A))
    peaks, _ = signal.find_peaks(spectrum[valid_indices])
    selected = valid_indices[peaks]
    ranked = selected[np.argsort(spectrum[selected])[::-1]][:max_candidates]
    return np.sort(thickness[ranked])


def autocorrelation_thickness_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
    max_candidates: int = 8,
) -> np.ndarray:
    """Infer thicknesses from unbiased positive-lag autocorrelation peaks."""
    _validate_max_candidates(max_candidates)
    uniform_qz, detrended, delta_q = uniform_feature_view(qz_a_inv, transformed)
    if span_cannot_resolve_bounded_thickness(uniform_qz):
        return np.empty(0, dtype=float)
    correlation = signal.fftconvolve(detrended, detrended[::-1], mode="full")
    correlation = correlation[uniform_qz.size - 1 :]
    correlation /= np.arange(uniform_qz.size, 0, -1)
    peaks, properties = signal.find_peaks(
        correlation[1:],
        prominence=max(float(np.ptp(correlation)) * 0.01, np.finfo(float).eps),
    )
    lags = peaks + 1
    thickness = 2.0 * np.pi / (lags * delta_q)
    valid = np.isfinite(thickness) & ((thickness >= MIN_THICKNESS_A) & (thickness <= MAX_THICKNESS_A))
    ranked = np.argsort(properties["prominences"][valid])[::-1][:max_candidates]
    return np.sort(thickness[valid][ranked])


def kiessig_spacing_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
    max_candidates: int = 8,
) -> np.ndarray:
    """Convert adjacent smoothed-fringe peak spacings into film thicknesses."""
    _validate_max_candidates(max_candidates)
    uniform_qz, detrended, _ = uniform_feature_view(qz_a_inv, transformed)
    if span_cannot_resolve_bounded_thickness(uniform_qz):
        return np.empty(0, dtype=float)
    window = min(51, uniform_qz.size // 2 * 2 - 1)
    smooth = signal.savgol_filter(detrended, window, 3)
    peaks, properties = signal.find_peaks(
        smooth,
        prominence=max(float(np.ptp(smooth)) * 0.02, np.finfo(float).eps),
    )
    if peaks.size < 2:
        return np.empty(0, dtype=float)
    thickness = 2.0 * np.pi / np.diff(uniform_qz[peaks])
    valid = np.isfinite(thickness) & ((thickness >= MIN_THICKNESS_A) & (thickness <= MAX_THICKNESS_A))
    strength = np.minimum(
        properties["prominences"][:-1],
        properties["prominences"][1:],
    )
    ranked = np.argsort(strength[valid])[::-1][:max_candidates]
    return np.sort(thickness[valid][ranked])


def bragg_period_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
) -> tuple[float, ...]:
    uniform_qz, detrended, _ = uniform_feature_view(qz_a_inv, transformed)
    if span_cannot_resolve_bounded_thickness(uniform_qz):
        return ()
    window = min(51, uniform_qz.size // 2 * 2 - 1)
    smooth = signal.savgol_filter(detrended, window, 3)
    peaks, properties = signal.find_peaks(
        smooth,
        prominence=max(float(np.ptp(smooth)) * 0.03, np.finfo(float).eps),
    )
    if peaks.size < 2:
        return ()
    periods = 2.0 * np.pi / np.diff(uniform_qz[peaks])
    strengths = np.minimum(
        properties["prominences"][:-1],
        properties["prominences"][1:],
    )
    valid = np.isfinite(periods) & ((periods >= MIN_THICKNESS_A) & (periods <= MAX_THICKNESS_A))
    ranked = np.argsort(strengths[valid])[::-1][:8]
    return tuple(float(value) for value in np.sort(periods[valid][ranked]))


def merge_candidates(
    groups: tuple[np.ndarray | tuple[float, ...], ...],
    relative_tolerance: float = 0.02,
) -> tuple[float, ...]:
    values = sorted(float(item) for group in groups for item in group if np.isfinite(item) and item > 0.0)
    clusters: list[list[float]] = []
    for value in values:
        for cluster in clusters:
            median = float(np.median(cluster))
            if abs(value - median) <= relative_tolerance * median:
                cluster.append(value)
                break
        else:
            clusters.append([value])
    return tuple(float(np.median(cluster)) for cluster in clusters)


def has_reliable_thickness_feature(
    spectral: np.ndarray,
    autocorrelation: np.ndarray,
    spacing: np.ndarray,
    spectral_resolution_a: float,
) -> bool:
    if spacing.size == 0:
        return False
    sorted_spacing = np.sort(spacing)
    if spacing.size > 1 and np.any(np.diff(sorted_spacing) <= 0.02 * sorted_spacing[:-1]):
        return True
    for candidate in spacing:
        fft_tolerance = max(0.02 * candidate, 0.5 * spectral_resolution_a)
        if spectral.size and np.any(np.abs(spectral - candidate) <= fft_tolerance):
            return True
        if autocorrelation.size and np.any(np.abs(autocorrelation - candidate) <= 0.02 * candidate):
            return True
    return False


def bounded_inverse_length(numerator: float, denominator: float) -> float:
    """Clip a positive ratio to thickness limits before risky division."""
    if not np.isfinite(denominator):
        return MIN_THICKNESS_A
    if denominator <= numerator / MAX_THICKNESS_A:
        return MAX_THICKNESS_A
    if denominator >= numerator / MIN_THICKNESS_A:
        return MIN_THICKNESS_A
    return numerator / denominator


def observable_thickness_bounds(
    qz_a_inv: np.ndarray,
) -> tuple[float, float, float, float]:
    observed_min = bounded_inverse_length(2.0 * np.pi, float(np.ptp(qz_a_inv)))
    observed_max = bounded_inverse_length(
        np.pi / 2.0,
        float(np.median(np.diff(qz_a_inv))),
    )
    return (
        observed_min,
        observed_max,
        max(MIN_THICKNESS_A, 0.25 * observed_min),
        min(MAX_THICKNESS_A, 4.0 * observed_max),
    )


__all__ = [
    "autocorrelation_thickness_candidates",
    "bounded_inverse_length",
    "bragg_period_candidates",
    "has_reliable_thickness_feature",
    "kiessig_spacing_candidates",
    "merge_candidates",
    "observable_thickness_bounds",
    "scaled_qz4_transform",
    "scaled_qz4_with_floor",
    "span_cannot_resolve_bounded_thickness",
    "spectral_thickness_candidates",
    "uniform_feature_view",
]
