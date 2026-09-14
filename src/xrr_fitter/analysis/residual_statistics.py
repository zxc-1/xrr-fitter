"""Predeclared continuous detectors and a row-symmetric max calibration.

The old Boolean screen still owns when calibration is requested. Every
instrument/geometry-applicable column enters a requested family, even when
that column did not trigger the screen. Scores retain their amplitude and sign;
unrepresentable input never becomes an innocuous zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt

import numpy as np
from scipy.stats import spearmanr

from xrr_fitter.analysis.diagnostics import (
    _normalized_centered_residuals,
    _ordered_data,
    _surface_spectrum,
)
from xrr_fitter.model.diagnostic_calibration import DiagnosticStatistic
from xrr_fitter.model.fitting import FitEvaluationContext


def _trend_score(coordinate: np.ndarray, values: np.ndarray, correlation_limit: float) -> float:
    constant = np.all(coordinate == coordinate[0]) or np.all(values == values[0])
    correlation = 0.0 if constant else float(spearmanr(coordinate, values).statistic)
    third = max(1, values.size // 3)
    drop = float(np.median(values[:third]) - np.median(values[-third:]))
    return max(0.0, -correlation) * drop / (correlation_limit * 0.05)


def _acf_score(values: np.ndarray) -> float:
    normalized = _normalized_centered_residuals(values)
    if normalized is None:
        return 0.0
    centered, denominator = normalized
    lags = sorted(
        abs(float(centered[:-lag] @ centered[lag:]) / denominator) for lag in range(1, min(20, values.size // 5) + 1)
    )
    return float(np.sqrt(values.size) / 3.0 * lags[-2])


def _scores(problem: FitEvaluationContext, data) -> tuple[tuple[str, float], ...]:
    values = []
    size = data.indices.size
    instrument = problem.instrument
    if instrument.footprint_mode == "none" and instrument.footprint_spill_angle_deg == 0.0:
        count = min(size, max(10, int(np.ceil(0.15 * size))))
        values.append(("footprint", _trend_score(data.theta[:count], data.residual[:count], 0.75)))
    if instrument.background_kind == "constant":
        count = min(size, max(20, int(np.ceil(0.20 * size))))
        values.append(("background", _trend_score(data.qz[-count:], data.residual[-count:], 0.70)))
    components = _surface_spectrum(data)
    if components is not None and np.any(components[2]):
        indices, spectrum, eligible = components
        median = float(np.median(spectrum[1:]))
        mad = float(np.median(np.abs(spectrum[1:] - median)))
        contrast = float(np.max(spectrum[eligible])) - median - 5.0 * 1.4826 * mad
        values.append(("surface", contrast / float(np.sqrt(indices.size))))
    values.append(("acf", _acf_score(data.residual)))
    return tuple(values)


def residual_statistics(
    problem: FitEvaluationContext,
    residuals: np.ndarray,
    dataset_id: str | None = None,
) -> tuple[DiagnosticStatistic, ...]:
    """Use the original stable q ordering, masks, windows, and FFT geometry."""
    data = _ordered_data(problem, residuals)
    if data.residual.size < 10 or data.residual.size != np.count_nonzero(problem.data.fit_mask):
        raise FloatingPointError("diagnostic statistics require complete finite fitted residuals")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scores = _scores(problem, data)
    if any(not np.isfinite(value) for _kind, value in scores):
        raise FloatingPointError("diagnostic statistics must be finite")
    return tuple(DiagnosticStatistic(dataset_id, kind, float(value)) for kind, value in scores)


@dataclass(frozen=True, slots=True)
class _SymmetricMax:
    centers: np.ndarray
    scales: np.ndarray
    scores: np.ndarray
    adjusted_p_values: tuple[float, ...]
    tail_count: int
    tie_count: int

    @property
    def p_value(self) -> float:
        return self.tail_count / self.scores.size

    @property
    def observed_score(self) -> float:
        return float(self.scores[0])


def _column_median(column: np.ndarray) -> float:
    ordered = np.sort(column)
    middle = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[middle])
    left, right = float(ordered[middle - 1]), float(ordered[middle])
    if left == right:
        return left
    half_max = np.finfo(float).max / 2.0
    safe_sum = abs(left) <= half_max and abs(right) <= half_max
    return (left + right) / 2.0 if left < 0.0 < right or safe_sum else left / 2.0 + right / 2.0


def _rms_normalization(difference: np.ndarray) -> tuple[float, np.ndarray]:
    peak = float(np.max(np.abs(difference)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise FloatingPointError("diagnostic RMS centered scale is not representable")
    relative = difference / peak
    if np.any((difference != 0.0) & (relative == 0.0)):
        raise FloatingPointError("diagnostic RMS normalization lost a nonzero value")
    # Fixed one-dimensional reduction, independent of row order and input layout.
    relative_rms = sqrt(fsum(np.sort(relative * relative)) / difference.size)
    if not np.isfinite(relative_rms) or not 0.0 < relative_rms <= 1.0:
        raise FloatingPointError("diagnostic relative RMS scale is not representable")
    scale = peak * relative_rms
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError("diagnostic RMS scale is not representable")
    # A rounded raw-unit scale need not reproduce these scores (e.g. subnormals).
    standardized = relative / relative_rms
    if np.any(~np.isfinite(standardized)):
        raise FloatingPointError("diagnostic RMS scores must be finite")
    standardized[standardized == 0.0] = 0.0
    return scale, standardized


def _rms_column(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    column = np.array(values, dtype=float, order="C", copy=True)
    if np.all(column == column[0]):
        center = float(column[0]) if column[0] != 0.0 else 0.0
        return center, 0.0, np.zeros(column.size)
    center = _column_median(column)
    center = center if center != 0.0 else 0.0
    scale, standardized = _rms_normalization(column - center)
    return center, scale, standardized


def _symmetric_rms(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, scales = np.empty(matrix.shape[1]), np.empty(matrix.shape[1])
    standardized = np.empty(matrix.shape)
    for index in range(matrix.shape[1]):
        centers[index], scales[index], standardized[:, index] = _rms_column(matrix[:, index])
    return centers, scales, standardized


def symmetric_max(values: np.ndarray) -> _SymmetricMax:
    """Calibrate an observed-plus-null matrix with inclusive, unjittered ranks.

    The observed row participates in every median/RMS. This equivariant
    normalization retains the rank property for exchangeable rows. Plug-in
    fitted means are only approximately exchangeable; this is not a claim of
    strong FWER under arbitrary partial nulls or a binomial MC error interval.
    Exact replay uses the full matrix and factorized RMS scores, not division
    by the returned rounded raw-unit scale. Constant columns are neutral.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("diagnostic matrix must contain observed/null rows and detector columns")
    if np.any(~np.isfinite(matrix)):
        raise FloatingPointError("diagnostic matrix must be finite")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        centers, scales, standardized = _symmetric_rms(matrix)
        scores = np.maximum(0.0, np.max(standardized, axis=1))
    adjusted = tuple(float(np.count_nonzero(scores >= max(0.0, value)) / scores.size) for value in standardized[0])
    for array in (centers, scales, scores):
        array.setflags(write=False)
    return _SymmetricMax(
        centers,
        scales,
        scores,
        adjusted,
        int(np.count_nonzero(scores >= scores[0])),
        int(np.count_nonzero(scores == scores[0])),
    )
