"""Numerically safe fitted-background bounds shared by fit compilation."""

from __future__ import annotations

from math import isfinite

import numpy as np

from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.instrument import InstrumentSpec


def _base_sensitivity_upper(data: PreparedData) -> float:
    upper = np.finfo(float).max / 8.0
    if data.r_floor < 1.0:
        upper *= data.r_floor
    return upper


def _linear_sensitivity_upper(upper: float, fitted_q: np.ndarray) -> float:
    maximum_q = float(np.max(fitted_q))
    return upper / maximum_q / 2.0 if maximum_q > 0.5 else upper


def _powerlaw_sensitivity_upper(upper: float, fitted_q: np.ndarray) -> float:
    positive_q = fitted_q[fitted_q > 0.0]
    if positive_q.size == 0:
        raise ValueError("power-law background requires positive fitted q")
    minimum_q = float(np.min(positive_q))
    if minimum_q < 1.0:
        upper *= minimum_q
        upper *= minimum_q
        upper *= minimum_q
    return upper


def _background_sensitivity_upper(
    data: PreparedData,
    instrument: InstrumentSpec,
) -> float:
    upper = _base_sensitivity_upper(data)
    fitted_q = np.abs(data.qz_a_inv[data.fit_mask])
    if instrument.background_kind == "linear":
        upper = _linear_sensitivity_upper(upper, fitted_q)
    elif instrument.background_kind == "powerlaw":
        upper = _powerlaw_sensitivity_upper(upper, fitted_q)
    if not isfinite(upper) or upper < 0.1:
        raise ValueError("background parameter sensitivity is not representable")
    return upper


def _stable_tail_median(tail: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        median = float(np.median(tail))
    if isfinite(median):
        return median
    scale = float(np.max(np.abs(tail)))
    if scale == 0.0 or not isfinite(scale):
        raise ValueError("fitted high-angle intensities must be finite")
    return scale * float(np.median(tail / scale))


def background_upper(
    data: PreparedData,
    instrument: InstrumentSpec,
) -> float:
    """Bound background coordinates while retaining a representable tangent."""
    fitted = data.intensity_normalized[data.fit_mask]
    high_count = max(1, int(np.ceil(0.20 * fitted.size)))
    tail = fitted[-high_count:]
    median = max(0.0, _stable_tail_median(tail))
    maximum = _background_sensitivity_upper(data, instrument)
    scaled = maximum if median > maximum / 10.0 else 10.0 * median
    return max(0.1, scaled)


__all__ = ["background_upper"]
