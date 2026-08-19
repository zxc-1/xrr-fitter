"""Hard-edge beam footprint correction."""

from __future__ import annotations

from math import isfinite

import numpy as np


def _validated_theta(theta_deg: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float)
    if np.any(~np.isfinite(theta)) or np.any((theta < 0.0) | (theta > 90.0)):
        raise ValueError("theta_deg must be finite and in [0, 90]")
    return theta


def _validated_spill_angle(spill_angle_deg: float) -> float:
    if not isfinite(spill_angle_deg) or not 0.0 <= spill_angle_deg <= 90.0:
        raise ValueError("spill_angle_deg must be finite and in [0, 90]")
    return spill_angle_deg


def footprint_factor(theta_deg: np.ndarray, spill_angle_deg: float) -> np.ndarray:
    theta = _validated_theta(theta_deg)
    spill_angle_deg = _validated_spill_angle(spill_angle_deg)
    if spill_angle_deg == 0.0:
        return np.ones_like(theta)
    numerator = np.sin(np.deg2rad(theta))
    denominator = np.sin(np.deg2rad(spill_angle_deg))
    # Decide the saturated branch before dividing.  A tiny but finite spill
    # angle can make the quotient overflow even though the capped result is
    # exactly one.
    result = np.ones_like(theta)
    active = numerator < denominator
    if np.any(active):
        result[active] = numerator[active] / denominator
    return result
