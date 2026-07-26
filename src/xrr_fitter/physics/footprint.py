"""Hard-edge beam footprint correction."""

from __future__ import annotations

from math import isfinite

import numpy as np


def footprint_factor(theta_deg: np.ndarray, spill_angle_deg: float) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float)
    if np.any(~np.isfinite(theta)):
        raise ValueError("theta_deg must be finite")
    if not isfinite(spill_angle_deg) or spill_angle_deg < 0.0:
        raise ValueError("spill_angle_deg must be finite and nonnegative")
    if spill_angle_deg == 0.0:
        return np.ones_like(theta)
    return np.minimum(1.0, np.sin(np.deg2rad(theta)) / np.sin(np.deg2rad(spill_angle_deg)))
