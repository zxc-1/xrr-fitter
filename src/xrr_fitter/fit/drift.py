"""Compile-time drift desugaring: per-copy coefficients and constraint rules."""

from __future__ import annotations

import math

import numpy as np

from xrr_fitter.model.structure import DriftSpec


def drift_coefficients(drift: DriftSpec, repeats: int) -> tuple[float, ...]:
    """Per-copy modulation constants c_k (c_0=0; copy 0 is the free base cell)."""
    coeffs: list[float] = [0.0]
    if drift.kind == "linear":
        coeffs.extend(float(k) for k in range(1, repeats))
    elif drift.kind == "sine":
        coeffs.extend(math.sin(2.0 * math.pi * k / drift.period + drift.phase) for k in range(1, repeats))
    else:  # random — deterministic, self-contained in drift.seed
        rng = np.random.default_rng(drift.seed)
        coeffs.extend(float(v) for v in rng.uniform(-1.0, 1.0, size=max(0, repeats - 1)))
    return tuple(coeffs)
