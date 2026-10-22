"""Deterministic reflected starts for bounded local search and recovery.

Proposals use the caller's exact Gaussian draw and unchanged unit-space sigma.
Only excursions outside the box are reflected; interior values keep their bytes.
Reflection avoids clipped point masses at stationary zero-width boundaries.
"""

from __future__ import annotations

from math import isfinite

import numpy as np


def _readonly(value: np.ndarray, dtype: type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_perturbation_center(center: np.ndarray) -> None:
    if center.ndim != 1 or not np.all(np.isfinite(center)):
        raise ValueError("perturbation center must be a finite unit vector")
    if np.any((center < 0.0) | (center > 1.0)):
        raise ValueError("perturbation center is outside unit bounds")


def _validate_perturbation_count(count: object) -> None:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("perturbation count must be a nonnegative integer")


def _validate_perturbation_sigma(sigma: float) -> None:
    if not isfinite(sigma) or sigma < 0.0:
        raise ValueError("perturbation sigma must be finite and nonnegative")


def bounded_perturbations(
    center: np.ndarray,
    count: int,
    *,
    seed: int,
    sigma: float = 0.1,
) -> tuple[np.ndarray, ...]:
    """Reflect bounded perturbations without mutating the carried center.

    An empty coordinate layout has one logical center but no useful perturbed
    vectors, so it returns an empty tuple before touching the RNG stream.
    """
    unit_center = np.asarray(center, dtype=float)
    _validate_perturbation_center(unit_center)
    _validate_perturbation_count(count)
    _validate_perturbation_sigma(sigma)
    if unit_center.size == 0:
        return ()
    with np.errstate(over="ignore", invalid="ignore"):
        perturbed = unit_center + np.random.default_rng(seed).normal(
            0.0,
            sigma,
            size=(int(count), unit_center.size),
        )
    if not np.all(np.isfinite(perturbed)):
        raise ValueError("perturbation proposals must be finite")
    # Clipping creates point masses at zero-width boundaries whose analytic
    # derivative vanishes. Reflect the same draw, preserving interior bytes.
    outside = (perturbed < 0.0) | (perturbed > 1.0)
    folded = np.remainder(np.abs(perturbed[outside]), 2.0)
    perturbed[outside] = np.where(folded <= 1.0, folded, 2.0 - folded)
    return tuple(_readonly(vector, float) for vector in perturbed)
