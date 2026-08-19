"""Normalized interface transition kernels.

The coordinate ``t`` is normalized depth across the transition region: ``0`` at
the incident (fronting) side, ``1`` at the layer material side. Each kernel
``f(t)`` returns the fraction of the layer material and satisfies ``f(0) == 0``,
``f(1) == 1`` and is monotone non-decreasing. This direction must match the
primal blend ``sld = (1 - f) * upper + f * lower`` used during stack expansion;
writing it backwards swaps the two adjacent materials.
"""

from __future__ import annotations

from collections.abc import Callable
from math import ceil, isfinite

import numpy as np
from scipy.special import erf

from xrr_fitter.model.structure import MAX_TRANSITION_SLABS, InterfaceTransition

ERF_HALF_WIDTH_SIGMAS = 2.0
TANH_HALF_WIDTH = 2.0
EXPONENTIAL_RATE = 4.0

ERF_DENOMINATOR = float(erf(ERF_HALF_WIDTH_SIGMAS / np.sqrt(2.0)))
TANH_DENOMINATOR = float(np.tanh(TANH_HALF_WIDTH))
EXPONENTIAL_DENOMINATOR = float(1.0 - np.exp(-EXPONENTIAL_RATE))


def _erf_profile(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(ERF_HALF_WIDTH_SIGMAS * (2.0 * t - 1.0) / np.sqrt(2.0)) / ERF_DENOMINATOR)


def _linear_profile(t: np.ndarray) -> np.ndarray:
    return t


def _tanh_profile(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(TANH_HALF_WIDTH * (2.0 * t - 1.0)) / TANH_DENOMINATOR)


def _sine_profile(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(np.pi * t))


def _exponential_profile(t: np.ndarray) -> np.ndarray:
    return (1.0 - np.exp(-EXPONENTIAL_RATE * t)) / EXPONENTIAL_DENOMINATOR


def _step_profile(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.0, 1.0)


PROFILES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "erf": _erf_profile,
    "linear": _linear_profile,
    "tanh": _tanh_profile,
    "sine": _sine_profile,
    "exponential": _exponential_profile,
    "step": _step_profile,
}

TRANSITION_KINDS = frozenset(PROFILES)


def transition_profile(kind: str, t: np.ndarray) -> np.ndarray:
    kernel = PROFILES.get(kind)
    if kernel is None:
        raise ValueError(f"unknown transition kind: {kind}")
    coordinates = np.asarray(t, dtype=float)
    if np.any(~np.isfinite(coordinates)) or np.any((coordinates < 0.0) | (coordinates > 1.0)):
        raise ValueError("transition coordinates must be finite and in [0,1]")
    return kernel(coordinates)


def transition_slab_count(width_a: float, microslab_max_a: float) -> int:
    if not isfinite(width_a) or width_a <= 0.0:
        raise ValueError(f"transition width must be finite and positive: {width_a}")
    if not isfinite(microslab_max_a) or microslab_max_a <= 0.0:
        raise ValueError(f"microslab_max_a must be finite and positive: {microslab_max_a}")
    ratio = width_a / microslab_max_a
    if not isfinite(ratio) or ceil(ratio) > MAX_TRANSITION_SLABS:
        raise ValueError(f"transition microslab count must not exceed {MAX_TRANSITION_SLABS}")
    return max(1, int(ceil(ratio)))


def transition_width(transition: InterfaceTransition) -> float:
    """Return the widest branch, which sets the region every path discretizes."""
    return max(branch.thickness_a for branch in transition.branches)


def transition_fractions(transition: InterfaceTransition, count: int) -> np.ndarray:
    """Blend the branch kernels at ``count`` slab centers of the widest branch.

    Narrow branches saturate at ``1`` before the region ends, which is how
    differently sized branches compose one profile. Sampling centers rather than
    edges keeps the first and last fraction strictly inside ``(0, 1)``.
    """
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or not 1 <= int(count) <= MAX_TRANSITION_SLABS
    ):
        raise ValueError(f"transition slab count must be a positive integer: {count}")
    count = int(count)
    width = transition_width(transition)
    depths = (np.arange(count, dtype=float) + 0.5) * (width / count)
    fractions = np.zeros(count, dtype=float)
    for branch in transition.branches:
        coordinate = np.ones_like(depths)
        active = depths < branch.thickness_a
        coordinate[active] = depths[active] / branch.thickness_a
        fractions += branch.weight * transition_profile(branch.kind, coordinate)
    return fractions
