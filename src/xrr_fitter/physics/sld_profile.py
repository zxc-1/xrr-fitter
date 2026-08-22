"""Complex scattering-length-density depth profiles."""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from xrr_fitter.model.slab_stack import SlabStack

MAX_SLD_PROFILE_POINTS = 1_000_000


def _depth_extents(stack: SlabStack) -> tuple[np.ndarray, float, float]:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        interfaces = np.r_[0.0, np.cumsum(stack.thickness_a[1:-1])]
        total = float(np.sum(stack.thickness_a[1:-1]))
        tail = 5.0 * stack.roughness_a
        left_extent = interfaces - tail
        right_extent = interfaces + tail
        start_value = float(np.min(left_extent))
        stop_value = float(np.max(right_extent))
        start = min(-10.0, start_value)
        stop = max(total + 10.0, stop_value)
    if not all(np.isfinite(value) for value in (total, start_value, stop_value, start, stop)):
        raise ValueError("SLD profile depth span must be finite")
    return interfaces, start, stop


def sld_profile_step_for_point_limit(
    stack: SlabStack,
    preferred_step_a: float = 0.5,
    point_limit: int = MAX_SLD_PROFILE_POINTS,
) -> float:
    """Keep a preferred profile resolution within a fixed point budget."""
    if not np.isfinite(preferred_step_a) or preferred_step_a <= 0.0:
        raise ValueError("preferred_step_a must be finite and positive")
    if (
        isinstance(point_limit, bool)
        or not isinstance(point_limit, int)
        or not 3 <= point_limit <= MAX_SLD_PROFILE_POINTS
    ):
        raise ValueError(f"point_limit must be an integer from 3 to {MAX_SLD_PROFILE_POINTS}")
    _interfaces, start, stop = _depth_extents(stack)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        required_step = (stop - start) / (point_limit - 2)
    if not np.isfinite(required_step):
        raise ValueError("SLD profile depth span must be finite")
    return max(float(preferred_step_a), float(np.nextafter(required_step, np.inf)))


def _depth_grid(stack: SlabStack, step_a: float) -> tuple[np.ndarray, np.ndarray]:
    interfaces, start, stop = _depth_extents(stack)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        estimated = (stop - start) / step_a
    if not np.isfinite(estimated):
        raise ValueError("SLD profile depth span must be finite")
    if estimated > MAX_SLD_PROFILE_POINTS - 1:
        raise ValueError(f"SLD profile grid exceeds {MAX_SLD_PROFILE_POINTS} points")
    count = int(np.ceil(estimated))
    depth = start + np.arange(count + 1, dtype=float) * step_a
    if depth[-1] < stop:
        depth = np.append(depth, stop)
    return depth, interfaces


def _transition(depth: np.ndarray, interface: float, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return np.where(depth < interface, 0.0, np.where(depth > interface, 1.0, 0.5))
    return 0.5 * (1.0 + erf((depth - interface) / (np.sqrt(2.0) * sigma)))


def _profile(depth: np.ndarray, interfaces: np.ndarray, stack: SlabStack) -> np.ndarray:
    first = np.clip(_transition(depth, interfaces[0], stack.roughness_a[0]), 0.0, 1.0)
    profile = stack.sld_a2[0] * (1.0 - first)
    previous = first
    for index in range(1, interfaces.size):
        current = np.minimum(
            previous,
            np.clip(_transition(depth, interfaces[index], stack.roughness_a[index]), 0.0, 1.0),
        )
        profile = profile + stack.sld_a2[index] * (previous - current)
        previous = current
    return profile + stack.sld_a2[-1] * previous


def _sharp_profile(depth: np.ndarray, interfaces: np.ndarray, stack: SlabStack) -> np.ndarray:
    # Expanded stacks can contain a zero-thickness body (for example when a
    # transition width equals the declared layer thickness).  Such media have
    # duplicate interface coordinates and no depth interval, so collapse them
    # before assigning the piecewise profile.  Otherwise the last duplicate
    # assignment would make the plotted value depend on declaration order.
    finite = stack.thickness_a[1:-1] > 0.0
    active = np.concatenate(([0], np.flatnonzero(finite).astype(int) + 1, [stack.sld_a2.size - 1]))
    active_sld = stack.sld_a2[active]
    # ``interfaces[j]`` is the boundary between media ``j`` and ``j + 1``;
    # the boundary before each retained lower medium therefore uses index
    # ``active[j + 1] - 1``.
    active_interfaces = interfaces[active[1:] - 1]

    positions = np.searchsorted(active_interfaces, depth, side="right")
    profile = active_sld[np.minimum(positions, active_sld.size - 1)].astype(np.complex128, copy=True)
    boundary_indices = np.clip(positions - 1, 0, active_interfaces.size - 1)
    exact = (positions > 0) & (depth == active_interfaces[boundary_indices])
    if np.any(exact):
        indices = boundary_indices[exact]
        profile[exact] = 0.5 * active_sld[indices] + 0.5 * active_sld[indices + 1]
    return profile


def sld_depth_profile(stack: SlabStack, step_a: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(step_a) or step_a <= 0.0:
        raise ValueError("step_a must be finite and positive")
    depth, interfaces = _depth_grid(stack, step_a)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            profile = (
                _sharp_profile(depth, interfaces, stack)
                if np.all(stack.roughness_a == 0.0)
                else _profile(depth, interfaces, stack)
            )
    except FloatingPointError as error:
        raise FloatingPointError("nonfinite SLD profile arithmetic") from error
    if np.any(~np.isfinite(profile.real)) or np.any(~np.isfinite(profile.imag)):
        raise FloatingPointError("nonfinite SLD profile")
    depth = np.array(depth, copy=True)
    profile = np.array(profile, copy=True)
    depth.setflags(write=False)
    profile.setflags(write=False)
    return depth, profile
