"""Complex scattering-length-density depth profiles."""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from xrr_fitter.model.slab_stack import SlabStack


def _depth_grid(stack: SlabStack, step_a: float) -> tuple[np.ndarray, np.ndarray]:
    interfaces = np.r_[0.0, np.cumsum(stack.thickness_a[1:-1])]
    total = float(np.sum(stack.thickness_a[1:-1]))
    start = -max(10.0, 5.0 * stack.roughness_a[0])
    stop = total + max(10.0, 5.0 * stack.roughness_a[-1])
    count = int(np.ceil((stop - start) / step_a))
    depth = start + np.arange(count + 1, dtype=float) * step_a
    if depth[-1] < stop:
        depth = np.append(depth, stop)
    return depth, interfaces


def _transition(depth: np.ndarray, interface: float, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return np.where(depth < interface, 0.0, np.where(depth > interface, 1.0, 0.5))
    return 0.5 * (1.0 + erf((depth - interface) / (np.sqrt(2.0) * sigma)))


def _profile(depth: np.ndarray, interfaces: np.ndarray, stack: SlabStack) -> np.ndarray:
    transitions = np.asarray(
        [
            np.clip(_transition(depth, interface, stack.roughness_a[index]), 0.0, 1.0)
            for index, interface in enumerate(interfaces)
        ]
    )
    ordered = np.minimum.accumulate(transitions, axis=0)
    weights = np.empty((stack.sld_a2.size, depth.size), dtype=float)
    weights[0] = 1.0 - ordered[0]
    weights[1:-1] = ordered[:-1] - ordered[1:]
    weights[-1] = ordered[-1]
    return np.sum(weights * stack.sld_a2[:, None], axis=0, dtype=np.complex128)


def _sharp_profile(depth: np.ndarray, interfaces: np.ndarray, stack: SlabStack) -> np.ndarray:
    profile = np.empty(depth.shape, dtype=np.complex128)
    profile[depth < interfaces[0]] = stack.sld_a2[0]
    profile[depth > interfaces[-1]] = stack.sld_a2[-1]
    for index in range(1, stack.sld_a2.size - 1):
        profile[(depth > interfaces[index - 1]) & (depth < interfaces[index])] = stack.sld_a2[index]
    for index, interface in enumerate(interfaces):
        profile[depth == interface] = (stack.sld_a2[index] + stack.sld_a2[index + 1]) / 2.0
    return profile


def sld_depth_profile(stack: SlabStack, step_a: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(step_a) or step_a <= 0.0:
        raise ValueError("step_a must be finite and positive")
    depth, interfaces = _depth_grid(stack, step_a)
    profile = (
        _sharp_profile(depth, interfaces, stack)
        if np.all(stack.roughness_a == 0.0)
        else _profile(depth, interfaces, stack)
    )
    depth = np.array(depth, copy=True)
    profile = np.array(profile, copy=True)
    depth.setflags(write=False)
    profile.setflags(write=False)
    return depth, profile
