from __future__ import annotations

import warnings
from importlib import import_module

import numpy as np
import pytest

from xrr_fitter.model.slab_stack import PeriodicSpan, SlabStack
from xrr_fitter.physics.derivatives import parratt_reflectivity_jacobian, smear_with_widths_jacobian


def _tangents(stack: SlabStack, q: np.ndarray, count: int = 3):
    q_jac = np.zeros((q.size, count))
    q_jac[:, 0] = 1e-3
    thickness_jac = np.zeros((stack.thickness_a.size, count))
    thickness_jac[1:-1:2, 1] = 0.7
    sld_jac = np.zeros((stack.sld_a2.size, count), complex)
    sld_jac[2:-1:2, 2] = 1e-7 + 2e-8j
    roughness_jac = np.zeros((stack.roughness_a.size, count))
    roughness_jac[:, 0] = 0.03
    return q_jac, thickness_jac, sld_jac, roughness_jac


__all__ = [
    "PeriodicSpan",
    "SlabStack",
    "_tangents",
    "import_module",
    "np",
    "parratt_reflectivity_jacobian",
    "pytest",
    "smear_with_widths_jacobian",
    "warnings",
]
