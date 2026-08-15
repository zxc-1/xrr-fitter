from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.drift_cases import (
    AIR,
    SILICON,
    drift_structure,
    drift_values,
    plain_periodic_structure,
    two_layer_block,
    wavelength,
)

from xrr_fitter.model.structure import DriftSpec, PeriodicSpan, StructureSpec
from xrr_fitter.physics.stack import expand_structure


def test_thickness_drift_expands_per_copy_and_suppresses_span() -> None:
    # linear thickness drift, repeats=3, amount=0.1 -> coeffs (0, 1, 2):
    # copy k thickness = base * (1 + 0.1 * k); copy 0 is the free base cell.
    structure = drift_structure()
    stack = expand_structure(structure, wavelength(), drift_values(structure))
    np.testing.assert_array_equal(
        stack.thickness_a,
        [0.0, 20.0, 500.0, 22.0, 550.0, 24.0, 600.0, 0.0],
    )
    # Thickness drift leaves per-layer roughness untouched.
    np.testing.assert_array_equal(stack.roughness_a, [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0])
    # A drifted block breaks bit-identical repetition: no matrix-power fast path.
    assert stack.periodic_spans == ()


def test_roughness_drift_expands_per_copy_roughness() -> None:
    block = replace(two_layer_block(3), drift=DriftSpec(kind="linear", target="roughness", amount=0.1))
    structure = StructureSpec(fronting=AIR, components=(block,), backing=SILICON, backing_roughness_a=3.0)
    values = {
        "component.0.repeat.1.layer.0.roughness_a": 2.2,
        "component.0.repeat.2.layer.0.roughness_a": 2.4,
        "component.0.repeat.1.layer.1.roughness_a": 3.3,
        "component.0.repeat.2.layer.1.roughness_a": 3.6,
    }
    stack = expand_structure(structure, wavelength(), values)
    np.testing.assert_allclose(stack.roughness_a, [2.0, 3.0, 2.2, 3.3, 2.4, 3.6, 3.0])
    # Roughness drift leaves per-copy thickness at the base cell values.
    np.testing.assert_array_equal(stack.thickness_a, [0.0, 20.0, 500.0, 20.0, 500.0, 20.0, 500.0, 0.0])
    assert stack.periodic_spans == ()


def test_plain_periodic_is_byte_identical_without_values() -> None:
    plain = plain_periodic_structure()
    stack = expand_structure(plain, wavelength())
    assert stack.periodic_spans == (PeriodicSpan(1, 2, 3),)
    np.testing.assert_array_equal(stack.thickness_a, [0.0, 20.0, 500.0, 20.0, 500.0, 20.0, 500.0, 0.0])
    # An empty value map must not perturb the non-drift fast path.
    twin = expand_structure(plain, wavelength(), {})
    assert twin.periodic_spans == stack.periodic_spans
    np.testing.assert_array_equal(twin.thickness_a, stack.thickness_a)


def test_drift_expansion_without_values_fails_loud() -> None:
    with pytest.raises(ValueError, match="per-copy"):
        expand_structure(drift_structure(), wavelength())
