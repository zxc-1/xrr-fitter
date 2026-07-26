from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    PeriodicSpan,
    SlabStack,
    StructureSpec,
)
from xrr_fitter.physics.stack import expand_structure


AIR = MaterialSpec("Air", None, None, 0j)
SI = MaterialSpec("Si", None, None, 20e-6 + 0.2e-6j)
MO = MaterialSpec("Mo", None, None, 55e-6 + 1.0e-6j)


def test_periodic_block_expands_in_declared_order() -> None:
    structure = StructureSpec(
        AIR,
        (PeriodicBlock("Mo/Si", (LayerSpec("Mo", MO, 28, roughness_a=3), LayerSpec("Si", SI, 42, roughness_a=4)), 3, top_roughness_a=2),),
        SI,
        backing_roughness_a=5,
    )
    stack = expand_structure(structure, 1.5406)
    np.testing.assert_array_equal(stack.thickness_a, [0, 28, 42, 28, 42, 28, 42, 0])
    np.testing.assert_array_equal(stack.roughness_a, [2, 4, 3, 4, 3, 4, 5])
    assert stack.periodic_spans == (PeriodicSpan(1, 2, 3),)


def test_gradient_layer_expands_to_midpoint_microslabs() -> None:
    structure = StructureSpec(
        AIR,
        (GradientLayerSpec("ramp", 10e-6 + 1e-6j, 30e-6 + 3e-6j, 5, roughness_a=1, microslab_max_a=2),),
        SI,
    )
    stack = expand_structure(structure, 1.5406)
    np.testing.assert_allclose(stack.thickness_a, [0, 5 / 3, 5 / 3, 5 / 3, 0])
    gradient = (30e-6 + 3e-6j) - (10e-6 + 1e-6j)
    np.testing.assert_allclose(stack.sld_a2[1:4], [(10e-6 + 1e-6j) + fraction * gradient for fraction in (1 / 6, 1 / 2, 5 / 6)])
    np.testing.assert_array_equal(stack.roughness_a, [1, 0, 0, 0])


def test_dynamic_interface_roughness_limit_is_rejected_during_expansion() -> None:
    with pytest.raises(ValueError, match="below 4.9"):
        expand_structure(StructureSpec(AIR, (LayerSpec("film", MO, 10, roughness_a=4.9),), SI), 1.5406)


def test_bare_substrate_uses_explicit_fifty_angstrom_roughness_limit() -> None:
    assert expand_structure(StructureSpec(AIR, (), SI, 50), 1.5406).roughness_a[0] == 50
    with pytest.raises(ValueError, match="below 50"):
        expand_structure(StructureSpec(AIR, (), SI, 50.01), 1.5406)


def test_material_sld_cache_is_scoped_to_one_expansion_and_wavelength(monkeypatch: pytest.MonkeyPatch) -> None:
    import xrr_fitter.physics.stack as stack_module
    calls = []
    original = stack_module.material_sld
    def counted(material, density_scale, wavelength_a):
        calls.append((material, density_scale, wavelength_a))
        return original(material, density_scale, wavelength_a)
    monkeypatch.setattr(stack_module, "material_sld", counted)
    structure = StructureSpec(AIR, (PeriodicBlock("repeat", (LayerSpec("Si", SI, 10),), 4),), SI)
    expand_structure(structure, 1.54056)
    expand_structure(structure, 1.54439)
    assert [wavelength for _material, _density, wavelength in calls] == [1.54056, 1.54056, 1.54439, 1.54439]


def test_single_periodic_cell_keeps_top_roughness_without_emitting_span() -> None:
    structure = StructureSpec(AIR, (PeriodicBlock("single", (LayerSpec("Mo", MO, 28, roughness_a=3),), 1, top_roughness_a=2),), SI)
    stack = expand_structure(structure, 1.5406)
    assert stack.periodic_spans == ()
    np.testing.assert_array_equal(stack.roughness_a, [2, 0])


def test_slab_stack_rejects_periodic_span_whose_cells_do_not_repeat() -> None:
    with pytest.raises(ValueError, match="does not repeat"):
        SlabStack([0, 10, 11, 0], [0j, 1e-6, 1e-6, 2e-6], [0, 0, 0], (PeriodicSpan(1, 1, 2),))


def test_slab_stack_rejects_overlapping_periodic_spans() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SlabStack([0, 10, 10, 10, 10, 0], [0j, 1e-6, 1e-6, 1e-6, 1e-6, 2e-6], [0] * 5, (PeriodicSpan(1, 1, 2), PeriodicSpan(2, 1, 2)))
