from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.parameters import PhysicalValueError
from xrr_fitter.model.slab_stack import PeriodicSpan, SlabStack
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    InterfaceTransition,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
    TransitionBranch,
)
from xrr_fitter.physics.stack import expand_structure

AIR = MaterialSpec("Air", None, None, 0j)
SI = MaterialSpec("Si", None, None, 20e-6 + 0.2e-6j)
MO = MaterialSpec("Mo", None, None, 55e-6 + 1.0e-6j)


def test_periodic_block_expands_in_declared_order() -> None:
    structure = StructureSpec(
        AIR,
        (
            PeriodicBlock(
                "Mo/Si",
                (LayerSpec("Mo", MO, 28, roughness_a=3), LayerSpec("Si", SI, 42, roughness_a=4)),
                3,
                top_roughness_a=2,
            ),
        ),
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
    np.testing.assert_allclose(
        stack.sld_a2[1:4], [(10e-6 + 1e-6j) + fraction * gradient for fraction in (1 / 6, 1 / 2, 5 / 6)]
    )
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
    structure = StructureSpec(
        AIR, (PeriodicBlock("single", (LayerSpec("Mo", MO, 28, roughness_a=3),), 1, top_roughness_a=2),), SI
    )
    stack = expand_structure(structure, 1.5406)
    assert stack.periodic_spans == ()
    np.testing.assert_array_equal(stack.roughness_a, [2, 0])


def test_slab_stack_rejects_periodic_span_whose_cells_do_not_repeat() -> None:
    with pytest.raises(ValueError, match="does not repeat"):
        SlabStack([0, 10, 11, 0], [0j, 1e-6, 1e-6, 2e-6], [0, 0, 0], (PeriodicSpan(1, 1, 2),))


def test_slab_stack_rejects_overlapping_periodic_spans() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SlabStack(
            [0, 10, 10, 10, 10, 0],
            [0j, 1e-6, 1e-6, 1e-6, 1e-6, 2e-6],
            [0] * 5,
            (PeriodicSpan(1, 1, 2), PeriodicSpan(2, 1, 2)),
        )


def transition_layer(
    thickness_a: float = 20.0,
    *,
    kind: str = "erf",
    width_a: float = 10.0,
    microslab_max_a: float = 2.5,
) -> LayerSpec:
    return LayerSpec(
        "film",
        MO,
        thickness_a,
        roughness_a=0.0,
        transition=InterfaceTransition(
            branches=(TransitionBranch(kind=kind, weight=1.0, thickness_a=width_a),),
            microslab_max_a=microslab_max_a,
        ),
    )


def test_transition_layer_expands_to_slab_count_plus_body() -> None:
    plain = expand_structure(StructureSpec(AIR, (LayerSpec("film", MO, 20.0),), SI), 1.5406)
    stack = expand_structure(StructureSpec(AIR, (transition_layer(),), SI), 1.5406)
    assert stack.thickness_a.size == plain.thickness_a.size + 4
    np.testing.assert_allclose(stack.thickness_a[1:5], [2.5, 2.5, 2.5, 2.5])
    assert stack.thickness_a[5] == pytest.approx(10.0)


def test_transition_body_slab_is_present_even_when_width_equals_thickness() -> None:
    stack = expand_structure(
        StructureSpec(AIR, (transition_layer(10.0, width_a=10.0),), SI),
        1.5406,
    )
    assert stack.thickness_a.size == 7
    assert stack.thickness_a[5] == 0.0


def test_transition_slabs_interpolate_between_neighbor_and_own_sld() -> None:
    stack = expand_structure(StructureSpec(AIR, (transition_layer(),), SI), 1.5406)
    upper = stack.sld_a2[0]
    lower = stack.sld_a2[5]
    slabs = stack.sld_a2[1:5]
    for part in (np.real, np.imag):
        values = part(slabs)
        low, high = sorted((float(part(upper)), float(part(lower))))
        assert np.all(values > low)
        assert np.all(values < high)
    assert abs(slabs[0] - upper) < abs(slabs[0] - lower)
    assert abs(slabs[-1] - lower) < abs(slabs[-1] - upper)


def test_transition_slabs_reuse_parent_thickness_for_roughness_limits() -> None:
    structure = StructureSpec(
        AIR,
        (transition_layer(), LayerSpec("under", SI, 30.0, roughness_a=3.0)),
        SI,
    )
    stack = expand_structure(structure, 1.5406)
    assert stack.roughness_a[5] == 3.0
    with pytest.raises(PhysicalValueError):
        expand_structure(
            StructureSpec(
                AIR,
                (transition_layer(), LayerSpec("under", SI, 30.0, roughness_a=9.9)),
                SI,
            ),
            1.5406,
        )


def test_transition_internal_interfaces_are_zero_roughness() -> None:
    stack = expand_structure(StructureSpec(AIR, (transition_layer(),), SI), 1.5406)
    np.testing.assert_array_equal(stack.roughness_a[:5], [0.0, 0.0, 0.0, 0.0, 0.0])


def test_structure_without_transitions_expands_bit_identically() -> None:
    structure = StructureSpec(
        AIR,
        (
            LayerSpec("cap", MO, 30.0, roughness_a=2.0),
            PeriodicBlock(
                "Mo/Si",
                (LayerSpec("Mo", MO, 28.0, roughness_a=3.0), LayerSpec("Si", SI, 42.0, roughness_a=4.0)),
                2,
                top_roughness_a=1.0,
            ),
            GradientLayerSpec("ramp", 10e-6 + 1e-6j, 30e-6 + 3e-6j, 6.0, roughness_a=1.0, microslab_max_a=2.0),
        ),
        SI,
        backing_roughness_a=2.0,
    )
    stack = expand_structure(structure, 1.5406)
    assert np.array_equal(stack.thickness_a, [0.0, 30.0, 28.0, 42.0, 28.0, 42.0, 2.0, 2.0, 2.0, 0.0])
    assert np.array_equal(
        stack.sld_a2,
        [
            0j,
            5.5e-05 + 1e-06j,
            5.5e-05 + 1e-06j,
            2e-05 + 2e-07j,
            5.5e-05 + 1e-06j,
            2e-05 + 2e-07j,
            1.3333333333333333e-05 + 1.3333333333333334e-06j,
            1.9999999999999998e-05 + 2.0000000000000003e-06j,
            2.6666666666666667e-05 + 2.666666666666667e-06j,
            2e-05 + 2e-07j,
        ],
    )
    assert np.array_equal(stack.roughness_a, [2.0, 1.0, 4.0, 3.0, 4.0, 1.0, 0.0, 0.0, 2.0])
    assert stack.periodic_spans == (PeriodicSpan(2, 2, 2),)
