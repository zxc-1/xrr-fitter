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


def _materials() -> tuple[MaterialSpec, MaterialSpec, MaterialSpec]:
    return (
        MaterialSpec("Air", None, None, 0.0j),
        MaterialSpec("Si", "Si", 2.329),
        MaterialSpec("Mo", "Mo", 10.28),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("thickness_a", 1.99, "film.thickness_a"),
        ("density_scale", 0.0, "film.density_scale"),
        ("density_scale", -0.01, "film.density_scale"),
        ("roughness_a", -0.01, "film.roughness_a"),
    ),
)
def test_layer_rejects_invalid_physical_values(
    field: str,
    value: float,
    message: str,
) -> None:
    _, silicon, _ = _materials()
    values = {"thickness_a": 20.0, "density_scale": 1.0, "roughness_a": 0.0}
    values[field] = value

    with pytest.raises(ValueError, match=message):
        LayerSpec("film", silicon, **values)


def test_periodic_block_requires_layers_and_positive_integer_repeats() -> None:
    with pytest.raises(ValueError, match="empty.layers"):
        PeriodicBlock("empty", (), repeats=1)
    _, silicon, _ = _materials()
    layer = LayerSpec("Si", silicon, 20.0)
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="block.repeats"):
            PeriodicBlock("block", (layer,), repeats=invalid)


def test_periodic_and_structure_sequences_are_defensively_tupleized() -> None:
    air, silicon, molybdenum = _materials()
    layers = [LayerSpec("Mo", molybdenum, 28.0)]
    block = PeriodicBlock("block", layers, repeats=2)
    components = [block]
    structure = StructureSpec(air, components, silicon)

    layers.clear()
    components.clear()

    assert len(block.layers) == 1
    assert structure.components == (block,)


def test_surface_termination_is_independent_of_internal_periodic_layer() -> None:
    _, _, molybdenum = _materials()
    internal = LayerSpec("internal Mo", molybdenum, 28.0, roughness_a=3.0)
    block = PeriodicBlock("Mo/Si", (internal,), repeats=2, top_roughness_a=1.5)

    assert block.top_roughness_a == 1.5
    assert block.layers[0].roughness_a == 3.0


def test_structure_accepts_large_finite_roughness_for_later_physics_validation() -> None:
    air, silicon, _ = _materials()

    assert StructureSpec(air, (), silicon, backing_roughness_a=50.01)
    with pytest.raises(ValueError, match="backing.roughness_a"):
        StructureSpec(air, (), silicon, backing_roughness_a=-0.01)


def test_layer_model_allows_expert_density_outside_default_fit_bounds() -> None:
    _, silicon, _ = _materials()
    assert LayerSpec("dense film", silicon, 20.0, density_scale=1.25).density_scale == 1.25


@pytest.mark.parametrize(
    ("field", "value"),
    (("thickness_a", 1.0), ("microslab_max_a", 0.0), ("microslab_max_a", 11.0)),
)
def test_gradient_rejects_invalid_discretization(field: str, value: float) -> None:
    values = {
        "upper_sld_a2": 1e-5 + 1e-7j,
        "lower_sld_a2": 3e-5 + 3e-7j,
        "thickness_a": 10.0,
        "roughness_a": 2.0,
        "microslab_max_a": 3.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=f"gradient.{field}"):
        GradientLayerSpec("gradient", **values)


def test_slab_stack_copies_arrays_and_validates_periodic_spans() -> None:
    thickness = np.array([0.0, 20.0, 20.0, 0.0])
    sld = np.array([0.0j, 2e-5, 2e-5, 3e-5])
    roughness = np.array([1.0, 1.0, 2.0])
    stack = SlabStack(thickness, sld, roughness, (PeriodicSpan(1, 1, 2),))

    thickness[1] = 99.0

    assert stack.thickness_a[1] == 20.0
    assert stack.thickness_a.flags.writeable is False
    with pytest.raises(ValueError, match="periodic span"):
        SlabStack(thickness, sld, roughness, (PeriodicSpan(0, 1, 2),))


def test_slab_stack_rejects_boolean_periodic_span_coordinates() -> None:
    with pytest.raises(ValueError, match="periodic span"):
        SlabStack(
            np.array([0.0, 20.0, 20.0, 0.0]),
            np.array([0.0j, 2e-5, 2e-5, 3e-5]),
            np.array([1.0, 1.0, 1.0]),
            (PeriodicSpan(True, 1, 2),),
        )


def test_slab_stack_preserves_finite_nonzero_boundary_thickness() -> None:
    stack = SlabStack(
        np.array([1.0, 20.0, 2.0]),
        np.array([0.0j, 2e-5, 3e-5]),
        np.array([1.0, 1.0]),
    )

    assert np.array_equal(stack.thickness_a, np.array([1.0, 20.0, 2.0]))
