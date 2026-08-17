from __future__ import annotations

import pickle

import numpy as np
import pytest

from xrr_fitter.model.slab_stack import PeriodicSpan, SlabStack
from xrr_fitter.model.structure import (
    MAX_EXPANDED_SLABS,
    MAX_TRANSITION_SLABS,
    DriftSpec,
    GradientLayerSpec,
    InterfaceTransition,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
    TransitionBranch,
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


def test_slab_stack_remains_read_only_after_pickle_round_trip() -> None:
    stack = SlabStack(
        np.array([0.0, 20.0, 20.0, 0.0]),
        np.array([0.0j, 2e-5, 2e-5, 3e-5]),
        np.array([1.0, 1.0, 2.0]),
        (PeriodicSpan(1, 1, 2),),
    )

    restored = pickle.loads(pickle.dumps(stack))

    assert restored.thickness_a.flags.writeable is False
    assert restored.sld_a2.flags.writeable is False
    assert restored.roughness_a.flags.writeable is False


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


def test_structure_accepts_expanded_slab_budget_boundary() -> None:
    air, silicon, _ = _materials()
    layer = LayerSpec("film", silicon, 20.0)
    block = PeriodicBlock("cell", (layer,), repeats=MAX_EXPANDED_SLABS)

    structure = StructureSpec(air, (block,), silicon)

    assert structure.components[0].repeats == MAX_EXPANDED_SLABS


def test_structure_rejects_periodic_expansion_over_slab_budget() -> None:
    air, silicon, _ = _materials()
    layer = LayerSpec("film", silicon, 20.0)

    with pytest.raises(ValueError, match="expanded slab count"):
        StructureSpec(
            air,
            (PeriodicBlock("cell", (layer,), repeats=MAX_EXPANDED_SLABS + 1),),
            silicon,
        )


def test_structure_rejects_gradient_expansion_over_slab_budget() -> None:
    air, silicon, _ = _materials()
    gradient = GradientLayerSpec(
        "ramp",
        1e-5 + 1e-7j,
        3e-5 + 3e-7j,
        float(MAX_EXPANDED_SLABS + 1),
        microslab_max_a=1.0,
    )

    with pytest.raises(ValueError, match="expanded slab count"):
        StructureSpec(air, (gradient,), silicon)


def test_structure_budget_sums_multiple_components() -> None:
    air, silicon, _ = _materials()
    layer = LayerSpec("film", silicon, 20.0)
    half = MAX_EXPANDED_SLABS // 2
    components = (
        PeriodicBlock("left", (layer,), repeats=half),
        PeriodicBlock("right", (layer,), repeats=half),
    )

    assert StructureSpec(air, components, silicon)
    with pytest.raises(ValueError, match="expanded slab count"):
        StructureSpec(air, (*components, layer), silicon)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", "gaussian"),
        ("weight", 0.0),
        ("weight", -1.0),
        ("weight", float("nan")),
        ("weight", float("inf")),
        ("thickness_a", 0.0),
        ("thickness_a", -2.0),
        ("thickness_a", float("inf")),
        ("thickness_a", float("nan")),
    ),
)
def test_transition_branch_rejects_invalid_declarations(field: str, value: object) -> None:
    values: dict[str, object] = {"kind": "erf", "weight": 1.0, "thickness_a": 10.0}
    values[field] = value

    with pytest.raises(ValueError):
        TransitionBranch(**values)


def test_transition_branch_accepts_widths_below_the_layer_thickness_floor() -> None:
    branch = TransitionBranch("erf", 1.0, 0.5)

    assert branch.thickness_a == 0.5


@pytest.mark.parametrize(
    ("weights", "expected"),
    (
        ((2.0, 2.0), (0.5, 0.5)),
        ((1.0, 3.0), (0.25, 0.75)),
    ),
)
def test_interface_transition_normalizes_weights(
    weights: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    transition = InterfaceTransition(tuple(TransitionBranch("erf", weight, 10.0) for weight in weights))

    assert tuple(branch.weight for branch in transition.branches) == expected


def test_interface_transition_requires_branches() -> None:
    with pytest.raises(ValueError, match="branches"):
        InterfaceTransition(())


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf"), 10.5))
def test_interface_transition_rejects_invalid_microslab_max(value: float) -> None:
    branches = (TransitionBranch("erf", 1.0, 10.0),)

    with pytest.raises(ValueError, match="microslab_max_a"):
        InterfaceTransition(branches, value)


def test_interface_transition_rejects_excessive_slab_count() -> None:
    with pytest.raises(ValueError, match="512"):
        InterfaceTransition((TransitionBranch("erf", 1.0, 4096.0),), 1.0)

    exact = InterfaceTransition((TransitionBranch("erf", 1.0, float(MAX_TRANSITION_SLABS)),), 1.0)

    assert exact.microslab_max_a == 1.0


def test_layer_with_transition_requires_zero_declared_roughness() -> None:
    _, silicon, _ = _materials()
    transition = InterfaceTransition((TransitionBranch("erf", 1.0, 10.0),))

    with pytest.raises(ValueError, match="roughness_a"):
        LayerSpec("film", silicon, 20.0, roughness_a=3.0, transition=transition)


def test_layer_transition_width_must_not_exceed_thickness() -> None:
    _, silicon, _ = _materials()
    transition = InterfaceTransition((TransitionBranch("erf", 1.0, 25.0),), 1.0)

    with pytest.raises(ValueError, match="film"):
        LayerSpec("film", silicon, 20.0, transition=transition)


def test_layer_transition_width_may_equal_thickness() -> None:
    _, silicon, _ = _materials()
    transition = InterfaceTransition((TransitionBranch("erf", 1.0, 20.0),), 1.0)
    layer = LayerSpec("film", silicon, 20.0, transition=transition)

    assert layer.transition is transition


def test_periodic_block_rejects_layers_with_transitions() -> None:
    _, silicon, molybdenum = _materials()
    transition = InterfaceTransition((TransitionBranch("erf", 1.0, 5.0),))
    plain = LayerSpec("plain", molybdenum, 20.0)
    textured = LayerSpec("textured", silicon, 20.0, transition=transition)

    with pytest.raises(ValueError, match="transition"):
        PeriodicBlock("cell", (plain, textured), repeats=3)


def test_layer_without_transition_keeps_existing_construction() -> None:
    _, silicon, _ = _materials()
    layer = LayerSpec("film", silicon, 20.0, 1.0, 3.0)

    assert layer.transition is None
    assert (layer.thickness_a, layer.density_scale, layer.roughness_a) == (20.0, 1.0, 3.0)


FILM = LayerSpec("film", MaterialSpec("SiO2", "SiO2", 2.2), 20.0, roughness_a=2.0)


def test_drift_spec_defaults_and_kind_validation() -> None:
    d = DriftSpec(kind="linear", target="thickness", amount=0.1)
    assert (d.period, d.phase, d.seed) == (0.0, 0.0, 0)
    with pytest.raises(ValueError):
        DriftSpec(kind="bogus", target="thickness")


def test_sine_requires_positive_period() -> None:
    with pytest.raises(ValueError):
        DriftSpec(kind="sine", target="thickness", amount=0.1, period=0.0)


def test_random_requires_nonneg_int_seed() -> None:
    with pytest.raises(ValueError):
        DriftSpec(kind="random", target="roughness", amount=0.1, seed=-1)


def test_drift_spec_normalizes_numpy_scalars_to_json_primitives() -> None:
    drift = DriftSpec(
        kind="random",
        target="roughness",
        amount=np.float32(0.2),
        period=np.float64(7.0),
        phase=np.float32(0.5),
        seed=np.int64(11),
    )

    assert type(drift.amount) is float
    assert type(drift.period) is float
    assert type(drift.phase) is float
    assert type(drift.seed) is int


@pytest.mark.parametrize(
    ("kind", "kwargs", "expected"),
    (
        pytest.param("linear", {"period": float("nan")}, ValueError, id="linear-period-nan"),
        pytest.param("linear", {"phase": float("nan")}, ValueError, id="linear-phase-nan"),
        pytest.param("linear", {"seed": 1.5}, TypeError, id="linear-seed-float"),
        pytest.param("sine", {"period": 4.0, "seed": True}, TypeError, id="sine-seed-bool"),
        pytest.param("random", {"period": "unused"}, TypeError, id="random-period-string"),
    ),
)
def test_drift_spec_rejects_invalid_unused_fields(
    kind: str, kwargs: dict[str, object], expected: type[Exception]
) -> None:
    with pytest.raises(expected, match="drift\\."):
        DriftSpec(kind=kind, target="thickness", amount=0.1, **kwargs)


def test_periodic_block_accepts_drift_and_requires_two_repeats() -> None:
    block = PeriodicBlock(
        name="p",
        layers=(FILM,),
        repeats=3,
        drift=DriftSpec(kind="linear", target="thickness", amount=0.05),
    )
    assert block.drift.kind == "linear"
    with pytest.raises(ValueError):
        PeriodicBlock(
            name="p",
            layers=(FILM,),
            repeats=1,
            drift=DriftSpec(kind="linear", target="thickness", amount=0.05),
        )


def test_periodic_block_without_drift_defaults_none() -> None:
    assert PeriodicBlock(name="p", layers=(FILM,), repeats=2).drift is None
