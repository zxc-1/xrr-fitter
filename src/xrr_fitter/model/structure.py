"""Immutable material, structure declaration, and expanded slab values.

The declaration side records formula or direct-SLD materials, homogeneous and
gradient layers, periodic blocks, and the fronting/backing media. The expanded
side owns read-only thickness, complex SLD, and interface-roughness arrays used
by the physics layer. No material lookup or stack expansion runs in this model
module; it only validates values at the boundary.

Periodic metadata is accepted only when every declared repeated cell is
exactly identical. The first repeated cell may retain its explicit top
roughness override, while later interfaces must match the normal repeated
cell. This keeps the optimized periodic representation equivalent to the full
slab sequence rather than treating metadata as an unchecked performance hint.

An interface transition replaces the Nevot-Croce roughness of the layer's
incident interface instead of adding to it; a single interface cannot be
broadened twice, so a layer carrying a transition must declare
``roughness_a == 0``. Only the declared value is checked here, because the
compiled fit locks that coordinate at zero and rebuilds layers from optimizer
values every iteration. Branch weights are normalized at construction so that
they always sum to one, and the transition occupies the incident side of the
layer, leaving ``thickness_a - width`` for the layer body. Widths are capped at
``MAX_TRANSITION_SLABS`` microslabs to keep expansion bounded. Kind names are
duplicated here rather than imported because this module deliberately depends
on nothing else in the package; ``physics.transitions`` owns the kernels and a
test pins the two name sets together.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np


def _finite_complex(value: complex) -> bool:
    return isfinite(value.real) and isfinite(value.imag)


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    """A formula-density material or a direct complex SLD override."""

    name: str
    formula: str | None
    bulk_density_g_cm3: float | None
    sld_override_a2: complex | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("material name must not be empty")
        formula_source = self.formula is not None or self.bulk_density_g_cm3 is not None
        override_source = self.sld_override_a2 is not None
        if formula_source == override_source:
            raise ValueError("material requires exactly one SLD source")
        if formula_source:
            self._validate_formula()
        else:
            self._validate_override()

    def _validate_formula(self) -> None:
        if self.formula is None or not self.formula.strip():
            raise ValueError("material formula must not be empty")
        density = self.bulk_density_g_cm3
        if density is None or not isfinite(density):
            raise ValueError("bulk density must be finite")
        if density <= 0.0:
            raise ValueError("bulk density must be positive")

    def _validate_override(self) -> None:
        assert self.sld_override_a2 is not None
        if not _finite_complex(self.sld_override_a2):
            raise ValueError("SLD override must be finite")
        if self.sld_override_a2.imag < 0.0:
            raise ValueError("SLD override requires nonnegative absorption")


def _oxide_bounds(values: object) -> tuple[float, float]:
    bounds = tuple(values)
    if len(bounds) != 2 or any(not isfinite(value) or value <= 0.0 for value in bounds):
        raise ValueError("oxide thickness_bounds_a must contain two positive finite values")
    if bounds[0] > bounds[1]:
        raise ValueError("oxide thickness_bounds_a must be ordered")
    return bounds


@dataclass(frozen=True, slots=True)
class OxideSuggestion:
    """One versioned proposal for a native oxide layer."""

    base_material: str
    oxide_material: MaterialSpec
    density_locked: bool
    thickness_initial_a: float
    thickness_bounds_a: tuple[float, float]
    oxide_table_version: str
    location: str

    def __post_init__(self) -> None:
        if not self.base_material.strip() or not self.oxide_table_version.strip():
            raise ValueError("oxide material and table version must not be empty")
        if not isinstance(self.oxide_material, MaterialSpec):
            raise TypeError("oxide_material must be MaterialSpec")
        if not isinstance(self.density_locked, bool):
            raise TypeError("density_locked must be bool")
        bounds = _oxide_bounds(self.thickness_bounds_a)
        if not isfinite(self.thickness_initial_a) or not bounds[0] <= self.thickness_initial_a <= bounds[1]:
            raise ValueError("oxide thickness_initial_a must be within bounds")
        if self.location not in {"surface", "backing"}:
            raise ValueError("oxide location must be surface or backing")
        object.__setattr__(self, "thickness_bounds_a", bounds)


def _component_name(name: str, kind: str) -> None:
    if not name.strip():
        raise ValueError(f"{kind} name must not be empty")


def _thickness(name: str, value: float) -> None:
    if not isfinite(value) or value < 2.0:
        raise ValueError(f"{name}.thickness_a must be at least 2 A")


def _roughness(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name}.roughness_a must be finite and nonnegative")


def _gradient_sld(name: str, fields: tuple[tuple[str, complex], ...]) -> None:
    for field, value in fields:
        if not _finite_complex(value) or value.imag < 0.0:
            raise ValueError(f"{name}.{field} must be finite with nonnegative absorption")


TRANSITION_KINDS = frozenset({"erf", "linear", "exponential", "tanh", "sine", "step"})
MAX_TRANSITION_SLABS = 512


@dataclass(frozen=True, slots=True)
class TransitionBranch:
    """One weighted kernel inside an interface transition."""

    kind: str
    weight: float = 1.0
    thickness_a: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in TRANSITION_KINDS:
            raise ValueError(f"transition kind must be one of {sorted(TRANSITION_KINDS)}: {self.kind}")
        for field, value in (("weight", self.weight), ("thickness_a", self.thickness_a)):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"transition branch {field} must be finite and positive: {value}")


def _transition_width(branches: tuple[TransitionBranch, ...]) -> float:
    return max(branch.thickness_a for branch in branches)


def _transition_branches(values: object) -> tuple[TransitionBranch, ...]:
    """Tupleize, type-check, and renormalize declared branch weights."""
    branches = tuple(values)
    if not branches:
        raise ValueError("transition branches must not be empty")
    if any(not isinstance(branch, TransitionBranch) for branch in branches):
        raise TypeError("transition branches must be TransitionBranch values")
    total = sum(branch.weight for branch in branches)
    return tuple(TransitionBranch(item.kind, item.weight / total, item.thickness_a) for item in branches)


@dataclass(frozen=True, slots=True)
class InterfaceTransition:
    """A normalizable interface width model discretized into microslabs."""

    branches: tuple[TransitionBranch, ...]
    microslab_max_a: float = 1.0

    def __post_init__(self) -> None:
        branches = _transition_branches(self.branches)
        width = _transition_width(branches)
        maximum = self.microslab_max_a
        if not isfinite(maximum) or maximum <= 0.0 or maximum > width:
            raise ValueError("transition microslab_max_a must be in (0,width]")
        if ceil(width / maximum) > MAX_TRANSITION_SLABS:
            raise ValueError(f"transition microslab count must not exceed {MAX_TRANSITION_SLABS}")
        object.__setattr__(self, "branches", branches)


def _layer_transition(name: str, thickness_a: float, roughness_a: float, value: object) -> None:
    """Reject declarations where a transition and a rough interface overlap."""
    if value is None:
        return
    if not isinstance(value, InterfaceTransition):
        raise TypeError("layer transition must be an InterfaceTransition")
    if roughness_a != 0.0:
        raise ValueError(f"{name}.roughness_a must be 0 when a transition sets the interface width")
    width = _transition_width(value.branches)
    if width > thickness_a:
        raise ValueError(f"{name} transition width {width} must not exceed thickness_a {thickness_a}")


def _periodic_layers(values: object, name: str) -> tuple[LayerSpec, ...]:
    layers = tuple(values)
    if not layers:
        raise ValueError(f"{name}.layers must not be empty")
    if any(not isinstance(layer, LayerSpec) for layer in layers):
        raise TypeError("periodic block layers must be LayerSpec values")
    if any(layer.transition is not None for layer in layers):
        raise ValueError(f"{name}.layers must not declare a transition inside a periodic block")
    return layers


def _positive_repeats(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name}.repeats must be a positive integer")


def _structure_components(values: object) -> tuple[StructureComponent, ...]:
    components = tuple(values)
    # ``_ExpandedDriftBlock`` is ephemeral yet flows through ``replace`` when
    # ``rebuild_structure`` bakes a drift, which re-runs this validation; admit it
    # here so the rebuilt spec is accepted while it never enters the public union.
    allowed = (LayerSpec, PeriodicBlock, GradientLayerSpec, _ExpandedDriftBlock)
    if any(not isinstance(value, allowed) for value in components):
        raise TypeError("structure components contain an unsupported value")
    return components


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """One homogeneous finite layer declaration."""

    name: str
    material: MaterialSpec
    thickness_a: float
    density_scale: float = 1.0
    roughness_a: float = 0.0
    transition: InterfaceTransition | None = None

    def __post_init__(self) -> None:
        _component_name(self.name, "layer")
        if not isinstance(self.material, MaterialSpec):
            raise TypeError("layer material must be a MaterialSpec")
        _thickness(self.name, self.thickness_a)
        if not isfinite(self.density_scale) or self.density_scale <= 0.0:
            raise ValueError(f"{self.name}.density_scale must be finite and positive")
        _roughness(self.name, self.roughness_a)
        _layer_transition(self.name, self.thickness_a, self.roughness_a, self.transition)


@dataclass(frozen=True, slots=True)
class GradientLayerSpec:
    """A continuous SLD transition with bounded microslab discretization."""

    name: str
    upper_sld_a2: complex
    lower_sld_a2: complex
    thickness_a: float
    roughness_a: float = 0.0
    microslab_max_a: float = 1.0

    def __post_init__(self) -> None:
        _component_name(self.name, "gradient")
        _gradient_sld(
            self.name,
            (("upper_sld_a2", self.upper_sld_a2), ("lower_sld_a2", self.lower_sld_a2)),
        )
        _thickness(self.name, self.thickness_a)
        _roughness(self.name, self.roughness_a)
        maximum = self.microslab_max_a
        if not isfinite(maximum) or maximum <= 0.0 or maximum > self.thickness_a:
            raise ValueError(f"{self.name}.microslab_max_a must be in (0,thickness]")


DRIFT_KINDS = frozenset({"linear", "sine", "random"})
DRIFT_TARGETS = frozenset({"thickness", "roughness"})


def _drift_float(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"drift.{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"drift.{field} must be finite")
    return result


def _drift_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("drift.seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("drift.seed must be a non-negative integer")
    return result


@dataclass(frozen=True, slots=True)
class DriftSpec:
    """Per-repeat drift law for a periodic block (primitive scalars only)."""

    kind: str
    target: str
    amount: float = 0.0
    period: float = 0.0
    phase: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in DRIFT_KINDS:
            raise ValueError(f"drift.kind must be one of {sorted(DRIFT_KINDS)}")
        if self.target not in DRIFT_TARGETS:
            raise ValueError(f"drift.target must be one of {sorted(DRIFT_TARGETS)}")
        amount = _drift_float("amount", self.amount)
        period = _drift_float("period", self.period)
        phase = _drift_float("phase", self.phase)
        seed = _drift_seed(self.seed)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "seed", seed)
        if self.kind == "sine":
            if period <= 0.0:
                raise ValueError("drift.period must be finite and positive for sine drift")


@dataclass(frozen=True, slots=True)
class PeriodicBlock:
    """A named cell repeated toward the substrate with an optional top termination."""

    name: str
    layers: tuple[LayerSpec, ...]
    repeats: int
    top_roughness_a: float | None = None
    drift: DriftSpec | None = None

    def __post_init__(self) -> None:
        _component_name(self.name, "periodic block")
        layers = _periodic_layers(self.layers, self.name)
        object.__setattr__(self, "layers", layers)
        _positive_repeats(self.repeats, self.name)
        if self.top_roughness_a is not None:
            _roughness(f"{self.name}.top", self.top_roughness_a)
        if self.drift is not None:
            if not isinstance(self.drift, DriftSpec):
                raise TypeError(f"{self.name}.drift must be a DriftSpec")
            if self.repeats < 2:
                raise ValueError(f"{self.name}.drift requires repeats >= 2")


@dataclass(frozen=True, slots=True)
class _ExpandedDriftBlock:
    """Ephemeral per-copy expansion of a drifted block.

    ``rebuild_structure`` produces this so that ``expand_structure`` stays a pure
    function of the structure: every per-copy thickness/roughness/SLD is already
    baked into ``layers`` (flattened copy-major, ``repeats * len(base.layers)``
    entries). ``layer_count`` records the base cell width so geometry can recover
    the ``(copy, layer)`` grid via ``divmod(flat_index, layer_count)``. NEVER
    persisted, codec-encoded, or checkpointed; it lives only between the rebuild
    and the expansion passes.
    """

    layers: tuple[LayerSpec, ...]
    layer_count: int
    top_roughness_a: float | None
    target: str


StructureComponent = LayerSpec | PeriodicBlock | GradientLayerSpec


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """Fronting, ordered components, backing, and backing roughness."""

    fronting: MaterialSpec
    components: tuple[StructureComponent, ...]
    backing: MaterialSpec
    backing_roughness_a: float = 0.0

    def __post_init__(self) -> None:
        components = _structure_components(self.components)
        object.__setattr__(self, "components", components)
        if not isinstance(self.fronting, MaterialSpec) or not isinstance(self.backing, MaterialSpec):
            raise TypeError("structure media must be MaterialSpec values")
        if not isfinite(self.backing_roughness_a) or self.backing_roughness_a < 0.0:
            raise ValueError("backing.roughness_a must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PeriodicSpan:
    """Coordinates of exactly repeated finite media in an expanded stack."""

    start_medium: int
    layer_count: int
    repeats: int


def _stack_arrays(
    thickness_a: object,
    sld_a2: object,
    roughness_a: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(thickness_a, dtype=float, copy=True),
        np.array(sld_a2, dtype=np.complex128, copy=True),
        np.array(roughness_a, dtype=float, copy=True),
    )


def _validate_stack_shapes(
    thickness: np.ndarray,
    sld: np.ndarray,
    roughness: np.ndarray,
) -> None:
    if thickness.ndim != 1 or sld.ndim != 1 or roughness.ndim != 1:
        raise ValueError("slab stack arrays must be one-dimensional")
    expected = thickness.size - 1
    if thickness.size < 2 or sld.shape != thickness.shape or roughness.size != expected:
        raise ValueError("slab stack array lengths are inconsistent")


def _validate_thickness(thickness: np.ndarray) -> None:
    if np.any(~np.isfinite(thickness)) or np.any(thickness < 0.0):
        raise ValueError("slab thickness must be finite and nonnegative")


def _validate_sld(sld: np.ndarray) -> None:
    invalid = np.any(~np.isfinite(sld.real)) or np.any(~np.isfinite(sld.imag))
    if invalid or np.any(sld.imag < 0.0):
        raise ValueError("slab SLD must be finite with nonnegative absorption")


def _validate_stack_roughness(roughness: np.ndarray) -> None:
    if np.any(~np.isfinite(roughness)) or np.any(roughness < 0.0):
        raise ValueError("interface roughness must be finite and nonnegative")


def _validate_span_coordinates(span: PeriodicSpan) -> None:
    values = (span.start_medium, span.layer_count, span.repeats)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("periodic span coordinates must be integers")
    if span.start_medium < 1 or span.layer_count < 1 or span.repeats < 2:
        raise ValueError("periodic span is outside finite media")


def _span_end(span: PeriodicSpan, finite_end: int, previous_end: int) -> int:
    end = span.start_medium + span.layer_count * span.repeats
    if end > finite_end:
        raise ValueError("periodic span is outside finite media")
    if span.start_medium < previous_end:
        raise ValueError("periodic spans overlap")
    return end


def _validate_media_repetition(
    thickness: np.ndarray,
    sld: np.ndarray,
    span: PeriodicSpan,
) -> None:
    start = span.start_medium
    stop = start + span.layer_count
    base_thickness = thickness[start:stop]
    base_sld = sld[start:stop]
    for repeat in range(1, span.repeats):
        offset = start + repeat * span.layer_count
        repeated = slice(offset, offset + span.layer_count)
        if not np.array_equal(thickness[repeated], base_thickness):
            raise ValueError("periodic span thickness does not repeat")
        if not np.array_equal(sld[repeated], base_sld):
            raise ValueError("periodic span SLD does not repeat")


def _validate_roughness_repetition(roughness: np.ndarray, span: PeriodicSpan) -> None:
    start = span.start_medium
    stop = start + span.layer_count
    first = roughness[start - 1 : stop - 1]
    normal_start = stop - 1
    normal = roughness[normal_start : normal_start + span.layer_count]
    if not np.array_equal(first[1:], normal[1:]):
        raise ValueError("periodic span roughness does not repeat")
    for repeat in range(2, span.repeats):
        offset = start + repeat * span.layer_count - 1
        if not np.array_equal(roughness[offset : offset + span.layer_count], normal):
            raise ValueError("periodic span roughness does not repeat")


def _validate_spans(
    thickness: np.ndarray,
    sld: np.ndarray,
    roughness: np.ndarray,
    spans: tuple[PeriodicSpan, ...],
) -> None:
    previous_end = 1
    finite_end = thickness.size - 1
    for span in spans:
        if not isinstance(span, PeriodicSpan):
            raise TypeError("periodic spans must be PeriodicSpan values")
        _validate_span_coordinates(span)
        previous_end = _span_end(span, finite_end, previous_end)
        _validate_media_repetition(thickness, sld, span)
        _validate_roughness_repetition(roughness, span)


@dataclass(frozen=True, slots=True)
class SlabStack:
    """Owned read-only slab arrays plus validated periodic fast-path metadata."""

    thickness_a: np.ndarray
    sld_a2: np.ndarray
    roughness_a: np.ndarray
    periodic_spans: tuple[PeriodicSpan, ...] = ()

    def __post_init__(self) -> None:
        thickness, sld, roughness = _stack_arrays(
            self.thickness_a,
            self.sld_a2,
            self.roughness_a,
        )
        spans = tuple(self.periodic_spans)
        _validate_stack_shapes(thickness, sld, roughness)
        _validate_thickness(thickness)
        _validate_sld(sld)
        _validate_stack_roughness(roughness)
        _validate_spans(thickness, sld, roughness, spans)
        for value in (thickness, sld, roughness):
            value.setflags(write=False)
        object.__setattr__(self, "thickness_a", thickness)
        object.__setattr__(self, "sld_a2", sld)
        object.__setattr__(self, "roughness_a", roughness)
        object.__setattr__(self, "periodic_spans", spans)
