"""Immutable material and structure declarations.

This module records formula or direct-SLD materials, homogeneous and gradient
layers, periodic blocks, and the fronting/backing media. Expanded read-only slab
arrays live in ``model.slab_stack``. No material lookup or stack expansion runs
in this model module; it only validates values at the boundary.

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


def _real_scalar(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{field} must be a real number")
    return float(value)


def _complex_scalar(field: str, value: object) -> complex:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex, np.number)):
        raise TypeError(f"{field} must be a complex number")
    return complex(value)


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    """A formula-density material or a direct complex SLD override."""

    name: str
    formula: str | None
    bulk_density_g_cm3: float | None
    sld_override_a2: complex | None = None

    def __post_init__(self) -> None:
        if self.bulk_density_g_cm3 is not None:
            object.__setattr__(self, "bulk_density_g_cm3", _real_scalar("bulk density", self.bulk_density_g_cm3))
        if self.sld_override_a2 is not None:
            object.__setattr__(self, "sld_override_a2", _complex_scalar("SLD override", self.sld_override_a2))
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
    bounds = tuple(_real_scalar("oxide thickness bound", value) for value in values)
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
        initial = _real_scalar("oxide thickness_initial_a", self.thickness_initial_a)
        if not isfinite(initial) or not bounds[0] <= initial <= bounds[1]:
            raise ValueError("oxide thickness_initial_a must be within bounds")
        if self.location not in {"surface", "backing"}:
            raise ValueError("oxide location must be surface or backing")
        object.__setattr__(self, "thickness_initial_a", initial)
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
MAX_EXPANDED_SLABS = 2048


class ExpandedSlabLimitError(ValueError):
    """A declared or candidate structure exceeds the finite-slab budget."""


@dataclass(frozen=True, slots=True)
class TransitionBranch:
    """One weighted kernel inside an interface transition."""

    kind: str
    weight: float = 1.0
    thickness_a: float = 1.0

    def __post_init__(self) -> None:
        weight = _real_scalar("transition branch weight", self.weight)
        thickness = _real_scalar("transition branch thickness_a", self.thickness_a)
        if self.kind not in TRANSITION_KINDS:
            raise ValueError(f"transition kind must be one of {sorted(TRANSITION_KINDS)}: {self.kind}")
        for field, value in (("weight", weight), ("thickness_a", thickness)):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"transition branch {field} must be finite and positive: {value}")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "thickness_a", thickness)


def _transition_width(branches: tuple[TransitionBranch, ...]) -> float:
    return max(branch.thickness_a for branch in branches)


def _transition_branches(values: object) -> tuple[TransitionBranch, ...]:
    """Tupleize, type-check, and renormalize declared branch weights."""
    branches = tuple(values)
    if not branches:
        raise ValueError("transition branches must not be empty")
    if any(not isinstance(branch, TransitionBranch) for branch in branches):
        raise TypeError("transition branches must be TransitionBranch values")
    maximum = max(branch.weight for branch in branches)
    scaled = tuple(branch.weight / maximum for branch in branches)
    total = sum(scaled)
    return tuple(
        TransitionBranch(item.kind, weight / total, item.thickness_a)
        for item, weight in zip(branches, scaled, strict=True)
    )


@dataclass(frozen=True, slots=True)
class InterfaceTransition:
    """A normalizable interface width model discretized into microslabs."""

    branches: tuple[TransitionBranch, ...]
    microslab_max_a: float = 1.0

    def __post_init__(self) -> None:
        branches = _transition_branches(self.branches)
        width = _transition_width(branches)
        maximum = _real_scalar("transition microslab_max_a", self.microslab_max_a)
        if not isfinite(maximum) or maximum <= 0.0 or maximum > width:
            raise ValueError("transition microslab_max_a must be in (0,width]")
        slab_ratio = width / maximum
        if not isfinite(slab_ratio) or ceil(slab_ratio) > MAX_TRANSITION_SLABS:
            raise ValueError(f"transition microslab count must not exceed {MAX_TRANSITION_SLABS}")
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "microslab_max_a", maximum)


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
        thickness = _real_scalar(f"{self.name}.thickness_a", self.thickness_a)
        density_scale = _real_scalar(f"{self.name}.density_scale", self.density_scale)
        roughness = _real_scalar(f"{self.name}.roughness_a", self.roughness_a)
        object.__setattr__(self, "thickness_a", thickness)
        object.__setattr__(self, "density_scale", density_scale)
        object.__setattr__(self, "roughness_a", roughness)
        _component_name(self.name, "layer")
        if not isinstance(self.material, MaterialSpec):
            raise TypeError("layer material must be a MaterialSpec")
        _thickness(self.name, thickness)
        if not isfinite(density_scale) or density_scale <= 0.0:
            raise ValueError(f"{self.name}.density_scale must be finite and positive")
        _roughness(self.name, roughness)
        _layer_transition(self.name, thickness, roughness, self.transition)


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
        upper = _complex_scalar(f"{self.name}.upper_sld_a2", self.upper_sld_a2)
        lower = _complex_scalar(f"{self.name}.lower_sld_a2", self.lower_sld_a2)
        thickness = _real_scalar(f"{self.name}.thickness_a", self.thickness_a)
        roughness = _real_scalar(f"{self.name}.roughness_a", self.roughness_a)
        maximum = _real_scalar(f"{self.name}.microslab_max_a", self.microslab_max_a)
        object.__setattr__(self, "upper_sld_a2", upper)
        object.__setattr__(self, "lower_sld_a2", lower)
        object.__setattr__(self, "thickness_a", thickness)
        object.__setattr__(self, "roughness_a", roughness)
        object.__setattr__(self, "microslab_max_a", maximum)
        _component_name(self.name, "gradient")
        _gradient_sld(
            self.name,
            (("upper_sld_a2", upper), ("lower_sld_a2", lower)),
        )
        _thickness(self.name, thickness)
        _roughness(self.name, roughness)
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
            top = _real_scalar(f"{self.name}.top.roughness_a", self.top_roughness_a)
            object.__setattr__(self, "top_roughness_a", top)
            _roughness(f"{self.name}.top", top)
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


def _layer_expanded_slab_count(layer: LayerSpec) -> int:
    if layer.transition is None:
        return 1
    transition = layer.transition
    return ceil(_transition_width(transition.branches) / transition.microslab_max_a) + 1


def _component_expanded_slab_count(component: StructureComponent | _ExpandedDriftBlock) -> int:
    if isinstance(component, LayerSpec):
        return _layer_expanded_slab_count(component)
    if isinstance(component, PeriodicBlock):
        return component.repeats * sum(_layer_expanded_slab_count(layer) for layer in component.layers)
    if isinstance(component, _ExpandedDriftBlock):
        return sum(_layer_expanded_slab_count(layer) for layer in component.layers)
    ratio = component.thickness_a / component.microslab_max_a
    if not isfinite(ratio) or ratio > MAX_EXPANDED_SLABS:
        raise ExpandedSlabLimitError(f"expanded slab count for {component.name} exceeds {MAX_EXPANDED_SLABS}")
    return max(1, ceil(ratio))


def _validate_expanded_slab_budget(components: tuple[StructureComponent, ...]) -> None:
    count = sum(_component_expanded_slab_count(component) for component in components)
    if count > MAX_EXPANDED_SLABS:
        raise ExpandedSlabLimitError(f"expanded slab count {count} exceeds {MAX_EXPANDED_SLABS}")


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """Fronting, ordered components, backing, and backing roughness."""

    fronting: MaterialSpec
    components: tuple[StructureComponent, ...]
    backing: MaterialSpec
    backing_roughness_a: float = 0.0

    def __post_init__(self) -> None:
        backing_roughness = _real_scalar("backing.roughness_a", self.backing_roughness_a)
        object.__setattr__(self, "backing_roughness_a", backing_roughness)
        components = _structure_components(self.components)
        object.__setattr__(self, "components", components)
        _validate_expanded_slab_budget(components)
        if not isinstance(self.fronting, MaterialSpec) or not isinstance(self.backing, MaterialSpec):
            raise TypeError("structure media must be MaterialSpec values")
        if not isfinite(backing_roughness) or backing_roughness < 0.0:
            raise ValueError("backing.roughness_a must be finite and nonnegative")
