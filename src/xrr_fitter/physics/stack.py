"""Deterministic expansion of structure declarations into slab stacks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from math import ceil, isfinite

import numpy as np

from xrr_fitter.model.parameters import PhysicalValueError
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    PeriodicSpan,
    SlabStack,
    StructureComponent,
    StructureSpec,
    _ExpandedDriftBlock,
)
from xrr_fitter.physics.materials import material_sld
from xrr_fitter.physics.transitions import (
    transition_fractions,
    transition_slab_count,
    transition_width,
)


def _replace_material(
    material: MaterialSpec,
    prefix: str,
    values: dict[str, float],
) -> MaterialSpec:
    """Rebuild an optional explicit complex-SLD material override.

    Formula-backed materials intentionally omit override coordinates and retain
    their original identity. Explicit real and imaginary parts are consumed as
    one value so a half-specified mapping cannot escape reconstruction.
    """
    real_name = f"{prefix}.sld_real_a2"
    if real_name not in values:
        return material
    return replace(
        material,
        sld_override_a2=complex(values[real_name], values[f"{prefix}.sld_imag_a2"]),
    )


def _replace_layer(
    layer: LayerSpec,
    prefix: str,
    values: dict[str, float],
) -> LayerSpec:
    """Apply one layer's material, thickness, density, and roughness values.

    Name construction remains positional because compiled definitions and
    periodic sharing both refer to the same component-index prefixes.
    """
    return replace(
        layer,
        material=_replace_material(layer.material, prefix, values),
        thickness_a=values[f"{prefix}.thickness_a"],
        density_scale=values[f"{prefix}.density_scale"],
        roughness_a=values[f"{prefix}.roughness_a"],
    )


def _replace_indexed_layer(
    item: tuple[int, LayerSpec],
    prefix: str,
    values: dict[str, float],
) -> LayerSpec:
    index, layer = item
    return _replace_layer(layer, f"{prefix}.layer.{index}", values)


def _replace_periodic(
    block: PeriodicBlock,
    prefix: str,
    values: dict[str, float],
) -> PeriodicBlock:
    """Rebuild shared periodic-cell coordinates without expanding repeats.

    Repeat count is persisted as numeric metadata but represents exact topology.
    ``None`` top roughness remains an inheritance sentinel rather than being
    materialized from the first layer during reconstruction.
    """
    layers = tuple(
        map(
            partial(_replace_indexed_layer, prefix=prefix, values=values),
            enumerate(block.layers),
        )
    )
    # A missing top override is semantic inheritance, not a zero roughness value.
    # Preserve the sentinel so later expansion chooses layer zero exactly once.
    top_roughness = None if block.top_roughness_a is None else values[f"{prefix}.top_roughness_a"]
    return replace(
        block,
        layers=layers,
        repeats=int(round(values[f"{prefix}.repeats"])),
        top_roughness_a=top_roughness,
    )


def _expand_drift(
    block: PeriodicBlock,
    prefix: str,
    values: dict[str, float],
) -> _ExpandedDriftBlock:
    """Bake a drifted block into flattened per-copy layers (copy-major).

    Copy zero is the free base cell (coefficient zero); later copies read the
    constraint-resolved ``.repeat.{k}`` coordinate for the drift target and share
    the base cell for every other family. Doing the value substitution once, here
    in the rebuild pass, lets the primal, geometry, and Jacobian expanders all
    consume the baked layers without threading a value map of their own.
    """
    assert block.drift is not None
    target = block.drift.target
    flat: list[LayerSpec] = []
    for repeat_index in range(block.repeats):
        for layer_index, layer in enumerate(block.layers):
            base = f"{prefix}.layer.{layer_index}"
            copy = f"{prefix}.repeat.{repeat_index}.layer.{layer_index}"
            drifting = repeat_index > 0
            thickness = (
                values[f"{copy}.thickness_a"] if drifting and target == "thickness" else values[f"{base}.thickness_a"]
            )
            roughness = (
                values[f"{copy}.roughness_a"] if drifting and target == "roughness" else values[f"{base}.roughness_a"]
            )
            flat.append(
                replace(
                    layer,
                    material=_replace_material(layer.material, base, values),
                    thickness_a=thickness,
                    density_scale=values[f"{base}.density_scale"],
                    roughness_a=roughness,
                )
            )
    top = None if block.top_roughness_a is None else values[f"{prefix}.top_roughness_a"]
    return _ExpandedDriftBlock(tuple(flat), len(block.layers), top, target)


def _replace_gradient(
    layer: GradientLayerSpec,
    prefix: str,
    values: dict[str, float],
) -> GradientLayerSpec:
    """Rebuild complex gradient endpoints and topology-defining dimensions.

    Thickness and microslab limit are independent coordinates because both can
    change the expanded slab count used by the primal and tangent mappings.
    """
    return replace(
        layer,
        upper_sld_a2=complex(
            values[f"{prefix}.upper_sld_real_a2"],
            values[f"{prefix}.upper_sld_imag_a2"],
        ),
        lower_sld_a2=complex(
            values[f"{prefix}.lower_sld_real_a2"],
            values[f"{prefix}.lower_sld_imag_a2"],
        ),
        thickness_a=values[f"{prefix}.thickness_a"],
        roughness_a=values[f"{prefix}.roughness_a"],
        microslab_max_a=values[f"{prefix}.microslab_max_a"],
    )


def _replace_component(
    component: StructureComponent,
    prefix: str,
    values: dict[str, float],
) -> StructureComponent:
    if isinstance(component, LayerSpec):
        return _replace_layer(component, prefix, values)
    if isinstance(component, PeriodicBlock):
        # A drifted block bakes into flattened per-copy layers here so that every
        # downstream expander stays a pure function of the rebuilt structure.
        if component.drift is not None:
            return _expand_drift(component, prefix, values)
        return _replace_periodic(component, prefix, values)
    return _replace_gradient(component, prefix, values)


def _replace_indexed_component(
    item: tuple[int, StructureComponent],
    values: dict[str, float],
) -> StructureComponent:
    index, component = item
    return _replace_component(component, f"component.{index}", values)


def rebuild_structure(
    structure: StructureSpec,
    values: dict[str, float],
) -> StructureSpec:
    """Return an immutable structure from one complete physical value map.

    Component order is retained exactly. Rebuilding never mutates the compiled
    problem's declared structure, allowing parallel candidates to share it.
    """
    components = tuple(
        map(
            partial(_replace_indexed_component, values=values),
            enumerate(structure.components),
        )
    )
    return replace(
        structure,
        components=components,
        backing=_replace_material(structure.backing, "backing", values),
        backing_roughness_a=values["backing.roughness_a"],
    )


@dataclass(slots=True)
class _Expansion:
    wavelength_a: float
    cache: dict[tuple[MaterialSpec, float], complex] = field(default_factory=dict)
    thickness: list[float] = field(default_factory=lambda: [0.0])
    limit_thickness: list[float] = field(default_factory=lambda: [0.0])
    sld: list[complex] = field(default_factory=list)
    roughness: list[float] = field(default_factory=list)
    spans: list[PeriodicSpan] = field(default_factory=list)

    def sld_for(self, material: MaterialSpec, density_scale: float) -> complex:
        key = (material, float(density_scale))
        if key not in self.cache:
            self.cache[key] = material_sld(material, density_scale, self.wavelength_a)
        return self.cache[key]


def _append_transition_layer(state: _Expansion, layer: LayerSpec) -> None:
    """Expand a graded incident interface into microslabs plus the layer body.

    The upper medium is read from the expansion state rather than the preceding
    declaration so that periodic blocks and gradients above this layer resolve
    correctly. The body slab is emitted unconditionally, even at zero thickness,
    keeping the expanded row count exactly ``count + 1``.
    """
    transition = layer.transition
    assert transition is not None
    upper = state.sld[-1]
    lower = state.sld_for(layer.material, layer.density_scale)
    width = transition_width(transition)
    count = transition_slab_count(width, transition.microslab_max_a)
    fractions = transition_fractions(transition, count)
    for index, fraction in enumerate(fractions):
        state.thickness.append(width / count)
        state.limit_thickness.append(layer.thickness_a)
        state.sld.append((1.0 - fraction) * upper + fraction * lower)
        state.roughness.append(layer.roughness_a if index == 0 else 0.0)
    state.thickness.append(layer.thickness_a - width)
    state.limit_thickness.append(layer.thickness_a)
    state.sld.append(lower)
    state.roughness.append(0.0)


def _append_layer(state: _Expansion, layer: LayerSpec, roughness: float | None = None) -> None:
    if layer.transition is not None:
        return _append_transition_layer(state, layer)
    state.thickness.append(layer.thickness_a)
    state.limit_thickness.append(layer.thickness_a)
    state.sld.append(state.sld_for(layer.material, layer.density_scale))
    state.roughness.append(layer.roughness_a if roughness is None else roughness)
    return None


def _append_periodic(state: _Expansion, block: PeriodicBlock) -> None:
    """Expand a plain (or raw, unbaked) periodic block from its base cell.

    A rebuilt drifted block arrives as ``_ExpandedDriftBlock`` and never reaches
    here. A raw drifted block (e.g. GUI structure validation before fit
    compilation) still expands its base cell for every copy, but suppresses the
    matrix-power span because drift makes the per-copy cells non-identical.
    """
    drifted = block.drift is not None
    start = len(state.thickness)
    for repeat_index in range(block.repeats):
        for layer_index, layer in enumerate(block.layers):
            override = block.top_roughness_a if repeat_index == layer_index == 0 else None
            _append_layer(state, layer, override)
    if block.repeats > 1 and not drifted:
        state.spans.append(PeriodicSpan(start, len(block.layers), block.repeats))


def _append_drift_block(state: _Expansion, block: _ExpandedDriftBlock) -> None:
    """Emit baked per-copy layers with no span; drift breaks bit-identical repetition.

    The top termination overrides only the very first interface of the block,
    which in the flattened copy-major layout is layer position zero.
    """
    for position, layer in enumerate(block.layers):
        _append_layer(state, layer, block.top_roughness_a if position == 0 else None)


def _append_gradient(state: _Expansion, gradient: GradientLayerSpec) -> None:
    count = ceil(gradient.thickness_a / gradient.microslab_max_a)
    thickness = gradient.thickness_a / count
    delta = gradient.lower_sld_a2 - gradient.upper_sld_a2
    for index in range(count):
        state.thickness.append(thickness)
        state.limit_thickness.append(gradient.thickness_a)
        state.sld.append(gradient.upper_sld_a2 + ((index + 0.5) / count) * delta)
        state.roughness.append(gradient.roughness_a if index == 0 else 0.0)


def _append_component(state: _Expansion, component: StructureComponent) -> None:
    if isinstance(component, LayerSpec):
        _append_layer(state, component)
    elif isinstance(component, _ExpandedDriftBlock):
        _append_drift_block(state, component)
    elif isinstance(component, PeriodicBlock):
        _append_periodic(state, component)
    else:
        _append_gradient(state, component)


def _finite_neighbors(thickness: np.ndarray, interface: int) -> list[float]:
    neighbors: list[float] = []
    if interface != 0:
        neighbors.append(float(thickness[interface]))
    if interface + 1 != thickness.size - 1:
        neighbors.append(float(thickness[interface + 1]))
    return neighbors


def _validate_roughness(thickness: np.ndarray, roughness: np.ndarray) -> None:
    for interface, sigma in enumerate(roughness):
        neighbors = _finite_neighbors(thickness, interface)
        limit = 0.49 * min(neighbors) if neighbors else 50.0
        invalid = sigma >= limit if neighbors else sigma > limit
        if invalid:
            raise PhysicalValueError(f"interface.{interface}.roughness_a must be below {limit:g} A")


def expand_structure(
    structure: StructureSpec,
    wavelength_a: float,
) -> SlabStack:
    """Expand fronting, declared components, and backing at one wavelength.

    This is a pure function of the structure: any per-copy drift geometry is
    already baked into ``_ExpandedDriftBlock`` components by ``rebuild_structure``,
    so no value map is threaded here. A raw (unbaked) drifted ``PeriodicBlock``
    expands from its base cell with the matrix-power span suppressed.
    """
    if not isfinite(wavelength_a) or wavelength_a <= 0.0:
        raise ValueError("wavelength_a must be positive")
    state = _Expansion(wavelength_a)
    state.sld.append(state.sld_for(structure.fronting, 1.0))
    for component in structure.components:
        _append_component(state, component)
    state.thickness.append(0.0)
    state.limit_thickness.append(0.0)
    state.sld.append(state.sld_for(structure.backing, 1.0))
    state.roughness.append(structure.backing_roughness_a)
    thickness = np.asarray(state.thickness, dtype=float)
    roughness = np.asarray(state.roughness, dtype=float)
    _validate_roughness(np.asarray(state.limit_thickness, dtype=float), roughness)
    return SlabStack(thickness, np.asarray(state.sld, dtype=np.complex128), roughness, tuple(state.spans))
