"""Deterministic expansion of structure declarations into slab stacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite

import numpy as np

from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    PeriodicSpan,
    SlabStack,
    StructureComponent,
    StructureSpec,
)
from xrr_fitter.physics.materials import material_sld


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


def _append_layer(state: _Expansion, layer: LayerSpec, roughness: float | None = None) -> None:
    state.thickness.append(layer.thickness_a)
    state.limit_thickness.append(layer.thickness_a)
    state.sld.append(state.sld_for(layer.material, layer.density_scale))
    state.roughness.append(layer.roughness_a if roughness is None else roughness)


def _append_periodic(state: _Expansion, block: PeriodicBlock) -> None:
    start = len(state.thickness)
    for repeat_index in range(block.repeats):
        for layer_index, layer in enumerate(block.layers):
            override = block.top_roughness_a if repeat_index == layer_index == 0 else None
            _append_layer(state, layer, override)
    if block.repeats > 1:
        state.spans.append(PeriodicSpan(start, len(block.layers), block.repeats))


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
            raise ValueError(f"interface.{interface}.roughness_a must be below {limit:g} A")


def expand_structure(structure: StructureSpec, wavelength_a: float) -> SlabStack:
    """Expand fronting, declared components, and backing at one wavelength."""
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
