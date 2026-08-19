"""Validation for fixed-topology gradient parameter bounds."""

from __future__ import annotations

from math import ceil

from xrr_fitter.model.parameters import ParameterDefinition
from xrr_fitter.model.structure import (
    MAX_EXPANDED_SLABS,
    GradientLayerSpec,
    LayerSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.transitions import transition_width


def component_slab_count(component: LayerSpec | PeriodicBlock | GradientLayerSpec) -> int:
    """Count declared finite slabs for the non-gradient budget reservation."""
    if isinstance(component, GradientLayerSpec):
        return 0
    if isinstance(component, LayerSpec):
        if component.transition is None:
            return 1
        return ceil(transition_width(component.transition) / component.transition.microslab_max_a) + 1
    return component.repeats * len(component.layers)


def _gradient_slab_count(
    by_name: dict[str, ParameterDefinition],
    index: int,
    component: GradientLayerSpec,
) -> int:
    prefix = f"component.{index}"
    thickness = by_name[f"{prefix}.thickness_a"]
    microslab = by_name[f"{prefix}.microslab_max_a"]
    fixed = (microslab.initial, microslab.lower, microslab.upper)
    if not microslab.locked or fixed != (component.microslab_max_a,) * 3:
        raise ValueError(f"microslab topology parameter must remain locked: {prefix}.microslab_max_a")
    if thickness.lower < microslab.initial:
        raise ValueError(f"gradient thickness lower bound must cover microslab maximum: {prefix}.thickness_a")
    return ceil(thickness.upper / microslab.initial)


def validate_gradient_modes(
    definitions: tuple[ParameterDefinition, ...],
    structure: StructureSpec,
) -> None:
    """Keep compiled gradient bounds inside one fixed expanded topology budget."""
    by_name = {definition.name: definition for definition in definitions}
    total = sum(component_slab_count(component) for component in structure.components)
    total += sum(
        _gradient_slab_count(by_name, index, component)
        for index, component in enumerate(structure.components)
        if isinstance(component, GradientLayerSpec)
    )
    if total > MAX_EXPANDED_SLABS:
        raise ValueError("gradient parameter bounds exceed expanded slab budget")
