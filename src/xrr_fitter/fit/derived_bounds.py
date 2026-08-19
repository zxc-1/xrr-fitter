"""Synchronize generated drift targets with edited base declarations."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from xrr_fitter.model.parameters import ParameterDefinition, ParameterSetting


def _derived_coordinate_parts(name: str) -> tuple[str, str, int] | None:
    marker = ".repeat."
    if marker not in name:
        return None
    prefix, suffix = name.split(marker, 1)
    parts = suffix.split(".")
    if len(parts) != 4 or parts[1] != "layer":
        return None
    repeat_index = _positive_repeat_index(parts[0])
    if repeat_index is None:
        return None
    return f"{prefix}.layer.{parts[2]}.{parts[3]}", prefix, repeat_index


def _positive_repeat_index(value: str) -> int | None:
    try:
        index = int(value)
    except ValueError:
        return None
    return index if index >= 1 else None


def _drift_factors(scale: ParameterDefinition, repeat_index: int) -> tuple[float, ...]:
    coefficient = float(repeat_index)
    return (
        1.0 + coefficient * scale.lower,
        1.0 + coefficient * scale.upper,
        1.0 - coefficient * scale.lower,
        1.0 - coefficient * scale.upper,
    )


def _derived_products(
    base: ParameterDefinition,
    scale: ParameterDefinition,
    repeat_index: int,
) -> tuple[float, ...]:
    factors = _drift_factors(scale, repeat_index)
    products = tuple(base_value * factor for base_value in (base.lower, base.upper) for factor in factors)
    return products if all(isfinite(value) for value in products) else (base.lower, base.upper, base.initial)


def _synchronized_definition(
    definition: ParameterDefinition,
    by_name: dict[str, ParameterDefinition],
    changed_names: set[str],
) -> ParameterDefinition:
    parts = _derived_coordinate_parts(definition.name)
    if parts is None:
        return definition
    base_name, prefix, repeat_index = parts
    scale_name = f"{prefix}.drift_scale"
    base = by_name.get(base_name)
    scale = by_name.get(scale_name)
    if base is None or scale is None or (base_name not in changed_names and scale_name not in changed_names):
        return definition
    products = _derived_products(base, scale, repeat_index)
    physical_lower = 2.0 if definition.transform == "log" else 0.0
    return replace(
        definition,
        initial=base.initial,
        lower=max(physical_lower, min(products)),
        upper=max(physical_lower, base.initial, max(products)),
    )


def synchronize_derived_bounds(
    definitions: tuple[ParameterDefinition, ...],
    settings: tuple[ParameterSetting, ...],
) -> tuple[ParameterDefinition, ...]:
    """Keep generated drift targets inside their effective expression domain."""
    by_name = {definition.name: definition for definition in definitions}
    changed_names = {setting.name for setting in settings}
    return tuple(_synchronized_definition(definition, by_name, changed_names) for definition in definitions)
