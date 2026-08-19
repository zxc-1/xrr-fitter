"""Physical search bounds for periodic drift scale parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, nextafter

from xrr_fitter.model.parameters import ParameterDefinition
from xrr_fitter.model.structure import PeriodicBlock

BoundaryPredicate = Callable[[float], bool]


@dataclass(slots=True)
class _Bounds:
    lower: float
    upper: float


def _strict_boundary(
    coefficient: float,
    threshold: float,
    direction: float,
    predicate: BoundaryPredicate,
) -> float:
    """Find a representable strict boundary for the runtime expression."""
    boundary = (threshold - 1.0) / coefficient
    if not isfinite(boundary):
        return boundary
    candidate = nextafter(boundary, direction)
    for _ in range(256):
        if predicate(candidate):
            return candidate
        candidate = nextafter(candidate, direction)
    nudge = max(1.0, abs(boundary)) * 1e-12
    candidate = boundary + (nudge if direction > boundary else -nudge)
    if predicate(candidate):
        return candidate
    raise ValueError("drift_scale has no representable strict physical boundary")


def _constant_factor_valid(threshold: float, *, upper: bool, strict: bool) -> bool:
    if upper:
        return 1.0 < threshold if strict else 1.0 <= threshold
    return 1.0 > threshold if strict else 1.0 >= threshold


def _factor_boundary(
    coefficient: float,
    threshold: float,
    *,
    strict: bool,
    direction: float,
    predicate: BoundaryPredicate | None,
    default_predicate: BoundaryPredicate,
) -> float:
    boundary = (threshold - 1.0) / coefficient
    if not strict:
        return boundary
    return _strict_boundary(
        coefficient,
        threshold,
        direction,
        predicate or default_predicate,
    )


def _require_factor(
    bounds: _Bounds,
    coefficient: float,
    threshold: float,
    *,
    strict: bool,
    predicate: BoundaryPredicate | None = None,
) -> None:
    """Intersect ``1 + coefficient * scale >=|> threshold``."""
    if coefficient == 0.0:
        if not _constant_factor_valid(threshold, upper=False, strict=strict):
            bounds.lower = bounds.upper
        return
    direction = float("inf") if coefficient > 0.0 else float("-inf")
    boundary = _factor_boundary(
        coefficient,
        threshold,
        strict=strict,
        direction=direction,
        predicate=predicate,
        default_predicate=lambda scale: 1.0 + coefficient * scale > threshold,
    )
    if coefficient > 0.0:
        bounds.lower = max(bounds.lower, boundary)
    else:
        bounds.upper = min(bounds.upper, boundary)


def _require_factor_upper(
    bounds: _Bounds,
    coefficient: float,
    threshold: float,
    *,
    strict: bool,
    predicate: BoundaryPredicate | None = None,
) -> None:
    """Intersect ``1 + coefficient * scale <=|< threshold``."""
    if coefficient == 0.0:
        if not _constant_factor_valid(threshold, upper=True, strict=strict):
            bounds.lower = bounds.upper
        return
    direction = float("-inf") if coefficient > 0.0 else float("inf")
    boundary = _factor_boundary(
        coefficient,
        threshold,
        strict=strict,
        direction=direction,
        predicate=predicate,
        default_predicate=lambda scale: 1.0 + coefficient * scale < threshold,
    )
    if coefficient > 0.0:
        bounds.upper = min(bounds.upper, boundary)
    else:
        bounds.lower = max(bounds.lower, boundary)


def _previous_thickness(
    thicknesses: tuple[float, ...],
    layer_index: int,
) -> float:
    return thicknesses[-1] if layer_index == 0 else thicknesses[layer_index - 1]


def _limit_thickness_drift(
    bounds: _Bounds,
    coefficient: float,
    previous_coefficient: float,
    layer_index: int,
    thicknesses: tuple[float, ...],
    roughness: float,
) -> None:
    thickness = thicknesses[layer_index]
    _require_factor(bounds, coefficient, 2.0 / thickness, strict=False)
    if roughness <= 0.0:
        return
    _require_factor(
        bounds,
        coefficient,
        roughness / (0.49 * thickness),
        strict=True,
        predicate=lambda scale: roughness < 0.49 * (thickness * (1.0 + coefficient * scale)),
    )
    previous_thickness = _previous_thickness(thicknesses, layer_index)
    adjacent_coefficient = previous_coefficient if layer_index == 0 else coefficient
    _require_factor(
        bounds,
        adjacent_coefficient,
        roughness / (0.49 * previous_thickness),
        strict=True,
        predicate=lambda scale: roughness < 0.49 * (previous_thickness * (1.0 + adjacent_coefficient * scale)),
    )


def _limit_roughness_drift(
    bounds: _Bounds,
    coefficient: float,
    layer_index: int,
    thicknesses: tuple[float, ...],
    roughness: float,
) -> None:
    if roughness <= 0.0:
        return
    _require_factor(bounds, coefficient, 0.0, strict=False)
    thickness = thicknesses[layer_index]
    cap = 0.49 * min(thickness, _previous_thickness(thicknesses, layer_index))
    _require_factor_upper(
        bounds,
        coefficient,
        cap / roughness,
        strict=True,
        predicate=lambda scale: roughness * (1.0 + coefficient * scale) < cap,
    )


def _limit_layer_drift(
    bounds: _Bounds,
    family: str,
    coefficient: float,
    previous_coefficient: float,
    layer_index: int,
    thicknesses: tuple[float, ...],
    roughness: float,
) -> None:
    if family == "thickness_a":
        _limit_thickness_drift(
            bounds,
            coefficient,
            previous_coefficient,
            layer_index,
            thicknesses,
            roughness,
        )
        return
    _limit_roughness_drift(bounds, coefficient, layer_index, thicknesses, roughness)


def drift_scale_bounds(
    block: PeriodicBlock,
    prefix: str,
    family: str,
    definitions: list[ParameterDefinition],
    coefficients: tuple[float, ...],
) -> tuple[float, float]:
    """Intersect the configured scale range with legal per-copy factors."""
    assert block.drift is not None
    bounds = _Bounds(min(-0.5, block.drift.amount), max(0.5, block.drift.amount))
    by_name = {definition.name: definition for definition in definitions}
    thicknesses = tuple(by_name[f"{prefix}.layer.{index}.thickness_a"].initial for index in range(len(block.layers)))
    for repeat_index, coefficient in enumerate(coefficients[1:], start=1):
        for layer_index in range(len(block.layers)):
            roughness = by_name[f"{prefix}.layer.{layer_index}.roughness_a"].initial
            _limit_layer_drift(
                bounds,
                family,
                coefficient,
                coefficients[repeat_index - 1],
                layer_index,
                thicknesses,
                roughness,
            )
    if bounds.lower >= bounds.upper or not bounds.lower <= block.drift.amount <= bounds.upper:
        raise ValueError(f"drift_scale initial value is outside the physical repeat domain: {block.name}")
    return bounds.lower, bounds.upper
