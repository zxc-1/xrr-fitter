from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

import numpy as np


TRANSFORMS = frozenset({"linear", "log", "roughness_fraction"})


def _name(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


def _bounds(name: str, initial: float, lower: float, upper: float) -> None:
    if not all(isfinite(value) for value in (initial, lower, upper)):
        raise ValueError(f"{name} values must be finite")
    if lower > upper or not lower <= initial <= upper:
        raise ValueError(f"{name} initial value must be within bounds")


@dataclass(frozen=True, slots=True)
class ParameterSetting:
    name: str
    initial: float
    lower: float
    upper: float
    locked: bool = False

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        _bounds(self.name, self.initial, self.lower, self.upper)
        if not isinstance(self.locked, bool):
            raise TypeError("locked must be bool")


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    name: str
    display_name: str
    unit: str
    category: str
    initial: float
    lower: float
    upper: float
    transform: str
    locked: bool
    integer: bool = False
    expert_only: bool = False
    sharing_key: str | None = None

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        _name(self.display_name, "display_name")
        _name(self.category, "category")
        _bounds(self.name, self.initial, self.lower, self.upper)
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unsupported parameter transform: {self.transform}")
        if not all(isinstance(value, bool) for value in (self.locked, self.integer, self.expert_only)):
            raise TypeError("parameter flags must be bool")
        if self.sharing_key is not None:
            _name(self.sharing_key, "sharing_key")


def _effective_upper(
    definition: ParameterDefinition,
    dynamic_upper: float | None,
) -> float:
    """Allow candidate geometry to tighten, but never widen, declared bounds.

    Dynamic roughness bounds may carry a ``nextafter`` value immediately below
    a physical limit. Taking the exact minimum preserves that strict endpoint.
    """
    # Static metadata remains authoritative whenever geometry imposes no cap.
    # A supplied cap may tighten only; it cannot silently broaden the project.
    return definition.upper if dynamic_upper is None else min(definition.upper, dynamic_upper)


def _zero_width_locked(definition: ParameterDefinition) -> bool:
    return (definition.locked, definition.lower == definition.upper) == (True, True)


class PhysicalValueError(ValueError):
    """An expected candidate value outside its compiled physical domain."""


def _validate_physical_value(
    definition: ParameterDefinition,
    value: float,
    upper: float,
) -> None:
    """Reject invalid physical coordinates instead of clipping them.

    Clipping would make a malformed project or candidate appear to have fitted
    successfully at a boundary and would break encode/decode invertibility.
    """
    if not all((isfinite(value), value >= definition.lower, value <= upper)):
        raise PhysicalValueError(f"value outside bounds: {definition.name}")


def _log_physical_to_unit(
    definition: ParameterDefinition,
    value: float,
    upper: float,
) -> float:
    """Normalize one positive value in logarithmic physical space.

    Natural logarithms are used in both directions; their base cancels in the
    unit ratio. A dynamic upper therefore participates in the inverse exactly.
    """
    if min(definition.lower, value) <= 0.0:
        raise PhysicalValueError(f"log parameter must be positive: {definition.name}")
    return (log(value) - log(definition.lower)) / (
        log(upper) - log(definition.lower)
    )


def _validate_unit_value(definition: ParameterDefinition, unit: float) -> None:
    """Reject nonfinite or out-of-domain solver coordinates before dispatch.

    Solvers normally honor box bounds, but imported checkpoints and direct API
    callers enter here too. No clipping or tolerance is applied at this layer.
    """
    if not all((isfinite(unit), unit >= 0.0, unit <= 1.0)):
        raise ValueError(f"unit value outside [0,1]: {definition.name}")


def _log_unit_to_physical(
    definition: ParameterDefinition,
    unit: float,
    upper: float,
) -> float:
    """Decode a logarithmic coordinate while preserving exact endpoints.

    Explicit branches at zero and one avoid exponential roundoff that could
    place a decoded value just outside its persisted physical interval.
    """
    if min(definition.lower, upper - definition.lower) <= 0.0:
        raise ValueError(f"invalid log bounds: {definition.name}")
    if unit == 0.0:
        return float(definition.lower)
    if unit == 1.0:
        return float(upper)
    decoded = float(
        np.exp(log(definition.lower) + unit * (log(upper) - log(definition.lower)))
    )
    return min(max(decoded, float(definition.lower)), float(upper))


def _linear_unit_to_physical(
    definition: ParameterDefinition,
    unit: float,
    upper: float,
) -> float:
    """Decode affine and geometry-tightened roughness coordinates exactly.

    Roughness shares this interpolation after its candidate-specific upper has
    been computed. Endpoint branches preserve project and checkpoint equality.
    """
    if unit == 0.0:
        return float(definition.lower)
    if unit == 1.0:
        return float(upper)
    return float(definition.lower + unit * (upper - definition.lower))


def physical_to_unit(
    definition: ParameterDefinition,
    value: float,
    dynamic_upper: float | None = None,
) -> float:
    """Encode one legal physical value in its declared unit interval.

    A locked zero-width declaration has no meaningful inverse coordinate and
    maps to zero. All active declarations reject out-of-range values before
    dispatching to their persisted transform identifier.
    """
    upper = _effective_upper(definition, dynamic_upper)
    if _zero_width_locked(definition):
        return 0.0
    _validate_physical_value(definition, value, upper)
    if definition.transform == "log":
        return _log_physical_to_unit(definition, value, upper)
    if definition.transform in {"linear", "roughness_fraction"}:
        return (value - definition.lower) / (upper - definition.lower)
    raise ValueError(f"unknown transform: {definition.transform}")


def unit_to_physical(
    definition: ParameterDefinition,
    unit: float,
    dynamic_upper: float | None = None,
) -> float:
    """Decode one finite unit coordinate using exact endpoint semantics.

    Dispatch is restricted to the three persisted transform identifiers.
    Unknown metadata remains an error instead of silently becoming affine.
    """
    _validate_unit_value(definition, unit)
    upper = _effective_upper(definition, dynamic_upper)
    if definition.transform == "log":
        return _log_unit_to_physical(definition, unit, upper)
    if definition.transform in {"linear", "roughness_fraction"}:
        return _linear_unit_to_physical(definition, unit, upper)
    raise ValueError(f"unknown transform: {definition.transform}")


@dataclass(frozen=True, slots=True)
class ParameterCoordinate:
    parameter_index: int
    name: str
    transform: str

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_index, int) or isinstance(self.parameter_index, bool) or self.parameter_index < 0:
            raise ValueError("parameter_index must be a nonnegative integer")
        _name(self.name, "parameter name")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unsupported parameter transform: {self.transform}")


@dataclass(frozen=True, slots=True)
class ParameterValue:
    name: str
    value: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        _bounds(self.name, self.value, self.lower, self.upper)


@dataclass(frozen=True, slots=True)
class ParameterReference:
    dataset_id: str
    parameter_name: str

    def __post_init__(self) -> None:
        _name(self.dataset_id, "dataset_id")
        _name(self.parameter_name, "parameter_name")


@dataclass(frozen=True, slots=True)
class SharingRule:
    sharing_key: str
    members: tuple[ParameterReference, ...]

    def __post_init__(self) -> None:
        _name(self.sharing_key, "sharing_key")
        members = tuple(self.members)
        object.__setattr__(self, "members", members)
        if len(members) < 2:
            raise ValueError("sharing rule requires at least two members")
        if any(not isinstance(member, ParameterReference) for member in members):
            raise TypeError("sharing rule members must be ParameterReference values")
        if len(members) != len(set(members)):
            raise ValueError("sharing rule members must be unique")
