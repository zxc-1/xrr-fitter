from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


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
