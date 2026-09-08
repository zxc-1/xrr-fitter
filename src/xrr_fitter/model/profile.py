"""Immutable profile traces with explicit total-objective threshold semantics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from xrr_fitter.model.fitting import _pickle_values


def _readonly(value: object, dtype: type, field: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _evidence_count(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class ParameterProfile:
    """Profile coordinates, objective evidence, and closure flags."""

    name: str
    values: np.ndarray
    objectives: np.ndarray
    lower_closed: bool
    upper_closed: bool
    interval_kind: str = "loss_support"
    confidence_level: float | None = None
    method: str = "objective_tolerance"
    unavailable_reason: str | None = None
    delta_total: float | None = None
    objective_point_count: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name must not be empty")
        values = _readonly(self.values, float, "profile values", 1)
        objectives = _readonly(self.objectives, float, "profile objectives", 1)
        if values.shape != objectives.shape:
            raise ValueError("profile values and objectives must have the same shape")
        if values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError("profile arrays contain invalid values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "objectives", objectives)
        self._validate_interval()

    def _validate_interval(self) -> None:
        if self.interval_kind not in {"loss_support", "likelihood_ratio", "unavailable"}:
            raise ValueError("unsupported profile interval_kind")
        if not self.method:
            raise ValueError("profile method must not be empty")
        self._validate_threshold()
        self._validate_confidence_level()

    def _validate_threshold(self) -> None:
        _evidence_count(self.objective_point_count, "objective_point_count")
        if self.objective_point_count == 0:
            raise ValueError("objective_point_count must be positive")
        if self.delta_total is not None and (not isfinite(self.delta_total) or self.delta_total <= 0.0):
            raise ValueError("profile delta_total must be positive and finite")

    def _validate_confidence_level(self) -> None:
        if self.interval_kind == "likelihood_ratio":
            if self.confidence_level != 0.95 or self.delta_total is None or self.unavailable_reason is not None:
                raise ValueError("likelihood profile requires a calibrated threshold and confidence level")
        elif self.confidence_level is not None:
            raise ValueError("support profiles cannot declare a confidence level")

    @property
    def objective_delta(self) -> float | None:
        return None if self.delta_total is None else self.delta_total / self.objective_point_count

    @property
    def successful_samples(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.objectives)))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)
