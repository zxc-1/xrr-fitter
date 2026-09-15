"""Immutable calibrated bootstrap samples with complete failed-replicate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from xrr_fitter.model.bootstrap_intervals import (
    LINEAR_INTERVAL_METHOD,
    BootstrapIntervalMetadata,
    validated_interval_ranks,
)
from xrr_fitter.model.fitting import _pickle_values

MIN_BOOTSTRAP_SUCCESS = 200


def bootstrap_calibration_reason(successful_samples: int, failure_rate: float) -> str | None:
    if failure_rate > 0.20:
        return "excessive_fit_failures"
    if successful_samples < MIN_BOOTSTRAP_SUCCESS:
        return "insufficient_successful_samples"
    return None


def _readonly(value: object, dtype: type, field: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _evidence_count(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _bootstrap_names(values: object) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not isinstance(value, str) or not value.strip() for value in names):
        raise ValueError("bootstrap parameter names must contain nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError("bootstrap parameter names must be unique")
    return names


def _bootstrap_intervals(
    names: tuple[str, ...],
    values: object,
) -> tuple[tuple[str, float, float], ...]:
    """Normalize optional bounds without erasing a gated empty result."""
    intervals = tuple(values)
    if not intervals:
        return ()
    valid_rows = all(isinstance(value, (tuple, list)) and len(value) == 3 for value in intervals)
    if not valid_rows:
        raise ValueError("bootstrap intervals must contain name, lower, and upper")
    interval_names = tuple(value[0] for value in intervals)
    if interval_names != names:
        raise ValueError("bootstrap interval names must match parameter names in order")
    bounds = _bootstrap_interval_bounds(intervals)
    return tuple((name, lower, upper) for name, (lower, upper) in zip(interval_names, bounds, strict=True))


def _bootstrap_interval_bounds(
    intervals: tuple[object, ...],
) -> tuple[tuple[float, float], ...]:
    try:
        bounds = tuple((float(value[1]), float(value[2])) for value in intervals)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("bootstrap interval bounds must be finite numbers") from error
    invalid = any(not isfinite(lower) or not isfinite(upper) or lower > upper for lower, upper in bounds)
    if invalid:
        raise ValueError("bootstrap interval bounds must be finite and ordered")
    return bounds


def _bootstrap_samples(names: tuple[str, ...], values: object) -> np.ndarray:
    samples = _readonly(values, float, "bootstrap samples", 2)
    if samples.shape[1] != len(names):
        raise ValueError("bootstrap samples do not match parameter names")
    if np.any(~np.isfinite(samples)):
        raise ValueError("bootstrap samples must be finite")
    return samples


def _validate_bootstrap_failure_rate(value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("failure_rate must be in [0, 1]")


def _validate_bootstrap_owner(
    candidate_id: str | None,
    provenance_sha256: str | None,
) -> None:
    """Require a complete candidate/provenance pair when ownership is sealed."""
    if (candidate_id is None) != (provenance_sha256 is None):
        raise ValueError("bootstrap candidate_id and provenance_sha256 must be paired")
    if candidate_id is None:
        return
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("bootstrap candidate_id must be nonempty or None")
    if not isinstance(provenance_sha256, str):
        raise ValueError("bootstrap provenance_sha256 must be a lowercase SHA-256")
    valid = len(provenance_sha256) == 64 and all(value in "0123456789abcdef" for value in provenance_sha256)
    if not valid:
        raise ValueError("bootstrap provenance_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class BootstrapResult(BootstrapIntervalMetadata):
    """Successful physical bootstrap samples and interval summary."""

    parameter_names: tuple[str, ...]
    samples: np.ndarray
    intervals: tuple[tuple[str, float, float], ...]
    failure_rate: float
    attempted_count: int
    failure_reasons: tuple[tuple[int, str], ...] = ()
    method: str = "custom_resampling"
    unavailable_reason: str | None = None
    candidate_id: str | None = None
    provenance_sha256: str | None = None
    diagnostic_unavailable_reason: str | None = None
    joint_owner_sha256: str | None = None
    interval_method: str = LINEAR_INTERVAL_METHOD
    interval_ranks: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        names = _bootstrap_names(self.parameter_names)
        samples = _bootstrap_samples(names, self.samples)
        _validate_bootstrap_failure_rate(self.failure_rate)
        intervals = _bootstrap_intervals(names, self.intervals)
        _validate_bootstrap_owner(self.candidate_id, self.provenance_sha256)
        if self.joint_owner_sha256 is not None:
            _validate_bootstrap_owner(self.candidate_id, self.joint_owner_sha256)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "intervals", intervals)
        self._validate_evidence()

    def _validate_evidence(self) -> None:
        _evidence_count(self.attempted_count, "attempted_count")
        failures = self._validated_failures()
        self._validate_counts(len(failures))
        self._validate_calibration()
        object.__setattr__(self, "interval_ranks", validated_interval_ranks(self))
        object.__setattr__(self, "failure_reasons", failures)

    def _validated_failures(self) -> tuple[tuple[int, str], ...]:
        failures = tuple(tuple(row) for row in self.failure_reasons)
        indices = []
        for index, reason in failures:
            _evidence_count(index, "failed sample index")
            if index >= self.attempted_count or not isinstance(reason, str) or not reason:
                raise ValueError("invalid bootstrap failure reason")
            indices.append(index)
        if indices != sorted(set(indices)) or len(failures) + self.successful_samples != self.attempted_count:
            raise ValueError("bootstrap attempt counts must match successful and failed samples")
        return failures

    def _validate_counts(self, failures: int) -> None:
        if self.attempted_count == 0 or not isfinite(self.failure_rate):
            raise ValueError("bootstrap must contain at least one attempted sample")
        if self.failure_rate != failures / self.attempted_count:
            raise ValueError("bootstrap failure_rate must match attempt counts")

    def _validate_calibration(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("bootstrap method must not be empty")
        diagnostic_reason = self.diagnostic_unavailable_reason
        if diagnostic_reason is not None and (not isinstance(diagnostic_reason, str) or not diagnostic_reason.strip()):
            raise ValueError("bootstrap diagnostic reason must be a nonempty string or None")
        sampling_reason = bootstrap_calibration_reason(self.successful_samples, self.failure_rate)
        reason = sampling_reason or diagnostic_reason
        if self.unavailable_reason != reason or bool(self.intervals) != (sampling_reason is None):
            raise ValueError("bootstrap intervals and unavailable_reason must match calibration gates")

    @property
    def successful_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def interval_kind(self) -> str:
        if self.intervals and self.diagnostic_unavailable_reason is None:
            return "percentile_bootstrap"
        return "exploratory_bootstrap" if self.successful_samples and self.failure_rate <= 0.20 else "unavailable"

    @property
    def confidence_level(self) -> float | None:
        return 0.95 if self.interval_kind == "percentile_bootstrap" else None

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)
