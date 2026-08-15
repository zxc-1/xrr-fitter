"""Validation for replay-only MCMC parameter state."""

from __future__ import annotations

from math import isfinite
from numbers import Real

import numpy as np


def _validated_names(
    parameter_names: tuple[str, ...],
    derived_names: object,
) -> tuple[str, ...]:
    names = tuple(derived_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("derived parameter names must be nonempty")
    if len(names) != len(set(names)):
        raise ValueError("derived parameter names must be unique")
    if set(names).intersection(parameter_names):
        raise ValueError("derived parameter names must be separate from parameter names")
    return names


def _validated_samples(
    sample_count: int,
    names: tuple[str, ...],
    derived_samples: object,
) -> np.ndarray:
    samples = np.array(derived_samples, dtype=float, copy=True)
    if samples.ndim != 2:
        raise ValueError("derived_samples_physical must be 2-dimensional")
    if samples.shape != (sample_count, len(names)):
        raise ValueError("derived samples must align with MCMC sample rows and names")
    if np.any(~np.isfinite(samples)):
        raise ValueError("derived samples contain nonfinite values")
    samples.setflags(write=False)
    return samples


def validate_derived_samples(
    parameter_names: tuple[str, ...],
    sample_count: int,
    derived_names: object,
    derived_samples: object | None,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    """Bind deterministic target samples without extending diagnostic axes."""
    names = tuple(derived_names)
    if not names:
        if derived_samples is not None:
            raise ValueError("derived samples require derived parameter names")
        return (), None
    names = _validated_names(parameter_names, names)
    if derived_samples is None:
        raise ValueError("derived samples must be provided with derived parameter names")
    return names, _validated_samples(sample_count, names, derived_samples)


def _fixed_pair(item: object) -> tuple[object, object]:
    try:
        name, value = item
    except (TypeError, ValueError) as error:
        raise ValueError("fixed parameter values must contain name/value pairs") from error
    return name, value


def _fixed_name(value: object) -> str:
    name = value if isinstance(value, str) else ""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("fixed parameter names must be nonempty")
    return name


def _fixed_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("fixed parameter values must be finite numbers")
    return float(value)


def _fixed_parameter_value(item: object) -> tuple[str, float]:
    name, value = _fixed_pair(item)
    return _fixed_name(name), _fixed_number(value)


def validate_fixed_parameter_values(
    parameter_names: tuple[str, ...],
    derived_names: tuple[str, ...],
    values: object,
) -> tuple[tuple[str, float], ...]:
    """Normalize fixed physical coordinates kept for deterministic replay."""
    normalized = tuple(_fixed_parameter_value(item) for item in tuple(values))
    names = tuple(name for name, _value in normalized)
    if len(names) != len(set(names)):
        raise ValueError("fixed parameter names must be unique")
    if set(names).intersection((*parameter_names, *derived_names)):
        raise ValueError("fixed parameter names must be separate from sampled and derived names")
    return normalized


__all__ = ["validate_derived_samples", "validate_fixed_parameter_values"]
