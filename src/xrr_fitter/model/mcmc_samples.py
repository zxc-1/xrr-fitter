"""Validation for replay-only MCMC parameter state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np

from xrr_fitter.model.mcmc_sampling import McmcConfig
from xrr_fitter.model.mcmc_sampling import pickle_values as _pickle_values
from xrr_fitter.model.mcmc_sampling import readonly_array as _readonly
from xrr_fitter.model.mcmc_sampling import set_array_fields as _set_array_fields


def _nonempty_unique_names(values: object, field: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not value.strip() for value in names):
        raise ValueError(f"{field} must be nonempty")
    if len(names) != len(set(names)):
        raise ValueError(f"{field} must be unique")
    return names


def _gradient_slab_counts(values: object) -> tuple[tuple[str, int], ...]:
    rows = tuple(values)
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("gradient slab counts must contain prefix and count")
        prefix, count = row
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("gradient slab count prefixes must be nonempty strings")
        if prefix in seen:
            raise ValueError("gradient slab count prefixes must be unique")
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("gradient slab counts must be positive integers")
        seen.add(prefix)
        normalized.append((prefix, int(count)))
    return tuple(normalized)


def _mcmc_arrays(value: McmcReport) -> tuple[np.ndarray, ...]:
    return (
        _readonly(value.samples_physical, float, "samples_physical", 2),
        _readonly(value.log_probability, float, "log_probability", 1),
        _readonly(value.acceptance_fraction, float, "acceptance_fraction", 1),
        _readonly(value.split_rhat, float, "split_rhat", 1),
        _readonly(value.effective_sample_size, float, "effective_sample_size", 1),
    )


def _validate_mcmc_identity(value: McmcReport) -> None:
    if not isinstance(value.config, McmcConfig):
        raise TypeError("config must be McmcConfig")
    if not isinstance(value.child_seed, int) or isinstance(value.child_seed, bool) or value.child_seed < 0:
        raise ValueError("child_seed must be a nonnegative integer")
    if not value.label.strip():
        raise ValueError("label must not be empty")
    if value.candidate_id is not None and not value.candidate_id.strip():
        raise ValueError("candidate_id must be nonempty or None")


def _validate_mcmc_axes(
    config: McmcConfig,
    names: tuple[str, ...],
    arrays: tuple[np.ndarray, ...],
) -> None:
    samples, probability, acceptance, rhat, effective = arrays
    parameter_axes = samples.shape[1] == len(names) == rhat.size == effective.size
    if not parameter_axes:
        raise ValueError("MCMC parameter axes do not match parameter names")
    if samples.shape[0] != probability.size:
        raise ValueError("MCMC sample and probability counts must match")
    if acceptance.size != config.walkers:
        raise ValueError("MCMC acceptance_fraction must match walker count")


def _validate_mcmc_values(arrays: tuple[np.ndarray, ...]) -> None:
    samples, probability, _acceptance, _rhat, _effective = arrays
    if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(probability)):
        raise ValueError("MCMC report arrays contain nonfinite values")


@dataclass(frozen=True, slots=True)
class McmcReport:
    """Physical retained samples bound to config, seed, and candidate identity."""

    config: McmcConfig
    child_seed: int
    parameter_names: tuple[str, ...]
    samples_physical: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: np.ndarray
    split_rhat: np.ndarray
    effective_sample_size: np.ndarray
    boundary_hits: tuple[str, ...]
    label: str = "目标函数伪后验"
    warnings: tuple[str, ...] = ()
    candidate_id: str | None = None
    prior_conflicts: tuple[str, ...] = ()
    derived_parameter_names: tuple[str, ...] = ()
    derived_samples_physical: np.ndarray | None = None
    fixed_parameter_values: tuple[tuple[str, float], ...] = ()
    gradient_slab_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        _validate_mcmc_identity(self)
        parameter_names = _nonempty_unique_names(
            self.parameter_names,
            "parameter names",
        )
        arrays = _mcmc_arrays(self)
        _validate_mcmc_axes(self.config, parameter_names, arrays)
        _validate_mcmc_values(arrays)
        object.__setattr__(self, "parameter_names", parameter_names)
        array_fields = (
            "samples_physical",
            "log_probability",
            "acceptance_fraction",
            "split_rhat",
            "effective_sample_size",
        )
        _set_array_fields(self, array_fields, arrays)
        object.__setattr__(self, "boundary_hits", tuple(self.boundary_hits))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "prior_conflicts", tuple(self.prior_conflicts))
        derived_names, derived = validate_derived_samples(
            parameter_names,
            arrays[0].shape[0],
            self.derived_parameter_names,
            self.derived_samples_physical,
        )
        object.__setattr__(self, "derived_parameter_names", derived_names)
        object.__setattr__(self, "derived_samples_physical", derived)
        object.__setattr__(
            self,
            "fixed_parameter_values",
            validate_fixed_parameter_values(
                parameter_names,
                derived_names,
                self.fixed_parameter_values,
            ),
        )
        object.__setattr__(
            self,
            "gradient_slab_counts",
            _gradient_slab_counts(self.gradient_slab_counts),
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


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


__all__ = [
    "McmcReport",
    "validate_derived_samples",
    "validate_fixed_parameter_values",
]
