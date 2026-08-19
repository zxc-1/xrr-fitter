"""Immutable affine-invariant sampler configuration and retained ensembles."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

import numpy as np


def readonly_array(value: object, dtype: type, field: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def pickle_values(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, item.name) for item in fields(value))


def set_array_fields(
    value: object,
    names: tuple[str, ...],
    arrays: tuple[np.ndarray, ...],
) -> None:
    for name, array in zip(names, arrays, strict=True):
        object.__setattr__(value, name, array)


def _positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class McmcConfig:
    """Affine-invariant sampling geometry and deterministic work budget."""

    walkers: int
    burn_in: int
    production_steps: int
    thin: int = 1
    stretch_scale: float = 2.0

    def __post_init__(self) -> None:
        for field in ("walkers", "production_steps", "thin"):
            _positive_integer(getattr(self, field), field)
        if not isinstance(self.burn_in, int) or isinstance(self.burn_in, bool) or self.burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer")
        if self.walkers % 2:
            raise ValueError("walkers must be even")
        if not isfinite(self.stretch_scale) or self.stretch_scale <= 1.0:
            raise ValueError("stretch_scale must be finite and greater than one")

    @classmethod
    def standard(cls, free_parameter_count: int) -> McmcConfig:
        if (
            not isinstance(free_parameter_count, int)
            or isinstance(free_parameter_count, bool)
            or free_parameter_count < 0
        ):
            raise ValueError("free_parameter_count must be a nonnegative integer")
        walkers = max(32, 2 * free_parameter_count + 2)
        if walkers % 2:
            walkers += 1
        return cls(walkers=walkers, burn_in=1000, production_steps=4000)


def _ensemble_arrays(value: EnsembleSamples) -> tuple[np.ndarray, ...]:
    return (
        readonly_array(value.samples_unit, float, "samples_unit", 3),
        readonly_array(value.log_probability, float, "log_probability", 2),
        readonly_array(value.acceptance_fraction, float, "acceptance_fraction", 1),
        readonly_array(value.split_rhat, float, "split_rhat", 1),
        readonly_array(value.effective_sample_size, float, "effective_sample_size", 1),
    )


def _validate_ensemble_axes(arrays: tuple[np.ndarray, ...]) -> None:
    draws, walkers, dimensions = arrays[0].shape
    valid = (
        draws > 0
        and dimensions > 0
        and arrays[1].shape == (draws, walkers)
        and arrays[2].shape == (walkers,)
        and arrays[3].shape == (dimensions,)
        and arrays[4].shape == (dimensions,)
    )
    if not valid:
        raise ValueError("ensemble diagnostic arrays have incompatible shapes")


@dataclass(frozen=True, slots=True)
class EnsembleSamples:
    """Retained unit-space ensemble and aligned convergence diagnostics."""

    samples_unit: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: np.ndarray
    split_rhat: np.ndarray
    effective_sample_size: np.ndarray

    def __post_init__(self) -> None:
        arrays = _ensemble_arrays(self)
        _validate_ensemble_axes(arrays)
        names = (
            "samples_unit",
            "log_probability",
            "acceptance_fraction",
            "split_rhat",
            "effective_sample_size",
        )
        set_array_fields(self, names, arrays)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), pickle_values(self)


__all__ = ["EnsembleSamples", "McmcConfig"]
