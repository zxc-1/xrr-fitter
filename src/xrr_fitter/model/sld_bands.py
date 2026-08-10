"""Immutable SLD uncertainty band values replayed from retained samples.

The band value object lives apart from the wider analysis contracts because it
owns quantile and depth-axis invariants that nothing else in that module shares,
and because ``model/analysis.py`` already sits close to the maintainability
gate; keeping the band here holds both files inside it.

This module contains one data contract only. It does not sample posteriors,
evaluate reflectivity, or read files: replay belongs to the analysis layer and
band rendering belongs to plotting. Construction copies every NumPy value and
restores read-only ownership after each pickle round trip.

Rows follow the quantile axis and columns follow one shared depth grid, so a
band can be drawn without consulting the sampler that produced it. Real and
imaginary envelopes stay separate throughout: absorption carries a large share
of the X-ray density contrast, so collapsing the two into a complex modulus
would erase evidence the depth profile exists to expose.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

import numpy as np


def _readonly(value: object, dtype: type, field: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _pickle_values(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, item.name) for item in fields(value))


def _positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _validate_quantile_axis(levels: tuple[float, ...]) -> None:
    if not levels:
        raise ValueError("quantiles must not be empty")
    if list(levels) != sorted(set(levels)):
        raise ValueError("quantiles must be sorted and unique")
    if not all(0.0 < value < 1.0 for value in levels):
        raise ValueError("quantiles must lie in (0, 1)")


def _validate_band_counts(value: object) -> None:
    total = value.total_samples
    count = value.sample_count
    _positive_integer(total, "total_samples")
    _positive_integer(count, "sample_count")
    if count > total:
        raise ValueError("sample_count must not exceed total_samples")
    if not isfinite(value.failure_rate) or not 0.0 <= value.failure_rate <= 1.0:
        raise ValueError("failure_rate must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SldUncertaintyBands:
    """Depth-aligned SLD quantile envelopes replayed from retained samples."""

    depth_a: np.ndarray
    quantiles: tuple[float, ...]
    real: np.ndarray
    imaginary: np.ndarray
    align_label: str
    sample_count: int
    total_samples: int
    failure_rate: float

    def __post_init__(self) -> None:
        levels = tuple(float(value) for value in self.quantiles)
        _validate_quantile_axis(levels)
        depth = _readonly(self.depth_a, float, "depth_a", 1)
        real = _readonly(self.real, float, "real", 2)
        imaginary = _readonly(self.imaginary, float, "imaginary", 2)
        expected = (len(levels), depth.size)
        if real.shape != expected or imaginary.shape != expected:
            raise ValueError("band arrays must be quantile-by-depth")
        _validate_band_counts(self)
        object.__setattr__(self, "quantiles", levels)
        object.__setattr__(self, "depth_a", depth)
        object.__setattr__(self, "real", real)
        object.__setattr__(self, "imaginary", imaginary)

    def caption(self) -> str:
        # The quantile text is literal rather than derived: the two published
        # bands are a fixed design decision, so deriving it would emit
        # meaningless wording if the quantile tuple were ever customized.
        return (
            f"对齐 {self.align_label}；抽样 {self.sample_count}/{self.total_samples}；带为 16–84% 与 2.5–97.5% 分位区间"
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)
