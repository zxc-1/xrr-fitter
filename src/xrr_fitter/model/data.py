"""Immutable beam declarations, column mappings, and prepared XRR data.

Prepared values retain raw provenance alongside owned read-only derived arrays.
Mask transitions create new values and never re-enable an invalid data point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, pi
from pathlib import Path
from typing import Any

import numpy as np

RESOLUTION_KINDS = frozenset(
    {
        "sigma_q_a_inv",
        "fwhm_q_a_inv",
        "sigma_two_theta_deg",
        "fwhm_two_theta_deg",
    }
)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _readonly_vector(value: Any, dtype: type, size: int, name: str) -> np.ndarray:
    vector = np.array(value, dtype=dtype, copy=True)
    if vector.shape != (size,):
        raise ValueError(f"{name} must match derived data length")
    vector.setflags(write=False)
    return vector


def _readonly(value: Any, dtype: type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_wavelengths(values: tuple[tuple[str, float], ...]) -> None:
    for name, value in values:
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")


def _column_indices(mapping: DataColumnMapping) -> tuple[int, ...]:
    values = (
        mapping.two_theta,
        mapping.intensity,
        mapping.intensity_sigma,
        mapping.resolution,
    )
    return tuple(value for value in values if value is not None)


def _validate_column_indices(indices: tuple[int, ...]) -> None:
    valid = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in indices)
    if not valid or len(indices) != len(set(indices)):
        raise ValueError("column indices must be distinct nonnegative integers")


def _validate_resolution_mapping(mapping: DataColumnMapping) -> None:
    if (mapping.resolution is None) != (mapping.resolution_kind is None):
        raise ValueError("resolution and resolution_kind must be specified together")
    if mapping.resolution_kind is not None and mapping.resolution_kind not in RESOLUTION_KINDS:
        raise ValueError(f"unsupported resolution kind: {mapping.resolution_kind}")


@dataclass(frozen=True, slots=True)
class BeamSpec:
    """Monochromatic or mixed K-alpha wavelength declaration."""

    kind: str
    wavelength_a: float = 1.5406
    wavelength_1_a: float = 1.54056
    wavelength_2_a: float = 1.54439
    intensity_ratio_21: float = 0.5

    def __post_init__(self) -> None:
        if self.kind not in {"monochromatic", "mixed_kalpha"}:
            raise ValueError(f"beam kind must be monochromatic or mixed_kalpha: {self.kind}")
        _validate_wavelengths(
            (
                ("wavelength_a", self.wavelength_a),
                ("wavelength_1_a", self.wavelength_1_a),
                ("wavelength_2_a", self.wavelength_2_a),
            )
        )
        if not isfinite(self.intensity_ratio_21) or self.intensity_ratio_21 < 0.0:
            raise ValueError("intensity_ratio_21 must be finite and nonnegative")

    @property
    def effective_wavelength_a(self) -> float:
        return self.wavelength_a if self.kind == "monochromatic" else self.wavelength_1_a


@dataclass(frozen=True, slots=True)
class DataColumnMapping:
    """Distinct source columns plus an optional resolution interpretation."""

    two_theta: int = 0
    intensity: int = 1
    intensity_sigma: int | None = None
    resolution: int | None = None
    resolution_kind: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_column_indices(_column_indices(self))
        _validate_resolution_mapping(self)


@dataclass(frozen=True, slots=True)
class DataSourceIdentity:
    """A source path bound to exact byte size and SHA-256 digest."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("source path must not be empty")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size <= 0:
            raise ValueError("source size must be a positive integer")
        if not _valid_sha256(self.sha256):
            raise ValueError("source SHA-256 must be lowercase hexadecimal")


def _prepared_groups(values: object) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(group) for group in values)


def _freeze_prepared_arrays(data: PreparedData, size: int) -> None:
    required = (
        ("two_theta_deg", data.two_theta_deg, float),
        ("intensity_raw", data.intensity_raw, float),
        ("qz_a_inv", data.qz_a_inv, float),
        ("intensity_normalized", data.intensity_normalized, float),
        ("validation_mask", data.validation_mask, bool),
        ("fit_mask", data.fit_mask, bool),
    )
    optional = (
        ("intensity_sigma_raw", data.intensity_sigma_raw),
        ("resolution_raw", data.resolution_raw),
        ("intensity_sigma_normalized", data.intensity_sigma_normalized),
        ("sigma_q_a_inv", data.sigma_q_a_inv),
    )
    for name, value, dtype in required:
        object.__setattr__(data, name, _readonly_vector(value, dtype, size, name))
    for name, value in optional:
        if value is not None:
            object.__setattr__(data, name, _readonly_vector(value, float, size, name))


def _validate_source_rows(data: PreparedData, groups: tuple[tuple[int, ...], ...]) -> None:
    if not _valid_sha256(data.source_sha256):
        raise ValueError("source_sha256 must be lowercase hexadecimal")
    if len(data.raw_rows) != len(data.raw_parse_status):
        raise ValueError("raw_rows and raw_parse_status must have equal length")
    indices = (index for group in groups for index in group)
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices):
        raise ValueError("source row groups require nonnegative integer indices")


def _validate_prepared_types(data: PreparedData) -> None:
    if not isinstance(data.beam, BeamSpec):
        raise TypeError("beam must be a BeamSpec")
    if not isinstance(data.column_mapping, DataColumnMapping):
        raise TypeError("column_mapping must be a DataColumnMapping")
    if not isinstance(data.fit_ready, bool):
        raise TypeError("fit_ready must be bool")


def _validate_prepared_scalars(data: PreparedData) -> None:
    if not isfinite(data.import_angle_offset_deg):
        raise ValueError("import_angle_offset_deg must be finite")
    if not isfinite(data.normalization) or data.normalization <= 0.0:
        raise ValueError("normalization must be positive and finite")
    if not isfinite(data.r_floor) or data.r_floor <= 0.0:
        raise ValueError("r_floor must be positive and finite")


def _validate_prepared_metadata(
    data: PreparedData,
    groups: tuple[tuple[int, ...], ...],
) -> None:
    _validate_source_rows(data, groups)
    _validate_prepared_types(data)
    if np.any(data.fit_mask & ~data.validation_mask):
        raise ValueError("fit mask cannot enable invalid points")
    _validate_prepared_scalars(data)


def _freeze_prepared_metadata(
    data: PreparedData,
    groups: tuple[tuple[int, ...], ...],
) -> None:
    object.__setattr__(data, "source_path", Path(data.source_path))
    object.__setattr__(data, "raw_rows", tuple(data.raw_rows))
    object.__setattr__(data, "raw_parse_status", tuple(data.raw_parse_status))
    object.__setattr__(data, "source_row_groups", groups)
    object.__setattr__(data, "warnings", tuple(data.warnings))


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Raw provenance and aligned immutable arrays produced by one import."""

    source_path: Path
    source_sha256: str
    raw_rows: tuple[str, ...]
    raw_parse_status: tuple[str, ...]
    source_row_groups: tuple[tuple[int, ...], ...]
    beam: BeamSpec
    import_angle_offset_deg: float
    two_theta_deg: np.ndarray
    intensity_raw: np.ndarray
    intensity_sigma_raw: np.ndarray | None
    resolution_raw: np.ndarray | None
    qz_a_inv: np.ndarray
    intensity_normalized: np.ndarray
    intensity_sigma_normalized: np.ndarray | None
    sigma_q_a_inv: np.ndarray | None
    validation_mask: np.ndarray
    fit_mask: np.ndarray
    normalization: float
    r_floor: float
    fit_ready: bool
    warnings: tuple[str, ...]
    column_mapping: DataColumnMapping

    def __post_init__(self) -> None:
        groups = _prepared_groups(self.source_row_groups)
        _freeze_prepared_arrays(self, len(groups))
        _validate_prepared_metadata(self, groups)
        _freeze_prepared_metadata(self, groups)


def qz_from_two_theta(
    two_theta_deg: np.ndarray,
    wavelength_a: float,
    angle_offset_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not isfinite(wavelength_a) or wavelength_a <= 0.0:
        raise ValueError("wavelength_a must be positive and finite")
    if not isfinite(angle_offset_deg):
        raise ValueError("angle_offset_deg must be finite")
    two_theta = np.asarray(two_theta_deg, dtype=float)
    if two_theta.ndim != 1:
        raise ValueError("two_theta_deg must be one-dimensional")
    with np.errstate(over="ignore", invalid="ignore"):
        theta_deg = two_theta / 2.0 + angle_offset_deg
    finite = np.isfinite(theta_deg)
    positive = finite & (theta_deg > 0.0) & (theta_deg <= 90.0)
    qz = np.full(theta_deg.shape, np.nan, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        converted = 4.0 * pi * np.sin(np.deg2rad(theta_deg[finite])) / wavelength_a
    converted_finite = np.isfinite(converted)
    finite_indices = np.flatnonzero(finite)
    qz[finite_indices[converted_finite]] = converted[converted_finite]
    positive[finite_indices[~converted_finite]] = False
    return _readonly(qz, float), _readonly(positive, bool)


def fit_ready(
    two_theta_deg: np.ndarray,
    qz_a_inv: np.ndarray,
    selected: np.ndarray,
) -> bool:
    angles = np.asarray(two_theta_deg, dtype=float)
    qz = np.asarray(qz_a_inv, dtype=float)
    mask = np.asarray(selected, dtype=bool)
    if angles.shape != qz.shape or angles.shape != mask.shape or angles.ndim != 1:
        raise ValueError("fit readiness arrays must be aligned one-dimensional vectors")
    if np.count_nonzero(mask) < 30 or np.unique(angles[mask]).size < 30:
        return False
    selected_qz = qz[mask]
    return bool(selected_qz.size >= 2 and np.isfinite(selected_qz).all() and np.all(np.diff(selected_qz) > 0.0))


def log_domain_mask(intensity_normalized: np.ndarray, r_floor: float) -> np.ndarray:
    if not isfinite(r_floor) or r_floor <= 0.0:
        raise ValueError("r_floor must be positive and finite")
    intensity = np.asarray(intensity_normalized, dtype=float)
    if intensity.ndim != 1:
        raise ValueError("intensity_normalized must be one-dimensional")
    return _readonly(np.isfinite(intensity) & (intensity > -r_floor), bool)


def with_fit_mask(data: PreparedData, mask: np.ndarray) -> PreparedData:
    requested = np.asarray(mask, dtype=bool)
    if requested.shape != data.fit_mask.shape:
        raise ValueError("fit mask must match derived data length")
    if np.any(requested & ~data.validation_mask):
        raise ValueError("fit mask cannot enable invalid points")
    ready = fit_ready(data.two_theta_deg, data.qz_a_inv, requested)
    return replace(data, fit_mask=requested, fit_ready=ready)
