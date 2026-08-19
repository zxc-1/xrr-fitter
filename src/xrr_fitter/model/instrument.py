"""Immutable instrument declarations and deterministic resolution mapping."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, degrees, isfinite, log, pi, sqrt

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * sqrt(2.0 * log(2.0)))
RESOLUTION_KINDS = frozenset({"sigma_q_a_inv", "fwhm_q_a_inv", "sigma_two_theta_deg", "fwhm_two_theta_deg"})


def _nonempty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


def _point_indices(values: object) -> tuple[int, ...]:
    indices = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in indices):
        raise ValueError("diagnostic point indices must be nonnegative integers")
    return indices


def _validate_modes(footprint: str, background: str, resolution: str) -> None:
    if footprint not in {"geometry", "fit", "none"}:
        raise ValueError("footprint_mode must be geometry, fit, or none")
    if background not in {"constant", "linear", "powerlaw"}:
        raise ValueError("background_kind must be constant, linear, or powerlaw")
    if resolution not in {"q", "theta"}:
        raise ValueError("resolution_domain must be q or theta")


def _geometry_sizes(length: float | None, width: float | None) -> tuple[float, float]:
    if length is None or width is None or not isfinite(length) or not isfinite(width):
        raise ValueError("geometry footprint mode requires finite sample and beam sizes")
    if length <= 0.0 or width <= 0.0 or width > length:
        raise ValueError("geometry requires 0 < beam_width_mm <= sample_length_mm")
    return length, width


def _resolution_arrays(
    two_theta_deg: np.ndarray,
    resolution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    two_theta = np.asarray(two_theta_deg, dtype=float)
    values = np.asarray(resolution, dtype=float)
    if two_theta.shape != values.shape:
        raise ValueError("resolution must match two_theta_deg")
    if two_theta.ndim != 1:
        raise ValueError("resolution inputs must be one-dimensional")
    return two_theta, values


def _validate_resolution(
    kind: str,
    wavelength_a: float,
    angle_offset_deg: float,
    values: np.ndarray,
) -> None:
    if kind not in RESOLUTION_KINDS:
        raise ValueError(f"unsupported resolution kind: {kind}")
    if not isfinite(wavelength_a) or wavelength_a <= 0.0:
        raise ValueError("wavelength_a must be positive")
    if not isfinite(angle_offset_deg):
        raise ValueError("angle_offset_deg must be finite")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("resolution values must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PhysicsDiagnostic:
    """A stable diagnostic code, message, and affected point indices."""

    code: str
    message: str
    point_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.code, "diagnostic code")
        _nonempty(self.message, "diagnostic message")
        object.__setattr__(self, "point_indices", _point_indices(self.point_indices))


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Persisted geometry, footprint, background, and resolution modes."""

    instrument_id: str | None = None
    footprint_mode: str = "fit"
    footprint_spill_angle_deg: float = 0.0
    sample_length_mm: float | None = None
    beam_width_mm: float | None = None
    background_kind: str = "constant"
    resolution_domain: str = "q"

    def __post_init__(self) -> None:
        if self.instrument_id is not None:
            _nonempty(self.instrument_id, "instrument_id")
        _validate_modes(self.footprint_mode, self.background_kind, self.resolution_domain)
        angle = self.footprint_spill_angle_deg
        if not isfinite(angle) or not 0.0 <= angle <= 90.0:
            raise ValueError("footprint_spill_angle_deg must be finite and in [0, 90]")
        self._validate_footprint()

    def _validate_footprint(self) -> None:
        if self.footprint_mode == "geometry":
            self._validate_geometry()
            return
        if self.sample_length_mm is not None or self.beam_width_mm is not None:
            raise ValueError("geometry dimensions are only valid in geometry footprint mode")
        if self.footprint_mode == "none" and self.footprint_spill_angle_deg != 0.0:
            raise ValueError("none footprint mode requires a zero spill angle")

    def _validate_geometry(self) -> None:
        length, width = _geometry_sizes(self.sample_length_mm, self.beam_width_mm)
        expected = degrees(asin(width / length))
        if abs(self.footprint_spill_angle_deg - expected) > 1e-12:
            raise ValueError("geometry spill angle does not match beam/sample dimensions")


def resolution_to_sigma_q(
    two_theta_deg: np.ndarray,
    resolution: np.ndarray,
    resolution_kind: str,
    wavelength_a: float,
    angle_offset_deg: float,
) -> np.ndarray:
    two_theta, values = _resolution_arrays(two_theta_deg, resolution)
    _validate_resolution(resolution_kind, wavelength_a, angle_offset_deg, values)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        sigma = values * FWHM_TO_SIGMA if resolution_kind.startswith("fwhm") else values
    result = (
        sigma
        if resolution_kind.endswith("q_a_inv")
        else _angular_sigma_q(
            two_theta,
            sigma,
            wavelength_a,
            angle_offset_deg,
        )
    )
    output = np.array(result, dtype=float, copy=True)
    output.setflags(write=False)
    return output


def _angular_sigma_q(
    two_theta_deg: np.ndarray,
    sigma_two_theta_deg: np.ndarray,
    wavelength_a: float,
    angle_offset_deg: float,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        sigma_theta_rad = np.deg2rad(sigma_two_theta_deg / 2.0)
        theta_deg = two_theta_deg / 2.0 + angle_offset_deg
        finite = np.isfinite(theta_deg)
        result = np.full(theta_deg.shape, np.nan, dtype=float)
        theta_rad = np.deg2rad(theta_deg[finite])
        result[finite] = np.abs((4.0 * pi / wavelength_a) * np.cos(theta_rad)) * sigma_theta_rad[finite]
        unstable = finite & ~np.isfinite(result)
        if np.any(unstable):
            unstable_theta = np.deg2rad(theta_deg[unstable])
            result[unstable] = 4.0 * pi * np.abs(np.cos(unstable_theta)) * sigma_theta_rad[unstable] / wavelength_a
    if np.any(~np.isfinite(result[finite])):
        raise ValueError("converted q resolution must be finite")
    return result
