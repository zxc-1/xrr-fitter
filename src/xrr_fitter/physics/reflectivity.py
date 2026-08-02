"""Instrument composition for one- or two-wavelength reflectivity."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.structure import SlabStack
from xrr_fitter.physics.footprint import footprint_factor
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.resolution import gaussian_smear, theta_domain_smear


def qz_from_theta_deg(theta_deg: np.ndarray, wavelength_a: float) -> np.ndarray:
    return 4.0 * np.pi * np.sin(np.deg2rad(theta_deg)) / wavelength_a


def _validate_resolution(domain: str, relative: float, absolute: float, sigma_theta: float, point: np.ndarray | None) -> None:
    if domain == "theta":
        if relative != 0.0 or absolute != 0.0 or point is not None:
            raise ValueError("q-domain and theta-domain resolution cannot be combined")
    elif domain != "q" or sigma_theta != 0.0:
        raise ValueError("resolution_domain must be q or theta with exclusive parameters")


def _single_wavelength(
    theta: np.ndarray,
    stack: SlabStack,
    wavelength: float,
    domain: str,
    relative: float,
    absolute: float,
    sigma_theta: float,
    point_sigma: np.ndarray | None,
    callback: Callable[[PhysicsDiagnostic], None] | None,
    emit_warning: bool,
) -> np.ndarray:
    _validate_resolution(domain, relative, absolute, sigma_theta, point_sigma)
    if domain == "theta":
        return theta_domain_smear(
            theta,
            lambda query: parratt_reflectivity(qz_from_theta_deg(query, wavelength), stack),
            sigma_theta,
            callback,
            emit_warning=emit_warning,
        )
    qz = qz_from_theta_deg(theta, wavelength)
    return gaussian_smear(
        qz,
        lambda query: parratt_reflectivity(query, stack),
        relative,
        absolute,
        point_sigma,
        callback,
        emit_warning=emit_warning,
    )


def _validate_scalars(scale: float, background: float, linear: float, amplitude: float, exponent: float) -> None:
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    if not np.isfinite(background) or background < 0.0:
        raise ValueError("background must be finite and nonnegative")
    if not np.isfinite(linear):
        raise ValueError("linear background must be finite")
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("powerlaw_background_amplitude must be finite and nonnegative")
    if not np.isfinite(exponent) or not 1.0 <= exponent <= 4.0:
        raise ValueError("powerlaw_background_exponent must be in [1,4]")


def _background(qz: np.ndarray, constant: float, linear: float, amplitude: float, exponent: float) -> np.ndarray:
    sampled = constant + linear * qz
    if amplitude > 0.0:
        sampled = sampled + amplitude * qz ** (-exponent)
    if np.any(sampled < 0.0):
        raise ValueError("sampled background must be nonnegative")
    return sampled


def instrument_reflectivity(
    theta_deg: np.ndarray,
    primary_stack: SlabStack,
    beam: BeamSpec,
    secondary_stack: SlabStack | None = None,
    scale: float = 1.0,
    background: float = 0.0,
    linear_background_per_a_inv: float = 0.0,
    powerlaw_background_amplitude: float = 0.0,
    powerlaw_background_exponent: float = 3.0,
    footprint_spill_angle_deg: float = 0.0,
    resolution_domain: str = "q",
    relative_sigma: float = 0.0,
    absolute_sigma_a_inv: float = 0.0,
    sigma_theta_deg: float = 0.0,
    sigma_q_a_inv: np.ndarray | None = None,
    secondary_sigma_q_a_inv: np.ndarray | None = None,
    diagnostic_callback: Callable[[PhysicsDiagnostic], None] | None = None,
    *,
    emit_warning: bool = True,
) -> np.ndarray:
    """Apply resolution, beam mixing, scale/footprint, then background."""
    theta = np.asarray(theta_deg, dtype=float)
    if np.any(~np.isfinite(theta)) or np.any(theta <= 0.0):
        raise ValueError("theta_deg must be finite and positive")
    _validate_scalars(scale, background, linear_background_per_a_inv, powerlaw_background_amplitude, powerlaw_background_exponent)
    resolution = (resolution_domain, relative_sigma, absolute_sigma_a_inv, sigma_theta_deg)
    if beam.kind == "monochromatic":
        if secondary_stack is not None or secondary_sigma_q_a_inv is not None:
            raise ValueError("secondary_stack/resolution is invalid for monochromatic beam")
        smeared = _single_wavelength(
            theta,
            primary_stack,
            beam.wavelength_a,
            *resolution,
            sigma_q_a_inv,
            diagnostic_callback,
            emit_warning,
        )
    else:
        if secondary_stack is None:
            raise ValueError("secondary_stack is required for mixed_kalpha")
        primary = _single_wavelength(
            theta,
            primary_stack,
            beam.wavelength_1_a,
            *resolution,
            sigma_q_a_inv,
            diagnostic_callback,
            emit_warning,
        )
        secondary = _single_wavelength(
            theta,
            secondary_stack,
            beam.wavelength_2_a,
            *resolution,
            secondary_sigma_q_a_inv,
            diagnostic_callback,
            emit_warning,
        )
        smeared = (primary + beam.intensity_ratio_21 * secondary) / (1.0 + beam.intensity_ratio_21)
    qz = qz_from_theta_deg(theta, beam.effective_wavelength_a)
    sampled_background = _background(qz, background, linear_background_per_a_inv, powerlaw_background_amplitude, powerlaw_background_exponent)
    return scale * smeared * footprint_factor(theta, footprint_spill_angle_deg) + sampled_background
