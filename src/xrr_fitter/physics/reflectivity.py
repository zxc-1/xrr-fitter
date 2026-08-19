"""Instrument composition for one- or two-wavelength reflectivity."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.physics.footprint import footprint_factor
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.resolution import gaussian_smear, theta_domain_smear


def qz_from_theta_deg(theta_deg: np.ndarray, wavelength_a: float) -> np.ndarray:
    if not np.isfinite(wavelength_a) or wavelength_a <= 0.0:
        raise ValueError("wavelength_a must be positive and finite")
    theta = np.asarray(theta_deg, dtype=float)
    if np.any(~np.isfinite(theta)):
        raise ValueError("theta_deg must be finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        qz = 4.0 * np.pi * np.sin(np.deg2rad(theta)) / wavelength_a
    if np.any(~np.isfinite(qz)):
        raise ValueError("qz conversion must be finite")
    return qz


def _mixed_kalpha_average(
    primary: np.ndarray,
    secondary: np.ndarray,
    intensity_ratio_21: float,
) -> np.ndarray:
    """Average two wavelength responses without overflowing before division."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            result = (primary + intensity_ratio_21 * secondary) / (1.0 + intensity_ratio_21)
    except FloatingPointError:
        denominator = 1.0 + intensity_ratio_21
        secondary_weight = intensity_ratio_21 / denominator
        primary_weight = 1.0 - secondary_weight
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            result = primary_weight * primary + secondary_weight * secondary
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("mixed K-alpha average is nonfinite")
    return result


def _validate_resolution(
    domain: str, relative: float, absolute: float, sigma_theta: float, point: np.ndarray | None
) -> None:
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


def _validated_theta(theta_deg: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float)
    if np.any(~np.isfinite(theta)) or np.any((theta <= 0.0) | (theta > 90.0)):
        raise ValueError("theta_deg must be finite, positive, and at most 90 degrees")
    return theta


def _validate_beam_stacks(
    beam: BeamSpec,
    secondary_stack: SlabStack | None,
    secondary_sigma_q_a_inv: np.ndarray | None,
) -> None:
    if beam.kind == "monochromatic" and (secondary_stack is not None or secondary_sigma_q_a_inv is not None):
        raise ValueError("secondary_stack/resolution is invalid for monochromatic beam")
    if beam.kind != "monochromatic" and secondary_stack is None:
        raise ValueError("secondary_stack is required for mixed_kalpha")


def _smeared_reflectivity(
    theta: np.ndarray,
    primary_stack: SlabStack,
    beam: BeamSpec,
    secondary_stack: SlabStack | None,
    resolution: tuple[str, float, float, float],
    sigma_q_a_inv: np.ndarray | None,
    secondary_sigma_q_a_inv: np.ndarray | None,
    callback: Callable[[PhysicsDiagnostic], None] | None,
    emit_warning: bool,
) -> np.ndarray:
    _validate_beam_stacks(beam, secondary_stack, secondary_sigma_q_a_inv)
    if beam.kind == "monochromatic":
        return _single_wavelength(
            theta,
            primary_stack,
            beam.wavelength_a,
            *resolution,
            sigma_q_a_inv,
            callback,
            emit_warning,
        )
    assert secondary_stack is not None
    primary = _single_wavelength(
        theta,
        primary_stack,
        beam.wavelength_1_a,
        *resolution,
        sigma_q_a_inv,
        callback,
        emit_warning,
    )
    secondary = _single_wavelength(
        theta,
        secondary_stack,
        beam.wavelength_2_a,
        *resolution,
        secondary_sigma_q_a_inv,
        callback,
        emit_warning,
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _mixed_kalpha_average(primary, secondary, beam.intensity_ratio_21)
    except FloatingPointError as error:
        raise FloatingPointError("instrument reflectivity arithmetic is nonfinite") from error


def _powerlaw_term(qz: np.ndarray, amplitude: float, exponent: float) -> np.ndarray:
    """Evaluate ``amplitude * q**(-exponent)`` without an overflowing power."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        result = amplitude * qz ** (-exponent)
    unstable = ~np.isfinite(result)
    if np.any(unstable):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            result[unstable] = np.exp(np.log(amplitude) - exponent * np.log(qz[unstable]))
    return result


def _background(qz: np.ndarray, constant: float, linear: float, amplitude: float, exponent: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sampled = constant + linear * qz
        if amplitude > 0.0:
            sampled = sampled + _powerlaw_term(qz, amplitude, exponent)
    if np.any(~np.isfinite(sampled)):
        raise ValueError("sampled background must be finite")
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
    theta = _validated_theta(theta_deg)
    _validate_scalars(
        scale, background, linear_background_per_a_inv, powerlaw_background_amplitude, powerlaw_background_exponent
    )
    resolution = (resolution_domain, relative_sigma, absolute_sigma_a_inv, sigma_theta_deg)
    smeared = _smeared_reflectivity(
        theta,
        primary_stack,
        beam,
        secondary_stack,
        resolution,
        sigma_q_a_inv,
        secondary_sigma_q_a_inv,
        diagnostic_callback,
        emit_warning,
    )
    qz = qz_from_theta_deg(theta, beam.effective_wavelength_a)
    sampled_background = _background(
        qz, background, linear_background_per_a_inv, powerlaw_background_amplitude, powerlaw_background_exponent
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            result = scale * smeared * footprint_factor(theta, footprint_spill_angle_deg) + sampled_background
    except FloatingPointError as error:
        raise FloatingPointError("instrument reflectivity arithmetic is nonfinite") from error
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("instrument reflectivity must be finite")
    return result
