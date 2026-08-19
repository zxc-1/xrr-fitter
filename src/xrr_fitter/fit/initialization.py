"""Deterministic assembly of data-derived initial hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xrr_fitter.fit.background_bounds import background_upper
from xrr_fitter.fit.reflectivity_features import (
    critical_angle_hypotheses_deg as _critical_angle_hypotheses_deg,
)
from xrr_fitter.fit.reflectivity_features import (
    critical_edge_candidates as critical_edge_candidates,
)
from xrr_fitter.fit.reflectivity_features import (
    critical_sld_candidates as critical_sld_candidates,
)
from xrr_fitter.fit.reflectivity_features import (
    direct_sld_start_rows as direct_sld_start_rows,
)
from xrr_fitter.fit.reflectivity_features import (
    footprint_angle_candidates as footprint_angle_candidates,
)
from xrr_fitter.fit.reflectivity_features import (
    ramp_inflection_estimate_deg as ramp_inflection_estimate_deg,
)
from xrr_fitter.fit.structure_evidence import (
    StructureEvidence as StructureEvidence,
)
from xrr_fitter.fit.structure_evidence import structure_evidence as structure_evidence
from xrr_fitter.fit.thickness_features import (
    autocorrelation_thickness_candidates as autocorrelation_thickness_candidates,
)
from xrr_fitter.fit.thickness_features import (
    bounded_inverse_length as _bounded_inverse_length,
)
from xrr_fitter.fit.thickness_features import (
    bragg_period_candidates as _bragg_period_candidates,
)
from xrr_fitter.fit.thickness_features import (
    has_reliable_thickness_feature as _has_reliable_thickness_feature,
)
from xrr_fitter.fit.thickness_features import (
    kiessig_spacing_candidates as kiessig_spacing_candidates,
)
from xrr_fitter.fit.thickness_features import merge_candidates as _merge_candidates
from xrr_fitter.fit.thickness_features import (
    observable_thickness_bounds as _observable_thickness_bounds,
)
from xrr_fitter.fit.thickness_features import (
    scaled_qz4_with_floor as _scaled_qz4_with_floor,
)
from xrr_fitter.fit.thickness_features import (
    spectral_thickness_candidates as spectral_thickness_candidates,
)
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import PeriodicBlock, StructureSpec


@dataclass(frozen=True, slots=True)
class InitialCandidates:
    """Versioned candidate axes derived from one prepared curve."""

    thickness_a: tuple[float, ...]
    period_a: tuple[float, ...]
    layer_fractions: tuple[float, ...]
    density_scales: tuple[float, ...]
    roughness_fractions: tuple[float, ...]
    angle_offsets_deg: tuple[float, ...]
    scales: tuple[float, ...]
    backgrounds: tuple[float, ...]
    relative_resolutions: tuple[float, ...]
    footprint_angles_deg: tuple[float, ...]
    direct_sld_rows: tuple[tuple[tuple[str, float], ...], ...]
    warnings: tuple[str, ...] = ()


def _validate_initialization_inputs(
    data: object,
    structure: object,
    instrument: object,
    rng: object,
) -> None:
    if not isinstance(data, PreparedData):
        raise TypeError("data must be PreparedData")
    if not isinstance(structure, StructureSpec):
        raise TypeError("structure must be StructureSpec")
    if not isinstance(instrument, InstrumentSpec):
        raise TypeError("instrument must be InstrumentSpec")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")


def _thickness_hypotheses(
    qz: np.ndarray,
    observed: np.ndarray,
    r_floor: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[str, ...]]:
    """Combine three thickness estimators and apply the wide fallback."""
    transformed = _scaled_qz4_with_floor(qz, observed, r_floor)
    spectral = spectral_thickness_candidates(qz, transformed)
    autocorrelation = autocorrelation_thickness_candidates(qz, transformed)
    spacing = kiessig_spacing_candidates(qz, transformed)
    thickness = _merge_candidates((spectral, autocorrelation, spacing))
    periods = _bragg_period_candidates(qz, transformed)
    reliable = _has_reliable_thickness_feature(
        spectral,
        autocorrelation,
        spacing,
        _bounded_inverse_length(2.0 * np.pi, float(np.ptp(qz))),
    )
    if reliable:
        return thickness, periods, ()
    _, _, hard_min, hard_max = _observable_thickness_bounds(qz)
    fallback = tuple(float(value) for value in np.geomspace(hard_min, hard_max, 8))
    return fallback, (), ("初始特征不足",)


def _angle_offset_hypotheses(
    data: PreparedData,
    structure: StructureSpec,
    observed: np.ndarray,
) -> tuple[tuple[float, ...], float | None, tuple[str, ...]]:
    """Pair observed and theoretical edges into bounded angle offsets."""
    mask = data.fit_mask
    observed_edges = critical_edge_candidates(data.two_theta_deg[mask] / 2.0, observed)
    theoretical_edges = _critical_angle_hypotheses_deg(
        structure,
        data.beam.effective_wavelength_a,
        (0.75, 0.85, 0.95, 1.0),
    )
    if observed_edges and theoretical_edges:
        offsets = tuple(
            sorted(
                {
                    float(np.clip(theoretical - edge, -0.1, 0.1))
                    for edge in observed_edges
                    for theoretical in theoretical_edges
                }
            )
        )
        critical_angle = theoretical_edges[0]
        warnings = ()
    else:
        offsets = (-0.02, 0.0, 0.02)
        critical_angle = None
        warnings = ("初始特征不足",)
    return tuple(sorted({*offsets, data.import_angle_offset_deg})), critical_angle, warnings


def _periodic_period_hypotheses(
    structure: StructureSpec,
    periods: tuple[float, ...],
    thickness: tuple[float, ...],
    warnings: list[str],
) -> tuple[float, ...]:
    """Reuse thickness evidence only when a periodic model lacks Bragg peaks."""
    periodic = any(isinstance(component, PeriodicBlock) for component in structure.components)
    if periodic and not periods:
        if "初始特征不足" not in warnings:
            warnings.append("初始特征不足")
        return thickness
    return periods


def _scale_background_hypotheses(
    qz: np.ndarray,
    observed: np.ndarray,
    maximum_background: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Estimate protected scale and tail-background candidates."""
    low_count = min(qz.size, max(20, int(np.ceil(0.10 * qz.size))))
    scale = float(np.clip(np.percentile(observed[:low_count], 95), 1e-3, 1e3))
    high_count = max(1, int(np.ceil(0.20 * qz.size)))
    tail = observed[-high_count:]
    with np.errstate(over="ignore", invalid="ignore"):
        high_median = float(np.median(tail))
    if not np.isfinite(high_median):
        tail_scale = float(np.max(np.abs(tail)))
        if tail_scale == 0.0 or not np.isfinite(tail_scale):
            raise ValueError("high-angle intensities must be finite")
        high_median = tail_scale * float(np.median(tail / tail_scale))
    high_median = min(maximum_background, max(0.0, high_median))
    protected_background = min(high_median, 0.1)
    scales = tuple(sorted({1.0, scale}))
    backgrounds = tuple(sorted({0.0, protected_background, high_median}))
    return scales, backgrounds


def estimate_initial_candidates(
    data: PreparedData,
    structure: StructureSpec,
    instrument: InstrumentSpec,
    rng: np.random.Generator,
) -> InitialCandidates:
    """Derive deterministic feature grids from one prepared data snapshot."""
    _validate_initialization_inputs(data, structure, instrument, rng)
    mask = data.fit_mask
    qz = data.qz_a_inv[mask]
    observed = data.intensity_normalized[mask]
    if qz.size < 16:
        raise ValueError("at least 16 fitted points are required for feature extraction")
    thickness, periods, thickness_warnings = _thickness_hypotheses(
        qz,
        observed,
        data.r_floor,
    )
    angle_offsets, critical_angle, angle_warnings = _angle_offset_hypotheses(
        data,
        structure,
        observed,
    )
    warnings = [*thickness_warnings, *angle_warnings]
    footprint_angles = footprint_angle_candidates(
        data,
        instrument,
        critical_angle,
        ramp_inflection_estimate_deg(data),
    )
    periods = _periodic_period_hypotheses(structure, periods, thickness, warnings)
    scales, backgrounds = _scale_background_hypotheses(
        qz,
        observed,
        background_upper(data, instrument),
    )
    return InitialCandidates(
        thickness_a=tuple(float(value) for value in thickness),
        period_a=tuple(float(value) for value in periods),
        layer_fractions=(0.20, 0.35, 0.50, 0.65, 0.80),
        density_scales=(0.75, 0.85, 0.95, 1.0),
        roughness_fractions=(0.02, 0.05, 0.10),
        angle_offsets_deg=angle_offsets,
        scales=scales,
        backgrounds=backgrounds,
        relative_resolutions=(0.0, 0.002, 0.005, 0.01, 0.02),
        footprint_angles_deg=footprint_angles,
        direct_sld_rows=direct_sld_start_rows(
            structure,
            critical_sld_candidates(data, structure),
        ),
        warnings=tuple(warnings),
    )


__all__ = [
    "InitialCandidates",
    "StructureEvidence",
    "autocorrelation_thickness_candidates",
    "critical_edge_candidates",
    "critical_sld_candidates",
    "direct_sld_start_rows",
    "estimate_initial_candidates",
    "footprint_angle_candidates",
    "kiessig_spacing_candidates",
    "ramp_inflection_estimate_deg",
    "spectral_thickness_candidates",
    "structure_evidence",
]
