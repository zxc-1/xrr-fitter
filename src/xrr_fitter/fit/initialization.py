"""Deterministic feature detection and data-derived initial hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.materials import material_sld


@dataclass(frozen=True, slots=True)
class InitialCandidates:
    """Versioned candidate axes derived from one prepared curve.

    Values retain deterministic ordering because downstream pool construction
    protects early declared and data-derived hypotheses before applying caps.
    """

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


@dataclass(frozen=True, slots=True)
class StructureEvidence:
    """Observed thickness modes compared with independent model geometry.

    Peak positions are published in ascending physical-thickness order so the
    evidence is stable across equivalent FFT ranking ties.
    """

    m_data: int
    m_model: int
    warning: str | None
    peak_positions_a: tuple[float, ...]


def _validate_max_candidates(max_candidates: int) -> None:
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, (int, np.integer))
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be a positive integer")


def _uniform_feature_view(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a validated curve onto one uniform, detrended q grid.

    All spectral estimators consume this exact representation so disagreement
    reflects the estimators rather than different sampling or detrending.
    """
    qz = np.asarray(qz_a_inv, dtype=float)
    values = np.asarray(transformed, dtype=float)
    if qz.ndim != 1 or qz.size < 16 or values.shape != qz.shape:
        raise ValueError("feature arrays must be equal vectors with at least 16 points")
    if np.any(~np.isfinite(qz)) or np.any(~np.isfinite(values)):
        raise ValueError("feature arrays must be finite and qz strictly increasing")
    if np.any(np.diff(qz) <= 0.0):
        raise ValueError("feature qz must be strictly increasing")
    uniform_qz = np.linspace(qz[0], qz[-1], qz.size)
    uniform_values = np.interp(uniform_qz, qz, values)
    detrended = signal.detrend(uniform_values)
    return uniform_qz, detrended, float(uniform_qz[1] - uniform_qz[0])


def spectral_thickness_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
    max_candidates: int = 8,
) -> np.ndarray:
    """Rank physical thicknesses by peaks in the Hann-windowed q spectrum.

    The returned values are sorted physically after peak-strength truncation;
    this keeps candidate order independent of FFT-bin traversal direction.
    """
    _validate_max_candidates(max_candidates)
    uniform_qz, detrended, delta_q = _uniform_feature_view(qz_a_inv, transformed)
    windowed = detrended * signal.windows.hann(detrended.size, sym=False)
    spectrum = np.abs(np.fft.rfft(windowed))
    thickness = 2.0 * np.pi * np.fft.rfftfreq(uniform_qz.size, delta_q)
    valid_indices = np.flatnonzero((thickness >= 2.0) & (thickness <= 2e5))
    peaks, _ = signal.find_peaks(spectrum[valid_indices])
    selected = valid_indices[peaks]
    ranked = selected[np.argsort(spectrum[selected])[::-1]][:max_candidates]
    return np.sort(thickness[ranked])


def autocorrelation_thickness_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
    max_candidates: int = 8,
) -> np.ndarray:
    """Infer thicknesses from unbiased positive-lag autocorrelation peaks.

    Dividing by the available sample count at each lag avoids systematically
    suppressing longer lags before prominence ranking.
    """
    _validate_max_candidates(max_candidates)
    uniform_qz, detrended, delta_q = _uniform_feature_view(qz_a_inv, transformed)
    correlation = signal.fftconvolve(detrended, detrended[::-1], mode="full")
    correlation = correlation[uniform_qz.size - 1 :]
    correlation /= np.arange(uniform_qz.size, 0, -1)
    peaks, properties = signal.find_peaks(
        correlation[1:],
        prominence=max(float(np.ptp(correlation)) * 0.01, np.finfo(float).eps),
    )
    lags = peaks + 1
    thickness = 2.0 * np.pi / (lags * delta_q)
    valid = np.isfinite(thickness) & (thickness >= 2.0) & (thickness <= 2e5)
    ranked = np.argsort(properties["prominences"][valid])[::-1][:max_candidates]
    return np.sort(thickness[valid][ranked])


def kiessig_spacing_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
    max_candidates: int = 8,
) -> np.ndarray:
    """Convert adjacent smoothed-fringe peak spacings into film thicknesses.

    Pair strength is limited by the weaker adjacent peak, preventing a single
    strong feature from promoting an unsupported spacing.
    """
    _validate_max_candidates(max_candidates)
    uniform_qz, detrended, _ = _uniform_feature_view(qz_a_inv, transformed)
    window = min(51, uniform_qz.size // 2 * 2 - 1)
    smooth = signal.savgol_filter(detrended, window, 3)
    peaks, properties = signal.find_peaks(
        smooth,
        prominence=max(float(np.ptp(smooth)) * 0.02, np.finfo(float).eps),
    )
    if peaks.size < 2:
        return np.empty(0, dtype=float)
    thickness = 2.0 * np.pi / np.diff(uniform_qz[peaks])
    valid = np.isfinite(thickness) & (thickness >= 2.0) & (thickness <= 2e5)
    strength = np.minimum(
        properties["prominences"][:-1],
        properties["prominences"][1:],
    )
    ranked = np.argsort(strength[valid])[::-1][:max_candidates]
    return np.sort(thickness[valid][ranked])


def _bragg_period_candidates(
    qz_a_inv: np.ndarray,
    transformed: np.ndarray,
) -> tuple[float, ...]:
    uniform_qz, detrended, _ = _uniform_feature_view(qz_a_inv, transformed)
    window = min(51, uniform_qz.size // 2 * 2 - 1)
    smooth = signal.savgol_filter(detrended, window, 3)
    peaks, properties = signal.find_peaks(
        smooth,
        prominence=max(float(np.ptp(smooth)) * 0.03, np.finfo(float).eps),
    )
    if peaks.size < 2:
        return ()
    periods = 2.0 * np.pi / np.diff(uniform_qz[peaks])
    strengths = np.minimum(
        properties["prominences"][:-1],
        properties["prominences"][1:],
    )
    valid = np.isfinite(periods) & (periods >= 2.0) & (periods <= 2e5)
    ranked = np.argsort(strengths[valid])[::-1][:8]
    return tuple(float(value) for value in np.sort(periods[valid][ranked]))


def critical_edge_candidates(
    coordinate: np.ndarray,
    reflectivity: np.ndarray,
    max_candidates: int = 4,
) -> tuple[float, ...]:
    """Find low-angle edge candidates from log-reflectivity curvature.

    Invalid or undersampled inputs intentionally produce no feature evidence;
    the caller then activates the declared bounded offset fallback.
    """
    _validate_max_candidates(max_candidates)
    positions = np.asarray(coordinate, dtype=float)
    values = np.asarray(reflectivity, dtype=float)
    invalid = (
        positions.shape != values.shape
        or positions.ndim != 1
        or positions.size < 11
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(positions) <= 0.0)
    )
    if invalid:
        return ()
    low_count = max(11, int(np.ceil(0.30 * positions.size)))
    low_positions = positions[:low_count]
    safe = np.log10(np.maximum(values[:low_count], np.finfo(float).tiny))
    window = min(31, low_count // 2 * 2 - 1)
    smooth = signal.savgol_filter(safe, window, 3)
    curvature = np.abs(np.gradient(np.gradient(smooth, low_positions), low_positions))
    peaks, _ = signal.find_peaks(curvature, distance=max(2, low_count // 40))
    if not peaks.size:
        return ()
    ranked = peaks[np.argsort(curvature[peaks])[::-1]][:max_candidates]
    return tuple(float(value) for value in np.sort(low_positions[ranked]))


_DIRECT_SLD_ANCHORS = tuple(
    value * 1e-6 for value in (-20.0, 0.0, 10.0, 20.0, 40.0, 80.0, 120.0)
)


def critical_sld_candidates(
    data: PreparedData,
    structure: StructureSpec,
) -> tuple[float, ...]:
    """Combine fixed SLD anchors with one bounded critical-edge estimate."""
    del structure
    mask = data.fit_mask
    edges = critical_edge_candidates(
        data.qz_a_inv[mask],
        data.intensity_normalized[mask],
    )
    candidates = set(_DIRECT_SLD_ANCHORS)
    if edges:
        estimate = edges[0] ** 2 / (16.0 * np.pi)
        candidates.add(float(np.clip(estimate, -150e-6, 150e-6)))
    return tuple(sorted(candidates))


def _direct_sld_paths(structure: StructureSpec) -> tuple[tuple[str, float], ...]:
    paths: list[tuple[str, float]] = []
    for component_index, component in enumerate(structure.components):
        if isinstance(component, LayerSpec):
            layers = ((f"component.{component_index}", component),)
        elif isinstance(component, PeriodicBlock):
            layers = tuple(
                (f"component.{component_index}.layer.{layer_index}", layer)
                for layer_index, layer in enumerate(component.layers)
            )
        else:
            layers = ()
        paths.extend(
            (f"{prefix}.sld_real_a2", layer.material.sld_override_a2.real)
            for prefix, layer in layers
            if layer.material.sld_override_a2 is not None
        )
    if structure.backing.sld_override_a2 is not None:
        paths.append(("backing.sld_real_a2", structure.backing.sld_override_a2.real))
    return tuple(paths)


def direct_sld_start_rows(
    structure: StructureSpec,
    candidates: tuple[float, ...],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Build at most eight stable direct-SLD hypotheses without a product."""
    declared = _direct_sld_paths(structure)
    if not declared:
        return ()
    estimate = next(
        (value for value in candidates if value not in _DIRECT_SLD_ANCHORS),
        candidates[0],
    )
    rows = [
        declared,
        tuple((name, estimate) for name, _value in declared),
    ]
    rows.extend(
        tuple(
            (name, _DIRECT_SLD_ANCHORS[(rotation + index) % len(_DIRECT_SLD_ANCHORS)])
            for index, (name, _value) in enumerate(declared)
        )
        for rotation in range(6)
    )
    return tuple(dict.fromkeys(rows))[:8]


def ramp_inflection_estimate_deg(data: PreparedData) -> float | None:
    mask = data.fit_mask & np.isfinite(data.intensity_normalized)
    theta = data.two_theta_deg[mask] / 2.0 + data.import_angle_offset_deg
    intensity = data.intensity_normalized[mask]
    positive = (theta > 0.0) & (intensity > 0.0)
    theta = theta[positive]
    intensity = intensity[positive]
    if theta.size < 20:
        return None
    count = min(theta.size, max(20, int(np.ceil(0.30 * theta.size))))
    theta = theta[:count]
    values = np.log10(np.maximum(intensity[:count], data.r_floor))
    window = min(11, theta.size // 2 * 2 - 1)
    smooth = signal.savgol_filter(values, window, 2)
    slope = np.gradient(smooth, theta)
    index = int(np.argmax(slope))
    if not (3 <= index <= slope.size - 5 and slope[index] > 0.0):
        return None
    rising_fraction = float(np.mean(np.diff(smooth[: index + 1]) >= 0.0))
    plateau_slope = float(np.median(np.abs(slope[index + 2 :])))
    if rising_fraction < 0.75 or plateau_slope > 0.25 * slope[index]:
        return None
    return float(theta[index])


def footprint_angle_candidates(
    data: PreparedData,
    instrument: InstrumentSpec,
    critical_angle_deg: float | None,
    ramp_inflection_deg: float | None,
) -> tuple[float, ...]:
    if not isinstance(data, PreparedData):
        raise TypeError("data must be PreparedData")
    if not isinstance(instrument, InstrumentSpec):
        raise TypeError("instrument must be InstrumentSpec")
    if instrument.footprint_mode in {"geometry", "none"}:
        return (instrument.footprint_spill_angle_deg,)
    for candidate in (ramp_inflection_deg, critical_angle_deg):
        if candidate is not None and np.isfinite(candidate) and candidate > 0.0:
            return 0.0, float(candidate)
    return (0.0,)


def _merge_candidates(
    groups: tuple[np.ndarray | tuple[float, ...], ...],
    relative_tolerance: float = 0.02,
) -> tuple[float, ...]:
    values = sorted(
        float(item)
        for group in groups
        for item in group
        if np.isfinite(item) and item > 0.0
    )
    clusters: list[list[float]] = []
    for value in values:
        for cluster in clusters:
            median = float(np.median(cluster))
            if abs(value - median) <= relative_tolerance * median:
                cluster.append(value)
                break
        else:
            clusters.append([value])
    return tuple(float(np.median(cluster)) for cluster in clusters)


def _has_reliable_thickness_feature(
    spectral: np.ndarray,
    autocorrelation: np.ndarray,
    spacing: np.ndarray,
    spectral_resolution_a: float,
) -> bool:
    if spacing.size == 0:
        return False
    sorted_spacing = np.sort(spacing)
    if spacing.size > 1 and np.any(
        np.diff(sorted_spacing) <= 0.02 * sorted_spacing[:-1]
    ):
        return True
    for candidate in spacing:
        fft_tolerance = max(0.02 * candidate, 0.5 * spectral_resolution_a)
        if spectral.size and np.any(np.abs(spectral - candidate) <= fft_tolerance):
            return True
        if autocorrelation.size and np.any(
            np.abs(autocorrelation - candidate) <= 0.02 * candidate
        ):
            return True
    return False


def _observable_thickness_bounds(qz_a_inv: np.ndarray) -> tuple[float, float, float, float]:
    observed_min = max(2.0, 2.0 * np.pi / np.ptp(qz_a_inv))
    observed_max = min(2e5, np.pi / (2.0 * np.median(np.diff(qz_a_inv))))
    return (
        observed_min,
        observed_max,
        max(2.0, 0.25 * observed_min),
        min(2e5, 4.0 * observed_max),
    )


def _critical_angle_hypotheses_deg(
    structure: StructureSpec,
    wavelength_a: float,
    density_scales: tuple[float, ...],
) -> tuple[float, ...]:
    first = structure.components[0] if structure.components else None
    if isinstance(first, PeriodicBlock):
        surface_layers = first.layers[:1]
    elif isinstance(first, LayerSpec):
        surface_layers = (first,)
    else:
        surface_layers = ()
    material_scales = tuple(
        (layer.material, density)
        for layer in surface_layers
        for density in density_scales
    ) + ((structure.backing, 1.0),)
    fronting_sld = material_sld(structure.fronting, 1.0, wavelength_a).real
    hypotheses: list[float] = []
    for material, density in material_scales:
        delta_sld = material_sld(material, density, wavelength_a).real - fronting_sld
        if delta_sld <= 0.0:
            continue
        argument = np.sqrt(16.0 * np.pi * delta_sld) * wavelength_a / (4.0 * np.pi)
        if 0.0 < argument < 1.0:
            hypotheses.append(float(np.rad2deg(np.arcsin(argument))))
    return tuple(sorted(set(hypotheses)))


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
    """Combine three thickness estimators and apply the wide fallback.

    Agreement between independent estimators is required before a narrow grid
    is trusted; otherwise the observable hard range receives eight log points.
    """
    transformed = qz**4 * np.maximum(observed + r_floor, r_floor)
    spectral = spectral_thickness_candidates(qz, transformed)
    autocorrelation = autocorrelation_thickness_candidates(qz, transformed)
    spacing = kiessig_spacing_candidates(qz, transformed)
    thickness = _merge_candidates((spectral, autocorrelation, spacing))
    periods = _bragg_period_candidates(qz, transformed)
    reliable = _has_reliable_thickness_feature(
        spectral,
        autocorrelation,
        spacing,
        2.0 * np.pi / np.ptp(qz),
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
    """Pair observed and theoretical edges into bounded angle offsets.

    The persisted import offset is added after feature pairing so it remains a
    protected hypothesis even when every feature-derived offset hits a bound.
    """
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
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Estimate protected scale and tail-background candidates.

    The low-q 95th percentile avoids a single plateau spike, while the high-q
    median is retained both raw and capped for a conservative background start.
    """
    low_count = min(qz.size, max(20, int(np.ceil(0.10 * qz.size))))
    scale = float(np.clip(np.percentile(observed[:low_count], 95), 1e-3, 1e3))
    high_count = max(1, int(np.ceil(0.20 * qz.size)))
    high_median = max(0.0, float(np.median(observed[-high_count:])))
    protected_background = min(high_median, 0.1)
    return tuple(sorted({1.0, scale})), tuple(sorted({0.0, protected_background, high_median}))


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
        qz, observed, data.r_floor
    )
    angle_offsets, critical_angle, angle_warnings = _angle_offset_hypotheses(
        data, structure, observed
    )
    warnings = [*thickness_warnings, *angle_warnings]
    footprint_angles = footprint_angle_candidates(
        data,
        instrument,
        critical_angle,
        ramp_inflection_estimate_deg(data),
    )
    periods = _periodic_period_hypotheses(structure, periods, thickness, warnings)
    scales, backgrounds = _scale_background_hypotheses(qz, observed)
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


def _independent_thickness_dof(structure: StructureSpec) -> int:
    return sum(
        len(component.layers) if isinstance(component, PeriodicBlock) else 1
        for component in structure.components
        if isinstance(component, (LayerSpec, GradientLayerSpec, PeriodicBlock))
    )


def structure_evidence(
    data: PreparedData,
    structure: StructureSpec,
) -> StructureEvidence:
    """Count resolvable thickness modes against independent structure DOF.

    FFT peaks must exceed a robust median-plus-MAD threshold and remain more
    than one spectral resolution apart before contributing evidence.
    """
    mask = data.fit_mask
    qz = data.qz_a_inv[mask]
    observed = data.intensity_normalized[mask]
    model_count = _independent_thickness_dof(structure)
    warning_text = "结构复杂度超过数据可分辨的厚度尺度，部分厚度参数预期不可辨识"
    if qz.size < 16 or np.ptp(qz) <= 0.0:
        return StructureEvidence(0, model_count, warning_text if model_count > 1 else None, ())
    transformed = qz**4 * np.maximum(observed + data.r_floor, data.r_floor)
    uniform_qz, detrended, delta_q = _uniform_feature_view(qz, transformed)
    spectrum = np.abs(
        np.fft.rfft(detrended * signal.windows.hann(detrended.size, sym=False))
    )
    thickness = 2.0 * np.pi * np.fft.rfftfreq(uniform_qz.size, delta_q)
    baseline_values = spectrum[1:]
    baseline = float(np.median(baseline_values))
    mad = float(np.median(np.abs(baseline_values - baseline)))
    threshold = baseline + 5.0 * max(mad, np.finfo(float).eps)
    valid = (thickness >= 2.0) & (thickness <= 2e5)
    peaks, properties = signal.find_peaks(spectrum, height=threshold)
    heights = properties["peak_heights"]
    keep = valid[peaks]
    peaks = peaks[keep]
    heights = heights[keep]
    resolution_a = 2.0 * np.pi / np.ptp(qz)
    accepted: list[float] = []
    for index in peaks[np.argsort(heights)[::-1]]:
        value = float(thickness[index])
        if all(abs(value - previous) > resolution_a for previous in accepted):
            accepted.append(value)
    accepted.sort()
    warning = warning_text if model_count > len(accepted) + 1 else None
    return StructureEvidence(len(accepted), model_count, warning, tuple(accepted))
