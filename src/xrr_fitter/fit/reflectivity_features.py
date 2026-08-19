"""Reflectivity-edge, instrument, and direct-SLD initialization features."""

from __future__ import annotations

import numpy as np
from scipy import signal

from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec
from xrr_fitter.physics.materials import material_sld

DIRECT_SLD_ANCHORS = (
    -20.0 * 1e-6,
    0.0 * 1e-6,
    10.0 * 1e-6,
    20.0 * 1e-6,
    40.0 * 1e-6,
    80.0 * 1e-6,
    120.0 * 1e-6,
)


def _validate_max_candidates(max_candidates: int) -> None:
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")


def _validated_edge_inputs(
    coordinate: np.ndarray,
    reflectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
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
    return None if invalid else (positions, values)


def critical_edge_candidates(
    coordinate: np.ndarray,
    reflectivity: np.ndarray,
    max_candidates: int = 4,
) -> tuple[float, ...]:
    """Find low-angle edge candidates from log-reflectivity curvature."""
    _validate_max_candidates(max_candidates)
    inputs = _validated_edge_inputs(coordinate, reflectivity)
    if inputs is None:
        return ()
    positions, values = inputs
    low_count = max(11, int(np.ceil(0.30 * positions.size)))
    low_positions = positions[:low_count]
    if low_positions[-1] - low_positions[0] < np.sqrt(np.finfo(float).tiny):
        return ()
    safe = np.log10(np.maximum(values[:low_count], np.finfo(float).tiny))
    window = min(31, low_count // 2 * 2 - 1)
    smooth = signal.savgol_filter(safe, window, 3)
    curvature = np.abs(np.gradient(np.gradient(smooth, low_positions), low_positions))
    peaks, _ = signal.find_peaks(curvature, distance=max(2, low_count // 40))
    if not peaks.size:
        return ()
    ranked = peaks[np.argsort(curvature[peaks])[::-1]][:max_candidates]
    return tuple(float(value) for value in np.sort(low_positions[ranked]))


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
    candidates = set(DIRECT_SLD_ANCHORS)
    if edges:
        estimate = edges[0] ** 2 / (16.0 * np.pi)
        candidates.add(float(np.clip(estimate, -150e-6, 150e-6)))
    return tuple(sorted(candidates))


def _component_layers(
    component_index: int,
    component: object,
) -> tuple[tuple[str, LayerSpec], ...]:
    if isinstance(component, LayerSpec):
        return ((f"component.{component_index}", component),)
    if isinstance(component, PeriodicBlock):
        return tuple(
            (f"component.{component_index}.layer.{layer_index}", layer)
            for layer_index, layer in enumerate(component.layers)
        )
    return ()


def _direct_sld_paths(structure: StructureSpec) -> tuple[tuple[str, float], ...]:
    paths: list[tuple[str, float]] = []
    for component_index, component in enumerate(structure.components):
        layers = _component_layers(component_index, component)
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
        (value for value in candidates if value not in DIRECT_SLD_ANCHORS),
        candidates[0],
    )
    rows = [
        declared,
        tuple((name, estimate) for name, _value in declared),
    ]
    rows.extend(
        tuple(
            (name, DIRECT_SLD_ANCHORS[(rotation + index) % len(DIRECT_SLD_ANCHORS)])
            for index, (name, _value) in enumerate(declared)
        )
        for rotation in range(6)
    )
    return tuple(dict.fromkeys(rows))[:8]


def _ramp_inputs(data: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    mask = data.fit_mask & np.isfinite(data.intensity_normalized)
    theta = data.two_theta_deg[mask] / 2.0 + data.import_angle_offset_deg
    intensity = data.intensity_normalized[mask]
    positive = (theta > 0.0) & (intensity > 0.0)
    return theta[positive], intensity[positive]


def ramp_inflection_estimate_deg(data: PreparedData) -> float | None:
    theta, intensity = _ramp_inputs(data)
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
    candidate = _first_positive_angle(ramp_inflection_deg, critical_angle_deg)
    if candidate is not None:
        return 0.0, candidate
    return (0.0,)


def _first_positive_angle(*values: float | None) -> float | None:
    return next(
        (float(value) for value in values if value is not None and np.isfinite(value) and value > 0.0),
        None,
    )


def _surface_layers(structure: StructureSpec) -> tuple[LayerSpec, ...]:
    first = structure.components[0] if structure.components else None
    if isinstance(first, PeriodicBlock):
        return first.layers[:1]
    if isinstance(first, LayerSpec):
        return (first,)
    return ()


def critical_angle_hypotheses_deg(
    structure: StructureSpec,
    wavelength_a: float,
    density_scales: tuple[float, ...],
) -> tuple[float, ...]:
    surface_layers = _surface_layers(structure)
    material_scales = tuple((layer.material, density) for layer in surface_layers for density in density_scales) + (
        (structure.backing, 1.0),
    )
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


__all__ = [
    "critical_angle_hypotheses_deg",
    "critical_edge_candidates",
    "critical_sld_candidates",
    "direct_sld_start_rows",
    "footprint_angle_candidates",
    "ramp_inflection_estimate_deg",
]
