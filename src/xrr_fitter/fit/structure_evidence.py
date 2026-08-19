"""Resolvable thickness-mode evidence for candidate structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from xrr_fitter.fit.thickness_features import (
    scaled_qz4_with_floor,
    span_cannot_resolve_bounded_thickness,
    uniform_feature_view,
)
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    PeriodicBlock,
    StructureSpec,
)


@dataclass(frozen=True, slots=True)
class StructureEvidence:
    """Observed thickness modes compared with independent model geometry."""

    m_data: int
    m_model: int
    warning: str | None
    peak_positions_a: tuple[float, ...]


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
    """Count resolvable thickness modes against independent structure DOF."""
    mask = data.fit_mask
    qz = data.qz_a_inv[mask]
    observed = data.intensity_normalized[mask]
    model_count = _independent_thickness_dof(structure)
    warning_text = "结构复杂度超过数据可分辨的厚度尺度，部分厚度参数预期不可辨识"
    if qz.size < 16 or np.ptp(qz) <= 0.0:
        return StructureEvidence(
            0,
            model_count,
            warning_text if model_count > 1 else None,
            (),
        )
    transformed = scaled_qz4_with_floor(qz, observed, data.r_floor)
    uniform_qz, detrended, delta_q = uniform_feature_view(qz, transformed)
    if span_cannot_resolve_bounded_thickness(uniform_qz):
        return StructureEvidence(
            0,
            model_count,
            warning_text if model_count > 1 else None,
            (),
        )
    spectrum = np.abs(np.fft.rfft(detrended * signal.windows.hann(detrended.size, sym=False)))
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


__all__ = ["StructureEvidence", "structure_evidence"]
