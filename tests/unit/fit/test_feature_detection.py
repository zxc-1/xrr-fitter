from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrr_fitter.fit.initialization import (
    autocorrelation_thickness_candidates,
    critical_edge_candidates,
    estimate_initial_candidates,
    kiessig_spacing_candidates,
    spectral_thickness_candidates,
    structure_evidence,
)
from xrr_fitter.model.data import BeamSpec, DataColumnMapping, PreparedData
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec

BEAM = BeamSpec(kind="monochromatic")
FIT_INSTRUMENT = InstrumentSpec(footprint_mode="fit")
AIR = MaterialSpec("Air", None, None, 0.0j)
SILICON = MaterialSpec("Si", "Si", 2.329)
MOLYBDENUM = MaterialSpec("Mo", "Mo", 10.28)


def _prepared_data(
    qz: np.ndarray,
    normalized: np.ndarray,
    *,
    two_theta_deg: np.ndarray | None = None,
    import_angle_offset_deg: float = 0.0,
) -> PreparedData:
    qz = np.asarray(qz, dtype=float)
    normalized = np.asarray(normalized, dtype=float)
    if two_theta_deg is None:
        argument = qz * BEAM.effective_wavelength_a / (4.0 * np.pi)
        two_theta_deg = 2.0 * np.rad2deg(np.arcsin(argument))
    size = qz.size
    return PreparedData(
        source_path=Path("synthetic.xy"),
        source_sha256="0" * 64,
        raw_rows=tuple(f"{angle} {value}" for angle, value in zip(two_theta_deg, normalized, strict=True)),
        raw_parse_status=("data",) * size,
        source_row_groups=tuple((index,) for index in range(size)),
        beam=BEAM,
        import_angle_offset_deg=import_angle_offset_deg,
        two_theta_deg=two_theta_deg,
        intensity_raw=normalized,
        intensity_sigma_raw=None,
        resolution_raw=None,
        qz_a_inv=qz,
        intensity_normalized=normalized,
        intensity_sigma_normalized=None,
        sigma_q_a_inv=None,
        validation_mask=np.ones(size, dtype=bool),
        fit_mask=np.ones(size, dtype=bool),
        normalization=1.0,
        r_floor=1e-10,
        fit_ready=bool(size >= 30 and np.ptp(qz) > 0.0),
        warnings=(),
        column_mapping=DataColumnMapping(),
    )


def _oscillatory_data(*thicknesses: float) -> PreparedData:
    qz = np.linspace(0.015, 0.45, 2048)
    transformed = np.ones_like(qz)
    for index, thickness in enumerate(thicknesses):
        transformed += (0.25 / (index + 1)) * np.cos(qz * thickness)
    return _prepared_data(qz, 1e-8 * transformed / qz**4)


def _single_layer_structure() -> StructureSpec:
    return StructureSpec(
        AIR,
        (LayerSpec("film", MOLYBDENUM, 173.0, roughness_a=3.0),),
        SILICON,
    )


def _periodic_structure() -> StructureSpec:
    from xrr_fitter.model.structure import PeriodicBlock

    return StructureSpec(
        AIR,
        (
            PeriodicBlock(
                "Mo/Si",
                (
                    LayerSpec("Mo", MOLYBDENUM, 28.0, roughness_a=2.0),
                    LayerSpec("Si", SILICON, 42.0, roughness_a=2.0),
                ),
                repeats=20,
                top_roughness_a=2.0,
            ),
        ),
        SILICON,
        backing_roughness_a=2.0,
    )


def test_fft_candidates_include_single_layer_thickness() -> None:
    qz = np.linspace(0.015, 0.45, 2048)
    thickness = 173.0
    transformed = 1.0 + 0.25 * np.cos(qz * thickness)

    candidates = spectral_thickness_candidates(qz, transformed, max_candidates=8)

    assert np.min(np.abs(candidates - thickness)) / thickness < 0.02


def test_feature_extractors_recover_single_layer_thickness() -> None:
    qz = np.linspace(0.015, 0.45, 2048)
    thickness = 173.0
    transformed = 1.0 + 0.25 * np.cos(qz * thickness)

    autocorrelation = autocorrelation_thickness_candidates(qz, transformed, max_candidates=8)
    spacing = kiessig_spacing_candidates(qz, transformed, max_candidates=8)

    assert np.min(np.abs(autocorrelation - thickness)) / thickness < 0.02
    assert np.min(np.abs(spacing - thickness)) / thickness < 0.02


def test_feature_extractors_reject_invalid_grids() -> None:
    values = np.ones(16)
    with pytest.raises(ValueError, match="strictly increasing"):
        spectral_thickness_candidates(np.zeros(16), values)
    with pytest.raises(ValueError, match="at least 16"):
        autocorrelation_thickness_candidates(np.arange(15.0), np.ones(15))


def test_critical_edge_curvature_finds_known_edge() -> None:
    qz = np.linspace(0.002, 0.12, 2000)
    edge = 0.031
    reflectivity = 1.0 / (1.0 + np.exp((qz - edge) / 0.0015))

    candidates = critical_edge_candidates(qz, reflectivity)

    assert min(abs(candidate - edge) for candidate in candidates) < 0.002


def test_angle_offset_candidates_keep_import_center_when_edges_hit_bound() -> None:
    import_offset = 0.013
    theta = np.linspace(0.03, 3.0, 520)
    qz = 4.0 * np.pi * np.sin(np.deg2rad(theta + import_offset)) / BEAM.effective_wavelength_a
    reflectivity = 1.0 / (1.0 + np.exp((theta - 0.7) / 0.005)) + 1e-8
    data = _prepared_data(
        qz,
        reflectivity,
        two_theta_deg=2.0 * theta,
        import_angle_offset_deg=import_offset,
    )

    initial = estimate_initial_candidates(
        data,
        _single_layer_structure(),
        FIT_INSTRUMENT,
        np.random.default_rng(5),
    )

    assert -0.1 in initial.angle_offsets_deg
    assert import_offset in initial.angle_offsets_deg


def test_featureless_data_warns_and_keeps_protected_background_candidate() -> None:
    qz = np.linspace(0.015, 0.45, 256)
    data = _prepared_data(qz, np.exp(-qz))

    initial = estimate_initial_candidates(
        data,
        _periodic_structure(),
        FIT_INSTRUMENT,
        np.random.default_rng(8),
    )

    low_count = max(20, int(np.ceil(0.10 * qz.size)))
    expected_scale = float(np.clip(np.percentile(data.intensity_normalized[:low_count], 95), 1e-3, 1e3))
    high_count = int(np.ceil(0.20 * qz.size))
    high_median = float(np.median(data.intensity_normalized[-high_count:]))
    assert "初始特征不足" in initial.warnings
    assert expected_scale in initial.scales
    assert min(high_median, 0.1) in initial.backgrounds
    assert high_median in initial.backgrounds


def test_structure_evidence_distinguishes_supported_and_overspecified_models() -> None:
    data = _oscillatory_data(173.0)
    supported = structure_evidence(data, _single_layer_structure())
    layers = tuple(
        LayerSpec(f"layer-{index}", MOLYBDENUM, thickness, roughness_a=1.0)
        for index, thickness in enumerate((20.0, 30.0, 40.0, 83.0))
    )
    overspecified = structure_evidence(data, StructureSpec(AIR, layers, SILICON))

    assert supported.m_data >= 1
    assert supported.m_model == 1
    assert not supported.warning
    assert overspecified.m_model > overspecified.m_data + 1
    assert overspecified.warning


def test_structure_evidence_scales_extreme_finite_q_before_qz4_transform() -> None:
    qz = np.linspace(1e80, 2e80, 256)
    data = _prepared_data(
        qz,
        np.full(qz.size, 1e-8),
        two_theta_deg=np.linspace(0.1, 1.0, qz.size),
    )

    with np.errstate(over="raise", invalid="raise"):
        evidence = structure_evidence(data, _single_layer_structure())

    assert evidence.m_data == 0
    assert all(np.isfinite(value) for value in evidence.peak_positions_a)
