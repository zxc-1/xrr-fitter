"""Candidate initialization contracts.

The fixtures distinguish declared geometry, data-derived hypotheses, and
deterministic cap/dedup behavior so those policies cannot collapse together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrr_fitter.fit.candidates import (
    CandidateStart,
    build_candidate_pool,
    select_coarse_candidates,
    select_full_search_candidates,
)
from xrr_fitter.fit.initialization import (
    estimate_initial_candidates,
    footprint_angle_candidates,
)
from xrr_fitter.model.data import BeamSpec, DataColumnMapping, PreparedData
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.materials import material_sld
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure

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
        raw_rows=tuple(
            f"{angle} {value}"
            for angle, value in zip(two_theta_deg, normalized, strict=True)
        ),
        raw_parse_status=("data",) * size,
        source_row_groups=tuple((index,) for index in range(size)),
        beam=BEAM,
        import_angle_offset_deg=0.0,
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
    block = PeriodicBlock(
        "Mo/Si",
        (
            LayerSpec("Mo", MOLYBDENUM, 28.0, roughness_a=2.0),
            LayerSpec("Si", SILICON, 42.0, roughness_a=2.0),
        ),
        repeats=20,
        top_roughness_a=2.0,
    )
    return StructureSpec(AIR, (block,), SILICON, backing_roughness_a=2.0)


def test_initial_candidate_grids_and_limits_are_versioned() -> None:
    initial = estimate_initial_candidates(
        _oscillatory_data(70.0),
        _periodic_structure(),
        FIT_INSTRUMENT,
        np.random.default_rng(20260715),
    )

    assert initial.layer_fractions == (0.20, 0.35, 0.50, 0.65, 0.80)
    assert initial.density_scales == (0.75, 0.85, 0.95, 1.0)
    assert initial.roughness_fractions == (0.02, 0.05, 0.10)
    assert initial.relative_resolutions[:5] == (0.0, 0.002, 0.005, 0.01, 0.02)
    assert 1.0 in initial.scales
    assert 0.0 in initial.backgrounds
    assert any(abs(period - 70.0) / 70.0 < 0.02 for period in initial.period_a)


def test_footprint_candidates_respect_fit_and_locked_modes() -> None:
    data = _oscillatory_data(70.0)
    geometry = InstrumentSpec(
        instrument_id="lab-1",
        footprint_mode="geometry",
        footprint_spill_angle_deg=np.rad2deg(np.arcsin(0.01)),
        sample_length_mm=10.0,
        beam_width_mm=0.1,
    )
    disabled = InstrumentSpec(footprint_mode="none", footprint_spill_angle_deg=0.0)

    assert footprint_angle_candidates(data, FIT_INSTRUMENT, 0.22, None) == (0.0, 0.22)
    assert footprint_angle_candidates(data, geometry, 0.22, 0.18) == (
        geometry.footprint_spill_angle_deg,
    )
    assert footprint_angle_candidates(data, disabled, 0.22, 0.18) == (0.0,)
    initial = estimate_initial_candidates(
        data,
        _periodic_structure(),
        FIT_INSTRUMENT,
        np.random.default_rng(1),
    )
    assert 0.0 in initial.footprint_angles_deg


def test_angle_offset_candidates_are_not_a_round_trip_of_import_qz() -> None:
    qz = np.linspace(0.002, 0.12, 2000)
    theta = np.rad2deg(np.arcsin(qz * BEAM.effective_wavelength_a / (4.0 * np.pi)))
    delta = material_sld(MOLYBDENUM, 1.0, BEAM.effective_wavelength_a).real
    theoretical = np.rad2deg(
        np.arcsin(
            np.sqrt(16.0 * np.pi * delta)
            * BEAM.effective_wavelength_a
            / (4.0 * np.pi)
        )
    )
    observed = theoretical - 0.018
    reflectivity = 1.0 / (1.0 + np.exp((theta - observed) / 0.0015))
    data = _prepared_data(qz, reflectivity, two_theta_deg=2.0 * theta)

    initial = estimate_initial_candidates(
        data,
        _single_layer_structure(),
        FIT_INSTRUMENT,
        np.random.default_rng(4),
    )

    assert any(abs(value - 0.018) < 0.005 for value in initial.angle_offsets_deg)
    assert any(abs(value) > 0.005 for value in initial.angle_offsets_deg)


def test_unreliable_single_method_peaks_fall_back_to_wide_observable_grid() -> None:
    qz = np.linspace(0.015, 0.45, 256)
    data = _prepared_data(qz, (1.0 + qz) / qz**4)

    initial = estimate_initial_candidates(
        data,
        _periodic_structure(),
        FIT_INSTRUMENT,
        np.random.default_rng(18),
    )

    observed_min = max(2.0, 2.0 * np.pi / np.ptp(qz))
    observed_max = min(2e5, np.pi / (2.0 * np.median(np.diff(qz))))
    expected = np.geomspace(
        max(2.0, 0.25 * observed_min),
        min(2e5, 4.0 * observed_max),
        8,
    )
    assert "初始特征不足" in initial.warnings
    np.testing.assert_allclose(initial.thickness_a, expected)


def test_candidate_pool_caps_are_deterministic() -> None:
    data = _oscillatory_data(70.0, 110.0)
    structure = _periodic_structure()

    first = build_candidate_pool(
        data, structure, FIT_INSTRUMENT, np.random.default_rng(91)
    )
    second = build_candidate_pool(
        data, structure, FIT_INSTRUMENT, np.random.default_rng(91)
    )

    assert len(first) == 512
    assert first == second


def test_candidate_pool_preserves_declared_periodic_geometry() -> None:
    structure = _periodic_structure()
    theta = np.linspace(0.03, 3.0, 600)
    qz = 4.0 * np.pi * np.sin(np.deg2rad(theta)) / BEAM.effective_wavelength_a
    normalized = instrument_reflectivity(
        theta,
        expand_structure(structure, BEAM.effective_wavelength_a),
        BEAM,
        scale=0.9,
        background=2e-8,
    )
    data = _prepared_data(qz, normalized, two_theta_deg=2.0 * theta)

    pool = build_candidate_pool(
        data,
        structure,
        InstrumentSpec(footprint_mode="none"),
        np.random.default_rng(2002),
    )
    block = structure.components[0]
    assert isinstance(block, PeriodicBlock)
    declared = tuple(layer.thickness_a for layer in block.layers)

    assert any(
        tuple(
            dict(start.values)[f"component.0.layer.{index}.thickness_a"]
            for index in range(len(declared))
        )
        == declared
        for start in pool
    )


def test_candidate_pool_preserves_declared_ordinary_geometry() -> None:
    pool = build_candidate_pool(
        _oscillatory_data(70.0),
        _single_layer_structure(),
        FIT_INSTRUMENT,
        np.random.default_rng(173),
    )

    assert any(dict(start.values)["component.0.thickness_a"] == 173.0 for start in pool)


def test_candidate_pool_protects_complete_declared_multilayer_baseline() -> None:
    structure = StructureSpec(
        AIR,
        (
            LayerSpec("cap", SILICON, 73.0, density_scale=0.93, roughness_a=2.5),
            LayerSpec("film", MOLYBDENUM, 201.0, density_scale=1.04, roughness_a=4.0),
        ),
        SILICON,
        backing_roughness_a=2.0,
    )
    data = _oscillatory_data(73.0, 201.0)

    pool = build_candidate_pool(
        data, structure, FIT_INSTRUMENT, np.random.default_rng(174), limit=32
    )

    declared = dict(pool[0].values)
    assert pool[0].feature_key == "declared-baseline"
    expected = {
        "component.0.thickness_a": 73.0,
        "component.0.density_scale": 0.93,
        "component.0.roughness_a": 2.5,
        "component.1.thickness_a": 201.0,
        "component.1.density_scale": 1.04,
        "component.1.roughness_a": 4.0,
        "backing.roughness_a": 2.0,
        "instrument.angle_offset_deg": data.import_angle_offset_deg,
        "instrument.scale": 1.0,
        "instrument.background": 0.0,
        "instrument.relative_sigma": 0.0,
    }
    assert {name: declared[name] for name in expected} == expected


def test_multilayer_geometry_preserves_declared_ratio_and_varies_layers() -> None:
    layers = (
        LayerSpec("top", MOLYBDENUM, 40.0, roughness_a=1.0),
        LayerSpec("middle", SILICON, 60.0, roughness_a=1.0),
        LayerSpec("bottom", MOLYBDENUM, 73.0, roughness_a=1.0),
    )
    pool = build_candidate_pool(
        _oscillatory_data(173.0),
        StructureSpec(AIR, layers, SILICON),
        FIT_INSTRUMENT,
        np.random.default_rng(17),
    )
    triples = {
        tuple(
            dict(start.values)[f"component.{index}.thickness_a"]
            for index in range(3)
        )
        for start in pool
    }
    declared_ratio = np.array([40.0, 60.0, 73.0]) / 173.0

    assert any(len(set(values)) > 1 for values in triples)
    assert any(
        np.allclose(np.array(values) / sum(values), declared_ratio)
        for values in triples
    )


def test_candidate_roughness_uses_adjacent_effective_thickness() -> None:
    structure = StructureSpec(
        AIR,
        (
            LayerSpec("thin", MOLYBDENUM, 2.0),
            LayerSpec("thick", SILICON, 171.0),
        ),
        SILICON,
    )

    pool = build_candidate_pool(
        _oscillatory_data(173.0),
        structure,
        FIT_INSTRUMENT,
        np.random.default_rng(31),
    )

    for start in pool:
        values = dict(start.values)
        first = values["component.0.thickness_a"]
        second = values["component.1.thickness_a"]
        assert values["component.0.roughness_a"] < 0.49 * first
        assert values["component.1.roughness_a"] < 0.49 * min(first, second)


def test_two_periodic_blocks_receive_independent_period_hypotheses() -> None:
    first = _periodic_structure().components[0]
    assert isinstance(first, PeriodicBlock)
    second = PeriodicBlock("second", first.layers, repeats=12, top_roughness_a=2.0)
    structure = StructureSpec(AIR, (first, second), SILICON)

    pool = build_candidate_pool(
        _oscillatory_data(70.0, 110.0),
        structure,
        FIT_INSTRUMENT,
        np.random.default_rng(23),
    )
    period_pairs = {
        (
            sum(
                dict(start.values)[f"component.0.layer.{index}.thickness_a"]
                for index in range(2)
            ),
            sum(
                dict(start.values)[f"component.1.layer.{index}.thickness_a"]
                for index in range(2)
            ),
        )
        for start in pool
    }

    assert any(first_period != second_period for first_period, second_period in period_pairs)
    assert all(
        not any(name.startswith("repeat.") for name, _value in start.values)
        for start in pool
    )


def test_curve_dedup_merges_far_parameter_degenerate_starts_globally() -> None:
    first = CandidateStart((("component.0.thickness_a", 100.0),), "peak-0")
    near = CandidateStart((("component.0.thickness_a", 101.0),), "peak-0")
    far = CandidateStart((("component.0.thickness_a", 10000.0),), "peak-1")
    distinct = CandidateStart((("component.0.thickness_a", 180.0),), "peak-2")
    scored = ((0.01, first), (0.02, near), (0.03, far), (0.015, distinct))
    curves = {
        first: np.array([-1.0, -2.0, -3.0]),
        near: np.array([-1.001, -2.001, -3.001]),
        far: np.array([-1.002, -2.002, -3.002]),
        distinct: np.array([-1.0, -1.2, -1.8]),
    }

    coarse = select_coarse_candidates(scored, curves)
    full = select_full_search_candidates(scored, curves)

    assert first in coarse
    assert near not in coarse
    assert far not in coarse
    assert len(coarse) <= 24
    assert len(full) <= 8
    for index, current in enumerate(full):
        for other in full[index + 1 :]:
            assert np.sqrt(np.mean((curves[current] - curves[other]) ** 2)) >= 0.02


def test_full_search_selection_reserves_declared_baseline() -> None:
    baseline = CandidateStart((("x", 0.0),), "declared-baseline")
    lower_cost = tuple(
        CandidateStart((("x", float(index)),), f"feature-{index}")
        for index in range(1, 10)
    )
    scored = tuple(
        [(1.0, baseline)]
        + [(0.01 * index, start) for index, start in enumerate(lower_cost, 1)]
    )
    curves = {
        start: np.array([0.0, float(index), float(index**2)])
        for index, (_cost, start) in enumerate(scored)
    }

    selected = select_full_search_candidates(scored, curves, limit=8)

    assert baseline in selected
    assert len(selected) == 8


def test_curve_dedup_rejects_inconsistent_curve_shapes() -> None:
    first = CandidateStart((("x", 1.0),), "a")
    second = CandidateStart((("x", 2.0),), "b")

    with pytest.raises(ValueError, match="same shape"):
        select_full_search_candidates(
            ((0.1, first), (0.2, second)),
            {first: np.array([1.0]), second: np.array([1.0, 2.0])},
        )
