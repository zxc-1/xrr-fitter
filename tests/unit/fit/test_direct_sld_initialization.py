import numpy as np
from tests.support.model_cases import prepared_data

from xrr_fitter.fit.candidates import build_candidate_pool
from xrr_fitter.fit.initialization import (
    critical_sld_candidates,
    direct_sld_start_rows,
)
from xrr_fitter.fit.parameters import (
    default_parameter_definitions as parameter_definitions,
)
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)

AIR = MaterialSpec("Air", None, None, 0j)
SI = MaterialSpec("Si", "Si", 2.329)


def _unknown_structure() -> StructureSpec:
    return StructureSpec(
        AIR,
        (
            LayerSpec("CrSiC", MaterialSpec("CrSiC", None, None, 20e-6 + 0j), 80.0),
            LayerSpec("SiCMo", MaterialSpec("SiCMo", None, None, 20e-6 + 0j), 100.0),
        ),
        SI,
    )


def _all_direct_locations_structure() -> StructureSpec:
    periodic_direct = MaterialSpec("periodic direct", None, None, 60e-6 + 2e-6j)
    direct_backing = MaterialSpec("direct backing", None, None, 30e-6 + 1e-6j)
    return StructureSpec(
        AIR,
        (
            LayerSpec("ordinary direct", MaterialSpec("direct", None, None, 20e-6 + 0j), 80.0),
            PeriodicBlock(
                "mixed cell",
                (
                    LayerSpec("periodic direct", periodic_direct, 25.0),
                    LayerSpec("periodic formula", SI, 35.0),
                ),
                repeats=3,
            ),
        ),
        direct_backing,
    )


def test_direct_sld_real_is_free_but_absorption_starts_locked() -> None:
    definitions = parameter_definitions(
        prepared_data(),
        _unknown_structure(),
        InstrumentSpec(),
        FitConfig.fast(41),
    )
    by_name = {value.name: value for value in definitions}
    assert by_name["component.0.sld_real_a2"].locked is False
    assert (
        by_name["component.0.sld_real_a2"].lower,
        by_name["component.0.sld_real_a2"].upper,
    ) == (-150e-6, 150e-6)
    assert by_name["component.0.sld_imag_a2"].locked is True


def test_direct_sld_density_is_locked_at_one_and_backing_is_compiled() -> None:
    definitions = parameter_definitions(
        prepared_data(),
        _all_direct_locations_structure(),
        InstrumentSpec(),
        FitConfig.fast(42),
    )
    by_name = {value.name: value for value in definitions}

    for name in ("component.0.density_scale", "component.1.layer.0.density_scale"):
        definition = by_name[name]
        assert definition.initial == definition.lower == definition.upper == 1.0
        assert definition.locked is True
    assert by_name["backing.sld_real_a2"].locked is False
    assert by_name["backing.sld_imag_a2"].locked is True


def test_direct_sld_rows_cover_layers_periodic_cells_and_backing_in_stable_order() -> None:
    structure = _all_direct_locations_structure()
    anchors = tuple(value * 1e-6 for value in (-20.0, 0.0, 10.0, 20.0, 40.0, 80.0, 120.0))
    candidates = tuple(sorted({*anchors, 33e-6}))

    rows = direct_sld_start_rows(structure, candidates)

    expected_names = (
        "component.0.sld_real_a2",
        "component.1.layer.0.sld_real_a2",
        "backing.sld_real_a2",
    )
    assert tuple(name for name, _value in rows[0]) == expected_names
    assert tuple(value for _name, value in rows[0]) == (20e-6, 60e-6, 30e-6)
    assert tuple(value for _name, value in rows[1]) == (33e-6, 33e-6, 33e-6)
    assert len(rows) == 8


def test_critical_sld_candidates_include_sorted_bounded_anchors() -> None:
    candidates = critical_sld_candidates(
        prepared_data(size=96),
        _all_direct_locations_structure(),
    )
    anchors = {value * 1e-6 for value in (-20.0, 0.0, 10.0, 20.0, 40.0, 80.0, 120.0)}

    assert candidates == tuple(sorted(set(candidates)))
    assert anchors <= set(candidates)
    assert all(-150e-6 <= value <= 150e-6 for value in candidates)


def test_direct_sld_candidate_rows_are_seed_independent_and_layer_distinct() -> None:
    data = prepared_data(size=96)
    first = build_candidate_pool(
        data,
        _unknown_structure(),
        InstrumentSpec(),
        np.random.default_rng(1),
        limit=64,
    )
    second = build_candidate_pool(
        data,
        _unknown_structure(),
        InstrumentSpec(),
        np.random.default_rng(99),
        limit=64,
    )
    first_sld = tuple(
        tuple((name, value) for name, value in start.values if "sld_real_a2" in name)
        for start in first
    )
    second_sld = tuple(
        tuple((name, value) for name, value in start.values if "sld_real_a2" in name)
        for start in second
    )
    baseline = dict(first[0].values)
    assert baseline["component.0.sld_imag_a2"] == 0.0
    assert baseline["component.1.sld_imag_a2"] == 0.0
    assert first_sld[:6] == second_sld[:6]
    assert any(len({value for _name, value in row}) > 1 for row in first_sld if row)
