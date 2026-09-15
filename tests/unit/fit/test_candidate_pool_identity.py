"""Start identity survives inactive parameter axes without resampling."""

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.fit.candidates as candidates
from xrr_fitter.fit.initialization import estimate_initial_candidates
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import MaterialSpec


def _direct_structure():
    structure = simple_structure()
    film = replace(structure.components[0], material=MaterialSpec("direct", None, None, 40e-6 + 0j))
    return replace(structure, components=(film,), backing=MaterialSpec("backing", None, None, 20e-6 + 0j))


@pytest.mark.parametrize("limit", (64, 512))
def test_direct_sld_pool_removes_only_repeated_complete_identities(limit, monkeypatch) -> None:
    data, structure = prepared_data(size=80), _direct_structure()
    instrument = InstrumentSpec(footprint_mode="none")
    constructed = []
    make_start, protected_starts = candidates._make_start, candidates._protected_starts

    def record_protected(*args):
        starts = protected_starts(*args)
        constructed.extend(starts)
        return starts

    def record_generated(*args):
        start = make_start(*args)
        constructed.append(start)
        return start

    monkeypatch.setattr(candidates, "_protected_starts", record_protected)
    monkeypatch.setattr(candidates, "_make_start", record_generated)
    rng = np.random.default_rng(0)

    pool = candidates.build_candidate_pool(data, structure, instrument, rng, limit=limit)

    assert len(constructed) == limit
    assert len(set(constructed)) < len(constructed)
    assert pool == tuple(dict.fromkeys(constructed))
    assert pool[0].feature_key == "declared-baseline"
    assert pool[0].value("component.0.thickness_a") == structure.components[0].thickness_a
    np.testing.assert_array_equal(rng.integers(2**32, size=3), [2195314465, 1158725112, 1322117304])


def test_equal_physical_values_keep_distinct_feature_families(monkeypatch) -> None:
    data, structure = prepared_data(size=80), _direct_structure()
    instrument = InstrumentSpec(footprint_mode="none")
    initial = replace(
        estimate_initial_candidates(data, structure, instrument, np.random.default_rng(0)),
        density_scales=(0.75, 1.0),
        roughness_fractions=(0.1,),
        angle_offsets_deg=(0.0,),
        scales=(1.0,),
        backgrounds=(0.0,),
        relative_resolutions=(0.0,),
        footprint_angles_deg=(0.0,),
        direct_sld_rows=(),
    )
    geometry = (("component.0.thickness_a", 20.0),)
    # Two upstream hypotheses can legitimately reach the same physical point.
    monkeypatch.setattr(candidates, "estimate_initial_candidates", lambda *_: initial)
    monkeypatch.setattr(candidates, "geometry_variants", lambda *_: (("family-a", geometry), ("family-b", geometry)))

    pool = candidates.build_candidate_pool(data, structure, instrument, np.random.default_rng(0), limit=10)

    assert tuple(start.feature_key for start in pool) == ("declared-baseline", "family-a", "family-b")
    assert pool[1].values == pool[2].values


def test_selection_still_rejects_repeated_complete_identity() -> None:
    start = candidates.CandidateStart((("x", 1.0),), "same-family")

    with pytest.raises(ValueError, match="scored starts must be unique"):
        candidates.select_coarse_candidates(((1.0, start), (2.0, start)), {start: np.zeros(4)})
