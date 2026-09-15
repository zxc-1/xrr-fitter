"""Search evidence dictionaries already satisfy the strict JSON wire contract."""

from __future__ import annotations

import json

import pytest

from xrr_fitter.io.codec_common import ProjectSchemaError
from xrr_fitter.io.codec_search import search_evidence_from_list, search_evidence_to_list
from xrr_fitter.model.search import GridReview, SearchAllocation, SearchEvidence


def _evidence() -> SearchEvidence:
    review = GridReview(
        grid_points=(128, 80),
        candidate_ids=("a", "b"),
        coarse_objectives=(1.0, float("inf")),
        full_objectives=(1.03, 1.0),
        promoted=True,
        grid_evaluations=12,
        full_evaluations=2,
    )
    allocation = SearchAllocation("lineage-0", "B-0", 0, 50, 12)
    return SearchEvidence("declared-baseline", 17, (review,), (allocation,), "budget_exhausted")


def test_search_evidence_encoder_emits_json_arrays_before_text_serialization() -> None:
    evidence = _evidence()
    payload = search_evidence_to_list((evidence,))

    assert payload == json.loads(json.dumps(payload, allow_nan=False))
    assert search_evidence_from_list(payload) == (evidence,)
    assert payload[0]["reviews"][0]["coarse_objectives"] == [1.0, None]


def test_search_evidence_reader_keeps_the_strict_json_array_boundary() -> None:
    payload = json.loads(json.dumps(search_evidence_to_list((_evidence(),)), allow_nan=False))
    payload[0]["reviews"][0]["candidate_ids"] = ("a", "b")

    with pytest.raises(ProjectSchemaError, match="candidate_ids must be a JSON array"):
        search_evidence_from_list(payload)
