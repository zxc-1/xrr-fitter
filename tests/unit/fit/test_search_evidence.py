"""Search work and full-grid decisions survive the actual result codec."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from importlib import import_module

import pytest
from tests.support.model_cases import fit_candidate

from xrr_fitter.io.codec_candidates import _candidate_from_dict, _candidate_to_dict, _stages_from_list, _stages_to_list
from xrr_fitter.io.project_codec import ProjectSchemaError
from xrr_fitter.model.fitting import FitStageSummary


def _types():
    module = import_module("xrr_fitter.model.fitting")
    for name in ("GridReview", "SearchAllocation", "SearchEvidence"):
        assert hasattr(module, name), f"missing immutable search evidence: {name}"
    return module


def _evidence():
    model = _types()
    review = model.GridReview(
        grid_points=(128, 80),
        candidate_ids=("a", "b"),
        coarse_objectives=(1.0, 1.01),
        full_objectives=(1.03, 1.0),
        promoted=True,
        grid_evaluations=12,
        full_evaluations=2,
    )
    allocation = model.SearchAllocation("lineage-0", "B-0", 0, 50, 12)
    return model.SearchEvidence("declared-baseline", 17, (review,), (allocation,), "budget_exhausted")


def test_search_evidence_derives_work_counts_and_is_pickle_safe() -> None:
    evidence = _evidence()
    assert evidence.grid_points == ((128, 80),)
    assert evidence.full_review_evaluations == 2
    assert evidence.grid_review_evaluations == 12
    assert evidence.optimizer_nfev == 12
    assert evidence.optimizer_nfev_limit == 50
    assert pickle.loads(pickle.dumps(evidence)) == evidence


def test_search_evidence_round_trips_with_candidates_and_stage_summaries() -> None:
    evidence = _evidence()
    candidate = replace(fit_candidate("B-0"), search_evidence=(evidence,))
    summary = FitStageSummary(
        "B", (candidate.candidate_id,), candidate.objective, candidate.nfev, (candidate.stop_reason,), (evidence,)
    )
    raw = json.loads(json.dumps(_candidate_to_dict(candidate), allow_nan=False))
    restored = _candidate_from_dict(raw)
    assert restored.search_evidence == (evidence,)
    stages = json.loads(json.dumps(_stages_to_list((summary,), (candidate,)), allow_nan=False))
    assert _stages_from_list(stages, (restored,)) == (summary,)
    raw.pop("search_evidence")
    with pytest.raises(ProjectSchemaError, match="search_evidence"):
        _candidate_from_dict(raw)
    stages[0].pop("search_evidence")
    with pytest.raises(ProjectSchemaError, match="search_evidence"):
        _stages_from_list(stages, (restored,))


def test_unavailable_review_cost_is_json_null_not_nan_or_fake_success() -> None:
    evidence = _evidence()
    review = replace(evidence.reviews[0], full_objectives=(float("inf"), 1.0))
    candidate = replace(fit_candidate(), search_evidence=(replace(evidence, reviews=(review,)),))
    raw = json.loads(json.dumps(_candidate_to_dict(candidate), allow_nan=False))
    assert raw["search_evidence"][0]["reviews"][0]["full_objectives"] == [None, 1.0]
    assert _candidate_from_dict(raw).search_evidence == candidate.search_evidence


@pytest.mark.parametrize(("allocated", "used"), [(2, 3), (-1, 0), (1, -1), (True, 1)])
def test_search_allocation_rejects_over_budget_or_invalid_work(allocated, used) -> None:
    model = _types()
    with pytest.raises(ValueError, match="nfev|budget"):
        model.SearchAllocation("lineage-0", "B-0", 0, allocated, used)


def test_review_evidence_rejects_misaligned_or_impossible_counts() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="align"):
        replace(evidence.reviews[0], full_objectives=(1.0,))
    with pytest.raises(ValueError, match="full_evaluations"):
        replace(evidence.reviews[0], full_evaluations=3)


def test_resume_normalization_preserves_validated_optimizer_and_grid_evidence() -> None:
    from tests.unit.fit.test_resume import _problem, _stage_b_checkpoint

    from xrr_fitter.fit.resume import validate_resume_checkpoint

    problem = _problem()
    checkpoint = _stage_b_checkpoint(problem)
    first, second = checkpoint.candidates
    evidence = _evidence()
    recorded = replace(evidence.budget_allocations[0], nfev=first.nfev)
    evidence = replace(evidence, budget_allocations=(recorded,))
    checkpoint = replace(
        checkpoint,
        candidates=(replace(first, search_evidence=(evidence,)), second),
        stage_summaries=(
            checkpoint.stage_summaries[0],
            replace(checkpoint.stage_summaries[1], search_evidence=(evidence,)),
        ),
    )

    plan = validate_resume_checkpoint(problem, checkpoint, reserved_child_seeds=(101, 102, 201, 202))

    assert plan.candidates[0].search_evidence == (evidence,)
    assert plan.stage_summaries == checkpoint.stage_summaries
