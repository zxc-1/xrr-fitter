"""A legal skip must not revive archived lineages or discard reclaimed work."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.unit.fit.test_stage_search import _candidate, _problem

from xrr_fitter.fit import pipeline, stages
from xrr_fitter.fit.candidates import CandidateStart, archive_stage_b_candidates


def _archived_state(problem, skipped):
    source = _candidate(problem, "source", np.full(len(problem.variables), 0.5))
    candidates = tuple(
        replace(source, candidate_id=candidate_id, objective=objective)
        for candidate_id, objective in (("B-declared-baseline", 3.0), ("B-0", 2.0), ("B-1", 100.0))
    )
    archive = archive_stage_b_candidates(candidates)
    by_id = {candidate.candidate_id: candidate for candidate in archive.active + archive.archived}
    retained = tuple(by_id[candidate.candidate_id] for candidate in candidates)
    return pipeline._SearchState(
        candidates=retained,
        summaries=(stages._summary("B", retained),),
        skipped_stages=skipped,
    )


@pytest.mark.parametrize("stage", ("D", "E"))
def test_skipped_refinement_keeps_the_committed_parent_continuation(monkeypatch, stage):
    problem = _problem(seed=833)
    request = pipeline.FitSearchRequest("curve", problem)
    state = _archived_state(problem, ("C",) if stage == "D" else ("C", "D"))
    expected_parents, expected_counts = stages.stage_b_continuation(state.candidates)
    name = "run_local_stage" if stage == "D" else "run_stage_e"
    run = getattr(pipeline, name)
    observed = []

    def capture(*args, **kwargs):
        parents = args[3] if stage == "D" else args[2]
        observed.append((tuple(candidate.candidate_id for candidate in parents), kwargs.get("perturbation_counts")))
        return run(*args, **kwargs)

    monkeypatch.setattr(pipeline, name, capture)
    outcome = pipeline._stage_outcome(
        request,
        stage,
        state,
        starts=None,
        seeds=pipeline._seed_ledger(request),
        perturbation_counts=(),
        progress=None,
        cancelled=None,
        task_runner=None,
    )

    assert observed == [
        (tuple(candidate.candidate_id for candidate in expected_parents), expected_counts if stage == "D" else None)
    ]
    assert outcome.candidates
    if stage == "D":
        allocations = tuple(
            allocation for evidence in outcome.summary.search_evidence for allocation in evidence.budget_allocations
        )
        assert len(allocations) == len(expected_parents) + sum(expected_counts)
        assert {allocation.lineage_id for allocation in allocations} == {"D-0", "D-1"}


def test_stage_a_completion_follows_full_review_and_records_its_work(monkeypatch):
    problem = _problem(seed=839, size=160)
    review = stages.review_stage_a
    reviewed = False
    events = []

    def full_review(*args, **kwargs):
        nonlocal reviewed
        result = review(*args, **kwargs)
        reviewed = True
        return result

    def progress(value):
        events.append(value)
        if value.completed == value.total:
            assert reviewed, "the coarse scan must not claim completion before full-grid review"

    monkeypatch.setattr(stages, "review_stage_a", full_review)
    _starts, summary, _warnings = stages.run_stage_a(problem, "curve", progress=progress, cancelled=None)

    assert sum(value.completed == value.total for value in events) == 1
    assert tuple(value.completed for value in events) == tuple(range(1, len(events) + 1))
    assert {value.total for value in events} == {len(events)}
    (evidence,) = summary.search_evidence
    assert events[-1].nfev == evidence.grid_review_evaluations + evidence.full_review_evaluations


def test_stage_a_scan_telemetry_excludes_physical_rejections(monkeypatch):
    problem = _problem(seed=853)
    evaluate = stages._stage_a_candidate
    events = []

    def reject_first(context, start, index):
        return None if index == 0 else evaluate(context, start, index)

    monkeypatch.setattr(stages, "_stage_a_candidate", reject_first)
    pool = (CandidateStart((), "rejected"), CandidateStart((), "retained"))
    _evaluated, rejected, invalid = stages._evaluate_stage_a_pool(problem, "curve", pool, events.append, None)

    assert (rejected, invalid) == (1, 0)
    assert [value.nfev for value in events] == [0, 1]
