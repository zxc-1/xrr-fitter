"""Declared Stage-B geometry must survive into local refinement and resume."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.unit.fit.test_resume import _assert_equivalent_results
from tests.unit.fit.test_stage_search import (
    _candidate,
    _problem,
    _stage_b_baseline_case,
    _unchanged_local_solution,
)

from xrr_fitter.fit import pipeline, stages
from xrr_fitter.fit.candidates import CandidateStart
from xrr_fitter.io.codec_results import _checkpoint_from_dict, _checkpoint_to_dict
from xrr_fitter.model.fitting import FitStageSummary


def _declared_case(monkeypatch, *, same_geometry=False):
    problem = _problem(seed=751)
    case = _stage_b_baseline_case(problem)
    original = stages.solve_global

    def controlled_global(context, unit, **_kwargs):
        if len(context.variables) != len(case.stage_problem.variables):
            return original(context, unit, **_kwargs)
        selected = unit if same_geometry else case.drift_unit
        return SimpleNamespace(
            unit_vector=selected,
            evaluation=stages.evaluate_vector(context, selected),
            population=np.vstack((selected,) * 5),
            trace=(),
            stop_reason="controlled DE",
            nfev=5,
        )

    monkeypatch.setattr(stages, "solve_global", controlled_global)
    return problem, case


def test_distinct_declared_geometry_receives_full_resolution_local_refinement(monkeypatch) -> None:
    problem, case = _declared_case(monkeypatch)
    outcome = stages.run_stage_b(problem, None, (case.start,), (123,), progress=None, cancelled=None)

    parents, counts = stages.stage_b_continuation(outcome.candidates, outcome.perturbation_counts)

    assert tuple(parent.candidate_id for parent in parents) == ("B-declared-start", "B-0")
    assert counts == (2, 2)
    monkeypatch.setattr(stages, "solve_local", _unchanged_local_solution)
    refined = stages.run_local_stage(problem, None, "C", parents, progress=None, cancelled=None)
    assert tuple(candidate.candidate_id for candidate in refined.candidates) == ("C-0-0", "C-1-0")
    np.testing.assert_array_equal(refined.candidates[0].unit_vector, case.baseline_unit)
    assert refined.candidates[0].qz_a_inv.size == problem.data.qz_a_inv.size


def test_same_geometry_declared_evaluation_merges_without_losing_optimizer_work(monkeypatch) -> None:
    problem, case = _declared_case(monkeypatch, same_geometry=True)

    outcome = stages.run_stage_b(problem, None, (case.start,), (123,), progress=None, cancelled=None)

    assert tuple(candidate.candidate_id for candidate in outcome.candidates) == ("B-0",)
    assert outcome.candidates[0].nfev == outcome.summary.total_nfev == 7
    allocations = tuple(
        allocation for evidence in outcome.summary.search_evidence for allocation in evidence.budget_allocations
    )
    assert len(allocations) == 1
    assert allocations[0].nfev == 5
    assert stages.stage_b_continuation(outcome.candidates)[1] == outcome.perturbation_counts == (2,)


@pytest.mark.parametrize("valid", [True, False], ids=["high-cost", "invalid"])
def test_ineligible_declared_geometry_is_archived_before_fresh_or_resumed_continuation(monkeypatch, valid) -> None:
    problem = _problem(seed=753)
    baseline = replace(
        _candidate(problem, "B-declared-start", np.full(len(problem.variables), 0.45)),
        objective=100.0 if valid else 1.0,
        valid=valid,
    )
    optimized = replace(
        _candidate(problem, "B-0", np.full(len(problem.variables), 0.55)),
        objective=2.0,
    )
    monkeypatch.setattr(stages, "_stage_b_candidate", lambda *_args, **_kwargs: optimized)
    monkeypatch.setattr(stages, "_stage_b_launch_evidence", lambda *_args: (baseline, optimized))

    outcome = stages.run_stage_b(
        problem, None, (CandidateStart((), "declared-baseline"),), (123,), progress=None, cancelled=None
    )

    declared = outcome.candidates[0]
    assert declared.candidate_id == "B-declared-start"
    assert declared.seed_index == -1
    assert declared.stop_reason == "early_eliminated"
    for counts in (outcome.perturbation_counts, ()):
        parents, recovered = stages.stage_b_continuation(outcome.candidates, counts)
        assert tuple(parent.candidate_id for parent in parents) == ("B-0",)
        assert recovered == (5,)


def test_declared_budget_stays_with_its_parent_when_archive_ranking_differs() -> None:
    problem = _problem(seed=755)
    base = _candidate(problem, "source", np.full(len(problem.variables), 0.5))
    candidates = tuple(
        replace(base, candidate_id=name, objective=cost)
        for name, cost in (("B-declared-start", 3.0), ("B-0", 2.0), ("B-1", 100.0))
    )

    parents, counts = stages.stage_b_continuation(candidates)

    assert tuple(parent.candidate_id for parent in parents) == ("B-declared-start", "B-0")
    assert counts == (3, 4)


def test_declared_lineage_replays_exact_suffix_after_strict_json_checkpoint(monkeypatch) -> None:
    problem, case = _declared_case(monkeypatch)
    summary = FitStageSummary("A", ("declared-baseline",), case.baseline.objective, 1, ("evaluated",))
    starts = (case.start, replace(case.start, feature_key="geometry-other"))
    monkeypatch.setattr(pipeline, "run_stage_a", lambda *_args, **_kwargs: (starts, summary, ()))
    checkpoints = []
    fresh = pipeline.run_fit_search(pipeline.FitSearchRequest(None, problem), checkpoint=checkpoints.append)
    checkpoint = checkpoints[0]
    assert checkpoint.stage == "B"
    parents, _counts = stages.stage_b_continuation(checkpoint.candidates)
    assert tuple(parent.candidate_id for parent in parents) == ("B-declared-start", "B-0")

    raw = json.loads(json.dumps(_checkpoint_to_dict(checkpoint), allow_nan=False))
    restored = _checkpoint_from_dict(raw)
    suffix = []
    resumed = pipeline.run_fit_search(pipeline.FitSearchRequest(None, problem, restored), checkpoint=suffix.append)

    assert tuple(item.stage for item in suffix) == ("C", "D", "E")
    _assert_equivalent_results(fresh, resumed)
    assert tuple(candidate.nfev for candidate in resumed.candidates) == tuple(
        candidate.nfev for candidate in fresh.candidates
    )
    assert tuple(candidate.search_evidence for candidate in resumed.candidates) == tuple(
        candidate.search_evidence for candidate in fresh.candidates
    )
