"""Actual optimizer accounting and full-cost-ranked local budget rounds."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.fit.test_joint_pipeline import _fully_locked_problem
from tests.unit.fit.test_stage_search import _candidate, _problem

from xrr_fitter.fit import stages
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.pipeline import FitSearchRequest, run_fit_search


def _parent_pair(problem):
    return tuple(
        _candidate(problem, f"B-{index}", np.full(len(problem.variables), unit))
        for index, unit in enumerate((0.45, 0.55))
    )


def _allocations(outcome):
    return tuple(item for evidence in outcome.summary.search_evidence for item in evidence.budget_allocations)


def _assert_distinct_round(allocations, round_index, expected):
    lineages = [item.lineage_id for item in allocations if item.round_index == round_index]
    assert len(lineages) == len(set(lineages)) == expected


def test_recovered_stage_b_budget_stays_aligned_with_publication_order() -> None:
    problem = _problem(seed=730)
    base = _candidate(problem, "source", np.full(len(problem.variables), 0.5))
    candidates = tuple(
        replace(base, candidate_id=f"B-{index}", objective=cost) for index, cost in enumerate((3.0, 2.0, 100.0))
    )

    parents, counts = stages.stage_b_continuation(candidates)

    assert tuple(parent.candidate_id for parent in parents) == ("B-0", "B-1")
    assert counts == (3, 4)


@pytest.mark.parametrize("locked", [False, True], ids=["free", "locked"])
def test_search_evidence_accounts_for_every_actual_optimizer_call(monkeypatch, locked) -> None:
    problem = _fully_locked_problem(seed=867, size=40) if locked else _problem(size=160)
    observed = []
    solve_local = stages.solve_local
    solve_global = stages.solve_global

    def local(context, start, *, max_nfev, **kwargs):
        solved = solve_local(context, start, max_nfev=max_nfev, **kwargs)
        observed.append((max_nfev, solved.nfev))
        return solved

    def global_search(context, start, *, population, maxiter, **kwargs):
        solved = solve_global(context, start, population=population, maxiter=maxiter, **kwargs)
        observed.append((len(population) * (maxiter + 1), solved.nfev))
        return solved

    monkeypatch.setattr(stages, "solve_local", local)
    monkeypatch.setattr(stages, "solve_global", global_search)

    result = run_fit_search(FitSearchRequest(None, problem))
    recorded = [
        (allocation.max_nfev, allocation.nfev)
        for summary in result.stage_summaries
        for evidence in summary.search_evidence
        for allocation in evidence.budget_allocations
    ]

    assert observed
    assert Counter(recorded) == Counter(observed), "budget evidence must include local work and locked paths"
    assert sum(used for _limit, used in recorded) <= sum(limit for limit, _used in recorded)


@pytest.mark.parametrize("stage", ["C", "D"])
def test_local_budget_refines_every_lineage_before_ranking_reclaimed_work(stage) -> None:
    problem = _problem(seed=762)
    parents = _parent_pair(problem)
    baseline = stages.run_local_stage(
        problem, None, stage, parents, perturbation_counts=(0, 0), progress=None, cancelled=None
    )
    preferred = min(range(2), key=lambda index: (baseline.candidates[index].objective, f"{stage}-{index}"))
    counts = tuple(0 if index == preferred else 1 for index in range(2))

    outcome = stages.run_local_stage(
        problem, None, stage, parents, perturbation_counts=counts, progress=None, cancelled=None
    )

    assert tuple(candidate.candidate_id for candidate in outcome.candidates[:2]) == (
        f"{stage}-0-0",
        f"{stage}-1-0",
    )
    assert outcome.candidates[2].candidate_id == f"{stage}-{preferred}-1"
    assert outcome.perturbation_counts == tuple(int(index == preferred) for index in range(2))
    allocations = _allocations(outcome)
    assert tuple(item.round_index for item in allocations) == (0, 0, 1)
    assert len({item.lineage_id for item in allocations[:2]}) == 2


def test_budget_rounds_never_allocate_twice_to_a_lineage_before_completing_the_round() -> None:
    problem = _problem(seed=767)
    parents = _parent_pair(problem)
    outcome = stages.run_local_stage(
        problem, None, "C", parents, perturbation_counts=(3, 0), progress=None, cancelled=None
    )
    allocations = _allocations(outcome)
    assert len(allocations) == 5
    assert tuple(item.round_index for item in allocations) == (0, 0, 1, 1, 2)
    for round_index in (0, 1):
        _assert_distinct_round(allocations, round_index, 2)
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * (len(problem.variables) + 1),
    )
    assert sum(item.max_nfev for item in allocations) <= 5 * maximum


@pytest.mark.parametrize(("seed", "stage", "size"), [(797, "C", 40), (730, "B", 160)])
def test_resume_keeps_each_completed_candidates_search_evidence(seed, stage, size) -> None:
    problem = _problem(seed=seed, size=size)
    checkpoints = []
    fresh = run_fit_search(FitSearchRequest(None, problem), checkpoint=checkpoints.append)
    completed = next(checkpoint for checkpoint in checkpoints if checkpoint.stage == stage)
    resumed = run_fit_search(FitSearchRequest(None, problem, completed))
    assert resumed.stage_summaries == fresh.stage_summaries
    assert tuple(candidate.search_evidence for candidate in resumed.candidates) == tuple(
        candidate.search_evidence for candidate in fresh.candidates
    )


def test_budget_rounds_replay_identically_with_parallel_lineages() -> None:
    problem = _problem(seed=797)
    sequential = run_fit_search(FitSearchRequest(None, problem))
    with ThreadPoolExecutor(max_workers=2) as executor:

        def run_tasks(tasks):
            return tuple(executor.map(lambda task: task(), tasks))

        parallel = run_fit_search(FitSearchRequest(None, problem), task_runner=run_tasks)

    assert parallel.stage_summaries == sequential.stage_summaries
    assert parallel.best_index == sequential.best_index
    for first, second in zip(sequential.candidates, parallel.candidates, strict=True):
        assert (first.candidate_id, first.nfev, first.objective) == (second.candidate_id, second.nfev, second.objective)
        assert first.search_evidence == second.search_evidence
        np.testing.assert_array_equal(first.unit_vector, second.unit_vector)


def test_cancel_during_last_local_solve_cannot_publish_a_completed_budget_round(monkeypatch) -> None:
    problem = _problem(seed=765)
    parent = _candidate(problem, "B-0", np.full(len(problem.variables), 0.5))
    cancelled = False
    original = stages.solve_local

    def cancel_after_solve(*args, **kwargs):
        nonlocal cancelled
        solved = original(*args, **kwargs)
        cancelled = True
        return solved

    monkeypatch.setattr(stages, "solve_local", cancel_after_solve)

    with pytest.raises(SearchCancelled):
        stages.run_local_stage(
            problem, None, "C", (parent,), perturbation_counts=(0,), progress=None, cancelled=lambda: cancelled
        )


@pytest.mark.parametrize(
    ("cancel_after", "completed_stages"),
    [(2, ()), (3, ("B",)), (4, ("B", "C")), (8, ("B", "C", "D"))],
)
def test_locked_stage_cancellation_keeps_only_previously_completed_checkpoints(
    monkeypatch, cancel_after, completed_stages
) -> None:
    problem = _fully_locked_problem(seed=867, size=40)
    original = stages.solve_local
    calls = 0
    cancelled = False
    checkpoints = []

    def solve(*args, **kwargs):
        nonlocal calls, cancelled
        solved = original(*args, **kwargs)
        calls += 1
        cancelled = calls == cancel_after
        return solved

    monkeypatch.setattr(stages, "solve_local", solve)
    with pytest.raises(SearchCancelled):
        run_fit_search(FitSearchRequest(None, problem), cancelled=lambda: cancelled, checkpoint=checkpoints.append)
    assert tuple(checkpoint.stage for checkpoint in checkpoints) == completed_stages
