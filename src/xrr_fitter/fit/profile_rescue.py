"""Atomic fit-owned profile-basin continuation with complete optimizer evidence.

Four starts reuse reserved Stage-E seed identities in the separate P namespace.
Only full-data evaluations can authorize replacements, and each replacement
keeps prior search work plus the actual budget used by its continuation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import numpy as np

from xrr_fitter.fit.candidates import (
    best_candidate_index,
    bounded_perturbations,
    candidate_from_evaluation,
    materially_improves,
)
from xrr_fitter.fit.local_budget import optimizer_evidence
from xrr_fitter.fit.local_search import LocalSearchResult, SearchCancelled, solve_local
from xrr_fitter.fit.tasking import TaskRunner
from xrr_fitter.fit.tasking import run_tasks as _run_tasks
from xrr_fitter.model.fitting import FitCandidate, FitEvaluationContext


def _poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


#
# Profile rescue start
#
def _profile_rescue_seed(child_seed: int, seed_index: int) -> int:
    return int(
        np.random.SeedSequence([int(child_seed), ord("P"), seed_index]).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )


def _profile_rescue_start(
    problem: FitEvaluationContext,
    center: np.ndarray,
    child_seed: int,
    seed_index: int,
) -> np.ndarray | None:
    """Derive one bounded profile-rescue start from a Stage-E child seed.

    The ``P`` namespace separates continuation perturbations from ordinary Stage-E
    incumbent and restart streams.
    """
    seed = _profile_rescue_seed(child_seed, seed_index)
    generated = bounded_perturbations(center, 1, seed=seed, sigma=0.002)
    if len(generated) != 1:
        return None
    start = np.asarray(generated[0], dtype=float)
    valid = (
        start.shape == (len(problem.variables),)
        and np.all(np.isfinite(start))
        and np.all((start >= 0.0) & (start <= 1.0))
    )
    return np.array(start, copy=True) if valid else None


#
# Profile rescue paths
#
def _profile_rescue_paths(
    problem: FitEvaluationContext,
    center: np.ndarray,
    child_seeds: tuple[int, ...],
    maximum: int,
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None,
) -> tuple[LocalSearchResult, ...] | None:
    """Run every distinct rescue start without publishing partial evidence.

    Duplicate or malformed starts reject the continuation before local search.
    Any missing local result rejects the complete four-path set; cancellation is
    polled before each start, each solve, and final return.
    """
    starts: list[np.ndarray] = []
    for seed_index, child_seed in enumerate(child_seeds):
        _poll(cancelled)
        start = _profile_rescue_start(problem, center, child_seed, seed_index)
        if start is None or any(np.array_equal(start, prior) for prior in starts):
            return None
        starts.append(start)
    tasks = tuple(
        partial(
            _solve_profile_rescue_path,
            problem,
            start,
            maximum,
            cancelled,
        )
        for start in starts
    )
    refined = _run_tasks(tasks, task_runner)
    if any(result is None for result in refined):
        return None
    _poll(cancelled)
    return refined


#
# Solve profile rescue path
#
def _solve_profile_rescue_path(
    problem: FitEvaluationContext,
    start: np.ndarray,
    maximum: int,
    cancelled: Callable[[], bool] | None,
) -> LocalSearchResult:
    _poll(cancelled)
    return solve_local(problem, start, max_nfev=maximum, cancelled=cancelled)


#
# Valid profile rescue result
#
def _valid_profile_rescue_result(problem: FitEvaluationContext, result: LocalSearchResult) -> bool:
    """Check one local result's coordinate layout and finite evaluation."""
    unit = np.asarray(result.unit_vector, dtype=float)
    evaluation = result.evaluation
    return bool(
        unit.shape == (len(problem.variables),)
        and np.all(np.isfinite(unit))
        and np.all((unit >= 0.0) & (unit <= 1.0))
        and evaluation.valid
        and np.isfinite(evaluation.objective)
    )


def _profile_rescue_evidence(original, result, child_seed, seed_index, maximum):
    rounds = (
        allocation.round_index for evidence in original.search_evidence for allocation in evidence.budget_allocations
    )
    round_index = max(rounds, default=-1) + 1
    evidence = optimizer_evidence(
        original.candidate_id,
        _profile_rescue_seed(child_seed, seed_index),
        f"{original.candidate_id}-profile-{round_index}",
        maximum,
        result,
        round_index,
    )
    return (*original.search_evidence, evidence)


#
# Publish profile rescue
#
def _publish_profile_rescue(
    problem: FitEvaluationContext,
    originals: tuple[FitCandidate, ...],
    refined: tuple[LocalSearchResult, ...],
    parameter_name: str,
    child_seeds: tuple[int, ...],
    maximum: int,
) -> tuple[FitCandidate, ...] | None:
    """Build four replacement candidates after all-or-nothing validation.

    The best refined evaluation must materially improve the original Stage-E
    incumbent. Every replacement retains its original path work and stable seed
    identity while recording the profile parameter in stop evidence.
    """
    if len(refined) != 4:
        return None
    if not all(_valid_profile_rescue_result(problem, result) for result in refined):
        return None
    incumbent_index = best_candidate_index(originals)
    if incumbent_index is None:
        return None
    best_result = min(refined, key=lambda result: result.evaluation.objective)
    if not materially_improves(
        problem,
        originals[incumbent_index],
        best_result.evaluation,
    ):
        return None
    return tuple(
        replace(
            candidate_from_evaluation(
                problem,
                result.unit_vector,
                result.evaluation,
                candidate_id=f"E-{seed_index}",
                seed_index=seed_index,
                stop_reason=(f"profile_basin_rescue:{parameter_name}:seed-{seed_index}:{result.stop_reason}"),
                nfev=int(original.nfev) + int(result.nfev),
            ),
            search_evidence=_profile_rescue_evidence(original, result, child_seed, seed_index, maximum),
        )
        for seed_index, (original, result, child_seed) in enumerate(zip(originals, refined, child_seeds, strict=True))
    )


#
# Reconverge profile basin
#
def reconverge_profile_basin(
    problem: FitEvaluationContext,
    stage_candidates: tuple[FitCandidate, ...],
    center_unit: np.ndarray,
    child_seeds: tuple[int, ...],
    *,
    parameter_name: str,
    cancelled: Callable[[], bool] | None = None,
    task_runner: TaskRunner | None = None,
) -> tuple[FitCandidate, ...] | None:
    """Rebuild all four Stage-E paths from one analysis-selected basin.

    Candidate and seed cardinality, seed indices, center shape, finiteness, and
    unit bounds are verified before any solve. The decision objective is ignored;
    only fit-owned full-data evaluations can authorize publication.
    """
    candidates = tuple(stage_candidates)
    seeds = tuple(child_seeds)
    if len(candidates) != 4 or len(seeds) != 4:
        return None
    if tuple(candidate.seed_index for candidate in candidates) != (0, 1, 2, 3):
        return None
    center = np.asarray(center_unit, dtype=float)
    valid_center = (
        center.shape == (len(problem.variables),)
        and np.all(np.isfinite(center))
        and np.all((center >= 0.0) & (center <= 1.0))
    )
    if not valid_center:
        return None
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * (center.size + 1),
    )
    refined = _profile_rescue_paths(problem, center, seeds, maximum, cancelled, task_runner)
    if refined is None:
        return None
    return _publish_profile_rescue(
        problem,
        candidates,
        refined,
        parameter_name,
        seeds,
        maximum,
    )
