"""Deterministic local restart rounds and per-call optimizer work records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.adaptive_grid import next_budget_lineage
from xrr_fitter.fit.candidates import best_candidate_index, bounded_perturbations
from xrr_fitter.fit.problem import compile_stage_problem
from xrr_fitter.model.fitting import FitCandidate, FitEvaluationContext, SearchAllocation, SearchEvidence


@dataclass(frozen=True, slots=True)
class LocalStageSetup:
    problem: FitEvaluationContext
    starts: tuple[np.ndarray, ...]
    seed: int


def optimizer_evidence(
    origin: str,
    seed: int,
    candidate_id: str,
    maximum: int,
    solved: object,
    round_index: int,
) -> SearchEvidence:
    allocation = SearchAllocation(origin, candidate_id, round_index, maximum, int(solved.nfev))
    return SearchEvidence(origin, seed, (), (allocation,), solved.stop_reason)


def local_stage_seed(problem: FitEvaluationContext, stage: str, cluster_index: int) -> int:
    return int(
        np.random.SeedSequence([problem.config.master_seed, ord(stage), cluster_index]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def local_stage_setups(
    problem: FitEvaluationContext,
    stage: str,
    parents: tuple[FitCandidate, ...],
    counts: tuple[int, ...],
) -> tuple[LocalStageSetup, ...]:
    if any(isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0 for count in counts):
        raise ValueError("local perturbation counts must be nonnegative integers")
    pool = int(sum(counts))
    setups = []
    for index, parent in enumerate(parents):
        values = {value.name: value.value for value in parent.parameters}
        stage_problem = compile_stage_problem(problem, stage, values)
        center = encode_physical_vector(stage_problem, values)
        seed = local_stage_seed(problem, stage, index)
        starts = (center, *bounded_perturbations(center, pool, seed=seed))
        setups.append(LocalStageSetup(stage_problem, starts, seed))
    return tuple(setups)


def _best_objective(candidates: tuple[FitCandidate, ...]) -> float:
    winner = best_candidate_index(candidates)
    return float("inf") if winner is None else candidates[winner].objective


def next_local_round(stage, setups, groups, remaining: int) -> tuple[int, ...]:
    indices = {
        f"{stage}-{index}": index for index in range(len(setups)) if len(groups[index]) < len(setups[index].starts)
    }
    costs = {lineage: _best_objective(tuple(groups[index])) for lineage, index in indices.items()}
    allocated: list[str] = []
    while len(allocated) < remaining:
        lineage = next_budget_lineage(costs, tuple(allocated))
        if lineage is None:
            break
        allocated.append(lineage)
    return tuple(indices[lineage] for lineage in allocated)
