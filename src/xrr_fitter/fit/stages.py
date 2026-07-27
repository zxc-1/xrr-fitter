"""Fixed fitting stage graph and deterministic child-stream reservation.

The stage graph keeps orchestration separate from solver implementation.
Stage A evaluates feature-diverse declared starts and applies screening.
Stage B performs short coarse global searches and protects the baseline.
Stage C refines active Stage-B lineages on the full dataset.
Stage D consumes reclaimed local budgets without reviving archived evidence.
Stage E runs four named global seeds, bounded local restarts, and elite carry.

Every published candidate is reevaluated against the full compiled problem.
Progress is emitted only after a seed completes, so cancellation cannot publish
partial work. Candidate IDs, child seeds, summaries, and continuation budgets
remain stable inputs for checkpoint and resume validation.
Resume suffixes therefore consume the same lineage and seed ledger as fresh runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from xrr_fitter.evaluation import PhysicalValueError, encode_physical_vector, values_by_name
from xrr_fitter.fit.candidates import (
    CandidateStart,
    archive_stage_b_candidates,
    best_candidate_index,
    bounded_perturbations,
    build_candidate_pool,
    candidate_from_evaluation,
    cluster_candidate_indices,
    rank_candidate_indices,
    select_coarse_candidates,
)
from xrr_fitter.fit.global_search import (
    build_de_population,
    build_stage_e_population,
    downsample_prepared_data,
    feature_grid_indices,
    solve_global,
)
from xrr_fitter.fit.local_search import SearchCancelled, solve_local
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_stage_problem
from xrr_fitter.fit.screening import fringe_count_screen
from xrr_fitter.model.fitting import FitCandidate, FitProgress, FitStageSummary


STAGE_ORDER = ("A", "B", "C", "D", "E")


@dataclass(frozen=True, slots=True)
class ChildSeed:
    stream_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class StageOutcome:
    candidates: tuple[FitCandidate, ...]
    summary: FitStageSummary
    warnings: tuple[str, ...] = ()
    perturbation_counts: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _StageESetup:
    coarse_problem: object
    full_problem: object
    centers: tuple[np.ndarray, ...]
    full_incumbents: tuple[np.ndarray, ...]
    population_size: int


def _stage_problems(
    problem: object,
    stage: str,
    current_values: dict[str, float],
) -> tuple[object, object]:
    full = compile_stage_problem(problem, stage, current_values)
    indices = feature_grid_indices(problem.data)
    if np.array_equal(indices, np.arange(problem.data.qz_a_inv.size)):
        return full, full
    coarse_data = downsample_prepared_data(problem.data, indices)
    coarse_base = replace(problem, data=coarse_data)
    return compile_stage_problem(coarse_base, stage, current_values), full


def _stream_order(stream_id: str) -> tuple[int, int, str]:
    prefix, separator, suffix = stream_id.partition("-")
    if not separator or not suffix.isdigit():
        return 2, 0, stream_id
    priority = {"E": 0, "B": 1}.get(prefix, 2)
    return priority, int(suffix), stream_id


def reserve_child_seeds(
    master_seed: int,
    stream_ids: tuple[str, ...],
) -> tuple[ChildSeed, ...]:
    """Map named streams to SeedSequence children independent of request order."""
    requested = tuple(stream_ids)
    if len(requested) != len(set(requested)) or any(not value for value in requested):
        raise ValueError("child stream IDs must be nonempty and unique")
    canonical = tuple(sorted(requested, key=_stream_order))
    spawned = np.random.SeedSequence(master_seed).spawn(len(canonical))
    by_stream = {
        stream_id: int(child.generate_state(1, dtype=np.uint64)[0])
        for stream_id, child in zip(canonical, spawned, strict=True)
    }
    return tuple(ChildSeed(stream_id, by_stream[stream_id]) for stream_id in requested)


def remaining_stages(completed_stage: str | None) -> tuple[str, ...]:
    if completed_stage is None:
        return STAGE_ORDER
    if completed_stage not in STAGE_ORDER:
        raise ValueError(f"unsupported fit stage: {completed_stage}")
    return STAGE_ORDER[STAGE_ORDER.index(completed_stage) + 1 :]


def _poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


def _emit(
    callback: Callable[[FitProgress], None] | None,
    dataset_id: str | None,
    stage: str,
    completed: int,
    total: int,
    best: float,
    message: str,
) -> None:
    if callback is not None:
        callback(FitProgress(dataset_id, stage, completed, total, best, message))


def _candidate_values(candidate: FitCandidate) -> dict[str, float]:
    return {value.name: value.value for value in candidate.parameters}


def _complete_values(problem: object, values: dict[str, float]) -> dict[str, float]:
    return {
        definition.name: values.get(definition.name, definition.initial)
        for definition in problem.parameter_definitions
    }


def _published_candidate(
    problem: object,
    stage_problem: object,
    stage_unit: np.ndarray,
    candidate_id: str,
    seed_index: int,
    stop_reason: str,
    nfev: int,
) -> FitCandidate:
    stage_evaluation = evaluate_vector(stage_problem, stage_unit)
    values = {value.name: value.value for value in stage_evaluation.parameters}
    full_unit = encode_physical_vector(problem, values)
    full_evaluation = evaluate_vector(problem, full_unit)
    return candidate_from_evaluation(
        problem,
        full_unit,
        full_evaluation,
        candidate_id,
        seed_index,
        stop_reason,
        nfev,
    )


def _summary(stage: str, candidates: tuple[FitCandidate, ...]) -> FitStageSummary:
    selectable = rank_candidate_indices(candidates)
    best = candidates[selectable[0]].objective if selectable else float("inf")
    return FitStageSummary(
        stage,
        tuple(candidate.candidate_id for candidate in candidates),
        best,
        sum(candidate.nfev for candidate in candidates),
        tuple(candidate.stop_reason for candidate in candidates),
    )


def _coarse_log_curve(problem: object, candidate: FitCandidate) -> np.ndarray:
    modeled = candidate.model_normalized[problem.data.fit_mask]
    return np.log10(np.maximum(modeled, problem.data.r_floor))


def _unique_feature_ids(starts: tuple[CandidateStart, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for start in starts:
        count = counts.get(start.feature_key, 0)
        result.append(start.feature_key if count == 0 else f"{start.feature_key}:{count}")
        counts[start.feature_key] = count + 1
    return tuple(result)


def _ensure_two_starts(starts: tuple[CandidateStart, ...]) -> tuple[CandidateStart, ...]:
    if len(starts) >= 2:
        return starts[:2]
    if not starts:
        raise ValueError("stage A produced no valid fitting candidates")
    return starts + (replace(starts[0], feature_key=f"{starts[0].feature_key}:alternate"),)


def _stage_a_candidate(problem: object, start: CandidateStart, index: int) -> FitCandidate | None:
    try:
        unit = encode_physical_vector(problem, dict(start.values))
    except PhysicalValueError:
        return None
    evaluation = evaluate_vector(problem, unit)
    return candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        f"A-{index}",
        -1,
        "evaluated",
        1,
    )


def _evaluate_stage_a_pool(
    problem: object,
    dataset_id: str | None,
    pool: tuple[CandidateStart, ...],
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[tuple[CandidateStart, FitCandidate], ...], int]:
    evaluated: list[tuple[CandidateStart, FitCandidate]] = []
    invalid_count = 0
    best = float("inf")
    for index, start in enumerate(pool):
        _poll(cancelled)
        candidate = _stage_a_candidate(problem, start, index)
        if candidate is None:
            invalid_count += 1
            _emit(progress, dataset_id, "A", index + 1, len(pool), best, "invalid_evaluation")
            continue
        invalid_count += int(not candidate.valid)
        if candidate.valid:
            evaluated.append((start, candidate))
            best = min(best, candidate.objective)
        _emit(progress, dataset_id, "A", index + 1, len(pool), best, candidate.stop_reason)
    return tuple(evaluated), invalid_count


def run_stage_a(
    problem: object,
    dataset_id: str | None,
    *,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[CandidateStart, ...], FitStageSummary, tuple[str, ...]]:
    """Evaluate, screen, and deterministically select the two coarse starts."""
    pool = build_candidate_pool(
        problem.data,
        problem.structure,
        problem.instrument,
        np.random.default_rng(problem.config.master_seed),
    )
    evaluated, invalid_count = _evaluate_stage_a_pool(
        problem,
        dataset_id,
        pool,
        progress,
        cancelled,
    )
    screen = fringe_count_screen(problem, tuple(candidate for _start, candidate in evaluated))
    survivors = {candidate.candidate_id for candidate in screen.candidates}
    accepted = tuple(item for item in evaluated if item[1].candidate_id in survivors)
    scored = tuple((candidate.objective, start) for start, candidate in accepted)
    curves = {start: _coarse_log_curve(problem, candidate) for start, candidate in accepted}
    selected = _ensure_two_starts(select_coarse_candidates(scored, curves, limit=2))
    warnings = tuple(screen.warnings)
    if invalid_count:
        warnings += ("stage_a_invalid_candidate_evaluation",)
    if len(accepted) < len(evaluated):
        warnings += ("stage_a_fringe_candidate_rejected",)
    summary = FitStageSummary(
        "A",
        _unique_feature_ids(selected),
        min(cost for cost, _start in scored),
        len(pool),
        ("evaluated",),
    )
    return selected, summary, warnings


def _stage_b_candidate(
    problem: object,
    start: CandidateStart,
    index: int,
    seed: int,
    cancelled: Callable[[], bool] | None,
) -> FitCandidate:
    values = _complete_values(problem, dict(start.values))
    coarse_problem, full_problem = _stage_problems(problem, "B", values)
    unit = encode_physical_vector(coarse_problem, values)
    if unit.size == 0:
        solved = solve_local(full_problem, unit, max_nfev=1, cancelled=cancelled)
    else:
        population = build_de_population(
            unit,
            seed=seed,
            population_size=max(32, 6 * unit.size),
        )
        solved = solve_global(
            coarse_problem,
            unit,
            population=population,
            seed=seed,
            maxiter=problem.config.budget.short_de_maxiter,
            cancelled=cancelled,
        )
    optimized = _published_candidate(
        problem,
        full_problem,
        solved.unit_vector,
        f"B-{index}",
        index,
        solved.stop_reason,
        solved.nfev,
    )
    if start.feature_key != "declared-baseline":
        return optimized
    baseline = _published_candidate(
        problem,
        full_problem,
        encode_physical_vector(full_problem, values),
        f"B-{index}",
        index,
        "declared_baseline",
        1,
    )
    winner = best_candidate_index((baseline, optimized))
    if winner != 0:
        return optimized
    return replace(
        baseline,
        stop_reason="declared_baseline_retained",
        nfev=baseline.nfev + optimized.nfev,
    )


def _best_objective(candidates: tuple[FitCandidate, ...]) -> float:
    winner = best_candidate_index(candidates)
    return float("inf") if winner is None else candidates[winner].objective


def run_stage_b(
    problem: object,
    dataset_id: str | None,
    starts: tuple[CandidateStart, ...],
    seeds: tuple[int, ...],
    *,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> StageOutcome:
    candidates: list[FitCandidate] = []
    for index, (start, seed) in enumerate(zip(starts, seeds, strict=True)):
        candidate = _stage_b_candidate(problem, start, index, seed, cancelled)
        candidates.append(candidate)
        best = _best_objective(tuple(candidates))
        _emit(progress, dataset_id, "B", index + 1, len(starts), best, candidate.stop_reason)
    archive = archive_stage_b_candidates(tuple(candidates))
    archived_by_id = {
        candidate.candidate_id: candidate
        for candidate in archive.active + archive.archived
    }
    values = tuple(archived_by_id[candidate.candidate_id] for candidate in candidates)
    return StageOutcome(
        values,
        _summary("B", values),
        perturbation_counts=archive.perturbation_counts,
    )


def _local_stage_candidate(
    problem: object,
    stage_problem: object,
    parent: FitCandidate,
    start: np.ndarray,
    candidate_id: str,
    cancelled: Callable[[], bool] | None,
) -> FitCandidate:
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * max(1, start.size),
    )
    solved = solve_local(stage_problem, start, max_nfev=maximum, cancelled=cancelled)
    return _published_candidate(
        problem,
        stage_problem,
        solved.unit_vector,
        candidate_id,
        parent.seed_index,
        solved.stop_reason,
        solved.nfev,
    )


def _local_stage_starts(
    problem: object,
    stage_problem: object,
    parent: FitCandidate,
    stage: str,
    cluster_index: int,
    perturbation_count: int,
) -> tuple[np.ndarray, ...]:
    center = encode_physical_vector(stage_problem, _candidate_values(parent))
    seed = int(
        np.random.SeedSequence(
            [problem.config.master_seed, ord(stage), cluster_index]
        ).generate_state(1, dtype=np.uint64)[0]
    )
    return (center, *bounded_perturbations(center, perturbation_count, seed=seed))


def run_local_stage(
    problem: object,
    dataset_id: str | None,
    stage: str,
    parents: tuple[FitCandidate, ...],
    *,
    perturbation_counts: tuple[int, ...] | None = None,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> StageOutcome:
    counts = (
        (0,) * len(parents)
        if perturbation_counts is None
        else tuple(perturbation_counts)
    )
    if len(counts) != len(parents):
        raise ValueError("local perturbation counts must align with parent clusters")
    total = sum(count + 1 for count in counts)
    candidates: list[FitCandidate] = []
    completed = 0
    for index, (parent, count) in enumerate(zip(parents, counts, strict=True)):
        values = _candidate_values(parent)
        stage_problem = compile_stage_problem(problem, stage, values)
        starts = _local_stage_starts(problem, stage_problem, parent, stage, index, count)
        lineage = parent.candidate_id.split("-")[1]
        for restart, start in enumerate(starts):
            candidate = _local_stage_candidate(
                problem,
                stage_problem,
                parent,
                start,
                f"{stage}-{lineage}-{restart}",
                cancelled,
            )
            candidates.append(candidate)
            completed += 1
            _emit(
                progress,
                dataset_id,
                stage,
                completed,
                total,
                _best_objective(tuple(candidates)),
                candidate.stop_reason,
            )
    values = tuple(candidates)
    return StageOutcome(values, _summary(stage, values), perturbation_counts=counts)


def stage_b_continuation(
    candidates: tuple[FitCandidate, ...],
    perturbation_counts: tuple[int, ...] = (),
) -> tuple[tuple[FitCandidate, ...], tuple[int, ...]]:
    """Recover active Stage-B parents and their reclaimed local budgets."""
    active = tuple(candidate for candidate in candidates if rank_candidate_indices((candidate,)))
    if perturbation_counts:
        if len(active) != len(perturbation_counts):
            raise ValueError("Stage-B perturbation counts do not match active candidates")
        return active, perturbation_counts
    archive = archive_stage_b_candidates(candidates)
    return archive.active, archive.perturbation_counts


def local_stage_continuation(
    candidates: tuple[FitCandidate, ...],
) -> tuple[tuple[FitCandidate, ...], tuple[int, ...]]:
    """Select one stable representative and retained budget per lineage."""
    grouped: dict[str, list[FitCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.candidate_id.split("-")[1], []).append(candidate)
    parents: list[FitCandidate] = []
    counts: list[int] = []
    for group in grouped.values():
        values = tuple(group)
        winner = best_candidate_index(values)
        parents.append(values[0 if winner is None else winner])
        counts.append(max(0, len(values) - 1))
    return tuple(parents), tuple(counts)


def _stage_e_setup(problem: object, parents: tuple[FitCandidate, ...]) -> _StageESetup:
    ranked = rank_candidate_indices(parents)
    if not ranked:
        raise ValueError("stage E has no selectable parent candidate")
    selected = tuple(parents[index] for index in ranked)
    values = _candidate_values(selected[0])
    coarse_problem, full_problem = _stage_problems(problem, "E", values)
    centers = tuple(
        encode_physical_vector(coarse_problem, _candidate_values(candidate))
        for candidate in selected
    )
    incumbents = [
        encode_physical_vector(full_problem, _candidate_values(candidate))
        for candidate in selected
    ]
    declared = encode_physical_vector(
        full_problem,
        {definition.name: definition.initial for definition in problem.parameter_definitions},
    )
    if not any(np.allclose(item, declared, rtol=0.0, atol=1e-14) for item in incumbents):
        incumbents.append(declared)
    return _StageESetup(
        coarse_problem,
        full_problem,
        centers,
        tuple(incumbents),
        max(64, 8 * len(coarse_problem.variables)),
    )


def _population_energies(problem: object, solved: object) -> tuple[np.ndarray, np.ndarray]:
    population = np.asarray(solved.population, dtype=float)
    if population.ndim != 2 or population.shape[1] != len(problem.variables):
        raise ValueError("Stage-E DE trace has an invalid population layout")
    supplied = getattr(solved, "population_energies", None)
    energies = (
        np.asarray([evaluate_vector(problem, row).objective for row in population])
        if supplied is None
        else np.asarray(supplied, dtype=float)
    )
    if population.shape[0] < 4 or energies.shape != (population.shape[0],):
        raise ValueError("Stage-E DE trace has an invalid population layout")
    return population, energies


def _population_starts(setup: _StageESetup, solved: object) -> tuple[np.ndarray, ...]:
    population, energies = _population_energies(setup.coarse_problem, solved)
    ranked = np.where(np.isfinite(energies), energies, np.inf)
    order = np.argsort(ranked, kind="stable")[:4]
    return tuple(
        encode_physical_vector(
            setup.full_problem,
            values_by_name(setup.coarse_problem, population[index]),
        )
        for index in order
    )


def _incumbent_starts(
    setup: _StageESetup,
    seed_index: int,
    child_seed: int,
    elite: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    if seed_index == 0:
        return setup.full_incumbents
    incumbents = setup.full_incumbents + (() if elite is None else (elite,))
    starts: list[np.ndarray] = []
    for index, incumbent in enumerate(incumbents):
        seed = int(
            np.random.SeedSequence([child_seed, ord("I"), index]).generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        starts.extend(bounded_perturbations(incumbent, 1, seed=seed, sigma=0.002))
    return tuple(starts)


def _stage_e_local_candidate(
    problem: object,
    setup: _StageESetup,
    start: np.ndarray,
    candidate_id: str,
    seed_index: int,
    cancelled: Callable[[], bool] | None,
) -> FitCandidate:
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * max(1, start.size),
    )
    solved = solve_local(setup.full_problem, start, max_nfev=maximum, cancelled=cancelled)
    return _published_candidate(
        problem,
        setup.full_problem,
        solved.unit_vector,
        candidate_id,
        seed_index,
        solved.stop_reason,
        solved.nfev,
    )


def _run_stage_e_locals(
    problem: object,
    setup: _StageESetup,
    starts: tuple[np.ndarray, ...],
    seed_index: int,
    kind: str,
    cancelled: Callable[[], bool] | None,
) -> list[FitCandidate]:
    return [
        _stage_e_local_candidate(
            problem,
            setup,
            start,
            f"E-{seed_index}-{kind}-{index}",
            seed_index,
            cancelled,
        )
        for index, start in enumerate(starts)
    ]


def _stage_e_seed(
    problem: object,
    setup: _StageESetup,
    seed_index: int,
    child_seed: int,
    elite: np.ndarray | None,
    cancelled: Callable[[], bool] | None,
) -> FitCandidate:
    if not setup.coarse_problem.variables:
        return _stage_e_local_candidate(
            problem,
            setup,
            setup.full_incumbents[0],
            f"E-{seed_index}",
            seed_index,
            cancelled,
        )
    population = build_stage_e_population(
        setup.centers,
        seed=child_seed,
        population_size=setup.population_size,
        perturbations_per_center=2,
    )
    solved = solve_global(
        setup.coarse_problem,
        setup.centers[0],
        population=population,
        seed=child_seed,
        maxiter=problem.config.budget.full_de_maxiter,
        cancelled=cancelled,
    )
    starts = _incumbent_starts(setup, seed_index, child_seed, elite)
    starts += _population_starts(setup, solved)
    attempts = _run_stage_e_locals(
        problem,
        setup,
        starts,
        seed_index,
        "local",
        cancelled,
    )
    winner_index = best_candidate_index(tuple(attempts))
    if winner_index is None:
        fallback = _published_candidate(
            problem,
            setup.full_problem,
            solved.unit_vector,
            f"E-{seed_index}-de-fallback",
            seed_index,
            solved.stop_reason,
            solved.nfev,
        )
    else:
        fallback = attempts[winner_index]
    restart_seed = int(
        np.random.SeedSequence([child_seed, ord("R")]).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )
    restart_starts = bounded_perturbations(
        encode_physical_vector(setup.full_problem, _candidate_values(fallback)),
        2,
        seed=restart_seed,
    )
    attempts.extend(
        _run_stage_e_locals(
            problem,
            setup,
            restart_starts,
            seed_index,
            "restart",
            cancelled,
        )
    )
    winner = best_candidate_index(tuple(attempts))
    selected = fallback if winner is None else attempts[winner]
    return replace(
        selected,
        candidate_id=f"E-{seed_index}",
        seed_index=seed_index,
        nfev=solved.nfev + sum(candidate.nfev for candidate in attempts),
    )


def _materially_improves(problem: object, incumbent: FitCandidate, candidate: FitCandidate) -> bool:
    thresholds = problem.config.confidence
    required = max(
        thresholds.equivalent_cost_fraction * abs(incumbent.objective),
        thresholds.equivalent_cost_floor,
    )
    return candidate.valid and candidate.objective + required < incumbent.objective


def run_stage_e(
    problem: object,
    dataset_id: str | None,
    parents: tuple[FitCandidate, ...],
    seeds: tuple[int, ...],
    *,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> StageOutcome:
    setup = _stage_e_setup(problem, parents)
    candidates: list[FitCandidate] = []
    best: FitCandidate | None = None
    elite: np.ndarray | None = None
    for index, seed in enumerate(seeds):
        _poll(cancelled)
        candidate = _stage_e_seed(
            problem,
            setup,
            index,
            seed,
            elite,
            cancelled,
        )
        candidates.append(candidate)
        winner = best_candidate_index(tuple(candidates))
        current = None if winner is None else candidates[winner]
        if best is not None and current is not None and current is not best:
            if _materially_improves(problem, best, current):
                elite = encode_physical_vector(
                    setup.full_problem,
                    _candidate_values(current),
                )
        best = current
        objective = float("inf") if best is None else best.objective
        _emit(progress, dataset_id, "E", index + 1, len(seeds), objective, candidate.stop_reason)
    values = tuple(candidates)
    return StageOutcome(values, _summary("E", values))
