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

Coarse contexts are compiled from the same declarations as the full problem;
only prepared observations are downsampled. Stage candidates are always decoded
through their stage context and then reevaluated on the full context before
publication. Coarse objectives therefore guide work but never become reported
full-data evidence.

Stage A retains a broad audit set while selecting a smaller, feature-diverse
launch set. Stage B preserves the declared launch beside differential-evolution
evidence and groups candidates only by structure coordinates. Invalid or
unselectable candidates remain auditable but do not become continuation parents.

Stages C and D preserve lineage identifiers while perturbing one representative
per parent. Reclaimed perturbation counts travel with the outcome instead of
being recomputed from an archived candidate list after resume.

Stage E combines ranked coarse population members, full-data incumbents, and
bounded deterministic restarts. A better candidate becomes an elite start for a
later seed only after exceeding the configured material-gain threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import numpy as np

from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    PhysicalValueError,
    encode_physical_vector,
)
from xrr_fitter.fit.adaptive_grid import initial_grid_points
from xrr_fitter.fit.adaptive_review import (
    compile_grid_problem,
    review_single_population,
    review_stage_a,
)
from xrr_fitter.fit.candidates import (
    CandidateStart,
    StageBArchive,
    archive_stage_b_candidates,
    best_candidate_index,
    bounded_perturbations,
    build_candidate_pool,
    candidate_from_evaluation,
    materially_improves,
    rank_candidate_indices,
    select_coarse_candidates,
    select_full_search_candidates,
)
from xrr_fitter.fit.global_search import (
    GlobalSearchResult,
    build_de_population,
    build_stage_e_population,
    solve_global,
)
from xrr_fitter.fit.local_budget import local_stage_setups, next_local_round, optimizer_evidence
from xrr_fitter.fit.local_search import LocalSearchResult, SearchCancelled, solve_local
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_stage_problem
from xrr_fitter.fit.progress import (
    best_preview_candidate as _best_candidate,
)
from xrr_fitter.fit.progress import (
    emit_progress as _emit,
)
from xrr_fitter.fit.screening import FringeScreenResult, fringe_count_screen
from xrr_fitter.fit.stage_schedule import (
    STAGE_ORDER,  # noqa: F401
    ChildSeed,  # noqa: F401
    remaining_stages,  # noqa: F401
    reserve_child_seeds,  # noqa: F401
)
from xrr_fitter.fit.tasking import TaskRunner
from xrr_fitter.fit.tasking import run_tasks as _run_tasks
from xrr_fitter.model.fitting import FitCandidate, FitEvaluationContext, FitProgress, FitStageSummary
from xrr_fitter.model.parameters import ParameterFreedom, ParameterSetting
from xrr_fitter.model.search import SearchEvidence


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Published stage evidence plus continuation-only budget metadata.

    Warnings and perturbation counts retain source order so fresh and resumed
    pipelines consume the same state.
    """

    candidates: tuple[FitCandidate, ...]
    summary: FitStageSummary
    warnings: tuple[str, ...] = ()
    perturbation_counts: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _StageESetup:
    """Shared full/coarse contexts and deterministic Stage-E launch geometry.

    Centers are coarse DE coordinates. Full incumbents are independently encoded
    against the complete context and may include the declared initial point.
    """

    coarse_problem: FitEvaluationContext
    full_problem: FitEvaluationContext
    centers: tuple[np.ndarray, ...]
    full_incumbents: tuple[np.ndarray, ...]
    population_size: int


#
# Parameter settings
#
def _parameter_settings(problem: FitEvaluationContext) -> tuple[ParameterSetting, ...]:
    return tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            ParameterFreedom.from_locked(definition.locked),
        )
        for definition in problem.parameter_definitions
    )


#
# Compile coarse problem
#
def compile_coarse_problem(problem: FitEvaluationContext) -> FitEvaluationContext:
    """Subset the numerical grid without recompiling full-data evidence."""
    return compile_grid_problem(problem, initial_grid_points(problem.data))


#
# Stage problems
#
def _stage_problems(
    problem: FitEvaluationContext,
    stage: str,
    current_values: dict[str, float],
) -> tuple[FitEvaluationContext, FitEvaluationContext]:
    full = compile_stage_problem(problem, stage, current_values)
    coarse = compile_coarse_problem(problem)
    return compile_stage_problem(coarse, stage, current_values), full


#
# Poll
#
def _poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


#
# Candidate values
#
def _candidate_values(candidate: FitCandidate) -> dict[str, float]:
    return {value.name: value.value for value in candidate.parameters}


#
# Complete values
#
def _complete_values(problem: FitEvaluationContext, values: dict[str, float]) -> dict[str, float]:
    return {
        definition.name: values.get(definition.name, definition.initial) for definition in problem.parameter_definitions
    }


#
# Published candidate
#
def _published_candidate(
    problem: FitEvaluationContext,
    stage_problem: FitEvaluationContext,
    stage_unit: np.ndarray,
    candidate_id: str,
    seed_index: int,
    stop_reason: str,
    nfev: int,
) -> FitCandidate:
    """Reevaluate a stage solution and publish only full-data evidence.

    Physical values bridge different stage coordinate layouts. The final
    evaluation receives one additional work count because publication itself
    performs a complete model evaluation.
    """
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
        nfev + 1,
    )


#
# Summary
#
def _summary(stage: str, candidates: tuple[FitCandidate, ...]) -> FitStageSummary:
    selectable = rank_candidate_indices(candidates)
    best = candidates[selectable[0]].objective if selectable else float("inf")
    return FitStageSummary(
        stage,
        tuple(candidate.candidate_id for candidate in candidates),
        best,
        sum(candidate.nfev for candidate in candidates),
        tuple(candidate.stop_reason for candidate in candidates),
        tuple(evidence for candidate in candidates for evidence in candidate.search_evidence),
    )


#
# Coarse log curve
#
def _coarse_log_curve(problem: FitEvaluationContext, candidate: FitCandidate) -> np.ndarray:
    modeled = candidate.model_normalized[problem.data.fit_mask]
    return np.log10(np.maximum(modeled, problem.data.r_floor))


#
# Unique feature ids
#
def _unique_feature_ids(starts: tuple[CandidateStart, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for start in starts:
        count = counts.get(start.feature_key, 0)
        result.append(start.feature_key if count == 0 else f"{start.feature_key}:{count}")
        counts[start.feature_key] = count + 1
    return tuple(result)


#
# Ensure two starts
#
def _ensure_two_starts(starts: tuple[CandidateStart, ...]) -> tuple[CandidateStart, ...]:
    if len(starts) >= 2:
        return starts[:2]
    if not starts:
        raise ValueError("stage A produced no valid fitting candidates")
    return starts + (replace(starts[0], feature_key=f"{starts[0].feature_key}:alternate"),)


#
# Stage a candidate
#
def _stage_a_candidate(problem: FitEvaluationContext, start: CandidateStart, index: int) -> FitCandidate | None:
    try:
        unit = encode_physical_vector(problem, dict(start.values))
    except (EvaluationConstraintError, PhysicalValueError):
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


#
# Evaluate stage a pool
#
def _evaluate_stage_a_pool(
    problem: FitEvaluationContext,
    dataset_id: str | None,
    pool: tuple[CandidateStart, ...],
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[tuple[CandidateStart, FitCandidate], ...], int, int]:
    evaluated: list[tuple[CandidateStart, FitCandidate]] = []
    rejected_count = 0
    invalid_count = 0
    best = float("inf")
    incumbent: FitCandidate | None = None
    for index, start in enumerate(pool):
        _poll(cancelled)
        candidate = _stage_a_candidate(problem, start, index)
        if candidate is None:
            rejected_count += 1
        elif candidate.valid and np.isfinite(candidate.objective):
            evaluated.append((start, candidate))
            if candidate.objective < best:
                best = candidate.objective
                incumbent = candidate
        else:
            invalid_count += 1
        message = (
            f"processed initial candidate {index + 1}; "
            f"physically rejected {rejected_count}; "
            f"invalid evaluations {invalid_count}"
        )
        # Always emit the current incumbent so the preview curve stays alive
        # even during long stretches without improvement. The panel-side
        # throttle (50ms) prevents canvas flicker.
        # Physically rejected starts never reach the objective. Reserve one
        # final progress unit for adaptive full-grid review and screening; the
        # coarse scan alone is not a committed Stage-A completion.
        _emit(
            progress,
            dataset_id,
            "A",
            index + 1,
            len(pool) + 1,
            best,
            message,
            incumbent,
            nfev=index + 1 - rejected_count,
        )
    return tuple(evaluated), rejected_count, invalid_count


#
# Stage a warnings
#
def _stage_a_warnings(
    screen_warnings: tuple[str, ...],
    evaluated_count: int,
    accepted_count: int,
    rejected_count: int,
    invalid_count: int,
) -> tuple[str, ...]:
    warnings: tuple[str, ...] = ()
    if rejected_count:
        warnings += ("stage_a_physical_candidate_rejected",)
    if invalid_count:
        warnings += ("stage_a_invalid_candidate_evaluation",)
    warnings += screen_warnings
    if accepted_count < evaluated_count:
        warnings += ("stage_a_fringe_candidate_rejected",)
    return warnings


#
# Stage a stop reasons
#
def _stage_a_stop_reasons(
    accepted: tuple[tuple[CandidateStart, FitCandidate], ...],
    rejected_count: int,
    invalid_count: int,
    fringe_rejected_count: int,
) -> tuple[str, ...]:
    reasons = ["evaluated" if accepted else "fallback_initial"]
    counts = (
        ("physical_rejected", rejected_count),
        ("invalid_evaluation", invalid_count),
        ("fringe_rejected", fringe_rejected_count),
    )
    reasons.extend(f"{label}:{count}" for label, count in counts if count)
    return tuple(reasons)


def _screen_stage_a(
    problem: FitEvaluationContext,
    reviewed: tuple[tuple[CandidateStart, FitCandidate], ...],
) -> tuple[tuple[tuple[CandidateStart, FitCandidate], ...], FringeScreenResult, int]:
    screen = fringe_count_screen(problem, tuple(candidate for _start, candidate in reviewed))
    survivors = {candidate.candidate_id for candidate in screen.candidates}
    accepted = tuple(item for item in reviewed if item[1].candidate_id in survivors)
    screen_accepted_count = len(accepted)
    if not accepted and reviewed:
        # A strict fringe screen can reject every launch after full-grid review
        # (for example, a user-supplied initial thickness one fringe away).
        # Keep the best complete objective as an auditable Stage-B baseline so
        # the run can recover instead of failing before any optimizer starts.
        accepted = reviewed[:1]
    return accepted, screen, screen_accepted_count


#
# Run stage a
#
def run_stage_a(
    problem: FitEvaluationContext,
    dataset_id: str | None,
    *,
    coarse_problem: FitEvaluationContext | None = None,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[CandidateStart, ...], FitStageSummary, tuple[str, ...]]:
    """Evaluate, screen, and deterministically select coarse Stage-B starts.

    The audit summary retains up to twenty-four feature-diverse IDs, while the
    full-search selector returns the two stable starts required by Stage B.
    Physical rejection, invalid evaluation, and fringe rejection remain distinct
    warning and stop-reason counts.
    """
    coarse = compile_coarse_problem(problem) if coarse_problem is None else coarse_problem
    pool = build_candidate_pool(
        coarse.data,
        coarse.structure,
        coarse.instrument,
        np.random.default_rng(problem.config.master_seed),
        limit=512,
    )
    evaluated, rejected_count, invalid_count = _evaluate_stage_a_pool(
        coarse,
        dataset_id,
        pool,
        progress,
        cancelled,
    )
    reviewed, evidence = review_stage_a(problem, evaluated, len(pool) - rejected_count, cancelled)
    accepted, screen, screen_accepted_count = _screen_stage_a(problem, reviewed)
    scored = tuple((candidate.objective, start) for start, candidate in accepted)
    curves = {start: _coarse_log_curve(problem, candidate) for start, candidate in accepted}
    reported = select_coarse_candidates(scored, curves, limit=24)
    selected = _ensure_two_starts(select_full_search_candidates(scored, curves, limit=8))
    warnings = _stage_a_warnings(
        tuple(screen.warnings),
        len(reviewed),
        screen_accepted_count,
        rejected_count,
        invalid_count,
    )
    if screen.stop_reason is not None and screen_accepted_count == 0:
        warnings = (*warnings, screen.stop_reason)
    stop_reasons = _stage_a_stop_reasons(
        accepted,
        rejected_count,
        invalid_count,
        len(reviewed) - screen_accepted_count,
    )
    summary = FitStageSummary(
        "A",
        _unique_feature_ids(reported),
        min(cost for cost, _start in scored),
        len(pool) - rejected_count,
        stop_reasons,
        (evidence,),
    )
    _emit(
        progress,
        dataset_id,
        "A",
        len(pool) + 1,
        len(pool) + 1,
        summary.best_objective,
        "completed initial full-grid review and screening",
        _best_candidate(tuple(candidate for _start, candidate in accepted)),
        nfev=evidence.grid_review_evaluations + evidence.full_review_evaluations,
    )
    return selected, summary, warnings


#
# Stage b candidate
#
def _stage_b_candidate(
    problem: FitEvaluationContext,
    start: CandidateStart,
    index: int,
    seed: int,
    cancelled: Callable[[], bool] | None,
    progress: Callable[[FitProgress], None] | None = None,
    dataset_id: str | None = None,
    total_launches: int = 1,
) -> FitCandidate:
    values = _complete_values(problem, dict(start.values))
    coarse_problem, full_problem = _stage_problems(problem, "B", values)
    unit = encode_physical_vector(coarse_problem, values)
    evidence: tuple[SearchEvidence, ...]
    solved: GlobalSearchResult | LocalSearchResult
    if unit.size == 0:
        solved = solve_local(full_problem, unit, max_nfev=1, cancelled=cancelled)
        evidence = (optimizer_evidence(f"B-{index}", seed, f"B-{index}", 1, solved, 0),)
    else:
        population = build_de_population(
            unit,
            seed=seed,
            population_size=max(32, 6 * unit.size),
        )

        #
        # B gen callback
        #
        def _b_gen_callback(xk: np.ndarray, best_obj: float, generation: int, nfev: int) -> None:
            preview = _published_candidate(problem, coarse_problem, xk, f"B-{index}", index, "running", 0)
            _emit(
                progress,
                dataset_id,
                "B",
                index + 1,
                total_launches,
                best_obj,
                f"DE generation (launch {index + 1})",
                preview,
                iteration=generation,
                nfev=nfev,
            )

        solved = solve_global(
            coarse_problem,
            unit,
            population=population,
            seed=seed,
            maxiter=problem.config.budget.short_de_maxiter,
            cancelled=cancelled,
            generation_callback=_b_gen_callback if progress else None,
        )
        reviewed, record = review_single_population(
            full_problem,
            solved,
            f"B-{index}",
            seed,
            len(population) * (problem.config.budget.short_de_maxiter + 1),
            cancelled=cancelled,
        )
        evidence = (record,)
        selected_unit = reviewed[0]
    if not unit.size:
        selected_unit = solved.unit_vector
    _poll(cancelled)
    candidate = _published_candidate(
        problem,
        full_problem,
        selected_unit,
        f"B-{index}",
        index,
        solved.stop_reason,
        solved.nfev,
    )
    _poll(cancelled)
    return replace(candidate, search_evidence=evidence)


#
# Stage b launch evidence
#
def _stage_b_launch_evidence(
    problem: FitEvaluationContext,
    start: CandidateStart,
    optimized: FitCandidate,
    index: int,
) -> tuple[FitCandidate, ...]:
    if start.feature_key != "declared-baseline":
        return (optimized,)
    values = _complete_values(problem, dict(start.values))
    baseline = _published_candidate(
        problem,
        problem,
        encode_physical_vector(problem, values),
        "B-declared-start",
        index,
        "declared_baseline",
        0,
    )
    return baseline, optimized


#
# Stage b geometry indices
#
def _stage_b_geometry_indices(problem: FitEvaluationContext) -> tuple[int, ...]:
    return tuple(
        index
        for index, variable in enumerate(problem.variables)
        if problem.parameter_definitions[variable.parameter_index].category == "structure"
    )


#
# Stage b geometry distance
#
def _stage_b_geometry_distance(
    first: FitCandidate,
    second: FitCandidate,
    indices: tuple[int, ...],
) -> float:
    if not indices:
        return 0.0
    positions = list(indices)
    difference = first.unit_vector[positions] - second.unit_vector[positions]
    return float(np.sqrt(np.mean(difference**2)))


#
# Stage b geometry group
#
def _stage_b_geometry_group(
    problem: FitEvaluationContext,
    candidates: tuple[FitCandidate, ...],
    groups: list[list[int]],
    candidate_index: int,
    geometry_indices: tuple[int, ...],
) -> list[int] | None:
    return next(
        (
            group
            for group in groups
            if _stage_b_geometry_distance(
                candidates[candidate_index],
                candidates[group[0]],
                geometry_indices,
            )
            <= problem.config.confidence.cluster_join_distance
        ),
        None,
    )


#
# Stage b representatives
#
def _stage_b_representatives(
    problem: FitEvaluationContext,
    candidates: tuple[FitCandidate, ...],
    limit: int = 4,
) -> tuple[FitCandidate, ...]:
    """Retain objective-ranked representatives of distinct structure geometry.

    Instrument-only coordinate differences cannot split a group. Published order
    follows original launch order after ranked grouping, and unselectable evidence
    remains visible even though it cannot represent a continuation group.
    """
    expected_shape = (len(problem.variables),)
    if any(candidate.unit_vector.shape != expected_shape for candidate in candidates):
        raise ValueError("Stage-B candidate unit layout does not match problem")
    ranked = sorted(
        rank_candidate_indices(candidates),
        key=lambda index: (
            candidates[index].objective,
            candidates[index].candidate_id,
            index,
        ),
    )
    geometry_indices = _stage_b_geometry_indices(problem)
    groups: list[list[int]] = []
    for index in ranked:
        target = _stage_b_geometry_group(
            problem,
            candidates,
            groups,
            index,
            geometry_indices,
        )
        if target is not None:
            target.append(index)
        elif len(groups) < limit:
            groups.append([index])
    representatives = {group[0] for group in groups}
    unselectable = set(range(len(candidates))) - set(ranked)
    retained = representatives | unselectable
    return tuple(candidate for index, candidate in enumerate(candidates) if index in retained)


#
# Best objective
#
def _best_objective(candidates: tuple[FitCandidate, ...]) -> float:
    winner = best_candidate_index(candidates)
    return float("inf") if winner is None else candidates[winner].objective


def _retain_stage_b_work(
    representatives: tuple[FitCandidate, ...],
    candidates: tuple[FitCandidate, ...],
) -> tuple[FitCandidate, ...]:
    """Geometry deduplication must not discard the work done by merged launches."""
    retained_ids = {candidate.candidate_id for candidate in representatives}
    merged = tuple(candidate for candidate in candidates if candidate.candidate_id not in retained_ids)
    if not merged:
        return representatives
    owner = representatives[0]
    evidence = tuple(
        record
        for candidate in candidates
        if candidate.candidate_id == owner.candidate_id or candidate.candidate_id not in retained_ids
        for record in candidate.search_evidence
    )
    retained = replace(owner, nfev=owner.nfev + sum(candidate.nfev for candidate in merged), search_evidence=evidence)
    return (retained, *representatives[1:])


#
# Run stage b
#
def run_stage_b(
    problem: FitEvaluationContext,
    dataset_id: str | None,
    starts: tuple[CandidateStart, ...],
    seeds: tuple[int, ...],
    *,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> StageOutcome:
    """Run each short DE launch and archive stable geometry representatives.

    A declared launch contributes both baseline and optimizer evidence. Grouping
    occurs only after every launch completes, then archive policy marks active
    representatives without changing publication order.
    """
    candidates: list[FitCandidate] = []
    for index, (start, seed) in enumerate(zip(starts, seeds, strict=True)):
        optimized = _stage_b_candidate(
            problem,
            start,
            index,
            seed,
            cancelled,
            progress=progress,
            dataset_id=dataset_id,
            total_launches=len(starts),
        )
        candidates.extend(_stage_b_launch_evidence(problem, start, optimized, index))
        current = tuple(candidates)
        _emit(
            progress,
            dataset_id,
            "B",
            index + 1,
            len(starts),
            _best_objective(current),
            f"completed short differential evolution {index + 1}",
            _best_candidate(current),
        )
    completed = tuple(candidates)
    representatives = _stage_b_representatives(problem, completed)
    representatives = _retain_stage_b_work(representatives, completed)
    archive = archive_stage_b_candidates(representatives)
    archived_by_id = {candidate.candidate_id: candidate for candidate in archive.active + archive.archived}
    values = tuple(archived_by_id[candidate.candidate_id] for candidate in representatives)
    _parents, counts = _ordered_stage_b_continuation(values, archive)
    return StageOutcome(
        values,
        _summary("B", values),
        perturbation_counts=counts,
    )


#
# Local stage candidate
#
def _local_stage_candidate(
    problem: FitEvaluationContext,
    stage_problem: FitEvaluationContext,
    start: np.ndarray,
    candidate_id: str,
    seed_index: int,
    cancelled: Callable[[], bool] | None,
    iteration_callback: Callable[[np.ndarray, int, int, float | None], None] | None = None,
    *,
    origin: str,
    seed: int,
    round_index: int,
) -> FitCandidate:
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * (start.size + 1),
    )
    solved = solve_local(
        stage_problem,
        start,
        max_nfev=maximum,
        cancelled=cancelled,
        iteration_callback=iteration_callback,
    )
    _poll(cancelled)
    candidate = _published_candidate(
        problem,
        stage_problem,
        solved.unit_vector,
        candidate_id,
        seed_index,
        solved.stop_reason,
        solved.nfev,
    )
    _poll(cancelled)
    evidence = optimizer_evidence(origin, seed, candidate_id, maximum, solved, round_index)
    return replace(candidate, search_evidence=(evidence,))


#
# Run local stage
#
def run_local_stage(
    problem: FitEvaluationContext,
    dataset_id: str | None,
    stage: str,
    parents: tuple[FitCandidate, ...],
    *,
    perturbation_counts: tuple[int, ...] | None = None,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None = None,
) -> StageOutcome:
    """Refine every lineage once, then allocate the bounded pool by full cost.

    Each subsequent round visits a lineage at most once. Independent lineages
    may run concurrently within a round; the next round waits for every result
    and ranks their complete objectives rather than worker completion order.
    """
    counts = (0,) * len(parents) if perturbation_counts is None else tuple(perturbation_counts)
    if len(counts) != len(parents):
        raise ValueError("local perturbation counts must align with parent clusters")
    setups = local_stage_setups(problem, stage, parents, counts)
    total = len(parents) + min(sum(counts), sum(len(setup.starts) - 1 for setup in setups))
    groups: list[list[FitCandidate]] = [[] for _parent in parents]
    candidates: list[FitCandidate] = []
    completed = 0
    message = {
        "C": "full-resolution density refinement",
        "D": "full-resolution roughness/instrument refinement",
    }.get(stage, f"completed local stage {stage}")

    def _make_local_cb(
        stg_problem: FitEvaluationContext,
        cid: str,
        sidx: int,
        position: int,
    ) -> Callable[[np.ndarray, int, int, float | None], None]:
        def _cb(unit_vector: np.ndarray, iteration: int, nfev: int, step: float | None) -> None:
            preview = _published_candidate(problem, stg_problem, unit_vector, cid, sidx, "running", 0)
            _emit(
                progress,
                dataset_id,
                stage,
                position,
                total,
                preview.objective,
                message,
                preview,
                iteration=iteration,
                nfev=nfev,
                step_size=step,
            )

        return _cb

    selected = tuple(range(len(parents)))
    while selected:
        tasks = tuple(
            partial(
                _local_stage_candidate,
                problem,
                setups[index].problem,
                setups[index].starts[len(groups[index])],
                f"{stage}-{index}-{len(groups[index])}",
                index,
                cancelled,
                _make_local_cb(setups[index].problem, f"{stage}-{index}-{len(groups[index])}", index, completed + 1)
                if progress
                else None,
                origin=f"{stage}-{index}",
                seed=setups[index].seed,
                round_index=len(groups[index]),
            )
            for index in selected
        )
        for index, candidate in zip(selected, _run_tasks(tasks, task_runner), strict=True):
            groups[index].append(candidate)
            candidates.append(candidate)
            completed += 1
            current = tuple(candidates)
            _emit(
                progress,
                dataset_id,
                stage,
                completed,
                total,
                _best_objective(current),
                message,
                _best_candidate(current),
            )
        selected = next_local_round(stage, setups, groups, total - completed)
    values = tuple(candidates)
    retained = tuple(max(0, len(group) - 1) for group in groups)
    return StageOutcome(values, _summary(stage, values), perturbation_counts=retained)


#
# Stage b continuation
#
def stage_b_continuation(
    candidates: tuple[FitCandidate, ...],
    perturbation_counts: tuple[int, ...] = (),
) -> tuple[tuple[FitCandidate, ...], tuple[int, ...]]:
    """Recover active Stage-B parents and their reclaimed local budgets.

    Fresh outcome counts follow publication order. Resume reconstructs that
    same order, keeping each count and index-derived seed on its original parent
    even when the archive's full-cost ranking differs from publication order.
    """
    active = _active_stage_b_candidates(candidates)
    if perturbation_counts:
        if len(active) != len(perturbation_counts):
            raise ValueError("Stage-B perturbation counts do not match active candidates")
        return active, perturbation_counts
    return _archive_stage_b_candidates(candidates)


def _active_stage_b_candidates(candidates: tuple[FitCandidate, ...]) -> tuple[FitCandidate, ...]:
    return tuple(candidate for candidate in candidates if rank_candidate_indices((candidate,)))


def _archive_stage_b_candidates(
    candidates: tuple[FitCandidate, ...],
) -> tuple[tuple[FitCandidate, ...], tuple[int, ...]]:
    archive = archive_stage_b_candidates(candidates)
    return _ordered_stage_b_continuation(candidates, archive)


def _ordered_stage_b_continuation(
    candidates: tuple[FitCandidate, ...],
    archive: StageBArchive,
) -> tuple[tuple[FitCandidate, ...], tuple[int, ...]]:
    counts = {
        candidate.candidate_id: count
        for candidate, count in zip(archive.active, archive.perturbation_counts, strict=True)
    }
    active = tuple(candidate for candidate in candidates if candidate.candidate_id in counts)
    return active, tuple(counts[candidate.candidate_id] for candidate in active)


#
# Local stage continuation
#
def local_stage_continuation(
    candidates: tuple[FitCandidate, ...],
    perturbation_counts: tuple[int, ...] = (),
) -> tuple[tuple[FitCandidate, ...], tuple[int, ...]]:
    """Select one stable representative and retained budget per lineage.

    Group insertion order preserves parent order. A supplied resumed budget is
    accepted only when every lineage has one corresponding count.
    """
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
    retained = tuple(perturbation_counts) if perturbation_counts else tuple(counts)
    if len(retained) != len(parents):
        raise ValueError("local perturbation counts do not match candidate lineages")
    return tuple(parents), retained


#
# Stage e setup
#
def _stage_e_setup(problem: FitEvaluationContext, parents: tuple[FitCandidate, ...]) -> _StageESetup:
    """Build Stage-E coarse centers and complete full-data incumbent starts.

    Only selectable parents participate. The declared initial vector is appended
    when no incumbent already represents it within the frozen unit tolerance.
    Population size scales deterministically with the coarse free dimension.
    """
    ranked = rank_candidate_indices(parents)
    if not ranked:
        raise ValueError("stage E has no selectable parent candidate")
    selected = tuple(parents[index] for index in ranked)
    values = _candidate_values(selected[0])
    coarse_problem, full_problem = _stage_problems(problem, "E", values)
    centers = tuple(encode_physical_vector(coarse_problem, _candidate_values(candidate)) for candidate in selected)
    incumbents = [encode_physical_vector(full_problem, _candidate_values(candidate)) for candidate in selected]
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


#
# Incumbent starts
#
def _incumbent_starts(
    setup: _StageESetup,
    seed_index: int,
    child_seed: int,
    elite: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    """Return exact first-seed incumbents or later seeded perturbations.

    Later seeds include a materially improved elite when available. Each source
    incumbent receives its own derived perturbation stream.
    """
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


#
# Stage e local candidate
#
def _stage_e_local_candidate(
    problem: FitEvaluationContext,
    setup: _StageESetup,
    start: np.ndarray,
    candidate_id: str,
    seed_index: int,
    cancelled: Callable[[], bool] | None,
    iteration_callback: Callable[[np.ndarray, int, int, float | None], None] | None = None,
    *,
    source_seed: int,
    round_index: int,
) -> FitCandidate:
    return _local_stage_candidate(
        problem,
        setup.full_problem,
        start,
        candidate_id,
        seed_index,
        cancelled,
        iteration_callback,
        origin=f"E-{seed_index}",
        seed=source_seed,
        round_index=round_index,
    )


#
# Run stage e locals
#
def _run_stage_e_locals(
    problem: FitEvaluationContext,
    setup: _StageESetup,
    starts: tuple[np.ndarray, ...],
    seed_index: int,
    kind: str,
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None,
    progress: Callable[[FitProgress], None] | None = None,
    dataset_id: str | None = None,
    total_seeds: int = 1,
    *,
    source_seed: int,
) -> list[FitCandidate]:
    # Result positions retain start order so winner selection and nfev totals
    # are independent of worker completion timing.

    def _make_e_local_cb(cid: str, sidx: int) -> Callable[[np.ndarray, int, int, float | None], None]:
        #
        # Cb
        #
        def _cb(unit_vector: np.ndarray, iteration: int, nfev: int, step: float | None) -> None:
            preview = _published_candidate(
                problem,
                setup.full_problem,
                unit_vector,
                cid,
                sidx,
                "running",
                0,
            )
            _emit(
                progress,
                dataset_id,
                "E",
                seed_index + 1,
                total_seeds,
                preview.objective,
                f"local refinement (seed {seed_index + 1})",
                preview,
                iteration=iteration,
                nfev=nfev,
                step_size=step,
            )

        return _cb

    tasks = tuple(
        partial(
            _stage_e_local_candidate,
            problem,
            setup,
            start,
            f"E-{seed_index}-{kind}-{index}",
            seed_index,
            cancelled,
            _make_e_local_cb(f"E-{seed_index}-{kind}-{index}", seed_index) if progress else None,
            source_seed=source_seed,
            round_index=1 if kind == "local" else 2,
        )
        for index, start in enumerate(starts)
    )
    return list(_run_tasks(tasks, task_runner))


#
# Stage e seed
#
def _stage_e_seed(
    problem: FitEvaluationContext,
    setup: _StageESetup,
    seed_index: int,
    child_seed: int,
    elite: np.ndarray | None,
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None,
    progress: Callable[[FitProgress], None] | None = None,
    dataset_id: str | None = None,
    total_seeds: int = 1,
) -> FitCandidate:
    """Run one complete Stage-E global, local, and restart path.

    Four ranked DE population members join incumbent starts. If no local attempt
    is selectable, the DE result is published as the fallback. Two deterministic
    perturbation restarts then challenge the current winner, and the final public
    candidate aggregates all work under the seed's stable ``E-N`` identity.
    """
    if not setup.coarse_problem.variables:
        return _stage_e_local_candidate(
            problem,
            setup,
            setup.full_incumbents[0],
            f"E-{seed_index}",
            seed_index,
            cancelled,
            source_seed=child_seed,
            round_index=0,
        )
    population = build_stage_e_population(
        setup.centers,
        seed=child_seed,
        population_size=setup.population_size,
        perturbations_per_center=2,
    )

    #
    # E gen callback
    #
    def _e_gen_callback(xk: np.ndarray, best_obj: float, generation: int, nfev: int) -> None:
        preview = _published_candidate(
            problem,
            setup.coarse_problem,
            xk,
            f"E-{seed_index}",
            seed_index,
            "running",
            0,
        )
        _emit(
            progress,
            dataset_id,
            "E",
            seed_index + 1,
            total_seeds,
            best_obj,
            f"DE generation (seed {seed_index + 1})",
            preview,
            iteration=generation,
            nfev=nfev,
        )

    solved = solve_global(
        setup.coarse_problem,
        setup.centers[0],
        population=population,
        seed=child_seed,
        maxiter=problem.config.budget.full_de_maxiter,
        cancelled=cancelled,
        generation_callback=_e_gen_callback if progress else None,
    )
    population_starts, evidence = review_single_population(
        setup.full_problem,
        solved,
        f"E-{seed_index}",
        child_seed,
        len(population) * (problem.config.budget.full_de_maxiter + 1),
        cancelled=cancelled,
    )
    starts = _incumbent_starts(setup, seed_index, child_seed, elite)
    starts += population_starts[:4]
    attempts = _run_stage_e_locals(
        problem,
        setup,
        starts,
        seed_index,
        "local",
        cancelled,
        task_runner,
        progress=progress,
        dataset_id=dataset_id,
        total_seeds=total_seeds,
        source_seed=child_seed,
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
            task_runner,
            progress=progress,
            dataset_id=dataset_id,
            total_seeds=total_seeds,
            source_seed=restart_seed,
        )
    )
    winner = best_candidate_index(tuple(attempts))
    selected = fallback if winner is None else attempts[winner]
    return replace(
        selected,
        candidate_id=f"E-{seed_index}",
        seed_index=seed_index,
        nfev=solved.nfev + sum(candidate.nfev for candidate in attempts),
        search_evidence=(evidence, *(record for candidate in attempts for record in candidate.search_evidence)),
    )


#
# Run stage e
#
def run_stage_e(
    problem: FitEvaluationContext,
    dataset_id: str | None,
    parents: tuple[FitCandidate, ...],
    seeds: tuple[int, ...],
    *,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None = None,
) -> StageOutcome:
    """Run final named seeds and carry only materially improved elite state.

    One public candidate is emitted per seed. Progress follows successful seed
    completion, and the outcome summary preserves seed order regardless of which
    candidate becomes the current best.
    """
    # Seed paths stay sequential because the next seed may consume the prior
    # elite; only local attempts within one path are independent.
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
            task_runner,
            progress=progress,
            dataset_id=dataset_id,
            total_seeds=len(seeds),
        )
        candidates.append(candidate)
        winner = best_candidate_index(tuple(candidates))
        current = None if winner is None else candidates[winner]
        if best is not None and current is not None and current is not best:
            if materially_improves(problem, best, current):
                elite = encode_physical_vector(
                    setup.full_problem,
                    _candidate_values(current),
                )
        best = current
        objective = float("inf") if best is None else best.objective
        _emit(
            progress,
            dataset_id,
            "E",
            index + 1,
            len(seeds),
            objective,
            f"completed final seed {index + 1}",
            best,
        )
    values = tuple(candidates)
    return StageOutcome(values, _summary("E", values))
