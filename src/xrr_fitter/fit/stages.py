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

Profile-basin recovery is fit-owned continuation. It accepts an opaque decision
value, validates its unit center, creates four distinct starts from the reserved
Stage-E seeds, and atomically returns four replacement candidates only after all
paths are valid and materially improve the incumbent. This module never imports
analysis or trusts an analysis-reported objective.
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
    values_by_name,
)
from xrr_fitter.fit.candidates import (
    CandidateStart,
    archive_stage_b_candidates,
    best_candidate_index,
    bounded_perturbations,
    build_candidate_pool,
    candidate_from_evaluation,
    rank_candidate_indices,
    select_coarse_candidates,
    select_full_search_candidates,
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
from xrr_fitter.fit.problem import compile_fit_problem, compile_stage_problem
from xrr_fitter.fit.progress import (
    best_preview_candidate as _best_candidate,
)
from xrr_fitter.fit.progress import (
    emit_progress as _emit,
)
from xrr_fitter.fit.screening import fringe_count_screen
from xrr_fitter.fit.stage_schedule import (
    STAGE_ORDER,  # noqa: F401
    ChildSeed,  # noqa: F401
    remaining_stages,  # noqa: F401
    reserve_child_seeds,  # noqa: F401
)
from xrr_fitter.fit.tasking import TaskRunner
from xrr_fitter.fit.tasking import run_tasks as _run_tasks
from xrr_fitter.model.fitting import FitCandidate, FitProgress, FitStageSummary
from xrr_fitter.model.parameters import ParameterSetting


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

    coarse_problem: object
    full_problem: object
    centers: tuple[np.ndarray, ...]
    full_incumbents: tuple[np.ndarray, ...]
    population_size: int


#
# Parameter settings
#
def _parameter_settings(problem: object) -> tuple[ParameterSetting, ...]:
    return tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            definition.locked,
        )
        for definition in problem.parameter_definitions
    )


#
# Compile coarse problem
#
def compile_coarse_problem(problem: object) -> object:
    """Compile the immutable feature-grid context used by coarse stages.

    A problem already on the selected grid is reused exactly. Otherwise all
    declarations and parameter settings are recompiled around downsampled
    prepared data instead of mutating or replacing one field in the context.
    """
    indices = feature_grid_indices(problem.data)
    if np.array_equal(indices, np.arange(problem.data.qz_a_inv.size)):
        return problem
    return compile_fit_problem(
        downsample_prepared_data(problem.data, indices),
        problem.structure,
        problem.instrument,
        problem.config,
        _parameter_settings(problem),
        problem.constraint_rules,
    )


#
# Stage problems
#
def _stage_problems(
    problem: object,
    stage: str,
    current_values: dict[str, float],
) -> tuple[object, object]:
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
def _complete_values(problem: object, values: dict[str, float]) -> dict[str, float]:
    return {
        definition.name: values.get(definition.name, definition.initial) for definition in problem.parameter_definitions
    }


#
# Published candidate
#
def _published_candidate(
    problem: object,
    stage_problem: object,
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
    )


#
# Coarse log curve
#
def _coarse_log_curve(problem: object, candidate: FitCandidate) -> np.ndarray:
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
def _stage_a_candidate(problem: object, start: CandidateStart, index: int) -> FitCandidate | None:
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
    problem: object,
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
        _emit(
            progress,
            dataset_id,
            "A",
            index + 1,
            len(pool),
            best,
            message,
            incumbent,
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


#
# Run stage a
#
def run_stage_a(
    problem: object,
    dataset_id: str | None,
    *,
    coarse_problem: object | None = None,
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
    screen = fringe_count_screen(coarse, tuple(candidate for _start, candidate in evaluated))
    survivors = {candidate.candidate_id for candidate in screen.candidates}
    accepted = tuple(item for item in evaluated if item[1].candidate_id in survivors)
    scored = tuple((candidate.objective, start) for start, candidate in accepted)
    curves = {start: _coarse_log_curve(coarse, candidate) for start, candidate in accepted}
    reported = select_coarse_candidates(scored, curves, limit=24)
    selected = _ensure_two_starts(select_full_search_candidates(scored, curves, limit=8))
    warnings = _stage_a_warnings(
        tuple(screen.warnings),
        len(evaluated),
        len(accepted),
        rejected_count,
        invalid_count,
    )
    stop_reasons = _stage_a_stop_reasons(
        accepted,
        rejected_count,
        invalid_count,
        len(evaluated) - len(accepted),
    )
    summary = FitStageSummary(
        "A",
        _unique_feature_ids(reported),
        min(cost for cost, _start in scored),
        len(pool) - rejected_count,
        stop_reasons,
    )
    return selected, summary, warnings


#
# Stage b candidate
#
def _stage_b_candidate(
    problem: object,
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
    if unit.size == 0:
        solved = solve_local(full_problem, unit, max_nfev=1, cancelled=cancelled)
    else:
        population = build_de_population(
            unit,
            seed=seed,
            population_size=max(32, 6 * unit.size),
        )

        #
        # B gen callback
        #
        def _b_gen_callback(xk: np.ndarray, best_obj: float) -> None:
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
    return _published_candidate(
        problem,
        full_problem,
        solved.unit_vector,
        f"B-{index}",
        index,
        solved.stop_reason,
        solved.nfev,
    )


#
# Stage b launch evidence
#
def _stage_b_launch_evidence(
    problem: object,
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
def _stage_b_geometry_indices(problem: object) -> tuple[int, ...]:
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
    problem: object,
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
    problem: object,
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


#
# Run stage b
#
def run_stage_b(
    problem: object,
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
    representatives = _stage_b_representatives(problem, tuple(candidates))
    archive = archive_stage_b_candidates(representatives)
    archived_by_id = {candidate.candidate_id: candidate for candidate in archive.active + archive.archived}
    values = tuple(archived_by_id[candidate.candidate_id] for candidate in representatives)
    return StageOutcome(
        values,
        _summary("B", values),
        perturbation_counts=archive.perturbation_counts,
    )


#
# Local stage candidate
#
def _local_stage_candidate(
    problem: object,
    stage_problem: object,
    start: np.ndarray,
    candidate_id: str,
    seed_index: int,
    cancelled: Callable[[], bool] | None,
    iteration_callback: Callable[[np.ndarray], None] | None = None,
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
    return _published_candidate(
        problem,
        stage_problem,
        solved.unit_vector,
        candidate_id,
        seed_index,
        solved.stop_reason,
        solved.nfev,
    )


#
# Local stage starts
#
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
        np.random.SeedSequence([problem.config.master_seed, ord(stage), cluster_index]).generate_state(
            1, dtype=np.uint64
        )[0]
    )
    return (center, *bounded_perturbations(center, perturbation_count, seed=seed))


#
# Run local stage
#
def run_local_stage(
    problem: object,
    dataset_id: str | None,
    stage: str,
    parents: tuple[FitCandidate, ...],
    *,
    perturbation_counts: tuple[int, ...] | None = None,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None = None,
) -> StageOutcome:
    """Refine each parent and its deterministic bounded perturbations.

    Stage-specific compilation releases only the intended parameter groups.
    Selected parents receive compact stage-local lineage identifiers. Retained
    perturbation counts preserve that order for the next stage and checkpoint
    resume.
    """
    counts = (0,) * len(parents) if perturbation_counts is None else tuple(perturbation_counts)
    if len(counts) != len(parents):
        raise ValueError("local perturbation counts must align with parent clusters")
    total = sum(count + 1 for count in counts)
    candidates: list[FitCandidate] = []
    completed = 0
    message = {
        "C": "full-resolution density refinement",
        "D": "full-resolution roughness/instrument refinement",
    }.get(stage, f"completed local stage {stage}")
    for index, (parent, count) in enumerate(zip(parents, counts, strict=True)):
        # A parent remains sequentially dependent, while its declared restarts
        # form one ordered batch that can safely execute concurrently.
        values = _candidate_values(parent)
        stage_problem = compile_stage_problem(problem, stage, values)
        starts = _local_stage_starts(problem, stage_problem, parent, stage, index, count)

        #
        # Make local cb
        #
        def _make_local_cb(stg_problem, cid, sidx, _completed=completed):
            #
            # Cb
            #
            def _cb(unit_vector: np.ndarray) -> None:
                preview = _published_candidate(
                    problem,
                    stg_problem,
                    unit_vector,
                    cid,
                    sidx,
                    "running",
                    0,
                )
                _emit(
                    progress,
                    dataset_id,
                    stage,
                    _completed + 1,
                    total,
                    preview.objective,
                    message,
                    preview,
                )

            return _cb

        tasks = tuple(
            partial(
                _local_stage_candidate,
                problem,
                stage_problem,
                start,
                f"{stage}-{index}-{restart}",
                index,
                cancelled,
                _make_local_cb(stage_problem, f"{stage}-{index}-{restart}", index) if progress else None,
            )
            for restart, start in enumerate(starts)
        )
        for candidate in _run_tasks(tasks, task_runner):
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
    values = tuple(candidates)
    return StageOutcome(values, _summary(stage, values), perturbation_counts=counts)


#
# Stage b continuation
#
def stage_b_continuation(
    candidates: tuple[FitCandidate, ...],
    perturbation_counts: tuple[int, ...] = (),
) -> tuple[tuple[FitCandidate, ...], tuple[int, ...]]:
    """Recover active Stage-B parents and their reclaimed local budgets.

    Supplied counts belong to a resumed outcome and must align exactly with the
    active candidates. Fresh continuation derives the same values from archive
    policy.
    """
    active = tuple(candidate for candidate in candidates if rank_candidate_indices((candidate,)))
    if perturbation_counts:
        if len(active) != len(perturbation_counts):
            raise ValueError("Stage-B perturbation counts do not match active candidates")
        return active, perturbation_counts
    archive = archive_stage_b_candidates(candidates)
    return archive.active, archive.perturbation_counts


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
def _stage_e_setup(problem: object, parents: tuple[FitCandidate, ...]) -> _StageESetup:
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
# Population energies
#
def _population_energies(problem: object, solved: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize one solver population trace.

    Missing energies are reevaluated in row order; supplied energies must align
    exactly and expose at least four population members.
    """
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


#
# Population starts
#
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
    problem: object,
    setup: _StageESetup,
    start: np.ndarray,
    candidate_id: str,
    seed_index: int,
    cancelled: Callable[[], bool] | None,
    iteration_callback: Callable[[np.ndarray], None] | None = None,
) -> FitCandidate:
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * (start.size + 1),
    )
    solved = solve_local(
        setup.full_problem,
        start,
        max_nfev=maximum,
        cancelled=cancelled,
        iteration_callback=iteration_callback,
    )
    return _published_candidate(
        problem,
        setup.full_problem,
        solved.unit_vector,
        candidate_id,
        seed_index,
        solved.stop_reason,
        solved.nfev,
    )


#
# Run stage e locals
#
def _run_stage_e_locals(
    problem: object,
    setup: _StageESetup,
    starts: tuple[np.ndarray, ...],
    seed_index: int,
    kind: str,
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None,
    progress: Callable[[FitProgress], None] | None = None,
    dataset_id: str | None = None,
    total_seeds: int = 1,
) -> list[FitCandidate]:
    # Result positions retain start order so winner selection and nfev totals
    # are independent of worker completion timing.

    def _make_e_local_cb(cid, sidx):
        #
        # Cb
        #
        def _cb(unit_vector: np.ndarray) -> None:
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
        )
        for index, start in enumerate(starts)
    )
    return list(_run_tasks(tasks, task_runner))


#
# Stage e seed
#
def _stage_e_seed(
    problem: object,
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
    def _e_gen_callback(xk: np.ndarray, best_obj: float) -> None:
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
    starts = _incumbent_starts(setup, seed_index, child_seed, elite)
    starts += _population_starts(setup, solved)
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


#
# Materially improves
#
def _materially_improves(problem: object, incumbent: FitCandidate, candidate: FitCandidate) -> bool:
    thresholds = problem.config.confidence
    required = max(
        thresholds.equivalent_cost_fraction * abs(incumbent.objective),
        thresholds.equivalent_cost_floor,
    )
    return candidate.valid and candidate.objective + required < incumbent.objective


#
# Profile rescue start
#
def _profile_rescue_start(
    problem: object,
    center: np.ndarray,
    child_seed: int,
    seed_index: int,
) -> np.ndarray | None:
    """Derive one bounded profile-rescue start from a Stage-E child seed.

    The ``P`` namespace separates continuation perturbations from ordinary Stage-E
    incumbent and restart streams.
    """
    seed = int(
        np.random.SeedSequence([int(child_seed), ord("P"), seed_index]).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )
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
    problem: object,
    center: np.ndarray,
    child_seeds: tuple[int, ...],
    cancelled: Callable[[], bool] | None,
    task_runner: TaskRunner | None,
) -> tuple[object, ...] | None:
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
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * (center.size + 1),
    )
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
    problem: object,
    start: np.ndarray,
    maximum: int,
    cancelled: Callable[[], bool] | None,
):
    _poll(cancelled)
    return solve_local(problem, start, max_nfev=maximum, cancelled=cancelled)


#
# Valid profile rescue result
#
def _valid_profile_rescue_result(problem: object, result: object) -> bool:
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


#
# Publish profile rescue
#
def _publish_profile_rescue(
    problem: object,
    originals: tuple[FitCandidate, ...],
    refined: tuple[object, ...],
    parameter_name: str,
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
    if not _materially_improves(
        problem,
        originals[incumbent_index],
        best_result.evaluation,
    ):
        return None
    return tuple(
        candidate_from_evaluation(
            problem,
            result.unit_vector,
            result.evaluation,
            candidate_id=f"E-{seed_index}",
            seed_index=seed_index,
            stop_reason=(f"profile_basin_rescue:{parameter_name}:seed-{seed_index}:{result.stop_reason}"),
            nfev=int(original.nfev) + int(result.nfev),
        )
        for seed_index, (original, result) in enumerate(zip(originals, refined, strict=True))
    )


#
# Reconverge profile basin
#
def reconverge_profile_basin(
    problem: object,
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
    refined = _profile_rescue_paths(problem, center, seeds, cancelled, task_runner)
    if refined is None:
        return None
    return _publish_profile_rescue(
        problem,
        candidates,
        refined,
        parameter_name,
    )


#
# Run stage e
#
def run_stage_e(
    problem: object,
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
            if _materially_improves(problem, best, current):
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
