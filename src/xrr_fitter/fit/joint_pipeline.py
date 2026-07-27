"""Pure joint fitting, atomic checkpoints, resume, and batch dispatch.

Each joint stage compiles only the coordinates that are free at that stage.
Sharing rules survive compilation only when every member remains jointly free.
Solvers work in stage coordinates, then solutions are encoded into the full
joint layout and reevaluated before candidates or progress are published.
Checkpoint batches transpose that full solution history back by dataset.
Resume accepts a batch only when every member describes the same joint state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from xrr_fitter.evaluation import (
    encode_physical_vector,
    values_by_name,
)
from xrr_fitter.fit.candidates import best_candidate_index, bounded_perturbations, candidate_from_evaluation
from xrr_fitter.fit.checkpoint import build_checkpoint
from xrr_fitter.fit.global_search import (
    build_de_population,
    build_stage_e_population,
    downsample_prepared_data,
    feature_grid_indices,
)
from xrr_fitter.fit.joint_evaluation import (
    JointEvaluation,
    evaluate_joint_jacobian,
    evaluate_joint_vector,
    joint_least_squares_loss,
)
from xrr_fitter.fit.joint_problem import JointFitProblem, compile_joint_problem
from xrr_fitter.fit.joint_sharing import (
    initial_joint_vector,
    joint_candidate_vectors,
    scatter_joint_vector,
    validate_joint_candidate_alignment,
)
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.pipeline import FitSearchRequest, run_fit_search
from xrr_fitter.fit.problem import compile_stage_problem
from xrr_fitter.fit.resume import validate_resume_checkpoint
from xrr_fitter.fit.stages import reserve_child_seeds
from xrr_fitter.model.fitting import (
    FitCandidate,
    FitCheckpoint,
    FitProgress,
    FitSearchResult,
    FitStageSummary,
)
from xrr_fitter.model.parameters import SharingRule
from xrr_fitter.model.provenance import fit_search_provenance_sha256


@dataclass(frozen=True, slots=True)
class JointFitRequest:
    problem: JointFitProblem
    resume_checkpoints: tuple[FitCheckpoint, ...] | None = None

    def __post_init__(self) -> None:
        if self.resume_checkpoints is None:
            return
        checkpoints = tuple(self.resume_checkpoints)
        if len(checkpoints) != len(self.problem.dataset_ids):
            raise ValueError("joint resume requires checkpoints for all datasets")
        object.__setattr__(self, "resume_checkpoints", checkpoints)


@dataclass(frozen=True, slots=True)
class FitBatchRequest:
    mode: str
    requests: tuple[FitSearchRequest, ...]
    sharing_rules: tuple[SharingRule, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(self, "sharing_rules", tuple(self.sharing_rules))


@dataclass(frozen=True, slots=True)
class _JointState:
    candidates: tuple[tuple[FitCandidate, ...], ...]
    warnings: tuple[str, ...]
    summaries: tuple[FitStageSummary, ...]


@dataclass(frozen=True, slots=True)
class _SolvedJoint:
    unit_vector: np.ndarray
    evaluation: JointEvaluation
    stop_reason: str
    nfev: int


def _poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


def _solve_joint(
    problem: JointFitProblem,
    start: np.ndarray,
    max_nfev: int,
    cancelled: Callable[[], bool] | None,
) -> _SolvedJoint:
    unit = np.asarray(start, dtype=float)
    _poll(cancelled)
    if unit.size == 0:
        return _SolvedJoint(unit, evaluate_joint_vector(problem, unit), "no_free_parameters", 1)

    def residual(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        return evaluate_joint_vector(problem, value).residuals

    def jacobian(value: np.ndarray) -> np.ndarray:
        _poll(cancelled)
        return evaluate_joint_jacobian(problem, value)

    solved = least_squares(
        residual,
        unit,
        jac=jacobian,
        bounds=(0.0, 1.0),
        loss=joint_least_squares_loss(problem),
        max_nfev=max_nfev,
        method="trf",
    )
    result_unit = np.array(solved.x, dtype=float, copy=True)
    return _SolvedJoint(
        result_unit,
        evaluate_joint_vector(problem, result_unit),
        str(solved.message),
        int(solved.nfev),
    )


def _solve_joint_global(
    problem: JointFitProblem,
    start: np.ndarray,
    population: np.ndarray,
    *,
    seed: int,
    maxiter: int,
    cancelled: Callable[[], bool] | None,
) -> _SolvedJoint:
    unit = np.asarray(start, dtype=float)
    _poll(cancelled)
    if unit.size == 0:
        return _SolvedJoint(unit, evaluate_joint_vector(problem, unit), "no_free_parameters", 1)

    def objective(value: np.ndarray) -> float:
        _poll(cancelled)
        return evaluate_joint_vector(problem, value).objective

    solved = differential_evolution(
        objective,
        tuple((0.0, 1.0) for _ in problem.global_variables),
        x0=unit,
        init=np.asarray(population, dtype=float),
        seed=np.random.default_rng(seed),
        maxiter=maxiter,
        updating="deferred",
        polish=False,
        workers=1,
    )
    result_unit = np.array(solved.x, dtype=float, copy=True)
    return _SolvedJoint(
        result_unit,
        evaluate_joint_vector(problem, result_unit),
        str(solved.message),
        int(solved.nfev),
    )


def _evaluation_values(evaluation: object) -> dict[str, float]:
    return {value.name: value.value for value in evaluation.parameters}


def _joint_unit_from_values(
    problem: JointFitProblem,
    values_by_dataset: tuple[dict[str, float], ...],
) -> np.ndarray:
    if len(values_by_dataset) != len(problem.problems):
        raise ValueError("joint physical values must align with every dataset")
    global_unit = np.full(len(problem.global_variables), np.nan, dtype=float)
    for local_problem, values, scatter in zip(
        problem.problems,
        values_by_dataset,
        problem.scatter_maps,
        strict=True,
    ):
        local_unit = encode_physical_vector(local_problem, values)
        for local_index, global_index in enumerate(scatter):
            value = local_unit[local_index]
            if np.isnan(global_unit[global_index]):
                global_unit[global_index] = value
            elif global_unit[global_index] != value:
                raise ValueError("joint shared physical values map to different unit coordinates")
    if np.any(~np.isfinite(global_unit)):
        raise ValueError("joint physical values leave a global coordinate unbound")
    return global_unit


def _values_from_full_vector(
    problem: JointFitProblem,
    global_unit: np.ndarray,
) -> tuple[dict[str, float], ...]:
    local_units = scatter_joint_vector(problem, global_unit)
    return tuple(
        values_by_name(local_problem, local_unit)
        for local_problem, local_unit in zip(problem.problems, local_units, strict=True)
    )


def _publish_full_solution(
    full_problem: JointFitProblem,
    solved: _SolvedJoint,
) -> _SolvedJoint:
    values = tuple(_evaluation_values(item) for item in solved.evaluation.local_evaluations)
    full_unit = _joint_unit_from_values(full_problem, values)
    return _SolvedJoint(
        full_unit,
        evaluate_joint_vector(full_problem, full_unit),
        solved.stop_reason,
        solved.nfev,
    )


def _project_candidate(
    problem: JointFitProblem,
    solved: _SolvedJoint,
    candidate_id: str,
    seed_index: int,
) -> tuple[FitCandidate, ...]:
    projected = []
    for local_problem, unit, evaluation in zip(
        problem.problems,
        solved.evaluation.local_unit_vectors,
        solved.evaluation.local_evaluations,
        strict=True,
    ):
        candidate = candidate_from_evaluation(
            local_problem,
            unit,
            evaluation,
            candidate_id,
            seed_index,
            solved.stop_reason,
            solved.nfev,
        )
        projected.append(replace(candidate, ranking_objective=solved.evaluation.objective))
    return tuple(projected)


def _summary(
    stage: str,
    projected: tuple[tuple[FitCandidate, ...], ...],
) -> FitStageSummary:
    first_dataset = tuple(aligned[0] for aligned in projected)
    best = min(candidate.ranking_objective for candidate in first_dataset)
    return FitStageSummary(
        stage,
        tuple(candidate.candidate_id for candidate in first_dataset),
        float(best),
        sum(candidate.nfev for candidate in first_dataset),
        tuple(candidate.stop_reason for candidate in first_dataset),
    )


def _transpose_candidates(
    projected: tuple[tuple[FitCandidate, ...], ...],
) -> tuple[tuple[FitCandidate, ...], ...]:
    if not projected:
        return ()
    return tuple(tuple(values) for values in zip(*projected, strict=True))


def _append_stage(
    state: _JointState,
    stage_candidates: tuple[tuple[FitCandidate, ...], ...],
    summary: FitStageSummary,
) -> _JointState:
    by_dataset = _transpose_candidates(stage_candidates)
    return _JointState(
        tuple(
            existing + additions
            for existing, additions in zip(state.candidates, by_dataset, strict=True)
        ),
        state.warnings,
        state.summaries + (summary,),
    )


def _emit(
    callback: Callable[[FitProgress], None] | None,
    stage: str,
    completed: int,
    total: int,
    best: float,
    message: str,
) -> None:
    if callback is not None:
        callback(FitProgress(None, stage, completed, total, best, message))


def _stage_base_problem(problem: object, stage: str) -> object:
    if stage != "B":
        return problem
    indices = feature_grid_indices(problem.data)
    if np.array_equal(indices, np.arange(problem.data.qz_a_inv.size)):
        return problem
    return replace(problem, data=downsample_prepared_data(problem.data, indices))


def _stage_sharing_rules(
    problem: JointFitProblem,
    local_problems: tuple[object, ...],
) -> tuple[SharingRule, ...]:
    by_dataset = dict(zip(problem.dataset_ids, local_problems, strict=True))
    selected = []
    for rule in problem.sharing_rules:
        free = tuple(
            member.parameter_name
            in {coordinate.name for coordinate in by_dataset[member.dataset_id].variables}
            for member in rule.members
        )
        if any(free) and not all(free):
            raise ValueError("joint stage sharing members must have aligned free state")
        if all(free):
            selected.append(rule)
    return tuple(selected)


def _compile_stage_joint(
    problem: JointFitProblem,
    stage: str,
    full_unit: np.ndarray,
) -> JointFitProblem:
    current = _values_from_full_vector(problem, full_unit)
    local_problems = tuple(
        compile_stage_problem(_stage_base_problem(local_problem, stage), stage, values)
        for local_problem, values in zip(problem.problems, current, strict=True)
    )
    rules = _stage_sharing_rules(problem, local_problems)
    return compile_joint_problem(problem.dataset_ids, local_problems, rules)


def _local_budget(problem: JointFitProblem) -> int:
    budget = problem.problems[0].config.budget
    return max(
        budget.local_min_nfev,
        budget.local_nfev_per_parameter * max(1, len(problem.global_variables)),
    )


def _append_solution(
    problem: JointFitProblem,
    solved: _SolvedJoint,
    candidate_id: str,
    seed_index: int,
    projected: list[tuple[FitCandidate, ...]],
    *,
    stage: str,
    completed: int,
    total: int,
    best: float,
    progress: Callable[[FitProgress], None] | None,
) -> float:
    published = _publish_full_solution(problem, solved)
    projected.append(_project_candidate(problem, published, candidate_id, seed_index))
    current_best = min(best, published.evaluation.objective)
    _emit(progress, stage, completed, total, current_best, published.stop_reason)
    return current_best


def _run_stage_b(
    problem: JointFitProblem,
    initial: np.ndarray | None,
    seeds: tuple[int, ...],
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[tuple[FitCandidate, ...], ...], FitStageSummary]:
    if initial is None:
        raise RuntimeError("joint stage B requires the fresh initial vector")
    stage_problem = _compile_stage_joint(problem, "B", initial)
    center = initial_joint_vector(stage_problem)
    projected: list[tuple[FitCandidate, ...]] = []
    best = float("inf")
    for index, seed in enumerate(seeds[:2]):
        start = _seeded_start(center, seed, 0.1)
        if stage_problem.global_variables:
            population = build_de_population(
                start,
                seed=seed,
                population_size=max(32, 6 * len(stage_problem.global_variables)),
            )
            solved = _solve_joint_global(
                stage_problem,
                start,
                population,
                seed=seed,
                maxiter=problem.problems[0].config.budget.short_de_maxiter,
                cancelled=cancelled,
            )
        else:
            solved = _solve_joint(stage_problem, start, 1, cancelled)
        best = _append_solution(
            problem,
            solved,
            f"B-{index}",
            index,
            projected,
            stage="B",
            completed=index + 1,
            total=2,
            best=best,
            progress=progress,
        )
    result = tuple(projected)
    return result, _summary("B", result)


def _run_local_joint_stage(
    problem: JointFitProblem,
    state: _JointState,
    stage: str,
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[tuple[FitCandidate, ...], ...], FitStageSummary]:
    parent_stage = "B" if stage == "C" else "C"
    parent_summary = next(value for value in reversed(state.summaries) if value.stage == parent_stage)
    parents = _vectors_from_state(problem, state, parent_stage)
    projected: list[tuple[FitCandidate, ...]] = []
    best = float("inf")
    for index, (parent_id, parent) in enumerate(
        zip(parent_summary.candidate_ids, parents, strict=True)
    ):
        stage_problem = _compile_stage_joint(problem, stage, parent)
        start = initial_joint_vector(stage_problem)
        solved = _solve_joint(stage_problem, start, _local_budget(stage_problem), cancelled)
        lineage = parent_id.split("-")[1]
        best = _append_solution(
            problem,
            solved,
            f"{stage}-{lineage}-0",
            index,
            projected,
            stage=stage,
            completed=index + 1,
            total=len(parents),
            best=best,
            progress=progress,
        )
    result = tuple(projected)
    return result, _summary(stage, result)


def _stage_e_solution(
    problem: JointFitProblem,
    stage_problem: JointFitProblem,
    start: np.ndarray,
    centers: tuple[np.ndarray, ...],
    seed: int,
    cancelled: Callable[[], bool] | None,
) -> _SolvedJoint:
    if not stage_problem.global_variables:
        return _solve_joint(stage_problem, start, 1, cancelled)
    population = build_stage_e_population(
        centers,
        seed=seed,
        population_size=max(64, 8 * len(stage_problem.global_variables)),
        perturbations_per_center=2,
    )
    global_solved = _solve_joint_global(
        stage_problem,
        start,
        population,
        seed=seed,
        maxiter=problem.problems[0].config.budget.full_de_maxiter,
        cancelled=cancelled,
    )
    local_solved = _solve_joint(
        stage_problem,
        global_solved.unit_vector,
        _local_budget(stage_problem),
        cancelled,
    )
    return _SolvedJoint(
        local_solved.unit_vector,
        local_solved.evaluation,
        local_solved.stop_reason,
        global_solved.nfev + local_solved.nfev,
    )


def _run_stage_e(
    problem: JointFitProblem,
    state: _JointState,
    seeds: tuple[int, ...],
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[tuple[FitCandidate, ...], ...], FitStageSummary]:
    parents = _vectors_from_state(problem, state, "D")
    parent_values = tuple(_values_from_full_vector(problem, parent) for parent in parents)
    final_seeds = seeds[2:]
    projected: list[tuple[FitCandidate, ...]] = []
    best = float("inf")
    for index, seed in enumerate(final_seeds):
        parent = parents[index % len(parents)]
        stage_problem = _compile_stage_joint(problem, "E", parent)
        start = initial_joint_vector(stage_problem)
        centers = tuple(
            _joint_unit_from_values(stage_problem, values)
            for values in parent_values
        )
        solved = _stage_e_solution(
            problem,
            stage_problem,
            start,
            centers,
            seed,
            cancelled,
        )
        best = _append_solution(
            problem,
            solved,
            f"E-{index}",
            index,
            projected,
            stage="E",
            completed=index + 1,
            total=len(final_seeds),
            best=best,
            progress=progress,
        )
    result = tuple(projected)
    return result, _summary("E", result)


def _vectors_from_state(
    problem: JointFitProblem,
    state: _JointState,
    stage: str,
) -> tuple[np.ndarray, ...]:
    summary = next(value for value in reversed(state.summaries) if value.stage == stage)
    return joint_candidate_vectors(problem, state.candidates, summary.candidate_ids)


def _seed_ledger(problem: JointFitProblem) -> tuple[int, ...]:
    config = problem.problems[0].config
    streams = ("B-0", "B-1", *(f"E-{index}" for index in range(config.final_seed_count)))
    return tuple(child.seed for child in reserve_child_seeds(config.master_seed, streams))


def _checkpoint_batch(
    request: JointFitRequest,
    state: _JointState,
    stage: str,
    seeds: tuple[int, ...],
) -> tuple[FitCheckpoint, ...]:
    consumed = seeds if stage == "E" else seeds[:2]
    return tuple(
        build_checkpoint(
            local_problem,
            stage=stage,
            candidates=candidates,
            child_seeds=consumed,
            runtime_warnings=state.warnings,
            stage_summaries=state.summaries,
            joint_layout_fingerprint=request.problem.layout_fingerprint,
        )
        for local_problem, candidates in zip(request.problem.problems, state.candidates, strict=True)
    )


def _validate_joint_resume(
    request: JointFitRequest,
    seeds: tuple[int, ...],
) -> tuple[_JointState, tuple[str, ...]]:
    checkpoints = request.resume_checkpoints
    if checkpoints is None:
        raise ValueError("joint resume checkpoints are missing")
    plans = []
    for local_problem, checkpoint in zip(request.problem.problems, checkpoints, strict=True):
        if checkpoint.joint_layout_fingerprint != request.problem.layout_fingerprint:
            raise ValueError("joint resume layout fingerprint mismatch")
        plans.append(
            validate_resume_checkpoint(
                local_problem,
                checkpoint,
                reserved_child_seeds=seeds,
                expected_joint_layout_fingerprint=request.problem.layout_fingerprint,
            )
        )
    first = plans[0]
    if any(
        (
            plan.completed_stage,
            plan.consumed_child_seeds,
            plan.runtime_warnings,
            plan.stage_summaries,
        )
        != (
            first.completed_stage,
            first.consumed_child_seeds,
            first.runtime_warnings,
            first.stage_summaries,
        )
        for plan in plans[1:]
    ):
        raise ValueError("joint resume stage, seed, warning, or history mismatch")
    state = _JointState(
        tuple(plan.candidates for plan in plans),
        first.runtime_warnings,
        first.stage_summaries,
    )
    validate_joint_candidate_alignment(request.problem, state.candidates, state.summaries)
    return state, first.remaining_stages


def _fresh_state(problem: JointFitProblem) -> tuple[_JointState, np.ndarray]:
    warnings = tuple(dict.fromkeys(warning for value in problem.problems for warning in value.warnings))
    initial = initial_joint_vector(problem)
    evaluation = evaluate_joint_vector(problem, initial)
    summary = FitStageSummary("A", ("joint-initial",), evaluation.objective, 1, ("evaluated",))
    state = _JointState(tuple(() for _ in problem.problems), warnings, (summary,))
    return state, initial


def _result_tuple(
    request: JointFitRequest,
    state: _JointState,
    seeds: tuple[int, ...],
) -> tuple[FitSearchResult, ...]:
    eligible_ids = next(summary.candidate_ids for summary in reversed(state.summaries) if summary.stage == "E")
    winner = best_candidate_index(state.candidates[0], eligible_ids=eligible_ids)
    results = tuple(
        FitSearchResult(
            local_problem.parameter_definitions,
            candidates,
            winner,
            state.warnings,
            seeds,
            state.summaries,
            local_problem.region_labels,
            local_problem.weights,
        )
        for local_problem, candidates in zip(request.problem.problems, state.candidates, strict=True)
    )
    return tuple(
        replace(
            result,
            provenance_sha256=fit_search_provenance_sha256(local_problem, result),
        )
        for local_problem, result in zip(request.problem.problems, results, strict=True)
    )


def _initial_joint_run(
    request: JointFitRequest,
    seeds: tuple[int, ...],
) -> tuple[_JointState, np.ndarray | None, tuple[str, ...]]:
    if request.resume_checkpoints is None:
        state, initial = _fresh_state(request.problem)
        return state, initial, ("B", "C", "D", "E")
    state, remaining = _validate_joint_resume(request, seeds)
    return state, None, remaining


def _seeded_start(center: np.ndarray, seed: int, sigma: float) -> np.ndarray:
    perturbations = bounded_perturbations(center, 1, seed=seed, sigma=sigma)
    return center if not perturbations else perturbations[0]


def _run_joint_stage(
    problem: JointFitProblem,
    state: _JointState,
    initial: np.ndarray | None,
    stage: str,
    seeds: tuple[int, ...],
    progress: Callable[[FitProgress], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[tuple[FitCandidate, ...], ...], FitStageSummary]:
    if stage == "B":
        return _run_stage_b(problem, initial, seeds, progress, cancelled)
    if stage in {"C", "D"}:
        return _run_local_joint_stage(problem, state, stage, progress, cancelled)
    return _run_stage_e(problem, state, seeds, progress, cancelled)


def run_joint_fit(
    request: JointFitRequest,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[FitProgress], None] | None = None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None = None,
) -> tuple[FitSearchResult, ...]:
    """Run a fresh joint search or an atomically validated suffix."""
    if not isinstance(request, JointFitRequest):
        raise TypeError("request must be a JointFitRequest")
    seeds = _seed_ledger(request.problem)
    state, initial, remaining = _initial_joint_run(request, seeds)
    for stage in remaining:
        projected, summary = _run_joint_stage(
            request.problem,
            state,
            initial,
            stage,
            seeds,
            progress,
            cancelled,
        )
        state = _append_stage(state, projected, summary)
        if checkpoint is not None:
            checkpoint(_checkpoint_batch(request, state, stage, seeds))
    return _result_tuple(request, state, seeds)


def _joint_resume_checkpoints(
    request: FitBatchRequest,
) -> tuple[FitCheckpoint, ...] | None:
    checkpoints = tuple(item.resume_checkpoint for item in request.requests)
    if all(value is None for value in checkpoints):
        return None
    if any(value is None for value in checkpoints):
        raise ValueError("joint resume requires checkpoints for all datasets")
    return tuple(value for value in checkpoints if value is not None)


def _run_joint_batch(
    request: FitBatchRequest,
    cancelled: Callable[[], bool] | None,
    progress: Callable[[FitProgress], None] | None,
) -> tuple[FitSearchResult, ...]:
    problem = compile_joint_problem(
        tuple(item.dataset_id for item in request.requests),
        tuple(item.problem for item in request.requests),
        request.sharing_rules,
    )
    return run_joint_fit(
        JointFitRequest(problem, _joint_resume_checkpoints(request)),
        cancelled=cancelled,
        progress=progress,
    )


def run_fit_batch(
    request: FitBatchRequest,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[FitProgress], None] | None = None,
) -> tuple[FitSearchResult, ...]:
    """Dispatch only the two declared batch modes without fallback."""
    if request.mode == "independent":
        return tuple(
            run_fit_search(item, cancelled=cancelled, progress=progress)
            for item in request.requests
        )
    if request.mode == "joint":
        return _run_joint_batch(request, cancelled, progress)
    raise ValueError("fit batch mode must be independent or joint")
