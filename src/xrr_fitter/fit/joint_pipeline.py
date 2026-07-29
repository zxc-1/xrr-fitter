"""Pure joint fitting, atomic checkpoints, resume, and batch dispatch.

Joint search uses one compiled global coordinate layout for every stage. Stage A
retains the declared initial candidate, Stage B consumes one reserved short-DE
seed, C and D refine that single lineage, and each Stage-E seed publishes an
atomic resumable prefix. Checkpoint batches transpose the aligned global history
back by dataset. Resume accepts a batch only when every member describes the
same joint state and deterministic child-seed prefix.
Stage-E summaries accumulate only completed seeds, so cancellation cannot expose
a candidate or checkpoint for an unfinished prefix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from xrr_fitter.fit.candidates import best_candidate_index, candidate_from_evaluation
from xrr_fitter.fit.checkpoint import build_checkpoint
from xrr_fitter.fit.global_search import build_de_population
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
    validate_joint_candidate_alignment,
)
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.pipeline import FitSearchRequest, run_fit_search
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
    objective_increased: bool = False


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
    initial_evaluation = evaluate_joint_vector(problem, unit)

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
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        x_scale="jac",
        max_nfev=max_nfev,
        callback=lambda *_args, **_kwargs: _poll(cancelled),
    )
    result_unit = np.array(solved.x, dtype=float, copy=True)
    evaluation = evaluate_joint_vector(problem, result_unit)
    tolerance = max(1e-12, 1e-8 * initial_evaluation.objective)
    objective_increased = bool(
        initial_evaluation.valid
        and (
            not evaluation.valid
            or evaluation.objective > initial_evaluation.objective + tolerance
        )
    )
    if objective_increased:
        return _SolvedJoint(
            np.array(unit, dtype=float, copy=True),
            initial_evaluation,
            "local_objective_increased",
            int(solved.nfev),
            True,
        )
    return _SolvedJoint(
        result_unit,
        evaluation,
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
        [(0.0, 1.0)] * len(problem.global_variables),
        init=np.asarray(population, dtype=float),
        seed=np.random.default_rng(seed),
        maxiter=maxiter,
        updating="deferred",
        polish=False,
        tol=1e-6,
        workers=1,
        callback=lambda *_args, **_kwargs: _poll(cancelled),
    )
    result_unit = np.array(solved.x, dtype=float, copy=True)
    return _SolvedJoint(
        result_unit,
        evaluate_joint_vector(problem, result_unit),
        str(solved.message),
        int(solved.nfev),
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
        ranking = None if solved.objective_increased else solved.evaluation.objective
        if solved.objective_increased:
            candidate = replace(candidate, valid=False)
        projected.append(replace(candidate, ranking_objective=ranking))
    return tuple(projected)


def _summary(
    stage: str,
    projected: tuple[tuple[FitCandidate, ...], ...],
) -> FitStageSummary:
    first_dataset = tuple(aligned[0] for aligned in projected)
    selectable = tuple(
        candidate.ranking_objective
        for candidate in first_dataset
        if candidate.valid and candidate.ranking_objective is not None
    )
    best = min(selectable, default=float("inf"))
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


def _append_stage_e(
    state: _JointState,
    candidate: tuple[FitCandidate, ...],
) -> _JointState:
    projected = (candidate,)
    if not state.summaries or state.summaries[-1].stage != "E":
        return _append_stage(state, projected, _summary("E", projected))
    previous = state.summaries[-1]
    primary = candidate[0]
    objective = (
        primary.ranking_objective
        if primary.valid and primary.ranking_objective is not None
        else float("inf")
    )
    summary = FitStageSummary(
        "E",
        previous.candidate_ids + (primary.candidate_id,),
        min(previous.best_objective, objective),
        previous.total_nfev + primary.nfev,
        previous.stop_reasons + (primary.stop_reason,),
    )
    return _JointState(
        tuple(
            existing + (addition,)
            for existing, addition in zip(state.candidates, candidate, strict=True)
        ),
        state.warnings,
        state.summaries[:-1] + (summary,),
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


def _local_budget(problem: JointFitProblem) -> int:
    budget = problem.problems[0].config.budget
    return max(
        budget.local_min_nfev,
        budget.local_nfev_per_parameter * (len(problem.global_variables) + 1),
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
    projected.append(_project_candidate(problem, solved, candidate_id, seed_index))
    objective = (
        solved.evaluation.objective
        if solved.evaluation.valid and not solved.objective_increased
        else float("inf")
    )
    current_best = min(best, objective)
    _emit(progress, stage, completed, total, current_best, f"joint {stage}")
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
    seed = seeds[0]
    projected: list[tuple[FitCandidate, ...]] = []
    if problem.global_variables:
        population = build_de_population(
            initial,
            seed=seed,
            population_size=max(32, 6 * len(problem.global_variables)),
        )
        solved = _solve_joint_global(
            problem,
            initial,
            population,
            seed=seed,
            maxiter=problem.problems[0].config.budget.short_de_maxiter,
            cancelled=cancelled,
        )
    else:
        solved = _SolvedJoint(
            initial,
            evaluate_joint_vector(problem, initial),
            "no_free_parameters",
            1,
        )
    _append_solution(
        problem,
        solved,
        "B-0",
        0,
        projected,
        stage="B",
        completed=1,
        total=1,
        best=float("inf"),
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
    parents = _vectors_from_state(problem, state, parent_stage)
    if len(parents) != 1:
        raise ValueError(f"joint stage {stage} requires one parent candidate")
    projected: list[tuple[FitCandidate, ...]] = []
    solved = _solve_joint(problem, parents[0], _local_budget(problem), cancelled)
    _append_solution(
        problem,
        solved,
        f"{stage}-0",
        0,
        projected,
        stage=stage,
        completed=1,
        total=1,
        best=float("inf"),
        progress=progress,
    )
    result = tuple(projected)
    return result, _summary(stage, result)


def _stage_e_solution(
    problem: JointFitProblem,
    start: np.ndarray,
    seed: int,
    cancelled: Callable[[], bool] | None,
) -> _SolvedJoint:
    if not problem.global_variables:
        return _SolvedJoint(
            start,
            evaluate_joint_vector(problem, start),
            "no_free_parameters",
            1,
        )
    population = build_de_population(
        start,
        seed=seed,
        population_size=max(64, 8 * len(problem.global_variables)),
    )
    global_solved = _solve_joint_global(
        problem,
        start,
        population,
        seed=seed,
        maxiter=problem.problems[0].config.budget.full_de_maxiter,
        cancelled=cancelled,
    )
    return _solve_joint(
        problem,
        global_solved.unit_vector,
        _local_budget(problem),
        cancelled,
    )


def _vectors_from_state(
    problem: JointFitProblem,
    state: _JointState,
    stage: str,
) -> tuple[np.ndarray, ...]:
    summary = next(value for value in reversed(state.summaries) if value.stage == stage)
    return joint_candidate_vectors(problem, state.candidates, summary.candidate_ids)


def _seed_ledger(problem: JointFitProblem) -> tuple[int, ...]:
    config = problem.problems[0].config
    streams = ("B-0", *(f"E-{index}" for index in range(config.final_seed_count)))
    return tuple(child.seed for child in reserve_child_seeds(config.master_seed, streams))


def _checkpoint_batch(
    request: JointFitRequest,
    state: _JointState,
    stage: str,
    seeds: tuple[int, ...],
) -> tuple[FitCheckpoint, ...]:
    if stage == "E":
        final_count = len(state.summaries[-1].candidate_ids)
        consumed = seeds[: final_count + 1]
    else:
        consumed = seeds[:1]
    checkpoint_summaries = state.summaries[1:]
    return tuple(
        build_checkpoint(
            local_problem,
            stage=stage,
            candidates=candidates[1:],
            child_seeds=consumed,
            runtime_warnings=state.warnings,
            stage_summaries=checkpoint_summaries,
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
    initial_state, _initial = _fresh_state(request.problem)
    state = _JointState(
        tuple(
            initial + plan.candidates
            for initial, plan in zip(initial_state.candidates, plans, strict=True)
        ),
        first.runtime_warnings,
        initial_state.summaries + first.stage_summaries,
    )
    validate_joint_candidate_alignment(request.problem, state.candidates, state.summaries)
    remaining = first.remaining_stages
    if first.completed_stage == "E":
        completed_final = len(first.stage_summaries[-1].candidate_ids)
        final_count = request.problem.problems[0].config.final_seed_count
        remaining = ("E",) if completed_final < final_count else ()
    return state, remaining


def _fresh_state(problem: JointFitProblem) -> tuple[_JointState, np.ndarray]:
    warnings = tuple(dict.fromkeys(warning for value in problem.problems for warning in value.warnings))
    initial = initial_joint_vector(problem)
    evaluation = evaluate_joint_vector(problem, initial)
    solved = _SolvedJoint(initial, evaluation, "declared_initial", 1)
    projected = _project_candidate(problem, solved, "A-0", 0)
    summary = _summary("A", (projected,))
    state = _JointState(
        tuple((candidate,) for candidate in projected),
        warnings,
        (summary,),
    )
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
    if stage not in {"C", "D"}:
        raise ValueError(f"unsupported single-candidate joint stage: {stage}")
    return _run_local_joint_stage(problem, state, stage, progress, cancelled)


def _completed_stage_e(state: _JointState) -> int:
    summary = next(
        (value for value in reversed(state.summaries) if value.stage == "E"),
        None,
    )
    return 0 if summary is None else len(summary.candidate_ids)


def _run_stage_e_prefix(
    request: JointFitRequest,
    state: _JointState,
    seeds: tuple[int, ...],
    progress: Callable[[FitProgress], None] | None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None,
    cancelled: Callable[[], bool] | None,
) -> _JointState:
    parents = _vectors_from_state(request.problem, state, "D")
    if len(parents) != 1:
        raise ValueError("joint stage E requires one stage-D parent candidate")
    start = parents[0]
    final_seeds = seeds[1:]
    completed = _completed_stage_e(state)
    best = (
        float("inf")
        if completed == 0
        else next(summary for summary in reversed(state.summaries) if summary.stage == "E").best_objective
    )
    for index in range(completed, len(final_seeds)):
        solved = _stage_e_solution(
            request.problem,
            start,
            final_seeds[index],
            cancelled,
        )
        projected: list[tuple[FitCandidate, ...]] = []
        best = _append_solution(
            request.problem,
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
        state = _append_stage_e(state, projected[0])
        if checkpoint is not None:
            checkpoint(_checkpoint_batch(request, state, "E", seeds))
    return state


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
    fresh = request.resume_checkpoints is None
    if fresh:
        _poll(cancelled)
    state, initial, remaining = _initial_joint_run(request, seeds)
    if fresh:
        summary = state.summaries[0]
        _emit(progress, "A", 1, 1, summary.best_objective, "joint A")
    for stage in remaining:
        if stage == "E":
            state = _run_stage_e_prefix(
                request,
                state,
                seeds,
                progress,
                checkpoint,
                cancelled,
            )
            continue
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
