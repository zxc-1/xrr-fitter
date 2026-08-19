"""Pure, pickle-safe single-dataset A-E fitting pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    evaluate_model,
)
from xrr_fitter.fit.candidates import best_candidate_index
from xrr_fitter.fit.checkpoint import build_checkpoint
from xrr_fitter.fit.resume import ResumePlan, validate_resume_checkpoint
from xrr_fitter.fit.stages import (
    StageOutcome,
    compile_coarse_problem,
    local_stage_continuation,
    reconverge_profile_basin,
    reserve_child_seeds,
    run_local_stage,
    run_stage_a,
    run_stage_b,
    run_stage_e,
    stage_b_continuation,
)
from xrr_fitter.fit.tasking import TaskRunner
from xrr_fitter.model.fitting import (
    FitCandidate,
    FitCheckpoint,
    FitEvaluationContext,
    FitProgress,
    FitSearchResult,
    FitStageSummary,
)
from xrr_fitter.model.provenance import fit_search_provenance_sha256


@dataclass(frozen=True, slots=True)
class FitSearchRequest:
    dataset_id: str | None
    problem: FitEvaluationContext
    resume_checkpoint: FitCheckpoint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.problem, FitEvaluationContext):
            raise TypeError("problem must be a FitEvaluationContext")


@dataclass(frozen=True, slots=True)
class _SearchState:
    candidates: tuple[FitCandidate, ...] = ()
    base_warnings: tuple[str, ...] = ()
    runtime_warnings: tuple[str, ...] = ()
    summaries: tuple[FitStageSummary, ...] = ()

    def append(self, outcome: StageOutcome) -> _SearchState:
        return _SearchState(
            self.candidates + outcome.candidates,
            self.base_warnings,
            self.runtime_warnings + outcome.warnings,
            self.summaries + (outcome.summary,),
        )


def _seed_ledger(request: FitSearchRequest) -> tuple[int, ...]:
    streams = (
        "B-0",
        "B-1",
        *(f"E-{index}" for index in range(request.problem.config.final_seed_count)),
    )
    return tuple(child.seed for child in reserve_child_seeds(request.problem.config.master_seed, streams))


def _dedupe_warnings(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for group in groups for value in group))


def _seal_result(
    problem: FitEvaluationContext,
    result: FitSearchResult,
) -> FitSearchResult:
    return replace(
        result,
        provenance_sha256=fit_search_provenance_sha256(problem, result),
    )


def _require_owned_result(
    problem: FitEvaluationContext,
    result: FitSearchResult,
) -> None:
    expected = fit_search_provenance_sha256(problem, result)
    if result.provenance_sha256 != expected:
        raise ValueError("search_result provenance does not match context")


def _base_warnings(problem: object, coarse_problem: object) -> tuple[str, ...]:
    return _dedupe_warnings(
        tuple(problem.data.warnings),
        tuple(problem.warnings),
        tuple(coarse_problem.warnings),
    )


def _state_from_resume(plan: ResumePlan, base_warnings: tuple[str, ...]) -> _SearchState:
    return _SearchState(
        plan.candidates,
        base_warnings,
        plan.runtime_warnings,
        plan.stage_summaries,
    )


def _stage_candidates(state: _SearchState, stage: str) -> tuple[FitCandidate, ...]:
    summary = next(value for value in reversed(state.summaries) if value.stage == stage)
    by_id = {candidate.candidate_id: candidate for candidate in state.candidates}
    return tuple(by_id[candidate_id] for candidate_id in summary.candidate_ids)


def _consumed_seeds(stage: str, seeds: tuple[int, ...]) -> tuple[int, ...]:
    return seeds if stage == "E" else seeds[:2]


def _publish_checkpoint(
    request: FitSearchRequest,
    state: _SearchState,
    stage: str,
    seeds: tuple[int, ...],
    callback: Callable[[FitCheckpoint], None] | None,
) -> None:
    if callback is None:
        return
    callback(
        build_checkpoint(
            request.problem,
            stage=stage,
            candidates=state.candidates,
            child_seeds=_consumed_seeds(stage, seeds),
            runtime_warnings=state.runtime_warnings,
            stage_summaries=state.summaries,
        )
    )


def _result(request: FitSearchRequest, state: _SearchState, seeds: tuple[int, ...]) -> FitSearchResult:
    eligible = _stage_candidates(state, "E")
    eligible_ids = tuple(candidate.candidate_id for candidate in eligible)
    result = FitSearchResult(
        parameter_definitions=request.problem.parameter_definitions,
        candidates=state.candidates,
        best_index=best_candidate_index(state.candidates, eligible_ids=eligible_ids),
        warnings=_dedupe_warnings(state.base_warnings, state.runtime_warnings),
        child_seeds=seeds,
        stage_summaries=state.summaries,
        region_labels=request.problem.region_labels,
        region_weights=request.problem.weights,
    )
    return _seal_result(request.problem, result)


def run_fit_search(
    request: FitSearchRequest,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[FitProgress], None] | None = None,
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
    task_runner: TaskRunner | None = None,
) -> FitSearchResult:
    """Run a fresh search or the exact suffix after a validated checkpoint."""
    if not isinstance(request, FitSearchRequest):
        raise TypeError("request must be a FitSearchRequest")
    coarse_problem = compile_coarse_problem(request.problem)
    base_warnings = _base_warnings(request.problem, coarse_problem)
    seeds = _seed_ledger(request)
    if request.resume_checkpoint is None:
        state = _SearchState(base_warnings=base_warnings)
        remaining = ("A", "B", "C", "D", "E")
    else:
        plan = validate_resume_checkpoint(
            request.problem,
            request.resume_checkpoint,
            reserved_child_seeds=seeds,
        )
        state = _state_from_resume(plan, base_warnings)
        remaining = plan.remaining_stages

    starts = None
    perturbation_counts: tuple[int, ...] = ()
    for stage in remaining:
        if stage == "A":
            starts, summary, warnings = run_stage_a(
                request.problem,
                request.dataset_id,
                coarse_problem=coarse_problem,
                progress=progress,
                cancelled=cancelled,
            )
            state = _SearchState(
                state.candidates,
                state.base_warnings,
                state.runtime_warnings + warnings,
                state.summaries + (summary,),
            )
            continue
        if stage == "B":
            if starts is None:
                raise RuntimeError("stage B requires committed stage-A starts")
            outcome = run_stage_b(
                request.problem,
                request.dataset_id,
                starts,
                seeds[:2],
                progress=progress,
                cancelled=cancelled,
            )
            perturbation_counts = outcome.perturbation_counts
        elif stage in {"C", "D"}:
            parent_stage = "B" if stage == "C" else "C"
            stage_candidates = _stage_candidates(state, parent_stage)
            if stage == "C":
                parents, counts = stage_b_continuation(
                    stage_candidates,
                    perturbation_counts,
                )
            else:
                parents, counts = local_stage_continuation(
                    stage_candidates,
                    perturbation_counts,
                )
            outcome = run_local_stage(
                request.problem,
                request.dataset_id,
                stage,
                parents,
                perturbation_counts=counts,
                progress=progress,
                cancelled=cancelled,
                task_runner=task_runner,
            )
        else:
            parents, _counts = local_stage_continuation(_stage_candidates(state, "D"))
            outcome = run_stage_e(
                request.problem,
                request.dataset_id,
                parents,
                seeds[2:],
                progress=progress,
                cancelled=cancelled,
                task_runner=task_runner,
            )
        state = state.append(outcome)
        perturbation_counts = outcome.perturbation_counts
        _publish_checkpoint(request, state, stage, seeds, checkpoint)
    return _result(request, state, seeds)


def _profile_stage_candidates(search_result: FitSearchResult) -> tuple[FitCandidate, ...]:
    summary = next(
        (value for value in reversed(search_result.stage_summaries) if value.stage in {"E", "stage-e"}),
        None,
    )
    if summary is None:
        return ()
    by_id = {candidate.candidate_id: candidate for candidate in search_result.candidates}
    if any(candidate_id not in by_id for candidate_id in summary.candidate_ids):
        return ()
    return tuple(by_id[candidate_id] for candidate_id in summary.candidate_ids)


def _profile_gain_required(problem: object, objective: float) -> float:
    thresholds = problem.config.confidence
    return max(
        thresholds.equivalent_cost_fraction * abs(objective),
        thresholds.equivalent_cost_floor,
    )


def _validated_profile_center(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
    center_unit: np.ndarray,
) -> bool:
    incumbent = search_result.best_candidate
    if incumbent is None:
        return False
    unit = np.asarray(center_unit, dtype=float)
    if unit.shape != (len(problem.variables),) or np.any(~np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
        return False
    try:
        evaluation = evaluate_model(problem, unit)
    except EvaluationConstraintError:
        return False
    required = _profile_gain_required(problem, incumbent.objective)
    return bool(
        evaluation.valid and np.isfinite(evaluation.objective) and evaluation.objective + required < incumbent.objective
    )


def _profile_replacements(
    originals: tuple[FitCandidate, ...],
    rebuilt: tuple[FitCandidate, ...],
) -> tuple[tuple[str, ...], dict[str, FitCandidate]] | None:
    original_ids = tuple(candidate.candidate_id for candidate in originals)
    if tuple(candidate.candidate_id for candidate in rebuilt) != original_ids:
        return None
    return original_ids, dict(zip(original_ids, rebuilt, strict=True))


def _replace_profile_candidates(
    search_result: FitSearchResult,
    replacements: dict[str, FitCandidate],
) -> tuple[FitCandidate, ...] | None:
    replaced_count = sum(candidate.candidate_id in replacements for candidate in search_result.candidates)
    if replaced_count != 4:
        return None
    return tuple(replacements.get(candidate.candidate_id, candidate) for candidate in search_result.candidates)


def _profile_summary(
    summary: FitStageSummary,
    candidate_ids: tuple[str, ...],
    rebuilt: tuple[FitCandidate, ...],
) -> FitStageSummary:
    if summary.stage not in {"E", "stage-e"}:
        return summary
    return FitStageSummary(
        summary.stage,
        candidate_ids,
        min(candidate.objective for candidate in rebuilt),
        sum(candidate.nfev for candidate in rebuilt),
        tuple(candidate.stop_reason for candidate in rebuilt),
    )


def _replace_profile_stage(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
    originals: tuple[FitCandidate, ...],
    rebuilt: tuple[FitCandidate, ...],
) -> FitSearchResult | None:
    replacement_data = _profile_replacements(originals, rebuilt)
    if replacement_data is None:
        return None
    original_ids, replacements = replacement_data
    candidates = _replace_profile_candidates(search_result, replacements)
    if candidates is None:
        return None
    summaries = tuple(_profile_summary(summary, original_ids, rebuilt) for summary in search_result.stage_summaries)
    best_index = best_candidate_index(candidates, eligible_ids=original_ids)
    if best_index is None:
        return None
    result = FitSearchResult(
        parameter_definitions=search_result.parameter_definitions,
        candidates=candidates,
        best_index=best_index,
        warnings=search_result.warnings,
        child_seeds=search_result.child_seeds,
        stage_summaries=summaries,
        region_labels=search_result.region_labels,
        region_weights=search_result.region_weights,
    )
    return _seal_result(problem, result)


def _profile_runtime_warnings(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
) -> tuple[str, ...]:
    coarse_problem = compile_coarse_problem(problem)
    base = frozenset(_base_warnings(problem, coarse_problem))
    return tuple(value for value in search_result.warnings if value not in base)


def _profile_checkpoint(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
) -> FitCheckpoint:
    return build_checkpoint(
        problem,
        stage="E",
        candidates=search_result.candidates,
        child_seeds=search_result.child_seeds,
        runtime_warnings=_profile_runtime_warnings(problem, search_result),
        stage_summaries=search_result.stage_summaries,
    )


def _validate_profile_continuation(
    problem: object,
    search_result: object,
    parameter_name: object,
) -> tuple[FitEvaluationContext, FitSearchResult, str]:
    if not isinstance(problem, FitEvaluationContext):
        raise TypeError("problem must be a FitEvaluationContext")
    if not isinstance(search_result, FitSearchResult):
        raise TypeError("search_result must be a FitSearchResult")
    if not isinstance(parameter_name, str) or not parameter_name.strip():
        raise ValueError("parameter_name must be a nonempty string")
    _require_owned_result(problem, search_result)
    return problem, search_result, parameter_name


def _raise_if_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is None or not cancelled():
        return
    from xrr_fitter.fit.local_search import SearchCancelled

    raise SearchCancelled("search cancelled")


def continue_profile_basin(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
    center_unit: np.ndarray,
    *,
    parameter_name: str,
    cancelled: Callable[[], bool] | None = None,
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
    task_runner: TaskRunner | None = None,
) -> FitSearchResult:
    """Atomically replace Stage-E evidence after verified basin recovery."""
    problem, search_result, parameter_name = _validate_profile_continuation(
        problem,
        search_result,
        parameter_name,
    )
    _raise_if_cancelled(cancelled)
    if not _validated_profile_center(problem, search_result, center_unit):
        return search_result
    originals = _profile_stage_candidates(search_result)
    final_count = problem.config.final_seed_count
    seeds = tuple(search_result.child_seeds[-final_count:])
    rebuilt = reconverge_profile_basin(
        problem,
        originals,
        center_unit,
        seeds,
        parameter_name=parameter_name,
        cancelled=cancelled,
        task_runner=task_runner,
    )
    if rebuilt is None:
        return search_result
    _raise_if_cancelled(cancelled)
    result = _replace_profile_stage(problem, search_result, originals, rebuilt)
    if result is None:
        return search_result
    if checkpoint is not None:
        checkpoint(_profile_checkpoint(problem, result))
    return result
