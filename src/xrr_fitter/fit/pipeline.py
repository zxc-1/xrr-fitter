"""Pure, pickle-safe single-dataset A-E fitting pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from xrr_fitter.fit.candidates import best_candidate_index
from xrr_fitter.fit.checkpoint import build_checkpoint
from xrr_fitter.fit.resume import ResumePlan, validate_resume_checkpoint
from xrr_fitter.fit.stages import (
    StageOutcome,
    local_stage_continuation,
    reserve_child_seeds,
    run_local_stage,
    run_stage_a,
    run_stage_b,
    run_stage_e,
    stage_b_continuation,
)
from xrr_fitter.model.fitting import (
    FitCandidate,
    FitCheckpoint,
    FitProgress,
    FitSearchResult,
    FitStageSummary,
)


@dataclass(frozen=True, slots=True)
class FitSearchRequest:
    dataset_id: str | None
    problem: object
    resume_checkpoint: FitCheckpoint | None = None


@dataclass(frozen=True, slots=True)
class _SearchState:
    candidates: tuple[FitCandidate, ...] = ()
    warnings: tuple[str, ...] = ()
    summaries: tuple[FitStageSummary, ...] = ()

    def append(self, outcome: StageOutcome) -> _SearchState:
        return _SearchState(
            self.candidates + outcome.candidates,
            self.warnings + outcome.warnings,
            self.summaries + (outcome.summary,),
        )


def _seed_ledger(request: FitSearchRequest) -> tuple[int, ...]:
    streams = (
        "B-0",
        "B-1",
        *(f"E-{index}" for index in range(request.problem.config.final_seed_count)),
    )
    return tuple(
        child.seed
        for child in reserve_child_seeds(request.problem.config.master_seed, streams)
    )


def _state_from_resume(plan: ResumePlan) -> _SearchState:
    return _SearchState(plan.candidates, plan.runtime_warnings, plan.stage_summaries)


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
            runtime_warnings=state.warnings,
            stage_summaries=state.summaries,
        )
    )


def _result(request: FitSearchRequest, state: _SearchState, seeds: tuple[int, ...]) -> FitSearchResult:
    eligible = _stage_candidates(state, "E")
    eligible_ids = tuple(candidate.candidate_id for candidate in eligible)
    return FitSearchResult(
        parameter_definitions=request.problem.parameter_definitions,
        candidates=state.candidates,
        best_index=best_candidate_index(state.candidates, eligible_ids=eligible_ids),
        warnings=state.warnings,
        child_seeds=seeds,
        stage_summaries=state.summaries,
        region_labels=request.problem.region_labels,
        region_weights=request.problem.weights,
    )


def run_fit_search(
    request: FitSearchRequest,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[FitProgress], None] | None = None,
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
) -> FitSearchResult:
    """Run a fresh search or the exact suffix after a validated checkpoint."""
    if not isinstance(request, FitSearchRequest):
        raise TypeError("request must be a FitSearchRequest")
    seeds = _seed_ledger(request)
    if request.resume_checkpoint is None:
        state = _SearchState(warnings=tuple(request.problem.warnings))
        remaining = ("A", "B", "C", "D", "E")
    else:
        plan = validate_resume_checkpoint(
            request.problem,
            request.resume_checkpoint,
            reserved_child_seeds=seeds,
        )
        state = _state_from_resume(plan)
        remaining = plan.remaining_stages

    starts = None
    perturbation_counts: tuple[int, ...] = ()
    for stage in remaining:
        if stage == "A":
            starts, summary, warnings = run_stage_a(
                request.problem,
                request.dataset_id,
                progress=progress,
                cancelled=cancelled,
            )
            state = _SearchState(
                state.candidates,
                state.warnings + warnings,
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
                parents, counts = local_stage_continuation(stage_candidates)
            outcome = run_local_stage(
                request.problem,
                request.dataset_id,
                stage,
                parents,
                perturbation_counts=counts,
                progress=progress,
                cancelled=cancelled,
            )
        else:
            parents, _counts = local_stage_continuation(
                _stage_candidates(state, "D")
            )
            outcome = run_stage_e(
                request.problem,
                request.dataset_id,
                parents,
                seeds[2:],
                progress=progress,
                cancelled=cancelled,
            )
        state = state.append(outcome)
        _publish_checkpoint(request, state, stage, seeds, checkpoint)
    return _result(request, state, seeds)
