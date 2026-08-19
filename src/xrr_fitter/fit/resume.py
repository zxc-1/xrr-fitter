"""Fail-closed validation of single-dataset search checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from xrr_fitter.fit.candidates import candidate_from_evaluation, rank_candidate_indices
from xrr_fitter.fit.checkpoint import checkpoint_identity
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.stages import remaining_stages
from xrr_fitter.model.fitting import FitCandidate, FitCheckpoint, FitStageSummary


@dataclass(frozen=True, slots=True)
class ResumePlan:
    completed_stage: str
    remaining_stages: tuple[str, ...]
    consumed_child_seeds: tuple[int, ...]
    candidates: tuple[FitCandidate, ...]
    runtime_warnings: tuple[str, ...]
    stage_summaries: tuple[FitStageSummary, ...]


def _validate_identity(
    problem: object,
    checkpoint: FitCheckpoint,
    expected_joint_layout_fingerprint: str,
) -> None:
    expected = checkpoint_identity(problem)
    fields = (
        "data_sha256",
        "structure_fingerprint",
        "instrument_fingerprint",
        "config_fingerprint",
        "parameter_settings_fingerprint",
    )
    mismatch = next(
        (field for field in fields if getattr(checkpoint, field) != getattr(expected, field)),
        None,
    )
    if mismatch is not None:
        raise ValueError(f"resume checkpoint {mismatch} mismatch")
    if checkpoint.joint_layout_fingerprint != expected_joint_layout_fingerprint:
        if expected_joint_layout_fingerprint:
            raise ValueError("joint resume layout fingerprint mismatch")
        raise ValueError("single-fit resume checkpoint contains a joint layout")


def _expected_stage_prefix(stage: str, *, joint: bool) -> tuple[str, ...]:
    order = ("B", "C", "D", "E") if joint else ("A", "B", "C", "D", "E")
    if stage not in {"B", "C", "D", "E"}:
        raise ValueError(f"unsupported resume checkpoint stage: {stage}")
    return order[: order.index(stage) + 1]


def _validate_history(checkpoint: FitCheckpoint, *, joint: bool) -> None:
    observed = tuple(summary.stage for summary in checkpoint.stage_summaries)
    if observed != _expected_stage_prefix(checkpoint.stage, joint=joint):
        raise ValueError("resume checkpoint history has a missing or reordered stage")
    expected_ids = tuple(
        candidate_id
        for summary in checkpoint.stage_summaries
        if joint or summary.stage != "A"
        for candidate_id in summary.candidate_ids
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in checkpoint.candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("resume checkpoint candidate IDs must be unique")
    if candidate_ids != expected_ids:
        raise ValueError("resume checkpoint candidate order does not match history")


def _array_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return first.shape == second.shape and np.array_equal(first, second, equal_nan=True)


def _stack_equal(first: object | None, second: object | None) -> bool:
    if first is None or second is None:
        return first is second
    return (
        first.periodic_spans == second.periodic_spans
        and _array_equal(first.thickness_a, second.thickness_a)
        and _array_equal(first.sld_a2, second.sld_a2)
        and _array_equal(first.roughness_a, second.roughness_a)
    )


def _candidate_metadata(candidate: FitCandidate) -> tuple[object, ...]:
    return (
        candidate.parameters,
        candidate.objective,
        candidate.valid,
        candidate.stop_reason,
    )


def _candidate_arrays(candidate: FitCandidate) -> tuple[np.ndarray, ...]:
    return (
        candidate.qz_a_inv,
        candidate.model_normalized,
        candidate.log_residuals_decades,
        candidate.weighted_residuals,
        candidate.sld_depth_a,
        candidate.sld_profile_a2,
    )


def _derived_candidate_equal(first: FitCandidate, second: FitCandidate) -> bool:
    return (
        _candidate_metadata(first) == _candidate_metadata(second)
        and all(
            _array_equal(first_array, second_array)
            for first_array, second_array in zip(
                _candidate_arrays(first),
                _candidate_arrays(second),
                strict=True,
            )
        )
        and _stack_equal(first.expanded_stack, second.expanded_stack)
    )


def _validate_candidate_layout(
    problem: object,
    candidates: tuple[FitCandidate, ...],
    *,
    joint: bool,
) -> tuple[FitCandidate, ...]:
    width = len(problem.variables)
    normalized = []
    for candidate in candidates:
        if candidate.unit_vector.shape != (width,):
            raise ValueError("resume checkpoint candidate unit width mismatch")
        if not joint and candidate.ranking_objective is not None:
            raise ValueError("single-fit resume candidate contains a ranking objective")
        evaluation = evaluate_vector(problem, candidate.unit_vector)
        expected = candidate_from_evaluation(
            problem,
            candidate.unit_vector,
            evaluation,
            candidate.candidate_id,
            candidate.seed_index,
            candidate.stop_reason,
            candidate.nfev,
        )
        if not _derived_candidate_equal(candidate, expected):
            raise ValueError("resume checkpoint candidate derived state mismatch")
        normalized.append(replace(expected, ranking_objective=candidate.ranking_objective))
    return tuple(normalized)


def _summary_for_candidates(
    summary: FitStageSummary,
    candidates: tuple[FitCandidate, ...],
    *,
    joint: bool,
) -> FitStageSummary:
    if joint:
        rankings = tuple(candidate.ranking_objective for candidate in candidates)
        if any(value is None for value in rankings):
            raise ValueError("joint resume candidate ranking objective is missing")
        best = min(rankings, default=float("inf"))
    else:
        ranked = rank_candidate_indices(candidates)
        best = candidates[ranked[0]].objective if ranked else float("inf")
    return FitStageSummary(
        summary.stage,
        tuple(candidate.candidate_id for candidate in candidates),
        float(best),
        sum(candidate.nfev for candidate in candidates),
        tuple(candidate.stop_reason for candidate in candidates),
    )


def _validate_summaries(checkpoint: FitCheckpoint, *, joint: bool) -> None:
    offset = 0
    for summary in checkpoint.stage_summaries:
        if summary.stage == "A" and not joint:
            continue
        stop = offset + len(summary.candidate_ids)
        candidates = checkpoint.candidates[offset:stop]
        if summary != _summary_for_candidates(summary, candidates, joint=joint):
            raise ValueError("resume checkpoint stage summary mismatch")
        offset = stop


def _validate_seeds(
    checkpoint: FitCheckpoint,
    reserved_child_seeds: tuple[int, ...],
    *,
    joint: bool,
) -> tuple[int, ...]:
    reserved = tuple(reserved_child_seeds)
    if joint:
        expected = reserved[:1]
        if checkpoint.stage == "E":
            final_ids = checkpoint.stage_summaries[-1].candidate_ids
            expected_ids = tuple(f"E-{index}" for index in range(len(final_ids)))
            if len(final_ids) > len(reserved) - 1 or final_ids != expected_ids:
                raise ValueError("joint resume Stage-E candidates must form a child seed prefix")
            expected = reserved[: len(final_ids) + 1]
    else:
        expected = reserved if checkpoint.stage == "E" else reserved[:2]
    consumed = tuple(checkpoint.child_seeds)
    if consumed != expected:
        raise ValueError("resume checkpoint child seed count or order mismatch")
    return consumed


def validate_resume_checkpoint(
    problem: object,
    checkpoint: FitCheckpoint,
    *,
    reserved_child_seeds: tuple[int, ...],
    expected_joint_layout_fingerprint: str = "",
) -> ResumePlan:
    """Validate a complete checkpoint before any resumed callback is emitted."""
    if not isinstance(checkpoint, FitCheckpoint):
        raise TypeError("resume checkpoint must be a FitCheckpoint")
    expected_joint_layout = str(expected_joint_layout_fingerprint)
    _validate_identity(problem, checkpoint, expected_joint_layout)
    joint = bool(expected_joint_layout)
    _validate_history(checkpoint, joint=joint)
    candidates = _validate_candidate_layout(problem, checkpoint.candidates, joint=joint)
    _validate_summaries(checkpoint, joint=joint)
    consumed = _validate_seeds(checkpoint, reserved_child_seeds, joint=joint)
    return ResumePlan(
        checkpoint.stage,
        remaining_stages(checkpoint.stage),
        consumed,
        candidates,
        checkpoint.runtime_warnings,
        checkpoint.stage_summaries,
    )
