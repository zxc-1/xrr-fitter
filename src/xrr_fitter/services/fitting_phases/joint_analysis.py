"""Joint candidate alignment and uncertainty projection."""

from __future__ import annotations

from collections.abc import Callable

from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.fitting import FitCheckpoint

from .common import PreparedDatasetFit

def _joint_checkpoints(
    prepared: tuple[PreparedDatasetFit, ...],
) -> tuple[FitCheckpoint, ...] | None:
    values = tuple(item.updated_dataset.checkpoint for item in prepared)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("joint resume requires checkpoints for all datasets")
    return tuple(value for value in values if value is not None)


def _joint_final_ids(searches: tuple[object, ...]) -> tuple[str, ...]:
    summaries = tuple(
        next(summary for summary in reversed(search.stage_summaries) if summary.stage == "E")
        for search in searches
    )
    if any(summary != summaries[0] for summary in summaries[1:]):
        raise ValueError("joint Stage-E history is not aligned")
    return summaries[0].candidate_ids


def _joint_candidate_maps(searches: tuple[object, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {candidate.candidate_id: candidate for candidate in search.candidates}
        for search in searches
    )


def _joint_candidate_rows(
    candidate_maps: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(candidate_map[candidate_id] for candidate_map in candidate_maps)
        for candidate_id in candidate_ids
    )


def _joint_objectives(rows: tuple[tuple[object, ...], ...]) -> tuple[float, ...]:
    return tuple(float(candidates[0].ranking_objective) for candidates in rows)


def _joint_validity(rows: tuple[tuple[object, ...], ...]) -> tuple[bool, ...]:
    return tuple(all(candidate.valid for candidate in candidates) for candidates in rows)


def _joint_diagnostics(
    rows: tuple[tuple[object, ...], ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(
            diagnostic
            for candidate in candidates
            for diagnostic in candidate.diagnostics
        )
        for candidates in rows
    )


def _joint_physical_values(
    problem: object,
    candidate_maps: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    dataset_indices = {
        dataset_id: index for index, dataset_id in enumerate(problem.dataset_ids)
    }
    rows = []
    for candidate_id in candidate_ids:
        values = []
        for variable in problem.global_variables:
            member = variable.members[0]
            candidate = candidate_maps[dataset_indices[member.dataset_id]][candidate_id]
            parameter = next(
                value for value in candidate.parameters if value.name == member.parameter_name
            )
            values.append(parameter.value)
        rows.append(tuple(values))
    return tuple(rows)


def _analyze_joint_searches(
    problem: object,
    searches: tuple[object, ...],
    *,
    joint_candidate_vectors: Callable,
    analyze_joint_ensemble: Callable,
) -> tuple[FitResult, ...]:
    candidate_ids = _joint_final_ids(searches)
    candidate_maps = _joint_candidate_maps(searches)
    vectors = joint_candidate_vectors(
        problem,
        tuple(search.candidates for search in searches),
        candidate_ids,
    )
    aligned = _joint_candidate_rows(candidate_maps, candidate_ids)
    report, confidence, evidence = analyze_joint_ensemble(
        variable_names=tuple(variable.name for variable in problem.global_variables),
        candidate_ids=candidate_ids,
        unit_vectors=vectors,
        physical_values=_joint_physical_values(problem, candidate_maps, candidate_ids),
        objectives=_joint_objectives(aligned),
        valid=_joint_validity(aligned),
        diagnostics=_joint_diagnostics(aligned),
        thresholds=problem.problems[0].config.confidence,
    )
    return tuple(
        FitResult.from_search(
            search,
            confidence=confidence,
            uncertainty=report,
            classification_evidence=evidence,
        )
        for search in searches
    )
