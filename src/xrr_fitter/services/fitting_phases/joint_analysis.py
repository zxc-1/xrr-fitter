"""Joint candidate alignment and uncertainty projection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

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
        next(summary for summary in reversed(search.stage_summaries) if summary.stage == "E") for search in searches
    )
    if any(summary != summaries[0] for summary in summaries[1:]):
        raise ValueError("joint Stage-E history is not aligned")
    return summaries[0].candidate_ids


def _joint_candidate_maps(searches: tuple[object, ...]) -> tuple[dict[str, object], ...]:
    return tuple({candidate.candidate_id: candidate for candidate in search.candidates} for search in searches)


def _joint_candidate_rows(
    candidate_maps: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(candidate_map[candidate_id] for candidate_map in candidate_maps) for candidate_id in candidate_ids
    )


def _joint_objectives(rows: tuple[tuple[object, ...], ...]) -> tuple[float, ...]:
    return tuple(float(candidates[0].ranking_objective) for candidates in rows)


def _joint_validity(rows: tuple[tuple[object, ...], ...]) -> tuple[bool, ...]:
    return tuple(all(candidate.valid for candidate in candidates) for candidates in rows)


def _joint_diagnostics(
    rows: tuple[tuple[object, ...], ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(diagnostic for candidate in candidates for diagnostic in candidate.diagnostics) for candidates in rows
    )


def _joint_physical_values(
    problem: object,
    candidate_maps: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    dataset_indices = {dataset_id: index for index, dataset_id in enumerate(problem.dataset_ids)}
    rows = []
    for candidate_id in candidate_ids:
        values = []
        for variable in problem.global_variables:
            member = variable.members[0]
            candidate = candidate_maps[dataset_indices[member.dataset_id]][candidate_id]
            parameter = next(value for value in candidate.parameters if value.name == member.parameter_name)
            values.append(parameter.value)
        rows.append(tuple(values))
    return tuple(rows)


def _joint_prior_conflicts(
    problem: object,
    candidate_maps: tuple[dict[str, object], ...],
    candidate_id: str | None,
    priors: tuple[tuple[object, ...], ...],
    *,
    with_parameter_priors: Callable,
    prior_conflicts: Callable,
) -> tuple[str, ...]:
    """Map the winning local prior conflicts onto ordered global variables."""
    if candidate_id is None:
        return ()
    if len(priors) != len(problem.dataset_ids):
        raise ValueError("joint prior batch size mismatch")
    global_by_member = {
        (member.dataset_id, member.parameter_name): variable.name
        for variable in problem.global_variables
        for member in variable.members
    }
    conflicts = set()
    for dataset_id, local_problem, candidate_map, local_priors in zip(
        problem.dataset_ids,
        problem.problems,
        candidate_maps,
        priors,
        strict=True,
    ):
        analysis_problem = with_parameter_priors(local_problem, local_priors)
        local = prior_conflicts(
            analysis_problem,
            candidate_map[candidate_id].unit_vector,
        )
        conflicts.update(global_by_member[(dataset_id, name)] for name in local)
    return tuple(variable.name for variable in problem.global_variables if variable.name in conflicts)


def _analyze_joint_searches(
    problem: object,
    searches: tuple[object, ...],
    priors: tuple[tuple[object, ...], ...],
    *,
    joint_candidate_vectors: Callable,
    analyze_joint_ensemble: Callable,
    with_parameter_priors: Callable,
    prior_conflicts: Callable,
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
    report = replace(
        report,
        prior_conflicts=_joint_prior_conflicts(
            problem,
            candidate_maps,
            report.candidate_id,
            tuple(tuple(values) for values in priors),
            with_parameter_priors=with_parameter_priors,
            prior_conflicts=prior_conflicts,
        ),
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
