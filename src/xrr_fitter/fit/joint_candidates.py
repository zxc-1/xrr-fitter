"""Build and validate joint vectors from aligned dataset candidates."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.joint_constraints import apply_joint_constraints
from xrr_fitter.fit.joint_evaluation import _finite_objective_mean
from xrr_fitter.fit.joint_roughness import (
    SHARED_ROUGHNESS_TRANSFORM,
    apply_consensus_roughness,
    rebuild_candidate_roughness,
)
from xrr_fitter.fit.joint_sharing import _raw_scatter, scatter_joint_vector


def _candidate_for_dataset(
    dataset_id: str,
    candidates_by_dataset: object,
) -> object:
    try:
        candidate = candidates_by_dataset[dataset_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"prefit candidate is missing: {dataset_id}") from error
    if candidate is None or not getattr(candidate, "valid", False):
        raise ValueError(f"prefit candidate is invalid: {dataset_id}")
    return candidate


def _candidate_local_layout(
    dataset_id: str,
    local_problem: object,
    candidates_by_dataset: object,
) -> tuple[np.ndarray, dict[str, int], dict[str, float]]:
    candidate = _candidate_for_dataset(dataset_id, candidates_by_dataset)
    physical = {parameter.name: parameter.value for parameter in candidate.parameters}
    indices = {coordinate.name: index for index, coordinate in enumerate(local_problem.variables)}
    missing = next((name for name in indices if name not in physical), None)
    if missing is not None:
        raise ValueError(f"prefit candidate parameter is missing: {dataset_id}/{missing}")
    return encode_physical_vector(local_problem, physical), indices, physical


def _consensus_value(
    variable: object,
    layouts: dict[str, tuple[np.ndarray, dict[str, int], dict[str, float]]],
) -> float:
    values = tuple(
        layouts[member.dataset_id][0][layouts[member.dataset_id][1][member.parameter_name]]
        for member in variable.members
    )
    return float(np.median(values))


def consensus_joint_vector(
    problem: object,
    candidates_by_dataset: object,
) -> np.ndarray:
    """Build a global start from the median of each variable's prefit members."""
    layouts = {
        dataset_id: _candidate_local_layout(
            dataset_id,
            local_problem,
            candidates_by_dataset,
        )
        for dataset_id, local_problem in zip(
            problem.dataset_ids,
            problem.problems,
            strict=True,
        )
    }
    consensus = np.asarray(
        [_consensus_value(variable, layouts) for variable in problem.global_variables],
        dtype=float,
    )
    local = _raw_scatter(problem, consensus)
    apply_joint_constraints(problem, local, roughness=False)
    apply_consensus_roughness(
        problem,
        consensus,
        local,
        {dataset_id: physical for dataset_id, (_unit, _indices, physical) in layouts.items()},
    )
    if np.any(~np.isfinite(consensus)) or np.any((consensus < 0.0) | (consensus > 1.0)):
        raise ValueError("prefit consensus must contain finite values within [0, 1]")
    consensus.setflags(write=False)
    return consensus


def _candidate_maps(
    candidates_by_dataset: tuple[tuple[object, ...], ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {candidate.candidate_id: candidate for candidate in candidates} for candidates in candidates_by_dataset
    )


def _nonrough_candidate_vector(
    problem: object,
    local_units: list[np.ndarray],
) -> np.ndarray:
    global_unit = np.full(len(problem.global_variables), np.nan, dtype=float)
    for dataset_index, scatter in enumerate(problem.scatter_maps):
        local = local_units[dataset_index]
        for local_index, global_index in enumerate(scatter):
            if global_index < 0:
                continue
            variable = problem.global_variables[global_index]
            if variable.transform == SHARED_ROUGHNESS_TRANSFORM:
                continue
            value = local[local_index]
            if np.isnan(global_unit[global_index]):
                global_unit[global_index] = value
            elif global_unit[global_index] != value:
                raise ValueError("joint candidate shared unit projection mismatch")
    return global_unit


def _validate_constraint_target_projection(
    problem: object,
    global_unit: np.ndarray,
    local_units: list[np.ndarray],
) -> None:
    if not problem.joint_constraint_rules:
        return
    try:
        projected = scatter_joint_vector(problem, global_unit)
    except EvaluationConstraintError as error:
        raise ValueError("joint candidate constraint target projection is invalid") from error
    dataset_indices = {dataset_id: index for index, dataset_id in enumerate(problem.dataset_ids)}
    for target in {rule.target for rule in problem.joint_constraint_rules}:
        dataset_index = dataset_indices[target.dataset_id]
        local_problem = problem.problems[dataset_index]
        local_index = next(
            index
            for index, coordinate in enumerate(local_problem.variables)
            if coordinate.name == target.parameter_name
        )
        if not np.isclose(
            local_units[dataset_index][local_index],
            projected[dataset_index][local_index],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"joint candidate constraint target projection mismatch: {target.dataset_id}::{target.parameter_name}"
            )


def _joint_candidate_vector(
    problem: object,
    candidates_by_id: tuple[dict[str, object], ...],
    candidate_id: str,
) -> np.ndarray:
    aligned_candidates = [candidates[candidate_id] for candidates in candidates_by_id]
    local_units = [np.asarray(candidate.unit_vector, dtype=float) for candidate in aligned_candidates]
    global_unit = _nonrough_candidate_vector(problem, local_units)
    rebuild_candidate_roughness(problem, global_unit, local_units)
    if np.any(~np.isfinite(global_unit)):
        raise ValueError("joint candidate global projection is incomplete")
    if all(getattr(candidate, "valid", True) for candidate in aligned_candidates):
        _validate_constraint_target_projection(
            problem,
            global_unit,
            local_units,
        )
    return global_unit


def joint_candidate_vectors(
    problem: object,
    candidates_by_dataset: tuple[tuple[object, ...], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    """Reconstruct aligned global vectors from dataset-local candidates."""
    candidates_by_id = _candidate_maps(candidates_by_dataset)
    return tuple(_joint_candidate_vector(problem, candidates_by_id, candidate_id) for candidate_id in candidate_ids)


def _validate_candidate_order(
    candidates_by_dataset: tuple[tuple[object, ...], ...],
) -> tuple[str, ...]:
    ids = tuple(candidate.candidate_id for candidate in candidates_by_dataset[0])
    if any(
        tuple(candidate.candidate_id for candidate in candidates) != ids for candidates in candidates_by_dataset[1:]
    ):
        raise ValueError("joint resume candidate order mismatch")
    return ids


def _validate_candidate_rankings(
    candidates_by_dataset: tuple[tuple[object, ...], ...],
    candidate_ids: tuple[str, ...],
) -> None:
    for candidate_index in range(len(candidate_ids)):
        aligned = tuple(candidates[candidate_index] for candidates in candidates_by_dataset)
        ranking = _finite_objective_mean(tuple(candidate.objective for candidate in aligned))
        if any(candidate.ranking_objective != ranking for candidate in aligned):
            raise ValueError("joint resume candidate ranking objective mismatch")


def validate_joint_candidate_alignment(
    problem: object,
    candidates_by_dataset: tuple[tuple[object, ...], ...],
    stage_summaries: tuple[object, ...],
) -> None:
    """Reject cross-dataset checkpoint drift before a resumed stage runs."""
    candidate_ids = _validate_candidate_order(candidates_by_dataset)
    _validate_candidate_rankings(candidates_by_dataset, candidate_ids)
    for summary in stage_summaries:
        if summary.stage != "A":
            joint_candidate_vectors(
                problem,
                candidates_by_dataset,
                summary.candidate_ids,
            )


__all__ = [
    "consensus_joint_vector",
    "joint_candidate_vectors",
    "validate_joint_candidate_alignment",
]
