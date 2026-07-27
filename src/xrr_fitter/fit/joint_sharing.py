"""Validation and immutable scatter operations for joint fitting."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.model.parameters import ParameterReference, SharingRule


def _definition(problem: object, parameter_name: str) -> object:
    by_name = {value.name: value for value in problem.parameter_definitions}
    try:
        return by_name[parameter_name]
    except KeyError as error:
        raise ValueError(f"sharing parameter is missing: {parameter_name}") from error


def _rule_definitions(
    rule: SharingRule,
    by_dataset: dict[str, object],
) -> tuple[tuple[object, object], ...]:
    datasets = tuple(member.dataset_id for member in rule.members)
    if len(datasets) != len(set(datasets)):
        raise ValueError("sharing group may contain at most one member per dataset")
    definitions = []
    for member in rule.members:
        if member.dataset_id not in by_dataset:
            raise ValueError(f"sharing member dataset is missing: {member.dataset_id}")
        problem = by_dataset[member.dataset_id]
        definition = _definition(problem, member.parameter_name)
        free_names = {coordinate.name for coordinate in problem.variables}
        if member.parameter_name not in free_names:
            raise ValueError(f"sharing coordinate is not free: {member.parameter_name}")
        definitions.append((definition, problem))
    return tuple(definitions)


def _definition_signature(definition: object) -> tuple[object, ...]:
    return (
        definition.category,
        definition.name.rsplit(".", 1)[-1],
        definition.transform,
        definition.integer,
        definition.unit,
        definition.lower,
        definition.upper,
    )


def _validate_compatible(definitions: tuple[tuple[object, object], ...]) -> None:
    baseline_definition, baseline_problem = definitions[0]
    baseline = _definition_signature(baseline_definition)
    if any(
        _definition_signature(definition) != baseline
        for definition, _problem in definitions[1:]
    ):
        raise ValueError("sharing coordinates must have compatible parameter families and bounds")
    if baseline_definition.category == "instrument" and any(
        problem.instrument != baseline_problem.instrument
        for _definition, problem in definitions[1:]
    ):
        raise ValueError("sharing instrument coordinates requires matching instrument identity and semantics")


def validate_sharing_rules(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    rules: tuple[SharingRule, ...],
) -> dict[ParameterReference, SharingRule]:
    """Validate declarations and return exact local-member ownership."""
    by_dataset = dict(zip(dataset_ids, problems, strict=True))
    keys = tuple(rule.sharing_key for rule in rules)
    if len(keys) != len(set(keys)):
        raise ValueError("sharing keys must be unique")
    owner: dict[ParameterReference, SharingRule] = {}
    for rule in rules:
        _validate_compatible(_rule_definitions(rule, by_dataset))
        for member in rule.members:
            if member in owner:
                raise ValueError("sharing coordinate belongs to multiple groups")
            owner[member] = rule
    return owner


def scatter_joint_vector(problem: object, global_unit: np.ndarray) -> tuple[np.ndarray, ...]:
    """Copy one global unit vector into every dataset-local coordinate layout."""
    unit = np.asarray(global_unit, dtype=float)
    valid = (
        unit.ndim == 1
        and unit.shape == (len(problem.global_variables),)
        and np.all(np.isfinite(unit))
        and np.all((unit >= 0.0) & (unit <= 1.0))
    )
    if not valid:
        raise ValueError("global unit vector must match the joint shape, finite values, and bounds")
    local = []
    for scatter in problem.scatter_maps:
        vector = np.array(unit[np.asarray(scatter, dtype=int)], dtype=float, copy=True)
        vector.setflags(write=False)
        local.append(vector)
    return tuple(local)


def initial_joint_vector(problem: object) -> np.ndarray:
    """Build the global unit vector from the first local owner of each coordinate."""
    global_unit = np.full(len(problem.global_variables), np.nan, dtype=float)
    for local_problem, scatter in zip(problem.problems, problem.scatter_maps, strict=True):
        local = encode_physical_vector(local_problem, {})
        for local_index, global_index in enumerate(scatter):
            if np.isnan(global_unit[global_index]):
                global_unit[global_index] = local[local_index]
    if np.any(~np.isfinite(global_unit)):
        raise ValueError("joint global layout contains an unbound coordinate")
    return global_unit


def _candidate_maps(
    candidates_by_dataset: tuple[tuple[object, ...], ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {candidate.candidate_id: candidate for candidate in candidates}
        for candidates in candidates_by_dataset
    )


def _joint_candidate_vector(
    problem: object,
    candidates_by_id: tuple[dict[str, object], ...],
    candidate_id: str,
) -> np.ndarray:
    global_unit = np.full(len(problem.global_variables), np.nan, dtype=float)
    for dataset_index, scatter in enumerate(problem.scatter_maps):
        local = candidates_by_id[dataset_index][candidate_id].unit_vector
        for local_index, global_index in enumerate(scatter):
            value = local[local_index]
            if np.isnan(global_unit[global_index]):
                global_unit[global_index] = value
            elif global_unit[global_index] != value:
                raise ValueError("joint candidate shared unit projection mismatch")
    return global_unit


def joint_candidate_vectors(
    problem: object,
    candidates_by_dataset: tuple[tuple[object, ...], ...],
    candidate_ids: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    """Reconstruct aligned global vectors from dataset-local candidates."""
    candidates_by_id = _candidate_maps(candidates_by_dataset)
    return tuple(
        _joint_candidate_vector(problem, candidates_by_id, candidate_id)
        for candidate_id in candidate_ids
    )


def _validate_candidate_order(
    candidates_by_dataset: tuple[tuple[object, ...], ...],
) -> tuple[str, ...]:
    ids = tuple(candidate.candidate_id for candidate in candidates_by_dataset[0])
    if any(
        tuple(candidate.candidate_id for candidate in candidates) != ids
        for candidates in candidates_by_dataset[1:]
    ):
        raise ValueError("joint resume candidate order mismatch")
    return ids


def _validate_candidate_rankings(
    candidates_by_dataset: tuple[tuple[object, ...], ...],
    candidate_ids: tuple[str, ...],
) -> None:
    for candidate_index in range(len(candidate_ids)):
        aligned = tuple(candidates[candidate_index] for candidates in candidates_by_dataset)
        ranking = float(np.mean([candidate.objective for candidate in aligned]))
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
            joint_candidate_vectors(problem, candidates_by_dataset, summary.candidate_ids)
