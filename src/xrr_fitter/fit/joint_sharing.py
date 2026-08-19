"""Validation and immutable scatter operations for joint fitting."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.joint_constraints import apply_joint_constraints
from xrr_fitter.fit.joint_roughness import (
    apply_shared_roughness,
    initialize_shared_roughness,
)
from xrr_fitter.model.parameters import (
    ParameterReference,
    SharingRule,
)


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
    if any(_definition_signature(definition) != baseline for definition, _problem in definitions[1:]):
        raise ValueError("sharing coordinates must have compatible parameter families and bounds")
    if baseline_definition.category == "instrument" and any(
        problem.instrument != baseline_problem.instrument for _definition, problem in definitions[1:]
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


def _validated_global_unit(problem: object, global_unit: np.ndarray) -> np.ndarray:
    unit = np.asarray(global_unit, dtype=float)
    valid = (
        unit.ndim == 1
        and unit.shape == (len(problem.global_variables),)
        and np.all(np.isfinite(unit))
        and np.all((unit >= 0.0) & (unit <= 1.0))
    )
    if not valid:
        raise ValueError("global unit vector must match the joint shape, finite values, and bounds")
    return unit


def _roughness_target_placeholders(
    local_problem: object,
    scatter: tuple[int, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for coordinate, global_index in zip(local_problem.variables, scatter, strict=True):
        if global_index >= 0:
            continue
        definition = local_problem.parameter_definitions[coordinate.parameter_index]
        if definition.transform == "roughness_fraction":
            values[definition.name] = definition.lower
    return values


def _encode_declared_local(
    local_problem: object,
    scatter: tuple[int, ...],
) -> np.ndarray:
    active_indices = tuple(index for index, global_index in enumerate(scatter) if global_index >= 0)
    active_problem = replace(
        local_problem,
        variables=tuple(local_problem.variables[index] for index in active_indices),
    )
    encoded = encode_physical_vector(
        active_problem,
        _roughness_target_placeholders(local_problem, scatter),
    )
    local = np.zeros(len(scatter), dtype=float)
    local[np.asarray(active_indices, dtype=int)] = encoded
    return local


def _raw_scatter(problem: object, unit: np.ndarray) -> list[np.ndarray]:
    local = []
    for local_problem, scatter in zip(
        problem.problems,
        problem.scatter_maps,
        strict=True,
    ):
        vector = _encode_declared_local(local_problem, scatter)
        for local_index, global_index in enumerate(scatter):
            if global_index >= 0:
                vector[local_index] = unit[global_index]
        local.append(vector)
    return local


def scatter_joint_vector(problem: object, global_unit: np.ndarray) -> tuple[np.ndarray, ...]:
    """Project global coordinates, preserving shared roughness in angstroms."""
    unit = _validated_global_unit(problem, global_unit)
    local = _raw_scatter(problem, unit)
    apply_joint_constraints(problem, local, roughness=False)
    apply_shared_roughness(problem, unit, local)
    apply_joint_constraints(problem, local, roughness=True)
    for vector in local:
        vector.setflags(write=False)
    return tuple(local)


def initial_joint_vector(problem: object) -> np.ndarray:
    """Build the global unit vector from the first local owner of each coordinate."""
    global_unit = np.full(len(problem.global_variables), np.nan, dtype=float)
    for local_problem, scatter in zip(problem.problems, problem.scatter_maps, strict=True):
        local = _encode_declared_local(local_problem, scatter)
        for local_index, global_index in enumerate(scatter):
            if global_index < 0:
                continue
            if np.isnan(global_unit[global_index]):
                global_unit[global_index] = local[local_index]
    if np.any(~np.isfinite(global_unit)):
        raise ValueError("joint global layout contains an unbound coordinate")
    local = _raw_scatter(problem, global_unit)
    apply_joint_constraints(problem, local, roughness=False)
    initialize_shared_roughness(problem, global_unit, local)
    return global_unit


def validated_joint_initial_vector(
    problem: object,
    initial_unit_vector: object | None,
) -> np.ndarray | None:
    """Own and validate an optional explicit unit-space joint start."""
    if initial_unit_vector is None:
        return None
    unit = np.array(initial_unit_vector, dtype=float, copy=True)
    valid = (
        unit.ndim == 1
        and unit.shape == (len(problem.global_variables),)
        and np.all(np.isfinite(unit))
        and np.all((unit >= 0.0) & (unit <= 1.0))
    )
    if not valid:
        raise ValueError("joint initial unit vector must match the joint shape, finite values, and bounds")
    unit.setflags(write=False)
    return unit
