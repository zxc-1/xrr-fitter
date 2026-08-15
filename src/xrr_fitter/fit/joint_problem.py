"""Compilation of stable global coordinate layouts for joint fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256

from xrr_fitter.fit.checkpoint import checkpoint_identity
from xrr_fitter.fit.drift import DRIFT_DATASET, rebind_drift_dataset
from xrr_fitter.fit.joint_constraint_compilation import (
    constraint_node_payload,
    joint_constraint_closure,
    merged_constraints,
    validate_joint_constraints,
    with_cross_target_coordinates,
)
from xrr_fitter.fit.joint_roughness import SHARED_ROUGHNESS_TRANSFORM
from xrr_fitter.fit.joint_sharing import validate_sharing_rules
from xrr_fitter.model.parameters import (
    ConstraintRule,
    ParameterReference,
    SharingRule,
)


@dataclass(frozen=True, slots=True)
class JointVariable:
    name: str
    transform: str
    sharing_key: str | None
    members: tuple[ParameterReference, ...]


@dataclass(frozen=True, slots=True)
class JointFitProblem:
    dataset_ids: tuple[str, ...]
    problems: tuple[object, ...]
    sharing_rules: tuple[SharingRule, ...]
    constraint_rules: tuple[ConstraintRule, ...]
    joint_constraint_rules: tuple[ConstraintRule, ...]
    global_variables: tuple[JointVariable, ...]
    scatter_maps: tuple[tuple[int, ...], ...]
    layout_fingerprint: str


def _validate_inputs(dataset_ids: tuple[str, ...], problems: tuple[object, ...]) -> None:
    if len(dataset_ids) != len(problems) or len(dataset_ids) < 2:
        raise ValueError("joint fitting requires aligned problems for at least two datasets")
    if any(not value for value in dataset_ids) or len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("joint dataset IDs must be nonempty and unique")
    if DRIFT_DATASET in dataset_ids:
        raise ValueError(f"joint dataset ID is reserved: {DRIFT_DATASET}")


def _global_variable(
    reference: ParameterReference,
    definition: object,
    rule: SharingRule | None,
) -> JointVariable:
    if rule is None:
        return JointVariable(
            f"{reference.dataset_id}:{reference.parameter_name}",
            definition.transform,
            None,
            (reference,),
        )
    transform = SHARED_ROUGHNESS_TRANSFORM if definition.transform == "roughness_fraction" else definition.transform
    return JointVariable(rule.sharing_key, transform, rule.sharing_key, rule.members)


def _layout(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    owners: dict[ParameterReference, SharingRule],
    constrained_targets: frozenset[ParameterReference],
) -> tuple[tuple[JointVariable, ...], tuple[tuple[int, ...], ...]]:
    variables: list[JointVariable] = []
    indices: dict[tuple[str, str], int] = {}
    group_indices: dict[str, int] = {}
    scatters: list[tuple[int, ...]] = []
    for dataset_id, problem in zip(dataset_ids, problems, strict=True):
        local: list[int] = []
        for coordinate in problem.variables:
            reference = ParameterReference(dataset_id, coordinate.name)
            if reference in constrained_targets:
                local.append(-1)
                continue
            rule = owners.get(reference)
            if rule is not None and rule.sharing_key in group_indices:
                index = group_indices[rule.sharing_key]
            else:
                definition = problem.parameter_definitions[coordinate.parameter_index]
                index = len(variables)
                variables.append(_global_variable(reference, definition, rule))
                if rule is not None:
                    group_indices[rule.sharing_key] = index
            indices[(dataset_id, coordinate.name)] = index
            local.append(index)
        scatters.append(tuple(local))
    return tuple(variables), tuple(scatters)


def _fingerprint_payload(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    rules: tuple[SharingRule, ...],
    constraint_rules: tuple[ConstraintRule, ...],
    variables: tuple[JointVariable, ...],
    scatters: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    identities = tuple(checkpoint_identity(problem) for problem in problems)
    return {
        "dataset_ids": dataset_ids,
        "identities": tuple(
            tuple(getattr(value, field) for field in value.__dataclass_fields__) for value in identities
        ),
        "rules": tuple(
            (rule.sharing_key, tuple((member.dataset_id, member.parameter_name) for member in rule.members))
            for rule in rules
        ),
        "constraints": tuple(
            (
                rule.target.dataset_id,
                rule.target.parameter_name,
                constraint_node_payload(rule.expression),
            )
            for rule in constraint_rules
        ),
        "variables": tuple(
            (
                value.name,
                value.transform,
                value.sharing_key,
                tuple((m.dataset_id, m.parameter_name) for m in value.members),
            )
            for value in variables
        ),
        "scatters": scatters,
    }


def _layout_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {"schema": "xrr-joint-layout-v1", **payload},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def compile_joint_problem(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    sharing_rules: tuple[SharingRule, ...],
    constraint_rules: tuple[ConstraintRule, ...] = (),
) -> JointFitProblem:
    """Compile validated sharing declarations into one global unit layout."""
    ids = tuple(dataset_ids)
    local_problems = tuple(problems)
    rules = tuple(sharing_rules)
    _validate_inputs(ids, local_problems)
    local_problems = tuple(
        rebind_drift_dataset(problem, dataset_id) for dataset_id, problem in zip(ids, local_problems, strict=True)
    )
    constraints = merged_constraints(
        ids,
        local_problems,
        tuple(constraint_rules),
    )
    validate_joint_constraints(
        ids,
        local_problems,
        rules,
        constraints,
    )
    joint_constraints = joint_constraint_closure(constraints)
    local_constraints = tuple(rule for rule in constraints if rule not in joint_constraints)
    local_problems = tuple(
        replace(
            problem,
            constraint_rules=tuple(rule for rule in local_constraints if rule.target.dataset_id == dataset_id),
        )
        for dataset_id, problem in zip(ids, local_problems, strict=True)
    )
    local_problems = with_cross_target_coordinates(
        ids,
        local_problems,
        joint_constraints,
    )
    owners = validate_sharing_rules(ids, local_problems, rules)
    constrained_targets = frozenset(rule.target for rule in joint_constraints)
    variables, scatters = _layout(
        ids,
        local_problems,
        owners,
        constrained_targets,
    )
    payload = _fingerprint_payload(
        ids,
        local_problems,
        rules,
        constraints,
        variables,
        scatters,
    )
    return JointFitProblem(
        ids,
        local_problems,
        rules,
        constraints,
        joint_constraints,
        variables,
        scatters,
        _layout_fingerprint(payload),
    )
