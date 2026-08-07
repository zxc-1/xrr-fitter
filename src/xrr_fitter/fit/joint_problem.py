"""Compilation of stable global coordinate layouts for joint fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from xrr_fitter.fit.checkpoint import checkpoint_identity
from xrr_fitter.fit.joint_roughness import SHARED_ROUGHNESS_TRANSFORM
from xrr_fitter.fit.joint_sharing import validate_sharing_rules
from xrr_fitter.model.parameters import ParameterReference, SharingRule


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
    global_variables: tuple[JointVariable, ...]
    scatter_maps: tuple[tuple[int, ...], ...]
    layout_fingerprint: str


def _validate_inputs(dataset_ids: tuple[str, ...], problems: tuple[object, ...]) -> None:
    if len(dataset_ids) != len(problems) or len(dataset_ids) < 2:
        raise ValueError("joint fitting requires aligned problems for at least two datasets")
    if any(not value for value in dataset_ids) or len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("joint dataset IDs must be nonempty and unique")


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
    transform = (
        SHARED_ROUGHNESS_TRANSFORM
        if definition.transform == "roughness_fraction"
        else definition.transform
    )
    return JointVariable(rule.sharing_key, transform, rule.sharing_key, rule.members)


def _layout(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    owners: dict[ParameterReference, SharingRule],
) -> tuple[tuple[JointVariable, ...], tuple[tuple[int, ...], ...]]:
    variables: list[JointVariable] = []
    indices: dict[tuple[str, str], int] = {}
    group_indices: dict[str, int] = {}
    scatters: list[tuple[int, ...]] = []
    for dataset_id, problem in zip(dataset_ids, problems, strict=True):
        local: list[int] = []
        for coordinate in problem.variables:
            reference = ParameterReference(dataset_id, coordinate.name)
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
    variables: tuple[JointVariable, ...],
    scatters: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    identities = tuple(checkpoint_identity(problem) for problem in problems)
    return {
        "dataset_ids": dataset_ids,
        "identities": tuple(tuple(getattr(value, field) for field in value.__dataclass_fields__) for value in identities),
        "rules": tuple(
            (rule.sharing_key, tuple((member.dataset_id, member.parameter_name) for member in rule.members))
            for rule in rules
        ),
        "variables": tuple(
            (value.name, value.transform, value.sharing_key, tuple((m.dataset_id, m.parameter_name) for m in value.members))
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
) -> JointFitProblem:
    """Compile validated sharing declarations into one global unit layout."""
    ids = tuple(dataset_ids)
    local_problems = tuple(problems)
    rules = tuple(sharing_rules)
    _validate_inputs(ids, local_problems)
    owners = validate_sharing_rules(ids, local_problems, rules)
    variables, scatters = _layout(ids, local_problems, owners)
    payload = _fingerprint_payload(ids, local_problems, rules, variables, scatters)
    return JointFitProblem(
        ids,
        local_problems,
        rules,
        variables,
        scatters,
        _layout_fingerprint(payload),
    )
