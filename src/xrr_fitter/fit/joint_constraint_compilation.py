"""Compile cross-dataset expression constraints into joint layouts."""

from __future__ import annotations

from dataclasses import replace

from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterCoordinate,
    ParameterReference,
    SharingRule,
    _iter_references,
    constraint_cycle_path,
    constraint_sharing_conflicts,
    validate_constraint_stage_split,
)


def constraint_node_payload(node: ConstraintNode) -> object:
    if node.op == "ref":
        return (
            "ref",
            node.reference.dataset_id,
            node.reference.parameter_name,
        )
    if node.op == "const":
        return ("const", node.value)
    return (node.op, tuple(constraint_node_payload(value) for value in node.operands))


def cross_dataset_rules(
    rules: tuple[ConstraintRule, ...],
) -> tuple[ConstraintRule, ...]:
    return tuple(
        rule
        for rule in rules
        if any(reference.dataset_id != rule.target.dataset_id for reference in _iter_references(rule.expression))
    )


def joint_constraint_closure(
    rules: tuple[ConstraintRule, ...],
) -> tuple[ConstraintRule, ...]:
    selected = set(cross_dataset_rules(rules))
    selected_targets = {rule.target for rule in selected}
    while True:
        dependent = {
            rule
            for rule in rules
            if any(reference in selected_targets for reference in _iter_references(rule.expression))
        }
        expanded = selected | dependent
        if expanded == selected:
            return tuple(rule for rule in rules if rule in selected)
        selected = expanded
        selected_targets = {rule.target for rule in selected}


def _local_rule_matches_dataset(rule: ConstraintRule, dataset_id: str) -> bool:
    return rule.target.dataset_id == dataset_id and all(
        reference.dataset_id == dataset_id for reference in _iter_references(rule.expression)
    )


def _compiled_local_constraints(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
) -> tuple[ConstraintRule, ...]:
    rules: list[ConstraintRule] = []
    for dataset_id, problem in zip(dataset_ids, problems, strict=True):
        for rule in problem.constraint_rules:
            if not _local_rule_matches_dataset(rule, dataset_id):
                raise ValueError(
                    f"compiled local constraint dataset identity does not match the joint member: {dataset_id}"
                )
            rules.append(rule)
    return tuple(rules)


def merged_constraints(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    declared: tuple[ConstraintRule, ...],
) -> tuple[ConstraintRule, ...]:
    _validate_rule_types(declared)
    compiled = _compiled_local_constraints(dataset_ids, problems)
    compiled_set = set(compiled)
    for rule in declared:
        if not cross_dataset_rules((rule,)) and rule not in compiled_set:
            raise ValueError("dataset-local constraint arguments must already be compiled into their local fit context")
    return (*declared, *(rule for rule in compiled if rule not in declared))


def definitions_by_reference(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
) -> dict[ParameterReference, object]:
    return {
        ParameterReference(dataset_id, definition.name): definition
        for dataset_id, problem in zip(dataset_ids, problems, strict=True)
        for definition in problem.parameter_definitions
    }


def _validate_rule_types(rules: tuple[ConstraintRule, ...]) -> None:
    if any(not isinstance(rule, ConstraintRule) for rule in rules):
        raise TypeError("constraint_rules must contain ConstraintRule values")


def _validate_unique_targets(rules: tuple[ConstraintRule, ...]) -> None:
    targets = tuple(rule.target for rule in rules)
    if len(targets) != len(set(targets)):
        raise ValueError("constraint targets must be unique")


def _validate_dependency_cycle(rules: tuple[ConstraintRule, ...]) -> None:
    cycle = constraint_cycle_path(rules)
    if cycle:
        raise ValueError(f"constraint rules form a dependency cycle: {' -> '.join(cycle)}")


def validate_joint_constraints(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    sharing_rules: tuple[SharingRule, ...],
    rules: tuple[ConstraintRule, ...],
) -> None:
    _validate_rule_types(rules)
    _validate_unique_targets(rules)
    _validate_dependency_cycle(rules)
    if constraint_sharing_conflicts(rules, sharing_rules):
        raise ValueError("constraint target cannot also be a sharing member")
    validate_constraint_stage_split(
        rules,
        definitions_by_reference(dataset_ids, problems),
    )


def _targets_by_dataset(
    rules: tuple[ConstraintRule, ...],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for rule in rules:
        result.setdefault(rule.target.dataset_id, set()).add(rule.target.parameter_name)
    return result


def _problem_with_target_coordinates(
    dataset_id: str,
    problem: object,
    targets: set[str],
) -> object:
    if not targets:
        return problem
    definitions = {
        definition.name: (index, definition) for index, definition in enumerate(problem.parameter_definitions)
    }
    unknown = targets - definitions.keys()
    if unknown:
        raise ValueError(f"cross-dataset constraint target is missing: {dataset_id}::{min(unknown)}")
    updated_definitions = tuple(
        replace(definition, constrained=True) if definition.name in targets else definition
        for definition in problem.parameter_definitions
    )
    coordinates = {coordinate.name: coordinate for coordinate in problem.variables}
    for name in targets:
        index, definition = definitions[name]
        coordinates.setdefault(
            name,
            ParameterCoordinate(index, name, definition.transform),
        )
    ordered = tuple(sorted(coordinates.values(), key=lambda coordinate: coordinate.parameter_index))
    return replace(
        problem,
        parameter_definitions=updated_definitions,
        variables=ordered,
    )


def with_cross_target_coordinates(
    dataset_ids: tuple[str, ...],
    problems: tuple[object, ...],
    rules: tuple[ConstraintRule, ...],
) -> tuple[object, ...]:
    targets_by_dataset = _targets_by_dataset(rules)
    return tuple(
        _problem_with_target_coordinates(
            dataset_id,
            problem,
            targets_by_dataset.get(dataset_id, set()),
        )
        for dataset_id, problem in zip(dataset_ids, problems, strict=True)
    )


__all__ = [
    "constraint_node_payload",
    "joint_constraint_closure",
    "merged_constraints",
    "validate_joint_constraints",
    "with_cross_target_coordinates",
]
