"""Validate persisted sharing and expression-constraint project graphs."""

from __future__ import annotations

from xrr_fitter.model.parameters import (
    ConstraintRule,
    SharingRule,
    _iter_references,
    constraint_cycle_path,
    constraint_sharing_conflicts,
)


def _validate_sharing_rules(
    rules: tuple[SharingRule, ...],
    dataset_ids: set[str],
) -> None:
    if any(not isinstance(rule, SharingRule) for rule in rules):
        raise TypeError("sharing_rules must contain SharingRule values")
    keys = tuple(rule.sharing_key for rule in rules)
    if len(keys) != len(set(keys)):
        raise ValueError("sharing_key values must be unique")
    for rule in rules:
        if any(member.dataset_id not in dataset_ids for member in rule.members):
            raise ValueError("sharing rule references a missing dataset")


def _validate_constraint_rule_types(rules: tuple[ConstraintRule, ...]) -> None:
    if any(not isinstance(rule, ConstraintRule) for rule in rules):
        raise TypeError("constraint_rules must contain ConstraintRule values")


def _validate_constraint_targets(rules: tuple[ConstraintRule, ...]) -> None:
    targets = tuple(rule.target for rule in rules)
    if len(targets) != len(set(targets)):
        raise ValueError("constraint targets must be unique")


def _validate_constraint_namespaces(
    rules: tuple[ConstraintRule, ...],
    dataset_ids: set[str],
) -> None:
    for rule in rules:
        references = (rule.target, *_iter_references(rule.expression))
        if any(reference.dataset_id not in dataset_ids for reference in references):
            raise ValueError("constraint rule references a missing dataset")


def _validate_constraint_cycle(rules: tuple[ConstraintRule, ...]) -> None:
    cycle = constraint_cycle_path(rules)
    if cycle:
        raise ValueError(f"constraint rules form a dependency cycle: {' -> '.join(cycle)}")


def _validate_constraint_sharing_conflicts(
    rules: tuple[ConstraintRule, ...],
    sharing_rules: tuple[SharingRule, ...],
) -> None:
    conflicts = constraint_sharing_conflicts(rules, sharing_rules)
    if not conflicts:
        return
    names = ", ".join(f"{reference.dataset_id}::{reference.parameter_name}" for reference in conflicts)
    raise ValueError("parameters are driven by both a sharing and a constraint rule: " + names)


def _validate_constraint_priors(
    datasets: tuple[object, ...],
    rules: tuple[ConstraintRule, ...],
) -> None:
    targets = {(rule.target.dataset_id, rule.target.parameter_name) for rule in rules}
    conflict = next(
        (
            (dataset.dataset_id, prior.name)
            for dataset in datasets
            for prior in dataset.parameter_priors
            if (dataset.dataset_id, prior.name) in targets
        ),
        None,
    )
    if conflict is not None:
        raise ValueError(f"constraint target must not also have a parameter prior: {conflict[0]}::{conflict[1]}")


def validate_project_parameter_graph(
    datasets: tuple[object, ...],
    sharing_rules: tuple[SharingRule, ...],
    constraint_rules: tuple[ConstraintRule, ...],
    dataset_ids: set[str],
) -> None:
    """Bind persisted parameter graphs to the project dataset namespace."""
    _validate_sharing_rules(sharing_rules, dataset_ids)
    _validate_constraint_rule_types(constraint_rules)
    _validate_constraint_targets(constraint_rules)
    _validate_constraint_namespaces(constraint_rules, dataset_ids)
    _validate_constraint_cycle(constraint_rules)
    _validate_constraint_sharing_conflicts(constraint_rules, sharing_rules)
    _validate_constraint_priors(datasets, constraint_rules)


__all__ = ["validate_project_parameter_graph"]
