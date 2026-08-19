"""Resolve cross-dataset expression constraints during joint projection."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    roughness_dynamic_uppers,
    values_by_name,
)
from xrr_fitter.model.constraint_expression import (
    ConstraintArithmeticError,
    evaluate_constraint_value,
)
from xrr_fitter.model.parameters import (
    ConstraintRule,
    ParameterReference,
    PhysicalValueError,
    _iter_references,
    physical_to_unit,
)


def _definitions_by_reference(problem: object) -> dict[ParameterReference, object]:
    return {
        ParameterReference(dataset_id, definition.name): definition
        for dataset_id, local_problem in zip(
            problem.dataset_ids,
            problem.problems,
            strict=True,
        )
        for definition in local_problem.parameter_definitions
    }


def _ordered_rules(
    problem: object,
    definitions: dict[ParameterReference, object],
    *,
    roughness: bool,
) -> tuple[ConstraintRule, ...]:
    selected = [
        rule
        for rule in problem.joint_constraint_rules
        if (definitions[rule.target].transform == "roughness_fraction") is roughness
    ]
    targets = {rule.target for rule in selected}
    ordered: list[ConstraintRule] = []
    placed: set[ParameterReference] = set()
    remaining = list(selected)
    while remaining:
        ready = [rule for rule in remaining if (set(_iter_references(rule.expression)) & targets) <= placed]
        if not ready:
            raise ValueError("constraint dependency cycle within joint phase")
        for rule in ready:
            ordered.append(rule)
            placed.add(rule.target)
            remaining.remove(rule)
    return tuple(ordered)


def _physical_values(
    problem: object,
    local_vectors: list[np.ndarray],
) -> dict[ParameterReference, float]:
    values: dict[ParameterReference, float] = {}
    for dataset_id, local_problem, vector in zip(
        problem.dataset_ids,
        problem.problems,
        local_vectors,
        strict=True,
    ):
        try:
            local_values = values_by_name(local_problem, vector)
        except PhysicalValueError as error:
            raise EvaluationConstraintError("constraint_violation:PhysicalValueError") from error
        values.update({ParameterReference(dataset_id, name): value for name, value in local_values.items()})
    return values


def _constraint_reason(kind: str, target: ParameterReference) -> str:
    return f"constraint_{kind}:{target.dataset_id}::{target.parameter_name}"


def _evaluated_value(
    rule: ConstraintRule,
    values: dict[ParameterReference, float],
) -> float:
    try:
        value = evaluate_constraint_value(rule.expression, values)
    except ConstraintArithmeticError as error:
        raise EvaluationConstraintError(_constraint_reason("nonfinite", rule.target)) from error
    if not np.isfinite(value):
        raise EvaluationConstraintError(_constraint_reason("nonfinite", rule.target))
    return value


def _local_coordinate_index(local_problem: object, parameter_name: str) -> int:
    return next(index for index, coordinate in enumerate(local_problem.variables) if coordinate.name == parameter_name)


def _active_upper(
    definition: object,
    local_problem: object,
    vector: np.ndarray,
    parameter_name: str,
    *,
    roughness: bool,
) -> tuple[float, float | None]:
    if not roughness:
        return definition.upper, None
    dynamic = roughness_dynamic_uppers(local_problem, vector)[parameter_name]
    return min(definition.upper, dynamic), dynamic


def _write_rule_value(
    problem: object,
    local_vectors: list[np.ndarray],
    definitions: dict[ParameterReference, object],
    dataset_indices: dict[str, int],
    rule: ConstraintRule,
    value: float,
    *,
    roughness: bool,
) -> None:
    target = rule.target
    definition = definitions[target]
    dataset_index = dataset_indices[target.dataset_id]
    local_problem = problem.problems[dataset_index]
    vector = local_vectors[dataset_index]
    upper, dynamic_upper = _active_upper(
        definition,
        local_problem,
        vector,
        target.parameter_name,
        roughness=roughness,
    )
    if not definition.lower <= value <= upper:
        raise EvaluationConstraintError(_constraint_reason("out_of_bounds", target))
    local_index = _local_coordinate_index(local_problem, target.parameter_name)
    try:
        vector[local_index] = physical_to_unit(
            definition,
            value,
            dynamic_upper=dynamic_upper,
        )
    except PhysicalValueError as error:
        raise EvaluationConstraintError("constraint_violation:PhysicalValueError") from error


def apply_joint_constraints(
    problem: object,
    local_vectors: list[np.ndarray],
    *,
    roughness: bool,
) -> None:
    """Resolve one joint constraint phase into mutable local unit vectors."""
    if not problem.joint_constraint_rules:
        return
    definitions = _definitions_by_reference(problem)
    rules = _ordered_rules(problem, definitions, roughness=roughness)
    if not rules:
        return
    dataset_indices = {dataset_id: index for index, dataset_id in enumerate(problem.dataset_ids)}
    values = _physical_values(problem, local_vectors)
    for rule in rules:
        value = _evaluated_value(rule, values)
        _write_rule_value(
            problem,
            local_vectors,
            definitions,
            dataset_indices,
            rule,
            value,
            roughness=roughness,
        )
        values = _physical_values(problem, local_vectors)


__all__ = ["apply_joint_constraints"]
