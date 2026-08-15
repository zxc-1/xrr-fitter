"""Resolve local constraint phases over compiled parameter definitions."""

from __future__ import annotations

from math import isfinite

import numpy as np

from xrr_fitter.model.constraint_expression import (
    ConstraintArithmeticError,
    constraint_value_and_grad,
    evaluate_constraint_value,
)
from xrr_fitter.model.parameters import ConstraintRule, _iter_references


class ConstraintResolutionError(ValueError):
    """A derived target left its compiled physical domain."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _reference_names(rule: ConstraintRule) -> set[str]:
    return {reference.parameter_name for reference in _iter_references(rule.expression)}


def _ordered_phase_rules(problem: object, *, roughness: bool) -> tuple[ConstraintRule, ...]:
    definitions = {item.name: item for item in problem.parameter_definitions}
    selected = [
        rule
        for rule in problem.constraint_rules
        if (definitions[rule.target.parameter_name].transform == "roughness_fraction") is roughness
    ]
    phase_targets = {rule.target.parameter_name for rule in selected}
    ordered: list[ConstraintRule] = []
    placed: set[str] = set()
    remaining = list(selected)
    while remaining:
        ready = [rule for rule in remaining if (_reference_names(rule) & phase_targets) <= placed]
        if not ready:
            raise ValueError("constraint dependency cycle within evaluation phase")
        for rule in ready:
            ordered.append(rule)
            placed.add(rule.target.parameter_name)
            remaining.remove(rule)
    return tuple(ordered)


def _target_closure(
    rules: tuple[ConstraintRule, ...],
    target_names: set[str],
) -> set[str]:
    by_target = {rule.target.parameter_name: rule for rule in rules}
    required = set(target_names)
    pending = list(target_names)
    while pending:
        rule = by_target.get(pending.pop())
        if rule is None:
            continue
        for reference in _iter_references(rule.expression):
            name = reference.parameter_name
            if name in by_target and name not in required:
                required.add(name)
                pending.append(name)
    return required


def geometry_constraint_targets(problem: object) -> set[str]:
    definitions = {item.name: item for item in problem.parameter_definitions}
    return {
        rule.target.parameter_name
        for rule in problem.constraint_rules
        if definitions[rule.target.parameter_name].category == "structure"
    }


def _selected_rules(
    problem: object,
    *,
    roughness: bool,
    target_names: set[str] | None,
) -> tuple[ConstraintRule, ...]:
    rules = _ordered_phase_rules(problem, roughness=roughness)
    if target_names is None:
        return rules
    required = _target_closure(rules, target_names)
    return tuple(rule for rule in rules if rule.target.parameter_name in required)


def apply_constraint_values(
    problem: object,
    values: dict[str, float],
    *,
    roughness: bool,
    dynamic_uppers: dict[str, float] | None = None,
    target_names: set[str] | None = None,
) -> None:
    definitions = {item.name: item for item in problem.parameter_definitions}
    rules = _selected_rules(
        problem,
        roughness=roughness,
        target_names=target_names,
    )
    for rule in rules:
        target = rule.target.parameter_name
        definition = definitions[target]
        try:
            value = evaluate_constraint_value(rule.expression, values)
        except ConstraintArithmeticError as error:
            raise ConstraintResolutionError(f"constraint_nonfinite:{target}") from error
        if not isfinite(value):
            raise ConstraintResolutionError(f"constraint_nonfinite:{target}")
        upper = definition.upper
        if roughness and dynamic_uppers is not None:
            upper = min(upper, dynamic_uppers.get(target, upper))
        if not definition.lower <= value <= upper:
            raise ConstraintResolutionError(f"constraint_out_of_bounds:{target}")
        values[target] = value


def constraint_value_jacobians(
    problem: object,
    value_jacobians: dict[str, np.ndarray],
    values: dict[str, float],
    *,
    roughness: bool,
) -> None:
    parameter_count = len(problem.variables)
    for rule in _ordered_phase_rules(problem, roughness=roughness):
        target = rule.target.parameter_name
        try:
            _value, gradient = constraint_value_and_grad(rule.expression, values)
        except ConstraintArithmeticError as error:
            raise ConstraintResolutionError(f"constraint_nonfinite:{target}") from error
        jacobian = np.zeros(parameter_count, dtype=float)
        for reference, partial in gradient.items():
            jacobian = jacobian + partial * value_jacobians[reference.parameter_name]
        value_jacobians[target] = jacobian


__all__ = [
    "ConstraintResolutionError",
    "apply_constraint_values",
    "constraint_value_jacobians",
    "geometry_constraint_targets",
]
