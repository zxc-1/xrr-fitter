"""Differentiate joint-to-local unit-vector projection maps."""

from __future__ import annotations

import numpy as np

from xrr_fitter.evaluation import EvaluationConstraintError, values_and_jacobians
from xrr_fitter.fit.joint_constraints import (
    _definitions_by_reference,
    _ordered_rules,
)
from xrr_fitter.fit.joint_sharing import scatter_joint_vector
from xrr_fitter.model.constraint_expression import (
    ConstraintArithmeticError,
    constraint_value_and_grad,
)
from xrr_fitter.model.parameters import ParameterReference


def _project_or_none(
    problem: object,
    unit: np.ndarray,
) -> tuple[np.ndarray, ...] | None:
    try:
        return scatter_joint_vector(problem, unit)
    except EvaluationConstraintError:
        return None


def _difference_window(
    baseline: tuple[np.ndarray, ...],
    plus_local: tuple[np.ndarray, ...] | None,
    minus_local: tuple[np.ndarray, ...] | None,
    *,
    center: float,
    lower: float,
    upper: float,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float] | None:
    if plus_local is None and minus_local is None:
        return None
    if plus_local is None:
        return baseline, minus_local, center - lower
    if minus_local is None:
        return plus_local, baseline, upper - center
    return plus_local, minus_local, upper - lower


def _perturbed_unit(unit: np.ndarray, index: int, value: float) -> np.ndarray:
    result = unit.copy()
    result[index] = value
    return result


def _fill_scatter_column(
    problem: object,
    unit: np.ndarray,
    baseline: tuple[np.ndarray, ...],
    jacobians: list[np.ndarray],
    global_index: int,
) -> None:
    center = float(unit[global_index])
    lower = max(0.0, center - 1e-6)
    upper = min(1.0, center + 1e-6)
    plus_local = _project_or_none(
        problem,
        _perturbed_unit(unit, global_index, upper),
    )
    minus_local = _project_or_none(
        problem,
        _perturbed_unit(unit, global_index, lower),
    )
    window = _difference_window(
        baseline,
        plus_local,
        minus_local,
        center=center,
        lower=lower,
        upper=upper,
    )
    if window is None:
        return
    high_local, low_local, scale = window
    if scale <= 0.0:
        return
    for dataset_index, (high, low) in enumerate(zip(high_local, low_local, strict=True)):
        jacobians[dataset_index][:, global_index] = (high - low) / scale


def _raw_scatter_jacobians(
    problem: object,
    width: int,
) -> list[np.ndarray]:
    jacobians = [np.zeros((len(local_problem.variables), width), dtype=float) for local_problem in problem.problems]
    for dataset_index, scatter in enumerate(problem.scatter_maps):
        for local_index, global_index in enumerate(scatter):
            if global_index >= 0:
                jacobians[dataset_index][local_index, global_index] = 1.0
    return jacobians


def _physical_states(
    problem: object,
    local_units: tuple[np.ndarray, ...],
    unit_jacobians: list[np.ndarray],
) -> tuple[
    dict[ParameterReference, float],
    dict[ParameterReference, np.ndarray],
    tuple[dict[str, np.ndarray], ...],
]:
    values: dict[ParameterReference, float] = {}
    derivatives: dict[ParameterReference, np.ndarray] = {}
    local_derivatives: list[dict[str, np.ndarray]] = []
    for dataset_index, (dataset_id, local_problem, unit) in enumerate(
        zip(problem.dataset_ids, problem.problems, local_units, strict=True)
    ):
        local_values, local_jacobians = values_and_jacobians(local_problem, unit)
        local_derivatives.append(local_jacobians)
        scatter_jacobian = unit_jacobians[dataset_index]
        for name, value in local_values.items():
            reference = ParameterReference(dataset_id, name)
            values[reference] = value
            derivatives[reference] = local_jacobians[name] @ scatter_jacobian
    return values, derivatives, tuple(local_derivatives)


def _local_coordinate_index(local_problem: object, parameter_name: str) -> int:
    return next(index for index, coordinate in enumerate(local_problem.variables) if coordinate.name == parameter_name)


def _analytic_constraint_jacobians(
    problem: object,
    local_units: tuple[np.ndarray, ...],
    unit_jacobians: list[np.ndarray],
) -> tuple[np.ndarray, ...]:
    definitions = _definitions_by_reference(problem)
    for roughness in (False, True):
        rules = _ordered_rules(problem, definitions, roughness=roughness)
        if not rules:
            continue
        for rule in rules:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    values, derivatives, local_derivatives = _physical_states(
                        problem,
                        local_units,
                        unit_jacobians,
                    )
                    try:
                        _value, gradient = constraint_value_and_grad(rule.expression, values)
                    except ConstraintArithmeticError as error:
                        raise EvaluationConstraintError(
                            f"constraint_nonfinite:{rule.target.dataset_id}::{rule.target.parameter_name}"
                        ) from error
                    target_derivative = sum(
                        (partial * derivatives[reference] for reference, partial in gradient.items()),
                        start=np.zeros(unit_jacobians[0].shape[1], dtype=float),
                    )
                    dataset_index = problem.dataset_ids.index(rule.target.dataset_id)
                    local_problem = problem.problems[dataset_index]
                    target_index = _local_coordinate_index(
                        local_problem,
                        rule.target.parameter_name,
                    )
                    target_local_jacobian = local_derivatives[dataset_index][rule.target.parameter_name]
                    own_derivative = float(target_local_jacobian[target_index])
                    if own_derivative == 0.0:
                        unit_jacobians[dataset_index][target_index] = 0.0
                        continue
                    current_derivative = target_local_jacobian @ unit_jacobians[dataset_index]
                    other_derivative = current_derivative - own_derivative * unit_jacobians[dataset_index][target_index]
                    unit_jacobians[dataset_index][target_index] = (
                        target_derivative - other_derivative
                    ) / own_derivative
            except FloatingPointError as error:
                target = f"{rule.target.dataset_id}::{rule.target.parameter_name}"
                raise FloatingPointError(f"nonfinite joint constraint Jacobian: {target}") from error
            if np.any(~np.isfinite(unit_jacobians[dataset_index][target_index])):
                target = f"{rule.target.dataset_id}::{rule.target.parameter_name}"
                raise FloatingPointError(f"nonfinite joint constraint Jacobian: {target}")
    return tuple(unit_jacobians)


def joint_scatter_jacobians(
    problem: object,
    global_unit: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return local-unit derivatives for nonlinear joint projection paths."""
    unit = np.asarray(global_unit, dtype=float)
    baseline = scatter_joint_vector(problem, unit)
    width = unit.size
    jacobians = _raw_scatter_jacobians(problem, width)
    if not any(variable.transform == "shared_roughness_physical" for variable in problem.global_variables):
        return _analytic_constraint_jacobians(problem, baseline, jacobians)
    for global_index in range(width):
        _fill_scatter_column(
            problem,
            unit,
            baseline,
            jacobians,
            global_index,
        )
    return tuple(jacobians)


__all__ = ["joint_scatter_jacobians"]
