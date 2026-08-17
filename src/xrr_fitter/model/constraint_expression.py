"""Pure scalar evaluation and differentiation for constraint expression trees."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, isfinite, sin

from xrr_fitter.model.parameters import CONSTRAINT_UNARY_OPS, ConstraintNode, ParameterReference


class ConstraintArithmeticError(Exception):
    """An expression left the finite real-number domain."""


def _safe_pow(base: float, exponent: float) -> float:
    try:
        result = base**exponent
    except ArithmeticError as error:
        raise ConstraintArithmeticError() from error
    if isinstance(result, complex) or not isfinite(result):
        raise ConstraintArithmeticError()
    return result


def _lookup(values: Mapping[object, float], reference: ParameterReference) -> float:
    if reference in values:
        return float(values[reference])
    return float(values[reference.parameter_name])


def _value_for_operator(op: str, left: float, right: float) -> float:
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        if right == 0.0:
            raise ConstraintArithmeticError()
        return left / right
    return _safe_pow(left, right)


def evaluate_constraint_value(
    node: ConstraintNode,
    values: Mapping[object, float],
) -> float:
    """Evaluate one expression node against resolved physical values."""
    if node.op == "const":
        return float(node.value)
    if node.op == "ref":
        return _lookup(values, node.reference)
    if node.op in CONSTRAINT_UNARY_OPS:
        inner = evaluate_constraint_value(node.operands[0], values)
        return sin(inner) if node.op == "sin" else cos(inner)
    left, right = node.operands
    return _value_for_operator(
        node.op,
        evaluate_constraint_value(left, values),
        evaluate_constraint_value(right, values),
    )


def _combine_grads(
    grad_left: dict[ParameterReference, float],
    scale_left: float,
    grad_right: dict[ParameterReference, float],
    scale_right: float,
) -> dict[ParameterReference, float]:
    combined: dict[ParameterReference, float] = {}
    for reference, partial in grad_left.items():
        combined[reference] = combined.get(reference, 0.0) + scale_left * partial
    for reference, partial in grad_right.items():
        combined[reference] = combined.get(reference, 0.0) + scale_right * partial
    return combined


def _gradient_for_operator(
    op: str,
    left: float,
    right: float,
    grad_left: dict[ParameterReference, float],
    grad_right: dict[ParameterReference, float],
) -> dict[ParameterReference, float]:
    if op == "add":
        return _combine_grads(grad_left, 1.0, grad_right, 1.0)
    if op == "sub":
        return _combine_grads(grad_left, 1.0, grad_right, -1.0)
    if op == "mul":
        return _combine_grads(grad_left, right, grad_right, left)
    if op == "div":
        if right == 0.0:
            raise ConstraintArithmeticError()
        inverse = 1.0 / right
        return _combine_grads(
            grad_left,
            inverse,
            grad_right,
            -left * inverse * inverse,
        )
    return _power_gradient(left, right, grad_left)


def _power_gradient(
    left: float,
    right: float,
    grad_left: dict[ParameterReference, float],
) -> dict[ParameterReference, float]:
    if right == 0.0:
        return {reference: 0.0 for reference in grad_left}
    if left == 0.0 and 0.0 < right < 1.0:
        # The primal value is finite at this real-domain boundary, but the
        # analytic derivative is infinite. Keep the legal candidate and publish
        # a finite no-step tangent rather than reclassifying it as invalid.
        return {reference: 0.0 for reference in grad_left}
    derivative = right * _safe_pow(left, right - 1.0)
    return {reference: partial * derivative for reference, partial in grad_left.items()}


def constraint_value_and_grad(
    node: ConstraintNode,
    values: Mapping[object, float],
) -> tuple[float, dict[ParameterReference, float]]:
    """Evaluate a node and return partials keyed by referenced parameters."""
    if node.op == "const":
        return float(node.value), {}
    if node.op == "ref":
        reference = node.reference
        return _lookup(values, reference), {reference: 1.0}
    if node.op in CONSTRAINT_UNARY_OPS:
        inner_value, inner_grad = constraint_value_and_grad(node.operands[0], values)
        if node.op == "sin":
            value, multiplier = sin(inner_value), cos(inner_value)
        else:
            value, multiplier = cos(inner_value), -sin(inner_value)
        gradient = {reference: multiplier * partial for reference, partial in inner_grad.items()}
        return value, gradient
    left, right = node.operands
    left_value, left_grad = constraint_value_and_grad(left, values)
    right_value, right_grad = constraint_value_and_grad(right, values)
    value = _value_for_operator(node.op, left_value, right_value)
    gradient = _gradient_for_operator(
        node.op,
        left_value,
        right_value,
        left_grad,
        right_grad,
    )
    return value, gradient


__all__ = [
    "ConstraintArithmeticError",
    "constraint_value_and_grad",
    "evaluate_constraint_value",
]
