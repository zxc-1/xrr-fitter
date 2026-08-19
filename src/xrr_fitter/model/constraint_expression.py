"""Pure scalar evaluation and differentiation for constraint expression trees."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, isfinite, sin

from xrr_fitter.model.parameters import CONSTRAINT_UNARY_OPS, ConstraintNode, ParameterReference


class ConstraintArithmeticError(Exception):
    """An expression left the finite real-number domain."""


def _finite(value: float) -> float:
    """Return one finite real scalar or normalize the arithmetic failure."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConstraintArithmeticError() from error
    if not isfinite(result):
        raise ConstraintArithmeticError()
    return result


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
        return _finite(values[reference])
    return _finite(values[reference.parameter_name])


def _value_for_operator(op: str, left: float, right: float) -> float:
    try:
        if op == "add":
            result = left + right
        elif op == "sub":
            result = left - right
        elif op == "mul":
            result = left * right
        elif op == "div":
            if right == 0.0:
                raise ConstraintArithmeticError()
            result = left / right
        else:
            result = _safe_pow(left, right)
    except (ArithmeticError, OverflowError) as error:
        if isinstance(error, ConstraintArithmeticError):
            raise
        raise ConstraintArithmeticError() from error
    return _finite(result)


def _unary_value(op: str, inner: float) -> float:
    try:
        result = sin(inner) if op == "sin" else cos(inner)
    except (ArithmeticError, ValueError, OverflowError) as error:
        raise ConstraintArithmeticError() from error
    return _finite(result)


def evaluate_constraint_value(
    node: ConstraintNode,
    values: Mapping[object, float],
) -> float:
    """Evaluate one expression node against resolved physical values."""
    if node.op == "const":
        return _finite(node.value)
    if node.op == "ref":
        return _lookup(values, node.reference)
    if node.op in CONSTRAINT_UNARY_OPS:
        inner = evaluate_constraint_value(node.operands[0], values)
        return _unary_value(node.op, inner)
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
        combined[reference] = _finite(combined.get(reference, 0.0) + scale_left * partial)
    for reference, partial in grad_right.items():
        combined[reference] = _finite(combined.get(reference, 0.0) + scale_right * partial)
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
        raise ConstraintArithmeticError()
    try:
        derivative = right * _safe_pow(left, right - 1.0)
    except (ArithmeticError, ValueError, OverflowError) as error:
        raise ConstraintArithmeticError() from error
    return {reference: _finite(partial * derivative) for reference, partial in grad_left.items()}


def constraint_value_and_grad(
    node: ConstraintNode,
    values: Mapping[object, float],
) -> tuple[float, dict[ParameterReference, float]]:
    """Evaluate a node and return partials keyed by referenced parameters."""
    if node.op == "const":
        return _finite(node.value), {}
    if node.op == "ref":
        reference = node.reference
        return _lookup(values, reference), {reference: 1.0}
    if node.op in CONSTRAINT_UNARY_OPS:
        inner_value, inner_grad = constraint_value_and_grad(node.operands[0], values)
        if node.op == "sin":
            value, multiplier = _unary_value("sin", inner_value), cos(inner_value)
        else:
            value, multiplier = _unary_value("cos", inner_value), -sin(inner_value)
        gradient = {reference: _finite(multiplier * partial) for reference, partial in inner_grad.items()}
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
