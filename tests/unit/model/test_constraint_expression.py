import math

import pytest

from xrr_fitter.model.constraint_expression import (
    constraint_value_and_grad,
    evaluate_constraint_value,
)
from xrr_fitter.model.parameters import ConstraintNode, ParameterReference


def _ref(name="x"):
    return ConstraintNode(op="ref", operands=(), reference=ParameterReference("d", name))


def test_sin_value():
    assert evaluate_constraint_value(ConstraintNode(op="sin", operands=(_ref(),)), {"x": 0.3}) == math.sin(0.3)


def test_cos_value():
    assert evaluate_constraint_value(ConstraintNode(op="cos", operands=(_ref(),)), {"x": 0.3}) == math.cos(0.3)


def test_sin_grad_matches_finite_difference():
    ref = ParameterReference("d", "x")
    node = ConstraintNode(op="sin", operands=(ConstraintNode(op="ref", operands=(), reference=ref),))
    x0, h = 0.3, 1e-6
    value, grad = constraint_value_and_grad(node, {ref: x0})
    assert value == math.sin(x0)
    fd = (evaluate_constraint_value(node, {ref: x0 + h}) - evaluate_constraint_value(node, {ref: x0 - h})) / (2 * h)
    assert grad[ref] == pytest.approx(fd, abs=1e-6)


def test_cos_grad_sign():
    ref = ParameterReference("d", "x")
    node = ConstraintNode(op="cos", operands=(ConstraintNode(op="ref", operands=(), reference=ref),))
    value, grad = constraint_value_and_grad(node, {ref: 0.7})
    assert value == math.cos(0.7)
    assert grad[ref] == pytest.approx(-math.sin(0.7))


def test_sin_chain_rule():  # sin(2*x): d/dx = 2*cos(2*x) —— 验证内层梯度经乘子传播
    ref = ParameterReference("d", "x")
    inner = ConstraintNode(
        op="mul",
        operands=(
            ConstraintNode(op="const", operands=(), value=2.0),
            ConstraintNode(op="ref", operands=(), reference=ref),
        ),
    )
    node = ConstraintNode(op="sin", operands=(inner,))
    value, grad = constraint_value_and_grad(node, {ref: 0.4})
    assert value == pytest.approx(math.sin(0.8))
    assert grad[ref] == pytest.approx(2 * math.cos(0.8))
