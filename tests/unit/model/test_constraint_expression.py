from __future__ import annotations

import math

import pytest

from xrr_fitter.model.constraint_expression import (
    constraint_value_and_grad,
    evaluate_constraint_value,
)
from xrr_fitter.model.parameters import ConstraintNode, ParameterReference


def _ref_node(reference: ParameterReference) -> ConstraintNode:
    return ConstraintNode("ref", reference=reference)


def test_sin_constraint_value_uses_inner_value() -> None:
    reference = ParameterReference("curve", "angle_rad")
    node = ConstraintNode("sin", operands=(_ref_node(reference),))

    assert evaluate_constraint_value(node, {reference: 0.3}) == pytest.approx(math.sin(0.3))


def test_cos_constraint_value_uses_inner_value() -> None:
    reference = ParameterReference("curve", "angle_rad")
    node = ConstraintNode("cos", operands=(_ref_node(reference),))

    assert evaluate_constraint_value(node, {reference: 0.3}) == pytest.approx(math.cos(0.3))


def test_sin_gradient_applies_chain_rule() -> None:
    reference = ParameterReference("curve", "angle_rad")
    inner = ConstraintNode(
        "mul",
        operands=(
            ConstraintNode("const", value=2.0),
            _ref_node(reference),
        ),
    )
    node = ConstraintNode("sin", operands=(inner,))

    value, gradient = constraint_value_and_grad(node, {reference: 0.4})

    assert value == pytest.approx(math.sin(0.8))
    assert gradient == {reference: pytest.approx(2.0 * math.cos(0.8))}


def test_cos_gradient_uses_negative_sine() -> None:
    reference = ParameterReference("curve", "angle_rad")
    node = ConstraintNode("cos", operands=(_ref_node(reference),))

    value, gradient = constraint_value_and_grad(node, {reference: 0.7})

    assert value == pytest.approx(math.cos(0.7))
    assert gradient == {reference: pytest.approx(-math.sin(0.7))}
