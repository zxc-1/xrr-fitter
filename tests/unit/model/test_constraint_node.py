from __future__ import annotations

import pytest

from xrr_fitter.model import parameters
from xrr_fitter.model.parameters import ConstraintNode, ParameterReference


def _ref(name: str = "x") -> ConstraintNode:
    return ConstraintNode("ref", reference=ParameterReference("curve", name))


def test_unary_constraint_ops_include_sin_and_cos() -> None:
    assert getattr(parameters, "CONSTRAINT_UNARY_OPS", None) == frozenset({"sin", "cos"})


def test_sin_accepts_one_constraint_operand() -> None:
    node = ConstraintNode("sin", operands=(_ref(),))

    assert node.op == "sin"
    assert node.operands == (_ref(),)


@pytest.mark.parametrize("operands", [(), (_ref("x"), _ref("y"))])
def test_sin_rejects_wrong_operand_count(operands: tuple[ConstraintNode, ...]) -> None:
    with pytest.raises(ValueError, match="unary op .* needs exactly one operand"):
        ConstraintNode("sin", operands=operands)


def test_cos_rejects_reference_field() -> None:
    with pytest.raises(ValueError, match="unary op .* takes no reference/value"):
        ConstraintNode("cos", reference=ParameterReference("curve", "x"), operands=(_ref(),))


def test_sin_rejects_value_field() -> None:
    with pytest.raises(ValueError, match="unary op .* takes no reference/value"):
        ConstraintNode("sin", value=1.0, operands=(_ref(),))
