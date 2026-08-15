import pytest

from xrr_fitter.model.parameters import (
    CONSTRAINT_UNARY_OPS,
    ConstraintNode,
    ParameterReference,
)


def _leaf(name="a"):
    return ConstraintNode(op="ref", operands=(), reference=ParameterReference("d", name))


def test_unary_ops_are_sin_cos():
    assert CONSTRAINT_UNARY_OPS == frozenset({"sin", "cos"})


def test_sin_accepts_single_operand():
    node = ConstraintNode(op="sin", operands=(_leaf(),))
    assert node.op == "sin" and len(node.operands) == 1


def test_cos_accepts_single_operand():
    node = ConstraintNode(op="cos", operands=(_leaf(),))
    assert node.op == "cos" and len(node.operands) == 1


def test_sin_rejects_two_operands():
    with pytest.raises(ValueError):
        ConstraintNode(op="sin", operands=(_leaf("a"), _leaf("b")))


def test_sin_rejects_reference_or_value():
    with pytest.raises(ValueError):
        ConstraintNode(op="cos", operands=(_leaf(),), value=1.0)
