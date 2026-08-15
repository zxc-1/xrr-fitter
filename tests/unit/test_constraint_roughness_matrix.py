from __future__ import annotations

import numpy as np
import pytest
from tests.unit.test_constraint_evaluation import (
    THICK_SOURCE,
    _binary,
    _closed_form,
    _constrained_problem,
    _rule,
    _variable_index,
)

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import encode_physical_vector

ROUGHNESS_SOURCE = "component.0.roughness_a"
ROUGHNESS_TARGET = "backing.roughness_a"
ROUGHNESS_MATRIX = (
    ("add", 0.5),
    ("sub", 0.5),
    ("mul", 1.5),
    ("div", 2.0),
    ("pow", 2.0),
)


@pytest.mark.parametrize(("op", "constant"), ROUGHNESS_MATRIX)
def test_roughness_constraint_operator_jacobian_matches_finite_difference(
    op: str,
    constant: float,
) -> None:
    source_value = 2.0
    problem = _constrained_problem((_rule(ROUGHNESS_TARGET, _binary(op, ROUGHNESS_SOURCE, constant)),))
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical[ROUGHNESS_SOURCE] = source_value
    unit = encode_physical_vector(problem, physical)

    values, jacobians = evaluation.values_and_jacobians(problem, unit)

    assert values[ROUGHNESS_TARGET] == pytest.approx(
        _closed_form(op, source_value, constant),
        rel=1e-12,
    )
    step = 1e-6
    for axis_name in (ROUGHNESS_SOURCE, THICK_SOURCE):
        index = _variable_index(problem, axis_name)
        forward = np.array(unit, copy=True)
        forward[index] += step
        backward = np.array(unit, copy=True)
        backward[index] -= step
        finite_difference = (
            evaluation.values_by_name(problem, forward)[ROUGHNESS_TARGET]
            - evaluation.values_by_name(problem, backward)[ROUGHNESS_TARGET]
        ) / (2.0 * step)

        assert jacobians[ROUGHNESS_TARGET][index] == pytest.approx(
            finite_difference,
            rel=1e-5,
            abs=1e-8,
        )
