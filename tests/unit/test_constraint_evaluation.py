from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
)
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec

CURVE = "curve"
TARGET_SCALE = "instrument.scale"
TARGET_BACKGROUND = "instrument.background"
THICK_SOURCE = "component.0.thickness_a"
DENS_SOURCE = "component.0.density_scale"


def _ref(name: str) -> ParameterReference:
    return ParameterReference(CURVE, name)


def _ref_node(name: str) -> ConstraintNode:
    return ConstraintNode("ref", reference=_ref(name))


def _const(value: float) -> ConstraintNode:
    return ConstraintNode("const", value=value)


def _binary(op: str, source: str, const: float) -> ConstraintNode:
    return ConstraintNode(op, operands=(_ref_node(source), _const(const)))


def _rule(target: str, expression: ConstraintNode) -> ConstraintRule:
    return ConstraintRule(target=_ref(target), expression=expression)


def _constrained_problem(
    rules: tuple[ConstraintRule, ...],
    *,
    structure=None,
    size: int = 64,
    seed: int = 1,
) -> FitEvaluationContext:
    return compile_fit_problem(
        prepared_data(size=size),
        simple_structure() if structure is None else structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=seed), scale_prior_enabled=False),
        constraint_rules=tuple(rules),
    )


def _variable_index(problem: FitEvaluationContext, name: str) -> int:
    return [variable.name for variable in problem.variables].index(name)


def _closed_form(op: str, source_value: float, const: float) -> float:
    return {
        "add": source_value + const,
        "sub": source_value - const,
        "mul": source_value * const,
        "div": source_value / const,
        "pow": source_value**const,
    }[op]


CONSTRAINT_MATRIX = (
    ("add", THICK_SOURCE, 20.0, 5.0),
    ("add", DENS_SOURCE, 0.8, 5.0),
    ("sub", THICK_SOURCE, 20.0, 5.0),
    ("sub", DENS_SOURCE, 0.8, 0.3),
    ("mul", THICK_SOURCE, 20.0, 2.0),
    ("mul", DENS_SOURCE, 0.8, 2.0),
    ("div", THICK_SOURCE, 20.0, 4.0),
    ("div", DENS_SOURCE, 0.8, 4.0),
    ("pow", THICK_SOURCE, 20.0, 0.5),
    ("pow", DENS_SOURCE, 0.8, 0.5),
)


@pytest.mark.parametrize(("op", "source", "source_value", "const"), CONSTRAINT_MATRIX)
def test_constraint_value_and_jacobian_match_finite_difference(
    op: str,
    source: str,
    source_value: float,
    const: float,
) -> None:
    problem = _constrained_problem((_rule(TARGET_SCALE, _binary(op, source, const)),))
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical[source] = source_value
    unit = encode_physical_vector(problem, physical)

    values, jacobian = evaluation.values_and_jacobians(problem, unit)

    assert values[TARGET_SCALE] == pytest.approx(
        _closed_form(op, source_value, const),
        rel=1e-12,
    )
    assert TARGET_SCALE not in {variable.name for variable in problem.variables}

    index = _variable_index(problem, source)
    step = 1e-6
    forward = np.array(unit, copy=True)
    forward[index] += step
    backward = np.array(unit, copy=True)
    backward[index] -= step
    finite_difference = (
        evaluation.values_by_name(problem, forward)[TARGET_SCALE]
        - evaluation.values_by_name(problem, backward)[TARGET_SCALE]
    ) / (2.0 * step)

    analytic = jacobian[TARGET_SCALE]
    assert analytic[index] == pytest.approx(finite_difference, rel=1e-5, abs=1e-8)
    others = [value for position, value in enumerate(analytic) if position != index]
    assert others == pytest.approx([0.0] * len(others), abs=1e-12)


def test_zero_power_constraint_is_constant_at_a_zero_source() -> None:
    source = "instrument.angle_offset_deg"
    problem = _constrained_problem((_rule(TARGET_SCALE, _binary("pow", source, 0.0)),))
    unit = encode_physical_vector(problem, {source: 0.0})

    values, jacobian = evaluation.values_and_jacobians(problem, unit)

    assert values[TARGET_SCALE] == 1.0
    np.testing.assert_array_equal(
        jacobian[TARGET_SCALE],
        np.zeros(len(problem.variables)),
    )


def test_fractional_power_constraint_is_invalid_at_zero_for_analytic_solver() -> None:
    source = "instrument.angle_offset_deg"
    problem = _constrained_problem((_rule(TARGET_BACKGROUND, _binary("pow", source, 0.5)),))
    unit = encode_physical_vector(problem, {source: 0.0})

    residual, jacobian = evaluation.least_squares_system(problem, unit)

    np.testing.assert_array_equal(residual, np.full(residual.shape, 1e6))
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))


def test_invalid_candidate_with_scale_prior_keeps_one_zero_prior_jacobian_row() -> None:
    problem = replace(
        _constrained_problem(
            (
                _rule(
                    TARGET_BACKGROUND,
                    _binary("pow", "instrument.angle_offset_deg", 0.5),
                ),
            )
        ),
        scale_prior_center=1.0,
    )
    unit = encode_physical_vector(problem, {"instrument.angle_offset_deg": 0.0})

    residual, jacobian = evaluation.least_squares_system(problem, unit)

    expected_rows = int(np.count_nonzero(problem.data.fit_mask)) + 1
    assert residual.shape == (expected_rows,)
    assert jacobian.shape == (expected_rows, len(problem.variables))
    np.testing.assert_array_equal(residual, np.full(expected_rows, 1e6))
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))


def test_roughness_constraint_jacobian_tracks_dynamic_upper_two_phase() -> None:
    source = "component.0.roughness_a"
    target = "backing.roughness_a"
    problem = _constrained_problem((_rule(target, _binary("mul", source, 1.5)),))
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical[source] = 2.0
    unit = encode_physical_vector(problem, physical)

    values, jacobian = evaluation.values_and_jacobians(problem, unit)

    assert values[target] == pytest.approx(1.5 * values[source], rel=1e-12)
    assert target not in {variable.name for variable in problem.variables}
    np.testing.assert_allclose(
        jacobian[target],
        1.5 * jacobian[source],
        rtol=1e-10,
        atol=1e-12,
    )

    step = 1e-6
    for axis_name in (source, THICK_SOURCE):
        index = _variable_index(problem, axis_name)
        forward = np.array(unit, copy=True)
        forward[index] += step
        backward = np.array(unit, copy=True)
        backward[index] -= step
        finite_difference = (
            evaluation.values_by_name(problem, forward)[target] - evaluation.values_by_name(problem, backward)[target]
        ) / (2.0 * step)
        assert jacobian[target][index] == pytest.approx(
            finite_difference,
            rel=1e-5,
            abs=1e-8,
        )


def test_constraint_values_agree_between_scalar_and_jacobian_paths() -> None:
    problem = _constrained_problem((_rule(TARGET_SCALE, _binary("mul", THICK_SOURCE, 2.0)),))
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical[THICK_SOURCE] = 20.0
    unit = encode_physical_vector(problem, physical)

    scalar_values = evaluation.values_by_name(problem, unit)
    jacobian_values, _ = evaluation.values_and_jacobians(problem, unit)

    assert scalar_values[TARGET_SCALE] == pytest.approx(40.0, rel=1e-12)
    assert scalar_values.keys() == jacobian_values.keys()
    for name in scalar_values:
        assert scalar_values[name] == pytest.approx(
            jacobian_values[name],
            rel=1e-12,
        ), name


def test_constraint_feeds_rebuilt_geometry_not_initial_value() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="first", thickness_a=24.0),
            replace(film, name="second", thickness_a=36.0),
        ),
        repeats=4,
        top_roughness_a=1.5,
    )
    structure = StructureSpec(
        base.fronting,
        (block,),
        base.backing,
        backing_roughness_a=2.0,
    )
    rule = _rule(
        "component.0.layer.1.thickness_a",
        _binary("mul", "component.0.layer.0.thickness_a", 2.0),
    )
    problem = _constrained_problem((rule,), structure=structure, size=80)
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical["component.0.layer.0.thickness_a"] = 27.0
    unit = encode_physical_vector(problem, physical)

    evaluated = evaluate_vector(problem, unit)

    assert evaluated.valid
    assert evaluated.expanded_stack is not None
    np.testing.assert_allclose(
        evaluated.expanded_stack.thickness_a[1:-1],
        np.tile([27.0, 54.0], 4),
    )


def test_constraint_out_of_bounds_raises_evaluation_constraint_error() -> None:
    target = "instrument.background"
    problem = _constrained_problem((_rule(target, _binary("mul", THICK_SOURCE, 2.0)),))
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical[THICK_SOURCE] = 20.0
    unit = encode_physical_vector(problem, physical)

    with pytest.raises(EvaluationConstraintError) as excinfo:
        evaluation.values_by_name(problem, unit)
    assert "constraint_out_of_bounds" in str(excinfo.value)


def test_constraint_nonfinite_raises_evaluation_constraint_error() -> None:
    target = "instrument.background"
    problem = _constrained_problem((_rule(target, _binary("div", THICK_SOURCE, 0.0)),))
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical[THICK_SOURCE] = 20.0
    unit = encode_physical_vector(problem, physical)

    with pytest.raises(EvaluationConstraintError) as excinfo:
        evaluation.values_by_name(problem, unit)
    assert "constraint_nonfinite" in str(excinfo.value)


def test_constraint_chain_jacobian_rejects_overflow_without_runtime_warning() -> None:
    data = replace(
        prepared_data(size=64),
        intensity_normalized=np.full(64, np.finfo(float).max),
    )
    scaled_background = ConstraintNode(
        "mul",
        operands=(_const(1e10), _ref_node(TARGET_BACKGROUND)),
    )
    expression = ConstraintNode(
        "add",
        operands=(
            _const(2.0),
            ConstraintNode("sin", operands=(scaled_background,)),
        ),
    )
    problem = _constrained_problem((_rule(TARGET_SCALE, expression),), structure=simple_structure())
    problem = compile_fit_problem(
        data,
        problem.structure,
        problem.instrument,
        problem.config,
        constraint_rules=problem.constraint_rules,
    )
    unit = encode_physical_vector(problem, {TARGET_BACKGROUND: 0.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="constraint Jacobian"):
            evaluation.values_and_jacobians(problem, unit)

    assert not any(item.category is RuntimeWarning for item in caught)
