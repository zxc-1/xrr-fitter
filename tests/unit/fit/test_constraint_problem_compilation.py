from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector, values_by_name
from xrr_fitter.fit.problem import (
    compile_fit_problem,
    compile_fixed_parameter_problem,
    compile_stage_problem,
)
from xrr_fitter.fit.stages import compile_coarse_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
    ParameterSetting,
)


def _config(seed: int = 11) -> FitConfig:
    return replace(FitConfig.fast(master_seed=seed), scale_prior_enabled=False)


def _problem(*, instrument=None, settings: tuple[ParameterSetting, ...] = ()):
    return compile_fit_problem(
        prepared_data(size=72),
        simple_structure(),
        instrument or InstrumentSpec(footprint_mode="fit"),
        _config(),
        settings,
    )


def _scaled_thickness_rule(target: str, factor: float) -> ConstraintRule:
    return ConstraintRule(
        target=ParameterReference("curve", target),
        expression=ConstraintNode(
            "mul",
            operands=(
                ConstraintNode(
                    "ref",
                    reference=ParameterReference(
                        "curve",
                        "component.0.thickness_a",
                    ),
                ),
                ConstraintNode("const", value=factor),
            ),
        ),
    )


def _variable_names(problem) -> set[str]:
    return {coordinate.name for coordinate in problem.variables}


def _definition(problem, name: str):
    return next(item for item in problem.parameter_definitions if item.name == name)


def test_coarse_recompile_preserves_expression_constraints() -> None:
    rule = ConstraintRule(
        ParameterReference("curve", "component.0.density_scale"),
        ConstraintNode("const", value=1.0),
    )
    problem = compile_fit_problem(
        prepared_data(size=240),
        simple_structure(),
        InstrumentSpec(footprint_mode="fit"),
        _config(),
        constraint_rules=(rule,),
    )

    coarse = compile_coarse_problem(problem)

    assert coarse is not problem
    assert coarse.constraint_rules == (rule,)
    assert rule.target.parameter_name not in {coordinate.name for coordinate in coarse.variables}


def test_constrained_target_leaves_the_free_variable_layout() -> None:
    instrument = InstrumentSpec(footprint_mode="none")
    baseline = _problem(instrument=instrument)
    baseline_names = _variable_names(baseline)
    assert "instrument.scale" in baseline_names

    constrained = compile_fit_problem(
        baseline.data,
        baseline.structure,
        baseline.instrument,
        baseline.config,
        (),
        (_scaled_thickness_rule("instrument.scale", 2.0),),
    )
    names = _variable_names(constrained)
    definition = _definition(constrained, "instrument.scale")

    assert "instrument.scale" not in names
    assert "component.0.thickness_a" in names
    assert len(constrained.variables) == len(baseline.variables) - 1
    assert definition.constrained is True
    assert definition.locked is False


def test_stage_compilation_preserves_constraint_target_domain() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    rule = _scaled_thickness_rule("component.0.density_scale", 0.04)
    constrained = compile_fit_problem(
        baseline.data,
        baseline.structure,
        baseline.instrument,
        baseline.config,
        constraint_rules=(rule,),
    )
    current = values_by_name(constrained, encode_physical_vector(constrained, {}))

    stage_b = compile_stage_problem(constrained, "B", current)
    target = next(
        definition for definition in stage_b.parameter_definitions if definition.name == rule.target.parameter_name
    )
    unit = encode_physical_vector(stage_b, {"component.0.thickness_a": 25.0})

    assert (target.lower, target.upper) == (0.5, 1.1)
    assert values_by_name(stage_b, unit)[target.name] == pytest.approx(1.0)


def test_physical_encoding_uses_constrained_thickness_for_roughness_domain() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    rule = ConstraintRule(
        target=ParameterReference("curve", "component.0.thickness_a"),
        expression=ConstraintNode(
            "mul",
            operands=(
                ConstraintNode(
                    "ref",
                    reference=ParameterReference(
                        "curve",
                        "component.0.density_scale",
                    ),
                ),
                ConstraintNode("const", value=10.0),
            ),
        ),
    )
    constrained = compile_fit_problem(
        baseline.data,
        baseline.structure,
        baseline.instrument,
        baseline.config,
        constraint_rules=(rule,),
    )

    unit = encode_physical_vector(
        constrained,
        {
            "component.0.density_scale": 1.0,
            "component.0.roughness_a": 4.0,
        },
    )
    values = values_by_name(constrained, unit)

    assert values["component.0.thickness_a"] == pytest.approx(10.0)
    assert values["component.0.roughness_a"] == pytest.approx(4.0)


def test_fixed_parameter_compilation_rejects_constrained_target() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    rule = _scaled_thickness_rule("component.0.density_scale", 0.04)
    constrained = compile_fit_problem(
        baseline.data,
        baseline.structure,
        baseline.instrument,
        baseline.config,
        constraint_rules=(rule,),
    )

    with pytest.raises(ValueError, match="constrained"):
        compile_fixed_parameter_problem(
            constrained,
            rule.target.parameter_name,
            0.91,
        )


def test_single_dataset_compilation_rejects_cross_dataset_constraint_rules() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    rule = ConstraintRule(
        ParameterReference("left", "instrument.scale"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("right", "component.0.thickness_a"),
        ),
    )

    with pytest.raises(ValueError, match="cross-dataset|single-dataset"):
        compile_fit_problem(
            baseline.data,
            baseline.structure,
            baseline.instrument,
            baseline.config,
            constraint_rules=(rule,),
        )


def test_single_dataset_compilation_rejects_mixed_constraint_namespaces() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    rules = (
        ConstraintRule(
            ParameterReference("curve", "instrument.scale"),
            ConstraintNode("const", value=1.0),
        ),
        ConstraintRule(
            ParameterReference("other", "instrument.background"),
            ConstraintNode("const", value=1.0e-7),
        ),
    )

    with pytest.raises(ValueError, match="cross-dataset|single-dataset"):
        compile_fit_problem(
            baseline.data,
            baseline.structure,
            baseline.instrument,
            baseline.config,
            constraint_rules=rules,
        )


def test_single_dataset_compilation_rejects_unknown_constraint_reference() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    rule = ConstraintRule(
        ParameterReference("curve", "instrument.scale"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("curve", "missing.parameter"),
        ),
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        compile_fit_problem(
            baseline.data,
            baseline.structure,
            baseline.instrument,
            baseline.config,
            constraint_rules=(rule,),
        )


def test_single_dataset_compilation_rejects_duplicate_or_cyclic_targets() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    duplicate = ConstraintRule(
        ParameterReference("curve", "instrument.scale"),
        ConstraintNode("const", value=1.0),
    )
    with pytest.raises(ValueError, match="target"):
        compile_fit_problem(
            baseline.data,
            baseline.structure,
            baseline.instrument,
            baseline.config,
            constraint_rules=(duplicate, duplicate),
        )

    cyclic = (
        ConstraintRule(
            ParameterReference("curve", "instrument.scale"),
            ConstraintNode(
                "ref",
                reference=ParameterReference("curve", "component.0.density_scale"),
            ),
        ),
        ConstraintRule(
            ParameterReference("curve", "component.0.density_scale"),
            ConstraintNode(
                "ref",
                reference=ParameterReference("curve", "instrument.scale"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="cycle"):
        compile_fit_problem(
            baseline.data,
            baseline.structure,
            baseline.instrument,
            baseline.config,
            constraint_rules=cyclic,
        )


def test_locked_flag_gates_variables_independently() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    locked_only = compile_fit_problem(
        baseline.data,
        baseline.structure,
        baseline.instrument,
        baseline.config,
        (ParameterSetting("instrument.scale", 1.25, 1.25, 1.25, locked=True),),
        (),
    )
    definition = _definition(locked_only, "instrument.scale")

    assert definition.locked is True
    assert definition.constrained is False
    assert "instrument.scale" not in _variable_names(locked_only)


def test_locked_and_constrained_flags_both_remain_visible() -> None:
    baseline = _problem(instrument=InstrumentSpec(footprint_mode="none"))
    both = compile_fit_problem(
        baseline.data,
        baseline.structure,
        baseline.instrument,
        baseline.config,
        (ParameterSetting("instrument.background", 0.0, 0.0, 0.0, locked=True),),
        (_scaled_thickness_rule("instrument.background", 0.0),),
    )
    definition = _definition(both, "instrument.background")

    assert definition.locked is True
    assert definition.constrained is True
    assert "instrument.background" not in _variable_names(both)
