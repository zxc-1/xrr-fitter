"""Compilation and evaluation contracts for generated periodic drift rules."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector, values_and_jacobians, values_by_name
from xrr_fitter.fit.problem import compile_fit_problem, compile_stage_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ConstraintNode, ConstraintRule, ParameterReference
from xrr_fitter.model.structure import DriftSpec, LayerSpec, PeriodicBlock, StructureSpec


def _config(seed: int = 11) -> FitConfig:
    return replace(FitConfig.fast(master_seed=seed), scale_prior_enabled=False)


def _problem(
    *,
    data=None,
    structure=None,
    instrument=None,
    settings=(),
    seed: int = 11,
):
    return compile_fit_problem(
        data or prepared_data(size=72),
        structure or simple_structure(),
        instrument or InstrumentSpec(footprint_mode="fit"),
        _config(seed),
        settings,
    )


def _initial_values(problem) -> dict[str, float]:
    return {definition.name: definition.initial for definition in problem.parameter_definitions}


def _periodic_structure() -> StructureSpec:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="a", thickness_a=22.0),
            replace(film, name="b", thickness_a=38.0),
        ),
        repeats=5,
        top_roughness_a=1.0,
    )
    return StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0)


def _drift_periodic_structure() -> StructureSpec:
    base = _periodic_structure()
    block = base.components[0]
    assert isinstance(block, PeriodicBlock)
    drifted = replace(block, drift=DriftSpec(kind="linear", target="thickness", amount=0.1))
    return replace(base, components=(drifted,))


def _roughness_drift_periodic_structure() -> StructureSpec:
    base = _periodic_structure()
    block = base.components[0]
    assert isinstance(block, PeriodicBlock)
    drifted = replace(
        block,
        top_roughness_a=1.0,
        drift=DriftSpec(kind="linear", target="roughness", amount=0.1),
    )
    return replace(base, components=(drifted,))


def _drift_problem():
    return _problem(
        structure=_drift_periodic_structure(),
        instrument=InstrumentSpec(footprint_mode="none"),
        seed=29,
    )


def test_periodic_block_compiles_shared_variables() -> None:
    problem = _problem(structure=_periodic_structure())
    names = tuple(coordinate.name for coordinate in problem.variables)

    assert names.count("component.0.layer.0.thickness_a") == 1
    assert names.count("component.0.layer.1.thickness_a") == 1
    assert not any("repeat.1" in name for name in names)


def test_compile_marks_repeat_targets_constrained() -> None:
    problem = _drift_problem()
    definitions = {item.name: item for item in problem.parameter_definitions}
    variables = {coordinate.name for coordinate in problem.variables}
    repeat_names = [name for name in definitions if ".repeat." in name]

    assert repeat_names  # per-copy targets were emitted
    assert all(definitions[name].constrained for name in repeat_names)
    assert not any(name in variables for name in repeat_names)
    assert "component.0.drift_scale" in variables  # the scale itself stays free


def test_drifted_single_dataset_compilation_accepts_local_constraints() -> None:
    local_rule = ConstraintRule(
        ParameterReference("curve", "component.0.layer.0.density_scale"),
        ConstraintNode("ref", reference=ParameterReference("curve", "instrument.scale")),
    )

    problem = _problem(
        structure=_drift_periodic_structure(),
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(),
    )
    # Recompile through the public entry point so generated rules share the
    # local namespace with a normal same-dataset constraint.
    compiled = compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        (),
        (local_rule,),
    )

    assert any(rule.target == local_rule.target for rule in compiled.constraint_rules)
    assert all(rule.target.dataset_id == "curve" for rule in compiled.constraint_rules)


def test_user_repeat_constraint_is_rejected_instead_of_silently_dropped() -> None:
    repeat_rule = ConstraintRule(
        ParameterReference("curve", "component.0.repeat.1.layer.0.thickness_a"),
        ConstraintNode("const", value=12.0),
    )
    problem = _problem(
        structure=_drift_periodic_structure(),
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(),
    )

    with pytest.raises(ValueError, match="generated drift"):
        compile_fit_problem(
            problem.data,
            problem.structure,
            problem.instrument,
            problem.config,
            (),
            (repeat_rule,),
        )


def test_drifted_local_constraint_cycle_is_rejected_at_compile_time() -> None:
    local_rule = ConstraintRule(
        ParameterReference("curve", "component.0.layer.0.thickness_a"),
        ConstraintNode(
            "ref",
            reference=ParameterReference(
                "curve",
                "component.0.repeat.1.layer.0.thickness_a",
            ),
        ),
    )

    problem = _problem(
        structure=_drift_periodic_structure(),
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(),
    )
    with pytest.raises(ValueError, match="dependency cycle"):
        compile_fit_problem(
            problem.data,
            problem.structure,
            problem.instrument,
            problem.config,
            (),
            (local_rule,),
        )


def test_stage_recompile_does_not_accumulate_drift_rules() -> None:
    problem = _drift_problem()

    restaged = compile_stage_problem(problem, "E", _initial_values(problem))

    assert len(restaged.constraint_rules) == len(problem.constraint_rules)
    assert all(".repeat." in rule.target.parameter_name for rule in restaged.constraint_rules)


def test_drift_rule_values_compose_base_scale_coeff() -> None:
    problem = _drift_problem()

    values = values_by_name(problem, encode_physical_vector(problem, {}))

    scale = values["component.0.drift_scale"]
    for layer_index in (0, 1):
        base = values[f"component.0.layer.{layer_index}.thickness_a"]
        for k in (1, 2, 3, 4):
            expected = base * (1.0 + scale * float(k))
            assert values[f"component.0.repeat.{k}.layer.{layer_index}.thickness_a"] == pytest.approx(
                expected, rel=1e-9
            )


def test_roughness_drift_with_explicit_top_round_trips_dynamic_values_and_jacobians() -> None:
    problem = _problem(
        structure=_roughness_drift_periodic_structure(),
        instrument=InstrumentSpec(footprint_mode="none"),
        seed=31,
    )
    unit = encode_physical_vector(problem, {})

    values = values_by_name(problem, unit)
    decoded, jacobians = values_and_jacobians(problem, unit)

    assert values["component.0.top_roughness_a"] == pytest.approx(1.0)
    assert decoded["component.0.layer.0.roughness_a"] == pytest.approx(2.0)
    assert values["component.0.layer.0.roughness_a"] == pytest.approx(2.0)
    assert decoded["component.0.repeat.1.layer.0.roughness_a"] == pytest.approx(2.2)
    assert jacobians["component.0.layer.0.roughness_a"].shape == (len(problem.variables),)
