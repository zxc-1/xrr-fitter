from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.support.model_cases import dataset_project, project, simple_structure
from tests.unit.services.test_fitting import (
    _automatic_dataset,
    _automatic_problem,
    _project,
    _source,
)

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.automation import AutomaticStatus, MeasurementPreset
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
    ParameterSetting,
)
from xrr_fitter.services import fitting
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.fitting_phases.automatic_absorption import (
    _automatic_absorption_problem,
    _fixed_absorption_problem,
)
from xrr_fitter.services.fitting_phases.operations import (
    _mcmc_problem as mcmc_problem_phase,
)
from xrr_fitter.services.fitting_phases.operations import (
    automatic_worker_handler as automatic_worker_handler_phase,
)
from xrr_fitter.services.fitting_phases.operations import (
    preflight_automatic_fit as preflight_automatic_fit_phase,
)
from xrr_fitter.services.structures import set_structure


def _density_constraint(dataset_id: str = "curve") -> ConstraintRule:
    return ConstraintRule(
        ParameterReference(dataset_id, "component.0.density_scale"),
        ConstraintNode(
            "mul",
            operands=(
                ConstraintNode(
                    "ref",
                    reference=ParameterReference(
                        dataset_id,
                        "component.0.thickness_a",
                    ),
                ),
                ConstraintNode("const", value=0.01),
            ),
        ),
    )


def test_prepare_dataset_fit_compiles_persisted_local_constraint_rules(
    tmp_path: Path,
) -> None:
    value = replace(_project(tmp_path), constraint_rules=(_density_constraint(),))

    prepared = fitting.prepare_dataset_fit(value, "curve", 13)

    assert prepared.problem.constraint_rules == value.constraint_rules
    assert "component.0.density_scale" not in {coordinate.name for coordinate in prepared.problem.variables}


def test_prepare_dataset_fit_rejects_cross_dataset_constraints_in_independent_mode(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    value = add_dataset(
        value,
        _source(tmp_path / "second.xy"),
        InstrumentSpec(instrument_id="fitting-service-2", footprint_mode="none"),
    )
    value = set_structure(value, "second", simple_structure())
    rule = ConstraintRule(
        ParameterReference("curve", "component.0.density_scale"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("second", "component.0.density_scale"),
        ),
    )
    value = replace(value, constraint_rules=(rule,))

    with pytest.raises(ValueError, match="cross-dataset.*joint"):
        fitting.prepare_dataset_fit(value, "curve", 13)


def test_automatic_absorption_problem_preserves_expression_constraints() -> None:
    base = _automatic_problem()
    rule = ConstraintRule(
        ParameterReference("curve", "instrument.scale"),
        ConstraintNode(
            "add",
            operands=(
                ConstraintNode("const", value=0.8),
                ConstraintNode(
                    "mul",
                    operands=(
                        ConstraintNode(
                            "ref",
                            reference=ParameterReference(
                                "curve",
                                "component.0.sld_imag_a2",
                            ),
                        ),
                        ConstraintNode("const", value=1e5),
                    ),
                ),
            ),
        ),
    )
    problem = compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        constraint_rules=(rule,),
    )
    values = {definition.name: definition.initial for definition in problem.parameter_definitions}

    trial = _automatic_absorption_problem(
        problem,
        ("component.0.sld_imag_a2",),
        values,
        compile_fit_problem=compile_fit_problem,
    )

    assert trial.constraint_rules == (rule,)
    assert rule.target.parameter_name not in {coordinate.name for coordinate in trial.variables}
    target = next(
        definition for definition in trial.parameter_definitions if definition.name == rule.target.parameter_name
    )
    assert target.lower < target.upper
    absorption = next(
        definition for definition in trial.parameter_definitions if definition.name == "component.0.sld_imag_a2"
    )
    unit = encode_physical_vector(trial, {absorption.name: 2.5e-6})
    assert evaluate_vector(trial, unit).valid is True


def test_absorption_recompilation_preserves_locked_constraint_target() -> None:
    base = _automatic_problem()
    target_name = "instrument.scale"
    target = next(definition for definition in base.parameter_definitions if definition.name == target_name)
    rule = ConstraintRule(
        ParameterReference("curve", target_name),
        ConstraintNode("const", value=target.initial),
    )
    problem = compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        (
            ParameterSetting(
                target.name,
                target.initial,
                target.lower,
                target.upper,
                locked=True,
            ),
        ),
        (rule,),
    )
    values = {definition.name: definition.initial for definition in problem.parameter_definitions}

    trial = _automatic_absorption_problem(
        problem,
        ("component.0.sld_imag_a2",),
        values,
        compile_fit_problem=compile_fit_problem,
    )
    accepted = _fixed_absorption_problem(
        problem,
        ("component.0.sld_imag_a2",),
        values,
        compile_fit_problem=compile_fit_problem,
    )

    for compiled in (trial, accepted):
        constrained = next(
            definition for definition in compiled.parameter_definitions if definition.name == target_name
        )
        assert constrained.constrained is True
        assert constrained.locked is True


def _cross_dataset_constraint() -> ConstraintRule:
    return ConstraintRule(
        ParameterReference("left", "component.0.density_scale"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("right", "component.0.density_scale"),
        ),
    )


def test_automatic_preflight_rejects_cross_dataset_constraints_before_routing() -> None:
    current = replace(
        project(
            _automatic_dataset("left", "batch-1", AutomaticStatus.PENDING),
            _automatic_dataset("right", "batch-1", AutomaticStatus.PENDING),
        ),
        batch_mode="joint",
        constraint_rules=(_cross_dataset_constraint(),),
        measurement_preset=MeasurementPreset(
            "lab",
            BeamSpec("monochromatic"),
            InstrumentSpec(instrument_id="lab"),
        ),
    )

    readiness = preflight_automatic_fit_phase(
        current,
        prepare_dataset_fit=lambda *_args, **_kwargs: pytest.fail(
            "automatic routing compiled a cross-dataset constrained project"
        ),
    )

    assert readiness.ready is False
    assert "cross-dataset constraints" in readiness.message


def test_mcmc_rejects_cross_dataset_constraints_before_single_dataset_compile() -> None:
    current = replace(
        project(dataset_project("left"), dataset_project("right")),
        batch_mode="joint",
        constraint_rules=(_cross_dataset_constraint(),),
    )

    with pytest.raises(ValueError, match="MCMC.*cross-dataset constraints"):
        mcmc_problem_phase(
            current,
            "left",
            compile_dataset=lambda *_args, **_kwargs: pytest.fail(
                "MCMC compiled a single dataset without its cross-dataset constraint"
            ),
        )


def test_mcmc_compiles_local_expression_constraints_before_analysis() -> None:
    rule = ConstraintRule(
        ParameterReference("left", "component.0.density_scale"),
        ConstraintNode("const", value=0.8),
    )
    current = replace(
        project(dataset_project("left")),
        constraint_rules=(rule,),
    )
    definitions = (object(),)
    result = SimpleNamespace(
        uncertainty=object(),
        parameter_definitions=definitions,
    )
    prepared = SimpleNamespace(
        problem=SimpleNamespace(parameter_definitions=definitions),
        updated_dataset=SimpleNamespace(last_valid_result=result),
    )

    compiled = mcmc_problem_phase(
        current,
        "left",
        compile_dataset=lambda *_args, **_kwargs: prepared,
    )

    assert compiled == (prepared, result)


def test_automatic_worker_rejects_cross_dataset_constraints_before_transaction() -> None:
    current = replace(
        project(
            _automatic_dataset("left", "batch-1", AutomaticStatus.PENDING),
            _automatic_dataset("right", "batch-1", AutomaticStatus.PENDING),
        ),
        batch_mode="joint",
        constraint_rules=(_cross_dataset_constraint(),),
    )

    with pytest.raises(ValueError, match="automatic fit.*cross-dataset constraints"):
        automatic_worker_handler_phase(
            current,
            "batch-1",
            None,
            None,
            lambda: False,
            fit_automatic_transaction=lambda *_args, **_kwargs: pytest.fail(
                "automatic worker started an unsupported constrained transaction"
            ),
            prepare_dataset_fit=lambda *_args, **_kwargs: None,
            fit_automatic_prepared_dataset=lambda *_args, **_kwargs: None,
            fit_automatic_joint_group=lambda *_args, **_kwargs: (),
        )
