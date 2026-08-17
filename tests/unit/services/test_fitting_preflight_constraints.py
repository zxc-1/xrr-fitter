from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.support.drift_cases import one_drift_block_structure
from tests.support.model_cases import simple_structure
from tests.unit.services.test_fitting import _project, _source

from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterPrior,
    ParameterReference,
    PriorSpec,
)
from xrr_fitter.services import batch, fitting
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.structures import set_structure

DENSITY = "component.0.density_scale"
ROUGHNESS = "component.0.roughness_a"


def _constant_rule(dataset_id: str, value: float) -> ConstraintRule:
    return ConstraintRule(
        ParameterReference(dataset_id, DENSITY),
        ConstraintNode("const", value=value),
    )


def _cross_dataset_rule(left: str, right: str) -> ConstraintRule:
    return ConstraintRule(
        ParameterReference(left, DENSITY),
        ConstraintNode(
            "mul",
            operands=(
                ConstraintNode(
                    "ref",
                    reference=ParameterReference(right, DENSITY),
                ),
                ConstraintNode("const", value=100.0),
            ),
        ),
    )


def _cross_roughness_rule(left: str, right: str) -> ConstraintRule:
    return ConstraintRule(
        ParameterReference(left, ROUGHNESS),
        ConstraintNode(
            "ref",
            reference=ParameterReference(right, ROUGHNESS),
        ),
    )


def test_preflight_rejects_out_of_bounds_local_constraint_initial(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    value = replace(
        value,
        constraint_rules=(_constant_rule("curve", 100.0),),
    )

    readiness = fitting.preflight_fit(value)

    assert readiness.ready is False
    assert readiness.message == f"constraint_out_of_bounds:{DENSITY}"


def test_preflight_rejects_out_of_bounds_joint_constraint_initial(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    value = add_dataset(
        value,
        _source(tmp_path / "right.xy"),
        InstrumentSpec(instrument_id="right", footprint_mode="none"),
        display_name="right",
    )
    value = set_structure(value, "right", simple_structure())
    value = replace(
        value,
        batch_mode="joint",
        constraint_rules=(_cross_dataset_rule("curve", "right"),),
    )

    readiness = fitting.preflight_fit(value)

    assert readiness.ready is False
    assert readiness.message == f"constraint_out_of_bounds:curve::{DENSITY}"


def test_joint_preflight_uses_joint_initial_when_local_roughness_initial_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _project(tmp_path)
    value = add_dataset(
        value,
        _source(tmp_path / "right.xy"),
        InstrumentSpec(instrument_id="right", footprint_mode="none"),
        display_name="right",
    )
    value = set_structure(value, "right", simple_structure())
    value = replace(
        value,
        batch_mode="joint",
        constraint_rules=(_cross_roughness_rule("curve", "right"),),
    )
    prepared = {
        dataset.dataset_id: SimpleNamespace(
            dataset_id=dataset.dataset_id,
            problem=SimpleNamespace(parameter_definitions=()),
            updated_dataset=SimpleNamespace(parameter_priors=()),
        )
        for dataset in value.datasets
    }
    joint = SimpleNamespace(name="joint")
    joint_initial = object()
    local_initial_calls = []
    joint_calls = []

    def compile_joint_problem(dataset_ids, problems, _sharing_rules, constraint_rules):
        joint_calls.append(("compile", dataset_ids, constraint_rules))
        assert problems == tuple(prepared[dataset_id].problem for dataset_id in dataset_ids)
        return joint

    def evaluate_declared_initial(problem):
        local_initial_calls.append(problem)
        return SimpleNamespace(
            valid=False,
            reason="local declared initial is invalid",
        )

    def initial_joint_vector(observed_joint):
        joint_calls.append(("initial", observed_joint is joint))
        return joint_initial

    def evaluate_joint_vector(observed_joint, observed_initial):
        joint_calls.append(
            (
                "evaluate",
                observed_joint is joint,
                observed_initial is joint_initial,
            )
        )
        return SimpleNamespace(valid=True)

    monkeypatch.setattr(
        fitting,
        "prepare_dataset_fit",
        lambda _project, dataset_id, _seed: prepared[dataset_id],
    )
    monkeypatch.setattr(fitting, "compile_joint_problem", compile_joint_problem)
    monkeypatch.setattr(fitting, "evaluate_declared_initial", evaluate_declared_initial)
    monkeypatch.setattr(fitting, "initial_joint_vector", initial_joint_vector)
    monkeypatch.setattr(fitting, "evaluate_joint_vector", evaluate_joint_vector)

    readiness = fitting.preflight_fit(value)

    assert readiness.ready is True, readiness.message
    assert readiness.message == "ready"
    assert local_initial_calls == []
    assert joint_calls == [
        ("compile", ("curve", "right"), value.constraint_rules),
        ("initial", True),
        ("evaluate", True, True),
    ]


def test_preflight_rejects_prior_for_generated_drift_constraint(
    tmp_path: Path,
) -> None:
    value = set_structure(_project(tmp_path), "curve", one_drift_block_structure())
    dataset = replace(
        value.datasets[0],
        parameter_priors=(
            ParameterPrior(
                "component.0.repeat.1.layer.0.thickness_a",
                PriorSpec("uniform"),
            ),
        ),
    )

    readiness = fitting.preflight_fit(replace(value, datasets=(dataset,)))

    assert readiness.ready is False
    assert readiness.message == (
        "cannot assign a prior to constrained parameter: component.0.repeat.1.layer.0.thickness_a"
    )


def test_automatic_preflight_rejects_out_of_bounds_constraint_initial(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    dataset = replace(
        value.datasets[0],
        automation=DatasetAutomation(
            import_batch_id="batch-1",
            role=AutomaticRole.UNROUTED,
            status=AutomaticStatus.PENDING,
        ),
    )
    value = replace(
        value,
        datasets=(dataset,),
        constraint_rules=(_constant_rule("curve", 100.0),),
        measurement_preset=MeasurementPreset(
            "lab",
            BeamSpec("monochromatic"),
            InstrumentSpec(instrument_id="lab"),
        ),
    )

    readiness = fitting.preflight_automatic_fit(value)

    assert readiness.ready is False
    assert readiness.message == f"constraint_out_of_bounds:{DENSITY}"


def test_automatic_preflight_rejects_stale_parameter_prior(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    dataset = replace(
        value.datasets[0],
        automation=DatasetAutomation(
            import_batch_id="batch-1",
            role=AutomaticRole.UNROUTED,
            status=AutomaticStatus.PENDING,
        ),
        parameter_priors=(ParameterPrior("component.99.thickness_a", PriorSpec("uniform")),),
    )
    value = replace(
        value,
        datasets=(dataset,),
        measurement_preset=MeasurementPreset(
            "lab",
            BeamSpec("monochromatic"),
            InstrumentSpec(instrument_id="lab"),
        ),
    )

    readiness = fitting.preflight_automatic_fit(value)

    assert readiness.ready is False
    assert readiness.message == "unknown parameter name: component.99.thickness_a"


def test_fit_project_rejects_invalid_initial_before_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = replace(
        _project(tmp_path),
        constraint_rules=(_constant_rule("curve", 100.0),),
    )
    monkeypatch.setattr(
        batch,
        "fit_project_transaction",
        lambda *_args, **_kwargs: pytest.fail("fit_project started a transaction with an invalid initial"),
    )

    with pytest.raises(ValueError, match=f"constraint_out_of_bounds:{DENSITY}"):
        fitting.fit_project(value)


def test_fit_worker_rejects_invalid_initial_before_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = replace(
        _project(tmp_path),
        constraint_rules=(_constant_rule("curve", 100.0),),
    )
    monkeypatch.setattr(
        batch,
        "fit_project_transaction",
        lambda *_args, **_kwargs: pytest.fail("fit worker started a transaction with an invalid initial"),
    )

    with pytest.raises(ValueError, match=f"constraint_out_of_bounds:{DENSITY}"):
        fitting.fit_worker_handler(
            value,
            None,
            None,
            lambda: False,
        )


def test_automatic_worker_rejects_invalid_initial_before_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _project(tmp_path)
    dataset = replace(
        value.datasets[0],
        automation=DatasetAutomation(
            import_batch_id="batch-1",
            role=AutomaticRole.UNROUTED,
            status=AutomaticStatus.PENDING,
        ),
    )
    value = replace(
        value,
        datasets=(dataset,),
        constraint_rules=(_constant_rule("curve", 100.0),),
        measurement_preset=MeasurementPreset(
            "lab",
            BeamSpec("monochromatic"),
            InstrumentSpec(instrument_id="lab"),
        ),
    )
    monkeypatch.setattr(
        batch,
        "fit_automatic_transaction",
        lambda *_args, **_kwargs: pytest.fail("automatic worker started a transaction with an invalid initial"),
    )

    with pytest.raises(ValueError, match=f"constraint_out_of_bounds:{DENSITY}"):
        fitting.automatic_worker_handler(
            value,
            "batch-1",
            None,
            None,
            lambda: False,
        )
