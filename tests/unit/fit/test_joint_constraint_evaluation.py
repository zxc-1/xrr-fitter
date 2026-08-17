from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.joint_constraint_cases import (
    SHARED_NAME,
    cross_constraint_chain_joint,
    cross_constraint_joint,
    cross_roughness_constraint_joint,
    local_problem,
)
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
    ParameterSetting,
)

ROUGHNESS_NAME = "component.0.roughness_a"


def test_cross_dataset_constraint_removes_target_and_drives_local_projection() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = cross_constraint_joint()

    assert len(joint.global_variables) == 1
    right_target = next(
        definition for definition in joint.problems[1].parameter_definitions if definition.name == SHARED_NAME
    )
    assert right_target.constrained is True
    evaluation = api.evaluate_joint_vector(joint, np.asarray([0.4]))
    parameters = tuple(
        {parameter.name: parameter.value for parameter in local.parameters} for local in evaluation.local_evaluations
    )

    assert parameters[1][SHARED_NAME] == pytest.approx(parameters[0][SHARED_NAME])


def test_local_rule_depending_on_cross_target_is_evaluated_in_joint_dag() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = cross_constraint_chain_joint()
    left_density_index = next(
        index
        for index, variable in enumerate(joint.global_variables)
        if variable.members == (ParameterReference("left", SHARED_NAME),)
    )
    unit = np.full(len(joint.global_variables), 0.5)
    unit[left_density_index] = 0.25

    evaluation = api.evaluate_joint_vector(joint, unit)
    right = {parameter.name: parameter.value for parameter in evaluation.local_evaluations[1].parameters}

    assert evaluation.valid
    assert set(joint.joint_constraint_rules) == set(joint.constraint_rules)
    assert right[SHARED_NAME] == pytest.approx(0.65)
    assert right["instrument.scale"] == pytest.approx(right[SHARED_NAME])


def test_cross_dataset_constraint_joint_jacobian_matches_finite_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = cross_constraint_joint()
    unit = np.asarray([0.45])

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    plus = api.evaluate_joint_vector(joint, unit + step).residuals
    minus = api.evaluate_joint_vector(joint, unit - step).residuals
    finite = ((plus - minus) / (2.0 * step))[:, None]

    np.testing.assert_allclose(analytic, finite, rtol=5e-5, atol=5e-8)


def _angle_offset_problem(*, seed: int, size: int, free: bool) -> object:
    base = compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        FitConfig.fast(seed),
    )
    free_names = {"instrument.angle_offset_deg"} if free else set()
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name in free_names else definition.initial,
            definition.upper if definition.name in free_names else definition.initial,
            locked=definition.name not in free_names,
        )
        for definition in base.parameter_definitions
    )
    return compile_fit_problem(base.data, base.structure, base.instrument, base.config, settings)


def test_cross_dataset_fractional_power_constraint_keeps_joint_callbacks_finite() -> None:
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    evaluation_api = import_module("xrr_fitter.fit.joint_evaluation")
    rule = ConstraintRule(
        ParameterReference("right", "instrument.background"),
        ConstraintNode(
            "pow",
            operands=(
                ConstraintNode(
                    "ref",
                    reference=ParameterReference("left", "instrument.angle_offset_deg"),
                ),
                ConstraintNode("const", value=0.5),
            ),
        ),
    )
    joint = joint_api.compile_joint_problem(
        ("left", "right"),
        (
            _angle_offset_problem(seed=862, size=40, free=True),
            _angle_offset_problem(seed=863, size=52, free=True),
        ),
        (),
        (rule,),
    )
    unit = np.asarray([0.5, 0.5])

    evaluation = evaluation_api.evaluate_joint_vector(joint, unit)
    jacobian = evaluation_api.evaluate_joint_jacobian(joint, unit)

    assert evaluation.valid
    assert not np.all(evaluation.residuals == 1e6)
    assert jacobian.shape == (evaluation.residuals.size, len(joint.global_variables))
    assert np.all(np.isfinite(jacobian))


def test_cross_dataset_constraint_scatter_jacobian_does_not_perturb_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_api = import_module("xrr_fitter.fit.joint_evaluation")
    scatter_api = import_module("xrr_fitter.fit.joint_scatter_jacobian")
    joint = cross_constraint_joint()
    monkeypatch.setattr(
        scatter_api,
        "_project_or_none",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("expression constraints must use analytic scatter derivatives")
        ),
    )

    jacobian = evaluation_api.evaluate_joint_jacobian(joint, np.asarray([0.45]))

    assert jacobian.shape[1] == 1


def test_cross_dataset_constraint_chain_jacobian_propagates_through_each_target() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = cross_constraint_chain_joint()
    unit = np.full(len(joint.global_variables), 0.45)

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    finite_columns = []
    for index in range(unit.size):
        plus = unit.copy()
        minus = unit.copy()
        plus[index] += step
        minus[index] -= step
        finite_columns.append(
            (api.evaluate_joint_vector(joint, plus).residuals - api.evaluate_joint_vector(joint, minus).residuals)
            / (2.0 * step)
        )

    np.testing.assert_allclose(
        analytic,
        np.column_stack(finite_columns),
        rtol=5e-5,
        atol=5e-8,
    )


def test_cross_dataset_roughness_constraint_jacobian_matches_finite_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = cross_roughness_constraint_joint()
    unit = np.full(len(joint.global_variables), 0.45)

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    finite_columns = []
    for index in range(unit.size):
        plus = unit.copy()
        minus = unit.copy()
        plus[index] += step
        minus[index] -= step
        finite_columns.append(
            (api.evaluate_joint_vector(joint, plus).residuals - api.evaluate_joint_vector(joint, minus).residuals)
            / (2.0 * step)
        )

    np.testing.assert_allclose(
        analytic,
        np.column_stack(finite_columns),
        rtol=5e-5,
        atol=5e-8,
    )


def _locked_roughness_problem(*, seed: int, size: int, roughness_initial: float | None = None) -> object:
    base = compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        FitConfig.fast(seed),
    )
    settings = []
    for definition in base.parameter_definitions:
        initial = (
            roughness_initial
            if definition.name == ROUGHNESS_NAME and roughness_initial is not None
            else definition.initial
        )
        lower = definition.lower if definition.name == ROUGHNESS_NAME and roughness_initial is not None else initial
        upper = definition.upper if definition.name == ROUGHNESS_NAME and roughness_initial is not None else initial
        settings.append(
            ParameterSetting(
                definition.name,
                initial,
                lower,
                upper,
                locked=True,
            )
        )
    return compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        tuple(settings),
    )


def test_initial_joint_vector_lets_roughness_constraint_replace_invalid_declared_target() -> None:
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    evaluation = import_module("xrr_fitter.fit.joint_evaluation")
    rule = ConstraintRule(
        ParameterReference("left", ROUGHNESS_NAME),
        ConstraintNode(
            "ref",
            reference=ParameterReference("right", ROUGHNESS_NAME),
        ),
    )
    joint = joint_api.compile_joint_problem(
        ("left", "right"),
        (
            _locked_roughness_problem(seed=858, size=40, roughness_initial=15.0),
            _locked_roughness_problem(seed=858, size=52),
        ),
        (),
        (rule,),
    )

    initial = sharing.initial_joint_vector(joint)
    result = evaluation.evaluate_joint_vector(joint, initial)
    left = {parameter.name: parameter.value for parameter in result.local_evaluations[0].parameters}
    right = {parameter.name: parameter.value for parameter in result.local_evaluations[1].parameters}

    assert result.valid
    assert left[ROUGHNESS_NAME] == pytest.approx(right[ROUGHNESS_NAME])
    assert left[ROUGHNESS_NAME] == pytest.approx(2.0)


def test_joint_layout_fingerprint_binds_cross_dataset_expression() -> None:
    baseline = cross_constraint_joint(multiplier=1.0)
    changed = cross_constraint_joint(multiplier=0.9)

    assert baseline.layout_fingerprint != changed.layout_fingerprint


def test_joint_result_provenance_binds_cross_dataset_expression_layout() -> None:
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    baseline = cross_constraint_joint(multiplier=1.0)
    source = ParameterReference("left", SHARED_NAME)
    target = ParameterReference("right", SHARED_NAME)
    equivalent_rule = ConstraintRule(
        target,
        ConstraintNode(
            "add",
            operands=(
                ConstraintNode("ref", reference=source),
                ConstraintNode("const", value=0.0),
            ),
        ),
    )
    equivalent = joint_api.compile_joint_problem(
        ("left", "right"),
        (
            local_problem(seed=854, size=40),
            local_problem(seed=854, size=52),
        ),
        (),
        (equivalent_rule,),
    )

    baseline_results = pipeline.run_joint_fit(pipeline.JointFitRequest(baseline))
    equivalent_results = pipeline.run_joint_fit(pipeline.JointFitRequest(equivalent))

    assert baseline.layout_fingerprint != equivalent.layout_fingerprint
    assert tuple(result.provenance_sha256 for result in baseline_results) != tuple(
        result.provenance_sha256 for result in equivalent_results
    )


def _candidate_rows(
    local_units: list[np.ndarray],
    right_unit: np.ndarray,
    *,
    candidate_id: str,
    valid: bool,
) -> tuple[tuple[SimpleNamespace, ...], ...]:
    objective = 1.0 if valid else float("inf")
    ranking = 1.0 if valid else None
    return (
        (
            SimpleNamespace(
                candidate_id=candidate_id,
                unit_vector=local_units[0],
                objective=objective,
                ranking_objective=ranking,
                valid=valid,
            ),
        ),
        (
            SimpleNamespace(
                candidate_id=candidate_id,
                unit_vector=right_unit,
                objective=objective,
                ranking_objective=ranking,
                valid=valid,
            ),
        ),
    )


def _joint_projection_context():
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = cross_constraint_joint()
    global_unit = sharing.initial_joint_vector(joint)
    local_units = list(sharing.scatter_joint_vector(joint, global_unit))
    target_index = next(
        index for index, coordinate in enumerate(joint.problems[1].variables) if coordinate.name == SHARED_NAME
    )
    return joint, global_unit, local_units, target_index


def test_joint_candidate_rebuild_rejects_cross_constraint_target_drift() -> None:
    candidates_api = import_module("xrr_fitter.fit.joint_candidates")
    joint, _global_unit, local_units, target_index = _joint_projection_context()
    tampered = np.array(local_units[1], copy=True)
    tampered[target_index] = 0.0 if tampered[target_index] != 0.0 else 1.0
    candidates = _candidate_rows(
        local_units,
        tampered,
        candidate_id="joint-a",
        valid=True,
    )

    with pytest.raises(ValueError, match="constraint target.*projection mismatch"):
        candidates_api.joint_candidate_vectors(joint, candidates, ("joint-a",))


def test_joint_candidate_rebuild_keeps_invalid_constraint_placeholders() -> None:
    candidates_api = import_module("xrr_fitter.fit.joint_candidates")
    joint, global_unit, local_units, target_index = _joint_projection_context()
    placeholder = np.array(local_units[1], copy=True)
    placeholder[target_index] = 0.0
    candidates = _candidate_rows(
        local_units,
        placeholder,
        candidate_id="joint-invalid",
        valid=False,
    )

    rebuilt = candidates_api.joint_candidate_vectors(
        joint,
        candidates,
        ("joint-invalid",),
    )

    np.testing.assert_allclose(rebuilt[0], global_unit)


def test_cross_dataset_constraint_runtime_domain_failure_is_an_invalid_candidate() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = cross_constraint_joint(multiplier=2.0)
    unit = np.asarray([0.9])

    evaluation = api.evaluate_joint_vector(joint, unit)
    jacobian = api.evaluate_joint_jacobian(joint, unit)

    assert evaluation.valid is False
    assert evaluation.objective == float("inf")
    assert all(not local.valid for local in evaluation.local_evaluations)
    assert evaluation.local_unit_vectors[0][0] == pytest.approx(unit[0])
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))


def test_cross_dataset_constraint_jacobian_uses_valid_one_sided_perturbation() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = cross_constraint_joint(multiplier=2.0)
    step = 1e-6
    unit = np.asarray([1.0 / 12.0 - step / 2.0])

    analytic = api.evaluate_joint_jacobian(joint, unit)
    baseline = api.evaluate_joint_vector(joint, unit)
    lower = api.evaluate_joint_vector(joint, unit - step)
    finite = ((baseline.residuals - lower.residuals) / step)[:, None]

    assert baseline.valid
    assert lower.valid
    np.testing.assert_allclose(analytic, finite, rtol=7e-5, atol=7e-8)


def test_dataset_local_constraints_keep_the_direct_joint_jacobian_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    left = local_problem(seed=855, size=40)
    right = local_problem(seed=855, size=52)
    rule = ConstraintRule(
        ParameterReference("left", SHARED_NAME),
        ConstraintNode("const", value=0.8),
    )
    left = compile_fit_problem(
        left.data,
        left.structure,
        left.instrument,
        left.config,
        tuple(
            ParameterSetting(
                definition.name,
                definition.initial,
                definition.lower,
                definition.upper,
                definition.locked,
            )
            for definition in left.parameter_definitions
        ),
        (rule,),
    )
    joint = joint_api.compile_joint_problem(
        ("left", "right"),
        (left, right),
        (),
        (rule,),
    )
    monkeypatch.setattr(
        api,
        "_joint_scatter_jacobians",
        lambda *_args: (_ for _ in ()).throw(AssertionError("local constraints do not need numerical scatter")),
    )

    jacobian = api.evaluate_joint_jacobian(joint, np.asarray([0.5]))

    assert jacobian.shape[1] == 1
