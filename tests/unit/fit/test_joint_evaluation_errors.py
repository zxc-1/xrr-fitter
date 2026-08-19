from __future__ import annotations

from tests.unit.fit.joint_evaluation_cases import *


def test_one_invalid_local_evaluation_invalidates_the_joint_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    values = {
        id(joint.problems[0]): _evaluation(joint.problems[0], objective=1.0, residual=0.2),
        id(joint.problems[1]): _evaluation(
            joint.problems[1],
            objective=float("inf"),
            residual=1e6,
            valid=False,
        ),
    }
    monkeypatch.setattr(api, "evaluate_vector", lambda problem, _unit: values[id(problem)])

    result = api.evaluate_joint_vector(joint, np.asarray([0.5]))

    assert not result.valid
    assert result.objective == float("inf")
    assert tuple(value.valid for value in result.local_evaluations) == (True, False)


def test_invalid_local_evaluation_with_scale_prior_remains_a_joint_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    unit = np.full(len(joint.global_variables), 0.55)
    baseline = api.evaluate_joint_vector(joint, unit)
    values = {
        id(joint.problems[0]): baseline.local_evaluations[0],
        id(joint.problems[1]): replace(
            baseline.local_evaluations[1],
            valid=False,
            objective=float("inf"),
            parameters=(),
        ),
    }
    monkeypatch.setattr(api, "evaluate_vector", lambda problem, _unit: values[id(problem)])

    result = api.evaluate_joint_vector(joint, unit)

    assert not result.valid
    assert result.objective == float("inf")
    assert result.residuals.size == sum(np.count_nonzero(problem.data.fit_mask) + 1 for problem in joint.problems)
    assert result.residuals[-1] == 1e6

    local_search = import_module("xrr_fitter.fit.local_search")
    original_jacobian = local_search.evaluate_jacobian

    def selective_jacobian(problem, candidate):
        if problem is joint.problems[1]:
            raise api.EvaluationConstraintError("constraint_violation:test")
        return original_jacobian(problem, candidate)

    monkeypatch.setattr(local_search, "evaluate_jacobian", selective_jacobian)
    jacobian = api.evaluate_joint_jacobian(joint, unit)
    first_rows = int(np.count_nonzero(joint.problems[0].data.fit_mask)) + 1
    second_data_rows = int(np.count_nonzero(joint.problems[1].data.fit_mask))
    np.testing.assert_array_equal(
        jacobian[first_rows + second_data_rows],
        np.zeros(len(joint.global_variables)),
    )


def test_joint_evaluation_propagates_unexpected_local_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    sentinel = RuntimeError("unexpected local evaluation failure")

    def fail(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(api, "evaluate_vector", fail)

    with pytest.raises(RuntimeError, match="unexpected local evaluation failure") as captured:
        api.evaluate_joint_vector(_joint(), np.asarray([0.5]))

    assert captured.value is sentinel


def test_shared_roughness_empty_member_domain_is_an_invalid_joint_candidate() -> None:
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    evaluation_api = import_module("xrr_fitter.fit.joint_evaluation")

    def build(seed: int, thickness_lower: float):
        config = replace(FitConfig.fast(seed), scale_prior_enabled=False)
        base = compile_fit_problem(
            prepared_data(size=40),
            simple_structure(),
            InstrumentSpec(footprint_mode="none", instrument_id="shared-lab"),
            config,
        )
        settings = []
        for definition in base.parameter_definitions:
            if definition.name == TIE_THICKNESS_NAME:
                settings.append(ParameterSetting(definition.name, 20.0, thickness_lower, 30.0))
            elif definition.name == "component.0.roughness_a":
                settings.append(ParameterSetting(definition.name, 4.5, 4.0, 9.0))
            else:
                settings.append(
                    ParameterSetting(
                        definition.name,
                        definition.initial,
                        definition.initial,
                        definition.initial,
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

    roughness = "component.0.roughness_a"
    joint = joint_api.compile_joint_problem(
        ("left", "right"),
        (build(901, 10.0), build(902, 5.0)),
        (
            SharingRule(
                "shared-roughness",
                (
                    ParameterReference("left", roughness),
                    ParameterReference("right", roughness),
                ),
            ),
        ),
    )
    unit = np.full(len(joint.global_variables), 0.5)
    for index, variable in enumerate(joint.global_variables):
        if variable.name in {"left:component.0.thickness_a", "right:component.0.thickness_a"}:
            unit[index] = 0.0
        elif variable.name == "shared-roughness":
            unit[index] = 1.0

    result = evaluation_api.evaluate_joint_vector(joint, unit)
    jacobian = evaluation_api.evaluate_joint_jacobian(joint, unit)

    assert result.valid is False
    assert all(local.reason == "constraint_violation:PhysicalValueError" for local in result.local_evaluations)
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))
