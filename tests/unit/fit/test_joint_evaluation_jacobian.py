from __future__ import annotations

from tests.unit.fit.joint_evaluation_cases import *


def test_joint_analytic_jacobian_matches_global_finite_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    unit = np.asarray([0.5])

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    plus = api.evaluate_joint_vector(joint, unit + step).residuals
    minus = api.evaluate_joint_vector(joint, unit - step).residuals
    finite = ((plus - minus) / (2.0 * step))[:, None]

    assert analytic.shape == finite.shape
    np.testing.assert_allclose(analytic, finite, rtol=2e-5, atol=2e-8)
    assert not analytic.flags.writeable


def test_joint_analytic_jacobian_matches_finite_difference_with_scale_prior() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    unit = np.full(len(joint.global_variables), 0.55)

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    finite = np.column_stack(
        [
            (
                api.evaluate_joint_vector(joint, unit + np.eye(unit.size)[index] * step).residuals
                - api.evaluate_joint_vector(
                    joint,
                    unit - np.eye(unit.size)[index] * step,
                ).residuals
            )
            / (2.0 * step)
            for index in range(unit.size)
        ]
    )
    prior_rows = np.cumsum([int(np.count_nonzero(problem.data.fit_mask)) + 1 for problem in joint.problems]) - 1

    assert np.all(np.any(np.abs(analytic[prior_rows]) > 0.0, axis=1))
    np.testing.assert_allclose(analytic, finite, rtol=2e-5, atol=2e-8)


def test_dynamic_roughness_tie_jacobian_matches_central_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _tie_joint()
    unit = sharing.initial_joint_vector(joint)
    thickness_index = next(
        index for index, variable in enumerate(joint.global_variables) if variable.name == "shared-thickness"
    )

    analytic = api.evaluate_joint_jacobian(joint, unit)[:, thickness_index]
    step = 1e-5
    plus = unit.copy()
    minus = unit.copy()
    plus[thickness_index] += step
    minus[thickness_index] -= step
    finite = (api.evaluate_joint_vector(joint, plus).residuals - api.evaluate_joint_vector(joint, minus).residuals) / (
        2.0 * step
    )

    np.testing.assert_allclose(analytic, finite, rtol=3e-5, atol=3e-8)


def test_shared_roughness_uses_one_physical_value_with_local_thicknesses() -> None:
    evaluation = import_module("xrr_fitter.fit.joint_evaluation")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _unequal_roughness_joint()
    unit = sharing.initial_joint_vector(joint)

    result = evaluation.evaluate_joint_vector(joint, unit)

    roughness = tuple(
        next(value.value for value in local.parameters if value.name == "component.0.roughness_a")
        for local in result.local_evaluations
    )
    assert roughness == pytest.approx((3.0, 3.0))
    roughness_index = next(
        index for index, variable in enumerate(joint.global_variables) if variable.name == "shared-physical-roughness"
    )
    assert joint.global_variables[roughness_index].transform == ("shared_roughness_physical")
