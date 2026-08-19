from __future__ import annotations

from tests.unit.fit.joint_evaluation_cases import *


def test_shared_roughness_consensus_uses_candidate_physical_values() -> None:
    candidates_api = import_module("xrr_fitter.fit.joint_candidates")
    evaluation = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _unequal_roughness_joint()
    candidates = {}
    for dataset_id, problem in zip(
        joint.dataset_ids,
        joint.problems,
        strict=True,
    ):
        unit = encode_physical_vector(
            problem,
            {"component.0.roughness_a": 4.0},
        )
        candidates[dataset_id] = SimpleNamespace(
            valid=True,
            parameters=evaluate_vector(problem, unit).parameters,
        )

    consensus = candidates_api.consensus_joint_vector(joint, candidates)
    result = evaluation.evaluate_joint_vector(joint, consensus)

    roughness = tuple(
        next(value.value for value in local.parameters if value.name == "component.0.roughness_a")
        for local in result.local_evaluations
    )
    assert roughness == pytest.approx((4.0, 4.0))


def test_shared_roughness_joint_candidate_rebuilds_from_physical_projection() -> None:
    candidates_api = import_module("xrr_fitter.fit.joint_candidates")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _unequal_roughness_joint()
    global_unit = sharing.initial_joint_vector(joint)
    local_units = sharing.scatter_joint_vector(joint, global_unit)
    candidates = tuple(
        (
            SimpleNamespace(
                candidate_id="joint-a",
                unit_vector=unit,
                objective=1.0,
                ranking_objective=1.0,
            ),
        )
        for unit in local_units
    )

    rebuilt = candidates_api.joint_candidate_vectors(
        joint,
        candidates,
        ("joint-a",),
    )

    np.testing.assert_allclose(rebuilt[0], global_unit)


def test_shared_physical_roughness_jacobian_matches_central_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _unequal_roughness_joint()
    unit = sharing.initial_joint_vector(joint)

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    finite = np.column_stack(
        [
            (
                api.evaluate_joint_vector(
                    joint,
                    unit + np.eye(unit.size)[index] * step,
                ).residuals
                - api.evaluate_joint_vector(
                    joint,
                    unit - np.eye(unit.size)[index] * step,
                ).residuals
            )
            / (2.0 * step)
            for index in range(unit.size)
        ]
    )

    np.testing.assert_allclose(analytic, finite, rtol=5e-5, atol=5e-8)


def test_shared_global_jacobian_column_contains_both_dataset_blocks() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()

    jacobian = api.evaluate_joint_jacobian(joint, np.asarray([0.55]))

    first_size = int(np.count_nonzero(joint.problems[0].data.fit_mask))
    assert np.any(np.abs(jacobian[:first_size, 0]) > 0.0)
    assert np.any(np.abs(jacobian[first_size:, 0]) > 0.0)
