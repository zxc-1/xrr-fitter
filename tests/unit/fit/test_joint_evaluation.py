from __future__ import annotations

from tests.unit.fit.joint_evaluation_cases import *


def test_joint_objective_is_the_arithmetic_mean_of_local_objectives() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()

    result = api.evaluate_joint_vector(joint, np.asarray([0.5]))

    assert result.valid
    assert len(result.local_evaluations) == 2
    assert result.objective == pytest.approx(np.mean([value.objective for value in result.local_evaluations]))
    assert result.local_unit_vectors[0][0] == result.local_unit_vectors[1][0] == 0.5
    assert not result.residuals.flags.writeable


def test_joint_objective_avoids_overflowing_a_finite_arithmetic_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    objectives = (1.6e308, 1.2e308)
    by_identity = {
        id(problem): _evaluation(problem, objective=objective, residual=1.0)
        for problem, objective in zip(joint.problems, objectives, strict=True)
    }
    monkeypatch.setattr(
        api,
        "evaluate_vector",
        lambda problem, _unit: by_identity[id(problem)],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = api.evaluate_joint_vector(joint, np.asarray([0.5]))

    assert result.valid
    assert result.objective == pytest.approx(
        objectives[0] / len(objectives) + objectives[1] / len(objectives),
    )
    assert np.isfinite(result.objective)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_joint_candidate_alignment_accepts_a_finite_rank_with_overflowing_local_sum() -> None:
    candidates_api = import_module("xrr_fitter.fit.joint_candidates")
    objectives = (1.6e308, 1.2e308)
    ranking = objectives[0] / len(objectives) + objectives[1] / len(objectives)
    candidates = tuple((SimpleNamespace(objective=objective, ranking_objective=ranking),) for objective in objectives)

    # The validator only consumes the aligned objective/ranking fields.
    candidates_api._validate_candidate_rankings(candidates, ("shared",))


def test_joint_residual_gives_each_dataset_equal_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    c_decades = joint.problems[0].config.c_decades
    objective = 2.0 * c_decades**2 * (np.sqrt(1.0 + 1.0 / c_decades**2) - 1.0)
    by_identity = {id(problem): _evaluation(problem, objective=objective, residual=1.0) for problem in joint.problems}
    monkeypatch.setattr(api, "evaluate_vector", lambda problem, _unit: by_identity[id(problem)])

    result = api.evaluate_joint_vector(joint, np.asarray([0.5]))

    first_size = len(by_identity[id(joint.problems[0])].fit_log_residuals_decades)
    first = result.residuals[:first_size]
    second = result.residuals[first_size:]
    np.testing.assert_array_equal(first, np.ones(first.size))
    np.testing.assert_array_equal(second, np.ones(second.size))
    rho = api.joint_least_squares_loss(joint)(result.residuals**2)
    assert 0.5 * np.sum(rho[0]) / result.residuals.size == pytest.approx(objective)
    assert result.objective == pytest.approx(objective)


def test_joint_loss_scales_data_mass_and_each_active_scale_prior_row() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    data_sizes = tuple(int(np.count_nonzero(problem.data.fit_mask)) for problem in joint.problems)
    row_count = sum(size + 1 for size in data_sizes)
    squared = np.linspace(0.001, 0.02, row_count)

    rho = api.joint_least_squares_loss(joint)(squared)

    total_data = sum(data_sizes)
    offset = 0
    for problem, size in zip(joint.problems, data_sizes, strict=True):
        alpha = total_data / (len(joint.problems) * size)
        weights = problem.weights[problem.data.fit_mask]
        data_squared = squared[offset : offset + size]
        scaled = 1.0 + data_squared / problem.config.c_decades**2
        np.testing.assert_allclose(
            rho[0, offset : offset + size],
            4.0 * alpha * weights**2 * problem.config.c_decades**2 * (np.sqrt(scaled) - 1.0),
        )
        np.testing.assert_allclose(
            rho[1, offset : offset + size],
            2.0 * alpha * weights**2 / np.sqrt(scaled),
        )
        np.testing.assert_allclose(
            rho[2, offset : offset + size],
            -(alpha * weights**2 / problem.config.c_decades**2) * scaled ** (-1.5),
        )
        prior_index = offset + size
        np.testing.assert_allclose(
            rho[:, prior_index],
            (2.0 * alpha * squared[prior_index], 2.0 * alpha, 0.0),
        )
        offset = prior_index + 1

    unit = np.full(len(joint.global_variables), 0.55)
    evaluation = api.evaluate_joint_vector(joint, unit)
    optimizer_rho = api.joint_least_squares_loss(joint)(evaluation.residuals**2)
    optimizer_objective = 0.5 * float(np.sum(optimizer_rho[0])) / total_data
    assert optimizer_objective == pytest.approx(
        evaluation.objective,
        rel=1e-12,
        abs=1e-14,
    )
    jacobian = api.evaluate_joint_jacobian(joint, unit)
    assert jacobian.shape == (row_count, len(joint.global_variables))
    assert np.any(np.abs(jacobian[np.cumsum(np.asarray(data_sizes) + 1) - 1]) > 0.0)
