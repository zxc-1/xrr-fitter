from __future__ import annotations

from tests.unit.fit.joint_evaluation_cases import *


def test_joint_loss_handles_extreme_positive_robust_scale_without_underflow() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    baseline = _joint()
    joint = replace(
        baseline,
        problems=tuple(
            replace(problem, config=replace(problem.config, c_decades=1e-200)) for problem in baseline.problems
        ),
    )
    sizes = tuple(int(np.count_nonzero(problem.data.fit_mask)) for problem in joint.problems)
    squared = np.ones(sum(sizes))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rho = api.joint_least_squares_loss(joint)(squared)

    assert np.all(np.isfinite(rho))
    assert np.all(rho[0] > 0.0)
    assert np.all(rho[1] > 0.0)
    assert np.all(rho[2] < 0.0)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_joint_loss_keeps_subnormal_residual_curvature_finite() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    baseline = _joint()
    joint = replace(
        baseline,
        problems=tuple(
            replace(problem, config=replace(problem.config, c_decades=1e-200)) for problem in baseline.problems
        ),
    )
    sizes = tuple(int(np.count_nonzero(problem.data.fit_mask)) for problem in joint.problems)
    squared = np.full(sum(sizes), 1e-320)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rho = api.joint_least_squares_loss(joint)(squared)

    total_data = sum(sizes)
    expected_blocks = []
    for size, problem in zip(sizes, joint.problems, strict=True):
        weights = problem.weights[problem.data.fit_mask]
        c_decades = problem.config.c_decades
        radius = np.hypot(c_decades, np.sqrt(np.full(size, 1e-320)))
        alpha = total_data / (len(sizes) * size)
        inverse_curvature_scale = ((c_decades / radius) / radius) / radius
        expected_blocks.append(-alpha * weights**2 * inverse_curvature_scale)
    np.testing.assert_allclose(
        rho[2],
        np.concatenate(expected_blocks),
        rtol=1e-15,
        atol=0.0,
    )
    assert np.all(np.isfinite(rho))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_joint_loss_preserves_nonzero_quadratic_value_near_zero() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    sizes = tuple(int(np.count_nonzero(problem.data.fit_mask)) for problem in joint.problems)
    squared = np.full(sum(sizes), 1e-24)

    rho = api.joint_least_squares_loss(joint)(squared)

    assert np.all(rho[0] > 0.0)


@pytest.mark.parametrize(
    "invalid_rows",
    [
        pytest.param(lambda count: np.zeros(count - 1), id="missing-row"),
        pytest.param(
            lambda count: np.concatenate((np.asarray([-1e-6]), np.zeros(count - 1))),
            id="negative",
        ),
        pytest.param(
            lambda count: np.concatenate((np.asarray([np.nan]), np.zeros(count - 1))),
            id="nonfinite",
        ),
    ],
)
def test_joint_loss_rejects_invalid_squared_residual_rows(invalid_rows) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    row_count = sum(int(np.count_nonzero(problem.data.fit_mask)) + 1 for problem in joint.problems)

    with pytest.raises(ValueError, match="joint loss|squared|row|finite|nonnegative"):
        api.joint_least_squares_loss(joint)(invalid_rows(row_count))


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda joint: replace(joint, problems=()), id="empty-layout"),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(replace(joint.problems[0], weights=np.empty(0)), *joint.problems[1:]),
            ),
            id="empty-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(joint.problems[0], weights=joint.problems[0].weights[None, :]),
                    *joint.problems[1:],
                ),
            ),
            id="two-dimensional-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(joint.problems[0], weights=np.zeros_like(joint.problems[0].weights)),
                    *joint.problems[1:],
                ),
            ),
            id="zero-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(
                        joint.problems[0],
                        weights=np.full_like(joint.problems[0].weights, np.nan),
                    ),
                    *joint.problems[1:],
                ),
            ),
            id="nonfinite-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(joint.problems[0], config=SimpleNamespace(c_decades=0.0)),
                    *joint.problems[1:],
                ),
            ),
            id="nonpositive-c-decades",
        ),
    ],
)
def test_joint_loss_rejects_invalid_compiled_layout(mutation) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")

    # The typed evaluation context now rejects several malformed dataset
    # layouts before the joint-loss boundary is reached. Both boundaries are
    # valid owners of this invariant and must keep the rejection explicit.
    with pytest.raises(
        (TypeError, ValueError),
        match="joint loss|layout|weight|c_decades|dataset|region arrays|config",
    ):
        api.joint_least_squares_loss(mutation(_joint(scale_prior=True)))
