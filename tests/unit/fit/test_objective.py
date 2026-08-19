from __future__ import annotations

from tests.unit.fit.objective_cases import *


def test_region_weights_give_each_present_region_equal_quadratic_mass() -> None:
    labels = np.array([0, 0, 1, 2, 2, 2])

    weights = region_weights(labels)

    masses = [np.sum(weights[labels == label] ** 2) for label in np.unique(labels)]
    np.testing.assert_allclose(masses, np.full(3, labels.size / 3.0))


def test_log_residual_uses_background_floor_and_is_unweighted() -> None:
    model = np.array([1.0, 1e-9, 0.0])
    observed = np.array([0.5, 2e-9, 1e-10])
    floor = 1e-8

    actual = log_residuals(model, observed, floor)

    expected = np.log10(model + floor) - np.log10(observed + floor)
    np.testing.assert_allclose(actual, expected)


def test_log_residual_preserves_adjacent_extreme_positive_difference() -> None:
    observed = 1e308
    model = np.nextafter(observed, np.inf)

    actual = log_residuals(
        np.array([model]),
        np.array([observed]),
        np.nextafter(0.0, 1.0),
    )

    expected = np.log1p((model - observed) / observed) / np.log(10.0)
    assert actual[0] == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert actual[0] > 0.0


def test_log_residual_avoids_intermediate_floor_addition_overflow() -> None:
    model = np.array([1e308])
    observed = np.array([5e307])
    floor = 1e308

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = log_residuals(model, observed, floor)

    expected = np.log10(4.0 / 3.0)
    assert actual[0] == pytest.approx(expected, rel=1e-14)
    assert np.all(np.isfinite(actual))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_robust_cost_places_weights_outside_the_loss() -> None:
    delta = np.array([0.01, 0.2])
    weights = np.array([3.0, 0.5])
    c = 0.05

    actual = robust_log_cost(delta, weights, c)

    pointwise = 2.0 * c**2 * (np.sqrt(1.0 + (delta / c) ** 2) - 1.0)
    assert actual == pytest.approx(np.mean(weights**2 * pointwise))


def test_robust_cost_equals_pointwise_threshold_scaling() -> None:
    delta = np.array([-0.1, 0.0, 0.1])
    weights = np.ones(3)
    c = 0.05

    actual = robust_log_cost(delta, weights, c)

    expected = np.mean(2.0 * c**2 * (np.sqrt(1.0 + (delta / c) ** 2) - 1.0))
    assert actual == pytest.approx(expected)


def test_robust_cost_handles_extreme_positive_threshold_without_underflowing_its_square() -> None:
    c = 1e-200

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = robust_log_cost(np.array([1.0]), np.ones(1), c)

    expected = 2.0 * c * (np.hypot(c, 1.0) - c)
    assert actual == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_robust_cost_preserves_nonzero_quadratic_loss_near_zero() -> None:
    delta = np.array([1e-12])
    c = 0.05

    actual = robust_log_cost(delta, np.ones(1), c)

    radius = np.hypot(c, delta[0])
    expected = 2.0 * c * abs(delta[0]) * (abs(delta[0]) / (radius + c))
    assert actual == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert actual > 0.0


def test_scale_prior_penalty_matches_versioned_form() -> None:
    scale = 1.2
    estimate = 1.05
    tau = 0.1
    count = 200

    actual = scale_prior_penalty(scale, estimate, tau, count)

    expected = ((np.log10(scale) - np.log10(estimate)) / tau) ** 2 / count
    assert actual == pytest.approx(expected)
