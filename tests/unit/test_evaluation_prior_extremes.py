from __future__ import annotations

from tests.unit.evaluation_prior_cases import *


def test_prior_cdf_short_circuits_outside_bounds_without_extreme_tail_work() -> None:
    spec = PriorSpec("lognormal", (0.0, 1e308))
    lower, upper = 1e-320, 1e-300

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        below = evaluation.prior_cdf(spec, lower / 2.0, lower, upper)
        above = evaluation.prior_cdf(spec, upper * 2.0, lower, upper)

    assert below == 0.0
    assert above == 1.0
    assert not any(item.category is RuntimeWarning for item in caught)


def test_broad_lognormal_prior_resolves_mass_in_log_coordinate_without_warning() -> None:
    spec = PriorSpec("lognormal", (0.0, 1e308))
    lower, upper = 1e-320, 1e-300
    midpoint = 1e-310

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability = evaluation.prior_cdf(spec, midpoint, lower, upper)
        median = evaluation.prior_inverse_cdf(spec, 0.5, lower, upper)

    assert probability == pytest.approx(0.5, abs=1e-3)
    assert median == pytest.approx(midpoint, rel=1e-3)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_soft_range_prior_handles_finite_cross_zero_bounds_with_overflowing_span() -> None:
    spec = PriorSpec("soft_range", (-1.0, 1.0, 1.0))
    lower, upper = -1e308, 1e308
    expected_total = 2.0 + np.sqrt(2.0 * np.pi)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density = evaluation.prior_log_density(spec, 0.0, lower, upper)
        probability = evaluation.prior_cdf(spec, 0.0, lower, upper)
        median = evaluation.prior_inverse_cdf(spec, 0.5, lower, upper)

    assert density == pytest.approx(-np.log(expected_total), rel=2e-4)
    assert probability == pytest.approx(0.5, abs=2e-4)
    assert median == pytest.approx(0.0, abs=2e-4)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_problem_log_probability_handles_extreme_positive_robust_scale() -> None:
    baseline = _prior_problem()
    problem = replace(baseline, config=replace(baseline.config, c_decades=1e-200))
    unit = np.full(len(problem.variables), 0.45)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = evaluation.evaluate_model(problem, unit)
        log_probability = evaluation.problem_log_probability(problem, unit)

    assert observed.valid
    assert np.isfinite(observed.objective)
    assert np.isfinite(log_probability)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_problem_log_probability_preserves_tiny_nonzero_residual_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    residual = 1e-12
    c_decades = 0.05
    problem = SimpleNamespace(
        variables=(),
        data=SimpleNamespace(fit_mask=np.array([True])),
        weights=np.ones(1),
        config=SimpleNamespace(c_decades=c_decades),
        scale_prior_center=None,
        parameter_definitions=(),
    )
    observed = SimpleNamespace(
        valid=True,
        objective=1.0,
        fit_log_residuals_decades=np.array([residual]),
        parameters=(),
    )
    monkeypatch.setattr(evaluation, "evaluate_model", lambda *_args, **_kwargs: observed)

    actual = evaluation.problem_log_probability(problem, np.empty(0))

    radius = np.hypot(c_decades, residual)
    expected = -(abs(residual) / c_decades) * (abs(residual) / (radius + c_decades))
    assert actual == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert actual < 0.0


@pytest.mark.parametrize(
    "spec",
    (
        PriorSpec("normal", (100.0, 1e-4)),
        PriorSpec("lognormal", (log(100.0), 1e-5)),
    ),
)
def test_prior_inverse_cdf_returns_exact_declared_endpoints(spec: PriorSpec) -> None:
    assert evaluation.prior_inverse_cdf(spec, 0.0, 1.0, 200.0) == 1.0
    assert evaluation.prior_inverse_cdf(spec, 1.0, 1.0, 200.0) == 200.0


def test_prior_bounds_are_respected() -> None:
    assert evaluation.prior_bounds(PriorSpec("uniform"), 2.0, 200.0) == (2.0, 200.0)
    assert evaluation.prior_bounds(PriorSpec("normal", (20.0, 2.0)), 2.0, 200.0) == (2.0, 200.0)
    assert evaluation.prior_bounds(PriorSpec("soft_range", (10.0, 30.0, 2.0)), 2.0, 200.0) == (2.0, 200.0)


def test_prior_center_and_spread_maps_each_kind() -> None:
    assert evaluation.prior_center_and_spread(PriorSpec("uniform")) is None
    assert evaluation.prior_center_and_spread(PriorSpec("normal", (20.0, 2.0))) == (20.0, 2.0)

    center, spread = evaluation.prior_center_and_spread(PriorSpec("lognormal", (log(20.0), 0.25)))

    assert (center, spread) == pytest.approx((20.0, 20.0 * 0.25))
    assert evaluation.prior_center_and_spread(PriorSpec("soft_range", (10.0, 30.0, 2.0))) == (20.0, 12.0)


def test_lognormal_center_mapping_handles_nonrepresentable_exp_consistently() -> None:
    spec = PriorSpec("lognormal", (710.0, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        center, spread = evaluation.prior_center_and_spread(spec)

    assert center == np.inf
    assert spread == np.inf
    assert not any(item.category is RuntimeWarning for item in caught)


def test_prior_center_and_spread_avoids_adjacent_extreme_midpoint_overflow() -> None:
    lower = 1e308
    upper = np.nextafter(lower, np.inf)

    center, spread = evaluation.prior_center_and_spread(PriorSpec("soft_range", (lower, upper, 1.0)))

    assert np.isfinite(center)
    assert lower <= center <= upper
    assert spread == pytest.approx((upper - lower) / 2.0 + 1.0)
