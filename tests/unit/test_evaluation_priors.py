from __future__ import annotations

from tests.unit.evaluation_prior_cases import *


def test_problem_log_probability_is_bitwise_unchanged_without_priors() -> None:
    problem = _prior_problem()
    unit = np.full(len(problem.variables), 0.45)
    residual = evaluation.least_squares_residual(problem, unit)
    weights = problem.weights[problem.data.fit_mask]
    c = problem.config.c_decades
    baseline = -float(np.sum(weights**2 * 2.0 * c**2 * (np.sqrt(1.0 + (residual / c) ** 2) - 1.0))) / (2.0 * c**2)

    assert all(definition.prior is None for definition in problem.parameter_definitions)
    assert evaluation.problem_log_probability(problem, unit) == baseline


def test_physical_uniform_prior_includes_log_parameter_coordinate_jacobian() -> None:
    problem = _prior_problem()
    target = "component.0.thickness_a"
    definitions = tuple(
        replace(definition, prior=PriorSpec("uniform")) if definition.name == target else definition
        for definition in problem.parameter_definitions
    )
    problem = replace(problem, parameter_definitions=definitions)
    variable_index = next(index for index, variable in enumerate(problem.variables) if variable.name == target)
    definition = definitions[problem.variables[variable_index].parameter_index]
    unit = np.full(len(problem.variables), 0.5)

    actual = evaluation._parameter_prior_log_density(problem, unit)
    physical = evaluation.unit_to_physical(definition, float(unit[variable_index]))
    expected = evaluation.prior_log_density(
        definition.prior,
        physical,
        definition.lower,
        definition.upper,
    ) + np.log(physical * np.log(definition.upper / definition.lower))

    assert actual == pytest.approx(expected)


def test_prior_log_density_matches_closed_form() -> None:
    uniform = PriorSpec("uniform")
    normal = PriorSpec("normal", (20.0, 2.0))
    lognormal = PriorSpec("lognormal", (log(20.0), 0.25))
    soft_range = PriorSpec("soft_range", (10.0, 30.0, 2.0))

    def unnormalized(spec: PriorSpec, x: float) -> float:
        return evaluation.prior_log_density(spec, x, 2.0, 200.0) - evaluation.prior_log_density(spec, 20.0, 2.0, 200.0)

    assert unnormalized(uniform, 100.0) == pytest.approx(0.0)
    assert unnormalized(normal, 22.0) == pytest.approx(-0.5)
    assert unnormalized(normal, 16.0) == pytest.approx(-2.0)
    assert unnormalized(lognormal, 20.0 * np.exp(0.25)) == pytest.approx(-0.5 - 0.25)
    assert unnormalized(soft_range, 25.0) == pytest.approx(0.0)
    assert unnormalized(soft_range, 34.0) == pytest.approx(-2.0)


def test_prior_cdf_is_monotone_and_spans_zero_to_one() -> None:
    lower, upper = 2.0, 200.0
    grid = np.linspace(lower, upper, 257)
    for spec in (
        PriorSpec("uniform"),
        PriorSpec("normal", (20.0, 2.0)),
        PriorSpec("lognormal", (log(20.0), 0.25)),
        PriorSpec("soft_range", (10.0, 30.0, 2.0)),
    ):
        values = np.array([evaluation.prior_cdf(spec, float(x), lower, upper) for x in grid])

        assert np.all(np.diff(values) >= -1e-12)
        assert values[0] == pytest.approx(0.0, abs=1e-12)
        assert values[-1] == pytest.approx(1.0, abs=1e-9)


def test_prior_inverse_cdf_round_trips() -> None:
    lower, upper = 2.0, 200.0
    for spec in (
        PriorSpec("uniform"),
        PriorSpec("normal", (20.0, 2.0)),
        PriorSpec("lognormal", (log(20.0), 0.25)),
        PriorSpec("soft_range", (10.0, 30.0, 2.0)),
    ):
        # Sample inside the strictly increasing part of the cdf: a truncated
        # prior is flat far out in its tails, where the inverse is not unique.
        for level in (0.1, 0.25, 0.5, 0.75, 0.9):
            x = evaluation.prior_inverse_cdf(spec, level, lower, upper)

            assert evaluation.prior_cdf(spec, x, lower, upper) == pytest.approx(level, rel=1e-10)
            assert evaluation.prior_inverse_cdf(spec, evaluation.prior_cdf(spec, x, lower, upper), lower, upper) == (
                pytest.approx(x, rel=1e-10)
            )


def test_uniform_prior_handles_finite_cross_zero_bounds_with_overflowing_span() -> None:
    spec = PriorSpec("uniform")
    lower, upper = -1e308, 1e308

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density = evaluation.prior_log_density(spec, 0.0, lower, upper)
        probability = evaluation.prior_cdf(spec, 0.0, lower, upper)
        median = evaluation.prior_inverse_cdf(spec, 0.5, lower, upper)

    assert density == pytest.approx(-(np.log(2.0) + np.log(1e308)))
    assert probability == pytest.approx(0.5)
    assert median == pytest.approx(0.0)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_uniform_prior_handles_numpy_scalar_bounds_without_overflow_warning() -> None:
    spec = PriorSpec("uniform")
    lower = np.float64(-np.finfo(float).max)
    upper = np.float64(np.finfo(float).max)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density = evaluation.prior_log_density(spec, np.float64(0.0), lower, upper)
        probability = evaluation.prior_cdf(spec, np.float64(0.0), lower, upper)
        median = evaluation.prior_inverse_cdf(spec, np.float64(0.5), lower, upper)

    assert np.isfinite(density)
    assert probability == pytest.approx(0.5)
    assert median == pytest.approx(0.0)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_normal_prior_handles_finite_cross_zero_bounds_with_overflowing_span() -> None:
    spec = PriorSpec("normal", (0.0, 1.0))
    lower, upper = -1e308, 1e308

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density = evaluation.prior_log_density(spec, 0.0, lower, upper)
        probability = evaluation.prior_cdf(spec, 0.0, lower, upper)
        median = evaluation.prior_inverse_cdf(spec, 0.5, lower, upper)

    assert density == pytest.approx(-0.5 * np.log(2.0 * np.pi), rel=2e-4)
    assert probability == pytest.approx(0.5, abs=2e-4)
    assert median == pytest.approx(0.0, abs=2e-4)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_broad_normal_prior_normalizes_without_linear_mass_overflow() -> None:
    scale = np.finfo(float).max
    spec = PriorSpec("normal", (0.0, scale))
    lower, upper = -scale, scale

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density = evaluation.prior_log_density(spec, 0.0, lower, upper)
        probability = evaluation.prior_cdf(spec, 0.0, lower, upper)
        median = evaluation.prior_inverse_cdf(spec, 0.5, lower, upper)

    truncated_mass = sqrt(2.0 * np.pi) * erf(1.0 / sqrt(2.0))
    assert density == pytest.approx(-log(scale) - log(truncated_mass), rel=2e-4)
    assert probability == pytest.approx(0.5, abs=2e-4)
    assert abs(median) <= scale * 2e-4
    assert not any(item.category is RuntimeWarning for item in caught)


def test_lognormal_prior_handles_extreme_finite_positive_bounds() -> None:
    spec = PriorSpec("lognormal", (0.0, 1.0))
    lower, upper = 1e-308, 1e308

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density = evaluation.prior_log_density(spec, 1.0, lower, upper)
        probability = evaluation.prior_cdf(spec, 1.0, lower, upper)
        median = evaluation.prior_inverse_cdf(spec, 0.5, lower, upper)

    assert density == pytest.approx(-0.5 * np.log(2.0 * np.pi), rel=2e-4)
    assert probability == pytest.approx(0.5, abs=2e-4)
    assert median == pytest.approx(1.0, rel=2e-4)
    assert not any(item.category is RuntimeWarning for item in caught)
