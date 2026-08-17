from __future__ import annotations

from dataclasses import replace
from math import log

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import PriorSpec, _prior_center


def _prior_problem() -> FitEvaluationContext:
    return compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=19), scale_prior_enabled=False),
    )


def test_problem_log_probability_is_bitwise_unchanged_without_priors() -> None:
    problem = _prior_problem()
    unit = np.full(len(problem.variables), 0.45)
    residual = evaluation.least_squares_residual(problem, unit)
    weights = problem.weights[problem.data.fit_mask]
    c = problem.config.c_decades
    baseline = -float(np.sum(weights**2 * 2.0 * c**2 * (np.sqrt(1.0 + (residual / c) ** 2) - 1.0))) / (2.0 * c**2)

    assert all(definition.prior is None for definition in problem.parameter_definitions)
    assert evaluation.problem_log_probability(problem, unit) == baseline


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


def test_prior_center_matches_the_model_side_lightweight_mapping() -> None:
    for spec in (
        PriorSpec("uniform"),
        PriorSpec("normal", (20.0, 2.0)),
        PriorSpec("lognormal", (log(20.0), 0.25)),
        PriorSpec("soft_range", (10.0, 30.0, 2.0)),
    ):
        expected = evaluation.prior_center_and_spread(spec)

        assert _prior_center(spec) == (None if expected is None else pytest.approx(expected[0]))


def test_prior_log_density_normalization_constant_is_cached() -> None:
    spec = PriorSpec("normal", (20.0, 2.0))
    evaluation.prior_log_density(spec, 20.0, 2.0, 200.0)
    before = evaluation._prior_norm.cache_info()
    for _ in range(8):
        evaluation.prior_log_density(spec, 21.0, 2.0, 200.0)
    after = evaluation._prior_norm.cache_info()

    assert after.hits > before.hits
    assert after.misses == before.misses


def test_extremely_narrow_normal_prior_retains_finite_mass_and_quantiles() -> None:
    spec = PriorSpec("normal", (100.03, 1e-4))

    density = evaluation.prior_log_density(spec, 100.03, 0.0, 200.0)
    median = evaluation.prior_inverse_cdf(spec, 0.5, 0.0, 200.0)

    assert np.isfinite(density)
    assert median == pytest.approx(100.03, abs=1e-7)
    assert evaluation.prior_cdf(spec, median, 0.0, 200.0) == pytest.approx(0.5)


def test_extremely_narrow_lognormal_prior_retains_finite_mass_and_quantiles() -> None:
    center = 75.07
    spec = PriorSpec("lognormal", (log(center), 1e-5))

    density = evaluation.prior_log_density(spec, center, 1.0, 200.0)
    median = evaluation.prior_inverse_cdf(spec, 0.5, 1.0, 200.0)

    assert np.isfinite(density)
    assert median == pytest.approx(center, rel=1e-7)
    assert evaluation.prior_cdf(spec, median, 1.0, 200.0) == pytest.approx(0.5)
