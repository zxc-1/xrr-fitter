from __future__ import annotations

from tests.unit.evaluation_prior_cases import *


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
