from __future__ import annotations

from tests.unit.analysis.mcmc_cases import *


def test_affine_sampler_recovers_bounded_gaussian_and_is_deterministic() -> None:
    mean = np.asarray([0.45, 0.60])
    covariance = np.asarray([[0.010, 0.004], [0.004, 0.020]])
    precision = np.linalg.inv(covariance)

    def log_probability(unit: np.ndarray) -> float:
        if np.any((unit < 0.0) | (unit > 1.0)):
            return -np.inf
        delta = unit - mean
        return float(-0.5 * delta @ precision @ delta)

    initial = np.random.default_rng(991).multivariate_normal(mean, covariance * 0.1, size=16)
    config = McmcConfig(walkers=16, burn_in=200, production_steps=500, thin=2)
    first = run_affine_invariant(log_probability, initial, config, child_seed=8128)
    second = run_affine_invariant(log_probability, initial, config, child_seed=8128)

    np.testing.assert_array_equal(first.samples_unit, second.samples_unit)
    np.testing.assert_array_equal(first.log_probability, second.log_probability)
    np.testing.assert_allclose(first.samples_unit.mean(axis=(0, 1)), mean, atol=0.05)
    assert np.all((first.samples_unit >= 0.0) & (first.samples_unit <= 1.0))


def test_mcmc_config_rejects_invalid_ensemble_shape() -> None:
    with pytest.raises(ValueError, match=r"walkers must be even and at least 2\*nfree\+2"):
        run_affine_invariant(
            lambda value: -float(value @ value),
            np.full((4, 2), 0.5),
            McmcConfig(walkers=4, burn_in=2, production_steps=4),
            child_seed=1,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        pytest.param(
            {"burn_in": -1},
            "invalid MCMC step configuration",
            id="negative-burn-in",
        ),
        pytest.param(
            {"production_steps": 1},
            "invalid MCMC step configuration",
            id="too-few-production-steps",
        ),
        pytest.param(
            {"thin": 0},
            "invalid MCMC step configuration",
            id="zero-thin",
        ),
        pytest.param(
            {"thin": 4},
            "invalid MCMC step configuration",
            id="thin-exceeds-production",
        ),
        pytest.param(
            {"stretch_scale": 1.0},
            r"stretch_scale must be in \(1,10\]",
            id="stretch-scale-lower-bound",
        ),
        pytest.param(
            {"stretch_scale": np.inf},
            r"stretch_scale must be in \(1,10\]",
            id="nonfinite-stretch-scale",
        ),
    ),
)
def test_affine_sampler_rejects_invalid_step_and_stretch_configuration(
    overrides: dict[str, float | int], message: str
) -> None:
    values: dict[str, float | int] = {
        "walkers": 6,
        "burn_in": 2,
        "production_steps": 4,
        "thin": 1,
        "stretch_scale": 2.0,
    }
    values.update(overrides)
    config = SimpleNamespace(**values)

    with pytest.raises(ValueError, match=message):
        run_affine_invariant(
            lambda value: -float(value @ value),
            np.full((6, 2), 0.5),
            config,
            child_seed=1,
        )


@pytest.mark.parametrize(
    "invalid_log_probability",
    (
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(np.nan, id="nan"),
    ),
)
def test_affine_sampler_rejects_nonfinite_initial_log_probability(
    invalid_log_probability: float,
) -> None:
    with pytest.raises(ValueError, match="initial walkers must have finite log probability"):
        run_affine_invariant(
            lambda _value: invalid_log_probability,
            np.full((6, 2), 0.5),
            McmcConfig(walkers=6, burn_in=2, production_steps=4),
            child_seed=1,
        )


@pytest.mark.parametrize(
    "proposal_log_probability",
    (
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(np.nan, id="nan"),
    ),
)
def test_affine_sampler_rejects_all_nonfinite_proposal_log_probabilities(
    proposal_log_probability: float,
) -> None:
    initial = np.full((6, 2), 0.5)
    calls = 0

    def log_probability(_value: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= initial.shape[0] else proposal_log_probability

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_affine_invariant(
            log_probability,
            initial,
            McmcConfig(walkers=6, burn_in=0, production_steps=4),
            child_seed=712,
        )

    expected = np.broadcast_to(initial, result.samples_unit.shape)
    np.testing.assert_array_equal(result.samples_unit, expected)
    np.testing.assert_array_equal(result.log_probability, np.zeros((4, 6)))
    np.testing.assert_array_equal(result.acceptance_fraction, np.zeros(6))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_affine_sampler_uses_warning_free_arithmetic_for_extreme_finite_log_probabilities() -> None:
    maximum = np.finfo(float).max
    initial = np.full((6, 2), 0.5)
    calls = 0

    def log_probability(_value: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return -maximum if calls <= initial.shape[0] else maximum

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_affine_invariant(
            log_probability,
            initial,
            McmcConfig(walkers=6, burn_in=0, production_steps=4),
            child_seed=713,
        )

    assert np.all(np.isfinite(result.log_probability))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_mcmc_split_rhat_and_ess_match_independent_numpy_calculations() -> None:
    module = _api()
    rng = np.random.default_rng(314)
    draws, walkers, dimension = 64, 6, 2
    innovations = rng.normal(size=(draws, walkers, dimension))
    samples = np.empty_like(innovations)
    samples[0] = innovations[0]
    for draw in range(1, draws):
        samples[draw] = 0.65 * samples[draw - 1] + innovations[draw]

    half = draws // 2
    chains = np.concatenate(
        (np.transpose(samples[:half], (1, 0, 2)), np.transpose(samples[-half:], (1, 0, 2))),
        axis=0,
    )
    within = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(chains, axis=1), axis=0, ddof=1)
    expected_rhat = np.sqrt((((half - 1.0) / half) * within + between / half) / within)
    expected_ess = np.empty(dimension)
    by_walker = np.transpose(samples, (1, 0, 2))
    for parameter_index in range(dimension):
        correlations = []
        for lag in range(draws):
            values = []
            for walker in range(walkers):
                series = by_walker[walker, :, parameter_index]
                centered = series - np.mean(series)
                variance = np.dot(centered, centered) / draws
                covariance = np.dot(centered[: draws - lag], centered[lag:]) / (draws - lag)
                values.append(covariance / variance)
            correlations.append(float(np.mean(values)))
        positive_sum = 0.0
        for lag in range(1, draws, 2):
            pair = correlations[lag] + (correlations[lag + 1] if lag + 1 < draws else 0.0)
            if pair <= 0.0:
                break
            positive_sum += pair
        expected_ess[parameter_index] = np.clip(
            draws * walkers / max(1.0, 1.0 + 2.0 * positive_sum),
            1.0,
            draws * walkers,
        )

    np.testing.assert_allclose(module.split_rhat(samples), expected_rhat, atol=1e-13)
    np.testing.assert_allclose(module.effective_sample_size(samples), expected_ess, atol=1e-11)
