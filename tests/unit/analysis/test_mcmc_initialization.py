from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.analysis import mcmc
from xrr_fitter.model.analysis import McmcConfig


def _narrow_target(center, sigma):
    def log_probability(_problem, value):
        if np.any((value < 0) | (value > 1)):
            return -np.inf
        return -0.5 * float(np.sum(((value - center) / sigma) ** 2))

    return log_probability


def test_initial_walkers_follow_local_width_even_with_a_wide_parameter_range(monkeypatch):
    center = np.array([0.5, 2e-7])
    sigma = np.array([1e-4, 1e-8])
    probability = _narrow_target(center, sigma)
    monkeypatch.setattr(mcmc, "problem_log_probability", probability)
    config = McmcConfig(walkers=32, burn_in=200, production_steps=500)

    walkers = mcmc._problem_walkers(object(), center, config, 11203671894797219712, None)

    assert min(probability(None, value) for value in walkers) > -8.0
    scaled = (walkers - center) / sigma
    assert np.all(np.std(scaled, axis=0) > 0.1)
    assert np.linalg.matrix_rank(scaled - scaled.mean(axis=0)) == center.size
    repeated = mcmc._problem_walkers(object(), center, config, 11203671894797219712, None)
    np.testing.assert_array_equal(walkers, repeated)
    np.testing.assert_array_equal(center, [0.5, 2e-7])


def test_local_initialization_preserves_the_sampled_target_distribution(monkeypatch):
    center = np.array([0.5, 2e-7])
    sigma = np.array([1e-4, 1e-8])
    probability = _narrow_target(center, sigma)
    monkeypatch.setattr(mcmc, "problem_log_probability", probability)
    config = McmcConfig(walkers=16, burn_in=200, production_steps=600)
    walkers = mcmc._problem_walkers(object(), center, config, 452, None)

    ensemble = mcmc.run_affine_invariant(lambda value: probability(None, value), walkers, config, 901)

    scaled = (ensemble.samples_unit - center) / sigma
    np.testing.assert_allclose(np.mean(scaled, axis=(0, 1)), [0, 0], atol=0.15)
    np.testing.assert_allclose(np.std(scaled, axis=(0, 1)), [1, 1], atol=0.15)
    assert np.all(ensemble.split_rhat < 1.10)


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_initialization_near_a_boundary_keeps_diverse_in_bounds_walkers(monkeypatch, boundary):
    center = np.array([boundary])
    monkeypatch.setattr(mcmc, "problem_log_probability", _narrow_target(center, np.array([1e-7])))
    walkers = mcmc._problem_walkers(object(), center, McmcConfig(8, 0, 4), 71, None)
    assert np.all((walkers >= 0) & (walkers <= 1))
    assert np.ptp(walkers) > 1e-8
    assert np.max(np.abs(walkers - center)) < 5e-7


def test_initialization_rejects_a_center_outside_the_probability_support(monkeypatch):
    monkeypatch.setattr(mcmc, "problem_log_probability", lambda *_args: -np.inf)
    with pytest.raises(ValueError, match="center.*finite"):
        mcmc._problem_walkers(object(), np.array([0.5]), McmcConfig(4, 0, 4), 7, None)


def test_local_scale_probe_honors_cancellation(monkeypatch):
    monkeypatch.setattr(mcmc, "problem_log_probability", _narrow_target(np.array([0.5]), np.array([1e-15])))
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(InterruptedError, match="cancelled"):
        mcmc._problem_walkers(object(), np.array([0.5]), McmcConfig(4, 0, 4), 7, cancelled)
    assert calls == 3


def test_unrepresentably_narrow_target_cannot_return_coincident_walkers(monkeypatch):
    center = np.array([0.5])
    probability = _narrow_target(center, np.array([1e-16]))
    monkeypatch.setattr(mcmc, "problem_log_probability", probability)

    with pytest.raises(RuntimeError, match="independent MCMC walkers"):
        mcmc._problem_walkers(object(), center, McmcConfig(8, 0, 4), 71, None)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_initialization_on_an_internal_support_boundary_keeps_diverse_valid_walkers(monkeypatch, direction):
    center = np.array([0.5])
    gaussian = _narrow_target(center, np.array([0.1]))

    def constrained_probability(problem, value):
        if direction * (value[0] - center[0]) < 0:
            return -np.inf
        return gaussian(problem, value)

    monkeypatch.setattr(mcmc, "problem_log_probability", constrained_probability)
    walkers = mcmc._problem_walkers(object(), center, McmcConfig(8, 0, 4), 71, None)

    assert np.all(direction * (walkers - center) >= 0)
    assert np.ptp(walkers) > 0.001
    assert all(np.isfinite(constrained_probability(None, walker)) for walker in walkers)


def test_initialization_enters_a_coupled_constraint_without_valid_coordinate_directions(monkeypatch):
    center = np.array([0.5, 0.5, 2e-7])
    sigma = np.array([0.1, 0.1, 1e-8])
    gaussian = _narrow_target(center, sigma)

    def constrained_probability(problem, value):
        dx, dy = (value - center)[:2]
        if not 0.5 * dx <= dy <= dx:
            return -np.inf
        return gaussian(problem, value)

    monkeypatch.setattr(mcmc, "problem_log_probability", constrained_probability)
    config = McmcConfig(16, 0, 4)
    walkers = mcmc._problem_walkers(object(), center, config, 71, None)

    assert np.all(np.ptp(walkers[:, :2], axis=0) > 0.001)
    assert all(np.isfinite(constrained_probability(None, walker)) for walker in walkers)
    assert np.max(np.abs(walkers[:, 2] - center[2])) < 5e-8
    scaled = (walkers - center) / sigma
    assert np.linalg.matrix_rank(scaled - scaled.mean(axis=0)) == center.size
    repeated = mcmc._problem_walkers(object(), center, config, 71, None)
    np.testing.assert_array_equal(walkers, repeated)
