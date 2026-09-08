from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.unit.fit.noise_mode_cases import _problem, _unit

from xrr_fitter.evaluation import evaluate_model, least_squares_loss, least_squares_system
from xrr_fitter.fit.checkpoint import checkpoint_identity
from xrr_fitter.fit.stages import compile_coarse_problem
from xrr_fitter.model.data import with_fit_mask


def test_poisson_residual_derivative_uses_zero_count_and_equal_mean_limits() -> None:
    boundary = import_module("xrr_fitter.evaluation_statistics")
    baseline = _problem("poisson")
    counts = np.tile(np.array([0.0, 1.0, 1e6, 1e12]), 20)
    data = replace(baseline.data, intensity_raw=counts, intensity_normalized=counts / 100)
    problem = _problem("poisson", data)
    mu = np.where(counts == 0.0, 2.0, counts)
    residual = boundary.data_residuals(problem, mu / 100)
    derivative = boundary.data_residual_model_derivative(problem, mu / 100, residual)
    expected = 100 / np.sqrt(np.where(counts == 0.0, 2 * mu, mu))
    np.testing.assert_allclose(derivative, expected, rtol=1e-14)


@pytest.mark.parametrize("mode", ["gaussian", "poisson"])
def test_coarse_likelihood_preserves_sampling_mass_and_prior(mode: str) -> None:
    full = replace(_problem(mode, size=1200), scale_prior_center=1.0, scale_prior_reason=None)
    coarse = compile_coarse_problem(full)
    assert coarse.scale_prior_center == full.scale_prior_center
    assert coarse.objective_point_count == 1200
    assert np.sum(coarse.sampling_multipliers) == pytest.approx(1200)
    unit = _unit(coarse)
    residual, _jacobian = least_squares_system(coarse, unit)
    total = np.sum(least_squares_loss(coarse)(residual**2)[0]) / 2
    assert total == pytest.approx(evaluate_model(coarse, unit).objective * 1200)


def test_noise_model_change_changes_checkpoint_without_reenabling_user_points() -> None:
    full = _problem("poisson")
    mask = full.data.fit_mask.copy()
    mask[::7] = False
    selected = with_fit_mask(full.data, mask)
    gaussian = _problem("gaussian", selected)
    poisson = _problem("poisson", selected)
    np.testing.assert_array_equal(gaussian.data.fit_mask, poisson.data.fit_mask)
    assert checkpoint_identity(gaussian).config_fingerprint != checkpoint_identity(poisson).config_fingerprint
