from __future__ import annotations

import warnings

import numpy as np
import pytest

from xrr_fitter.physics.resolution import GaussHermiteConvergenceWarning, gaussian_smear, theta_domain_smear


def _manual(samples: np.ndarray, width: float, function, order: int) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    query = samples[:, None] + np.sqrt(2) * width * nodes
    keep = query >= 0
    retained = weights * keep
    return np.sum(function(np.where(keep, query, 0)) * retained, axis=1) / np.sum(retained, axis=1)


def test_zero_resolution_returns_exact_samples() -> None:
    q = np.linspace(0.01, 0.4, 101)
    np.testing.assert_array_equal(gaussian_smear(q, lambda x: np.exp(-x)), np.exp(-q))


def test_smooth_function_converges_at_first_escalation_level() -> None:
    q = np.linspace(0.05, 0.4, 50)
    actual = gaussian_smear(q, lambda x: x**2, absolute_sigma_a_inv=0.002)
    np.testing.assert_allclose(actual, _manual(q, 0.002, lambda x: x**2, 33), rtol=1e-13, atol=1e-15)


def test_gauss_hermite_escalates_to_sixty_five_points_when_seventeen_is_not_converged() -> None:
    q = np.array([0.02])
    spike = lambda x: np.exp(-43150 * (x - 0.0336) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GaussHermiteConvergenceWarning)
        actual = gaussian_smear(q, spike, absolute_sigma_a_inv=0.012)
    np.testing.assert_allclose(actual, _manual(q, 0.012, spike, 65), rtol=1e-13, atol=1e-15)


def test_unconverged_sixty_five_point_result_records_warning() -> None:
    q = np.array([0.02])
    needle = lambda x: np.exp(-4e6 * (x - 0.021) ** 2)
    with pytest.warns(GaussHermiteConvergenceWarning, match="65"):
        actual = gaussian_smear(q, needle, absolute_sigma_a_inv=0.012)
    np.testing.assert_allclose(actual, _manual(q, 0.012, needle, 65), rtol=1e-13, atol=1e-15)


def test_adaptive_quadrature_evaluates_sixty_five_nodes_only_for_unresolved_points() -> None:
    calls: list[tuple[int, ...]] = []
    def needle(query: np.ndarray) -> np.ndarray:
        calls.append(query.shape)
        return np.exp(-4e6 * (query - 0.02137) ** 2)
    with pytest.warns(GaussHermiteConvergenceWarning):
        gaussian_smear(np.array([0.02, 0.2]), needle, absolute_sigma_a_inv=0.011)
    assert calls == [(2, 17), (2, 33), (1, 65)]


def test_unconverged_quadrature_reports_structured_point_indices() -> None:
    diagnostics = []
    with pytest.warns(GaussHermiteConvergenceWarning):
        gaussian_smear(np.array([0.02, 0.2]), lambda x: np.exp(-4e6 * (x - 0.02137) ** 2), absolute_sigma_a_inv=0.011, diagnostic_callback=diagnostics.append)
    assert diagnostics[0].code == "gauss_hermite_unconverged"
    assert diagnostics[0].point_indices == (0,)


def test_point_resolution_combines_in_quadrature_and_preserves_zero_width_points() -> None:
    q = np.array([0.05, 0.1, 0.2])
    points = np.array([0, 0.002, 0.004])
    actual = gaussian_smear(q, lambda x: x**2, relative_sigma=0.01, absolute_sigma_a_inv=0.001, sigma_q_a_inv=points)
    variance = (0.01 * q) ** 2 + 0.001**2 + points**2
    np.testing.assert_allclose(actual, q**2 + variance, rtol=2e-4, atol=1e-10)


def test_point_resolution_requires_matching_shape() -> None:
    with pytest.raises(ValueError, match="sigma_q_a_inv must match qz_a_inv"):
        gaussian_smear(np.array([0.1, 0.2]), lambda x: x, sigma_q_a_inv=np.array([0.001]))


def test_negative_gauss_queries_are_zero_weighted_and_retained_weights_renormalized() -> None:
    actual = gaussian_smear(
        np.array([0.001]),
        lambda query: np.where(query == 0.0, 1e12, 1.0),
        absolute_sigma_a_inv=0.02,
    )
    np.testing.assert_allclose(actual, 1.0, rtol=0, atol=1e-14)


def test_quadrature_chunks_large_query_grids_without_changing_values() -> None:
    q = np.linspace(0.02, 0.4, 600)
    calls: list[tuple[int, ...]] = []
    def smooth(query: np.ndarray) -> np.ndarray:
        calls.append(query.shape)
        return query**2
    np.testing.assert_allclose(gaussian_smear(q, smooth, absolute_sigma_a_inv=0.002), q**2 + 0.002**2, rtol=2e-13, atol=2e-15)
    assert max(np.prod(shape) for shape in calls) <= 1024


def test_theta_domain_smear_leaves_constant_function_unchanged() -> None:
    theta = np.linspace(0.05, 2, 50)
    np.testing.assert_allclose(theta_domain_smear(theta, lambda x: np.ones_like(x), 0.01), 1)


def test_theta_domain_zero_sigma_returns_exact_samples() -> None:
    theta = np.linspace(0.05, 2, 80)
    np.testing.assert_array_equal(theta_domain_smear(theta, lambda x: np.exp(-x), 0), np.exp(-theta))


def test_theta_domain_agrees_with_q_domain_under_small_angle_conversion() -> None:
    wavelength = 1.5406
    theta = np.linspace(0.3, 1.2, 60)
    sigma_theta = 0.005
    model_q = lambda q: 1 / (1 + (180 * q) ** 2)
    model_theta = lambda angle: model_q(4 * np.pi * np.sin(np.deg2rad(angle)) / wavelength)
    theta_smeared = theta_domain_smear(theta, model_theta, sigma_theta)
    q = 4 * np.pi * np.sin(np.deg2rad(theta)) / wavelength
    sigma_q = (4 * np.pi / wavelength) * np.cos(np.deg2rad(theta)) * np.deg2rad(sigma_theta)
    np.testing.assert_allclose(theta_smeared, gaussian_smear(q, model_q, sigma_q_a_inv=sigma_q), rtol=1e-3)


def test_theta_domain_smear_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sigma_theta_deg"):
        theta_domain_smear(np.array([0.5]), lambda x: x, -0.01)
    with pytest.raises(ValueError, match="theta_deg"):
        theta_domain_smear(np.array([-0.5]), lambda x: x, 0.01)


def test_theta_domain_unconverged_diagnostic_indices_match_theta_points() -> None:
    diagnostics = []
    with pytest.warns(GaussHermiteConvergenceWarning):
        theta_domain_smear(np.array([0.02, 0.2]), lambda x: np.exp(-4e6 * (x - 0.02137) ** 2), 0.011, diagnostic_callback=diagnostics.append)
    assert diagnostics[0].code == "gauss_hermite_unconverged"
    assert diagnostics[0].point_indices == (0,)
