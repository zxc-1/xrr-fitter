"""Adaptive Gauss-Hermite instrument resolution."""

from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from xrr_fitter.model.instrument import PhysicsDiagnostic


def _gauss_rules(orders: tuple[int, ...]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {order: np.polynomial.hermite.hermgauss(order) for order in orders}


ORDERS = (17, 33, 65)
RULES = _gauss_rules(ORDERS)
MAX_QUERY_VALUES = 4096


class GaussHermiteConvergenceWarning(UserWarning):
    """The retained 65-point result did not meet the adaptive tolerance."""


def gh_converged(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    return np.abs(fine - coarse) <= np.maximum(1e-12, 1e-4 * np.abs(fine))


def _nonnegative(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _nonnegative_scalar(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def gauss_hermite_values(
    samples: np.ndarray,
    widths: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
    order: int,
) -> np.ndarray:
    nodes, weights = RULES[order]
    result = np.empty(samples.size, dtype=float)
    chunk_size = max(1, MAX_QUERY_VALUES // order)
    for start in range(0, samples.size, chunk_size):
        stop = min(start + chunk_size, samples.size)
        query = samples[start:stop, None] + np.sqrt(2.0) * widths[start:stop, None] * nodes
        keep = query >= 0.0
        safe_query = np.where(keep, query, 0.0)
        values = np.asarray(function(safe_query), dtype=float)
        if values.shape != safe_query.shape or np.any(~np.isfinite(values)):
            raise ValueError("ideal_reflectivity must return finite values with query shape")
        retained = weights * keep
        normalizer = retained.sum(axis=1)
        if np.any(normalizer <= 0.0):
            raise FloatingPointError("resolution kernel has no nonnegative q samples")
        result[start:stop] = np.sum(values * retained, axis=1) / normalizer
    return result


def _adaptive_values(
    samples: np.ndarray,
    widths: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    values_17 = gauss_hermite_values(samples, widths, function, 17)
    values_33 = gauss_hermite_values(samples, widths, function, 33)
    accepted = values_33.copy()
    needs_65 = ~gh_converged(values_17, values_33)
    unresolved = np.empty(0, dtype=int)
    if np.any(needs_65):
        values_65 = gauss_hermite_values(samples[needs_65], widths[needs_65], function, 65)
        accepted[needs_65] = values_65
        selected = np.flatnonzero(needs_65)
        unresolved = selected[~gh_converged(values_33[needs_65], values_65)]
    return accepted, unresolved


def _emit_unconverged(indices: tuple[int, ...], callback: Callable[[PhysicsDiagnostic], None] | None) -> None:
    message = f"Gauss-Hermite quadrature did not converge at the 65-point ceiling for {len(indices)} point(s)"
    warnings.warn(message + "; using the 65-point result", GaussHermiteConvergenceWarning, stacklevel=3)
    if callback is not None:
        callback(PhysicsDiagnostic("gauss_hermite_unconverged", message, indices))


def smear_with_widths(
    samples: np.ndarray,
    widths: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
    diagnostic_callback: Callable[[PhysicsDiagnostic], None] | None = None,
) -> np.ndarray:
    flat_samples = samples.ravel()
    flat_widths = widths.ravel()
    if np.all(flat_widths == 0.0):
        exact = np.asarray(function(flat_samples), dtype=float)
        if exact.shape != flat_samples.shape or np.any(~np.isfinite(exact)):
            raise ValueError("ideal_reflectivity must return finite values with query shape")
        return np.array(exact, copy=True).reshape(samples.shape)
    result = np.empty_like(flat_samples)
    zero = flat_widths == 0.0
    if np.any(zero):
        exact = np.asarray(function(flat_samples[zero]), dtype=float)
        if exact.shape != flat_samples[zero].shape or np.any(~np.isfinite(exact)):
            raise ValueError("ideal_reflectivity must return finite values with query shape")
        result[zero] = exact
    if np.any(~zero):
        accepted, unresolved = _adaptive_values(flat_samples[~zero], flat_widths[~zero], function)
        result[~zero] = accepted
        if unresolved.size:
            selected = np.flatnonzero(~zero)
            _emit_unconverged(tuple(int(value) for value in selected[unresolved]), diagnostic_callback)
    return result.reshape(samples.shape)


def gaussian_smear(
    qz_a_inv: np.ndarray,
    ideal_reflectivity: Callable[[np.ndarray], np.ndarray],
    relative_sigma: float = 0.0,
    absolute_sigma_a_inv: float = 0.0,
    sigma_q_a_inv: np.ndarray | None = None,
    diagnostic_callback: Callable[[PhysicsDiagnostic], None] | None = None,
) -> np.ndarray:
    qz = _nonnegative(qz_a_inv, "qz_a_inv")
    _nonnegative_scalar(relative_sigma, "relative_sigma")
    _nonnegative_scalar(absolute_sigma_a_inv, "absolute_sigma_a_inv")
    point_width = np.zeros_like(qz) if sigma_q_a_inv is None else _nonnegative(sigma_q_a_inv, "sigma_q_a_inv")
    if point_width.shape != qz.shape:
        raise ValueError("sigma_q_a_inv must match qz_a_inv")
    widths = np.sqrt((relative_sigma * qz) ** 2 + absolute_sigma_a_inv**2 + point_width**2)
    return smear_with_widths(qz, widths, ideal_reflectivity, diagnostic_callback)


def theta_domain_smear(
    theta_deg: np.ndarray,
    ideal_reflectivity: Callable[[np.ndarray], np.ndarray],
    sigma_theta_deg: float = 0.0,
    diagnostic_callback: Callable[[PhysicsDiagnostic], None] | None = None,
) -> np.ndarray:
    theta = _nonnegative(theta_deg, "theta_deg")
    _nonnegative_scalar(sigma_theta_deg, "sigma_theta_deg")
    return smear_with_widths(theta, np.full_like(theta, sigma_theta_deg), ideal_reflectivity, diagnostic_callback)
