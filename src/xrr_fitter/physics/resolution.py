"""Adaptive Gauss-Hermite instrument resolution."""

from __future__ import annotations

import warnings
from collections.abc import Callable

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


def _gauss_hermite_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average over the node axis without overflowing a finite result."""
    weight_shape = (1, weights.size) + (1,) * (values.ndim - 2)
    broadcast_weights = weights.reshape(weight_shape)
    normalizer = np.sum(weights)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.sum(values * broadcast_weights, axis=1) / normalizer
    if np.all(np.isfinite(result)):
        return result
    scale = np.max(np.abs(values), axis=1)
    expanded_scale = np.expand_dims(scale, axis=1)
    scaled = np.divide(
        values,
        expanded_scale,
        out=np.zeros_like(values),
        where=expanded_scale != 0.0,
    )
    normalized_weights = (weights / normalizer).reshape(weight_shape)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = scale * np.sum(scaled * normalized_weights, axis=1)
    if np.any(~np.isfinite(result)):
        raise ValueError("Gauss-Hermite weighted average must be finite")
    return result


def _validated_gauss_inputs(
    samples: np.ndarray,
    widths: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or int(order) not in RULES:
        raise ValueError(f"unsupported Gauss-Hermite order: {order}")
    samples = np.asarray(samples, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if samples.ndim != 1 or widths.ndim != 1:
        raise ValueError("gauss-hermite samples and widths must be one-dimensional arrays")
    if widths.shape != samples.shape:
        raise ValueError("gauss-hermite samples and widths must have the same shape")
    if np.any(~np.isfinite(samples)):
        raise ValueError("gauss-hermite samples must be finite")
    return samples, _nonnegative(widths, "gauss-hermite widths"), int(order)


def _evaluated_values(
    function: Callable[[np.ndarray], np.ndarray],
    query: np.ndarray,
) -> np.ndarray:
    values = np.asarray(function(query), dtype=float)
    if values.shape != query.shape or np.any(~np.isfinite(values)):
        raise ValueError("ideal_reflectivity must return finite values with query shape")
    return values


def _gauss_chunk(
    samples: np.ndarray,
    widths: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        query = samples[:, None] + np.sqrt(2.0) * widths[:, None] * nodes
    if np.any(~np.isfinite(query)):
        raise ValueError("resolution query must be finite")
    reflected_query = np.abs(query)
    return _gauss_hermite_average(_evaluated_values(function, reflected_query), weights)


def gauss_hermite_values(
    samples: np.ndarray,
    widths: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
    order: int,
) -> np.ndarray:
    samples, widths, order = _validated_gauss_inputs(samples, widths, order)
    nodes, weights = RULES[order]
    result = np.empty(samples.size, dtype=float)
    chunk_size = max(1, MAX_QUERY_VALUES // order)
    for start in range(0, samples.size, chunk_size):
        stop = min(start + chunk_size, samples.size)
        result[start:stop] = _gauss_chunk(
            samples[start:stop],
            widths[start:stop],
            nodes,
            weights,
            function,
        )
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


def _emit_unconverged(
    indices: tuple[int, ...],
    callback: Callable[[PhysicsDiagnostic], None] | None,
    *,
    emit_warning: bool,
) -> None:
    message = f"Gauss-Hermite quadrature did not converge at the 65-point ceiling for {len(indices)} point(s)"
    if emit_warning:
        warnings.warn(message + "; using the 65-point result", GaussHermiteConvergenceWarning, stacklevel=4)
    if callback is not None:
        callback(PhysicsDiagnostic("gauss_hermite_unconverged", message, indices))


def _validated_smearing_inputs(
    samples: np.ndarray,
    widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if samples.shape != widths.shape:
        raise ValueError("smearing samples and widths must have the same shape")
    if np.any(~np.isfinite(samples)):
        raise ValueError("smearing samples must be finite")
    return samples, _nonnegative(widths, "smearing widths")


def _smear_nonzero_widths(
    samples: np.ndarray,
    widths: np.ndarray,
    point_indices: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
    callback: Callable[[PhysicsDiagnostic], None] | None,
    *,
    emit_warning: bool,
) -> np.ndarray:
    accepted, unresolved = _adaptive_values(samples, widths, function)
    if unresolved.size:
        _emit_unconverged(
            tuple(int(point_indices[index]) for index in unresolved),
            callback,
            emit_warning=emit_warning,
        )
    return accepted


def smear_with_widths(
    samples: np.ndarray,
    widths: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
    diagnostic_callback: Callable[[PhysicsDiagnostic], None] | None = None,
    *,
    emit_warning: bool = True,
) -> np.ndarray:
    samples, widths = _validated_smearing_inputs(samples, widths)
    flat_samples = samples.ravel()
    flat_widths = widths.ravel()
    if np.all(flat_widths == 0.0):
        exact = _evaluated_values(function, flat_samples)
        return np.array(exact, copy=True).reshape(samples.shape)
    result = np.empty_like(flat_samples)
    zero = flat_widths == 0.0
    if np.any(zero):
        result[zero] = _evaluated_values(function, flat_samples[zero])
    if np.any(~zero):
        selected = np.flatnonzero(~zero)
        result[~zero] = _smear_nonzero_widths(
            flat_samples[~zero],
            flat_widths[~zero],
            selected,
            function,
            diagnostic_callback,
            emit_warning=emit_warning,
        )
    return result.reshape(samples.shape)


def gaussian_smear(
    qz_a_inv: np.ndarray,
    ideal_reflectivity: Callable[[np.ndarray], np.ndarray],
    relative_sigma: float = 0.0,
    absolute_sigma_a_inv: float = 0.0,
    sigma_q_a_inv: np.ndarray | None = None,
    diagnostic_callback: Callable[[PhysicsDiagnostic], None] | None = None,
    *,
    emit_warning: bool = True,
) -> np.ndarray:
    qz = _nonnegative(qz_a_inv, "qz_a_inv")
    _nonnegative_scalar(relative_sigma, "relative_sigma")
    _nonnegative_scalar(absolute_sigma_a_inv, "absolute_sigma_a_inv")
    point_width = np.zeros_like(qz) if sigma_q_a_inv is None else _nonnegative(sigma_q_a_inv, "sigma_q_a_inv")
    if point_width.shape != qz.shape:
        raise ValueError("sigma_q_a_inv must match qz_a_inv")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        relative_width = relative_sigma * qz
        widths = np.sqrt(relative_width**2 + absolute_sigma_a_inv**2 + point_width**2)
    overflowed = ~np.isfinite(widths)
    if np.any(overflowed):
        stable = np.hypot(np.hypot(relative_width[overflowed], absolute_sigma_a_inv), point_width[overflowed])
        widths[overflowed] = stable
    return smear_with_widths(
        qz,
        widths,
        ideal_reflectivity,
        diagnostic_callback,
        emit_warning=emit_warning,
    )


def theta_domain_smear(
    theta_deg: np.ndarray,
    ideal_reflectivity: Callable[[np.ndarray], np.ndarray],
    sigma_theta_deg: float = 0.0,
    diagnostic_callback: Callable[[PhysicsDiagnostic], None] | None = None,
    *,
    emit_warning: bool = True,
) -> np.ndarray:
    theta = _nonnegative(theta_deg, "theta_deg")
    _nonnegative_scalar(sigma_theta_deg, "sigma_theta_deg")
    return smear_with_widths(
        theta,
        np.full_like(theta, sigma_theta_deg),
        ideal_reflectivity,
        diagnostic_callback,
        emit_warning=emit_warning,
    )
