"""Internal objective implementation."""

from __future__ import annotations

from math import isfinite, log

import numpy as np

from xrr_fitter.evaluation_geometry import (
    _active_upper_tangent as _active_upper_tangent,
)
from xrr_fitter.evaluation_geometry import (
    _add_latent_roughness_cap_jacobians as _add_latent_roughness_cap_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _add_latent_roughness_caps as _add_latent_roughness_caps,
)
from xrr_fitter.evaluation_geometry import (
    _allowed_missing_roughness_names as _allowed_missing_roughness_names,
)
from xrr_fitter.evaluation_geometry import (
    _definition_name as _definition_name,
)
from xrr_fitter.evaluation_geometry import (
    _expand_structure_with_jacobian as _expand_structure_with_jacobian,
)
from xrr_fitter.evaluation_geometry import (
    _fill_missing_roughness_caps as _fill_missing_roughness_caps,
)
from xrr_fitter.evaluation_geometry import (
    _fill_missing_roughness_jacobians as _fill_missing_roughness_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _gradient_slab_counts as _gradient_slab_counts,
)
from xrr_fitter.evaluation_geometry import (
    _interface_neighbor_indices as _interface_neighbor_indices,
)
from xrr_fitter.evaluation_geometry import (
    _interface_upper as _interface_upper,
)
from xrr_fitter.evaluation_geometry import (
    _is_public_interface as _is_public_interface,
)
from xrr_fitter.evaluation_geometry import (
    _is_roughness_definition as _is_roughness_definition,
)
from xrr_fitter.evaluation_geometry import (
    _latent_periodic_roughness_names as _latent_periodic_roughness_names,
)
from xrr_fitter.evaluation_geometry import (
    _record_active_upper as _record_active_upper,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_definition_map as _roughness_definition_map,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_definitions as _roughness_definitions,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_dynamic_upper_jacobians as _roughness_dynamic_upper_jacobians,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_dynamic_uppers as _roughness_dynamic_uppers,
)
from xrr_fitter.evaluation_geometry import (
    _roughness_geometry_context as _roughness_geometry_context,
)
from xrr_fitter.evaluation_geometry import (
    _zero_roughness_values as _zero_roughness_values,
)
from xrr_fitter.model.parameters import (
    _log10_ratio,
)


def _float_vectors(
    first: np.ndarray,
    second: np.ndarray,
    message: str,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if left.shape != right.shape:
        raise ValueError(message)
    return left, right


def region_weights(labels: np.ndarray) -> np.ndarray:
    """Give every present region equal total squared residual weight.

    Region cardinalities may differ, so each point receives the square root of
    the inverse regional mass. This makes the squared weights of every region
    sum to the same value while preserving the full fitted-point count.
    """
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("region labels must be a finite integer vector")
    valid = all(
        (
            np.all(np.isfinite(values)),
            np.all(values == np.floor(values)),
        )
    )
    if not valid:
        raise ValueError("region labels must be a finite integer vector")
    integer = values.astype(int)
    present = np.unique(integer)
    weights = np.empty(integer.size, dtype=float)
    # Squared point weights, not point weights themselves, define residual mass.
    # Multiplying by the global fitted count keeps the all-region sum unchanged.
    # The normalization therefore remains comparable to an ordinary mean loss.
    for label in present:
        selected = integer == label
        weights[selected] = np.sqrt(integer.size / (present.size * selected.sum()))
    return weights


def _validated_qz(qz_a_inv: np.ndarray) -> np.ndarray:
    qz = np.asarray(qz_a_inv, dtype=float)
    valid = all((qz.ndim == 1, qz.size > 0, np.all(np.isfinite(qz))))
    if not valid:
        raise ValueError("qz_a_inv must be a nonempty finite vector")
    return qz


def assign_fit_regions(
    qz_a_inv: np.ndarray,
    critical_candidates: tuple[tuple[float, float], ...] = (),
    bragg_candidates: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """Assign stable region labels from declared intervals or q quartiles.

    Explicit critical and Bragg intervals are consumed in caller order and do
    not overwrite points already claimed by an earlier interval. Unclaimed
    points receive deterministic equal-width labels. With no features, the
    frozen four-quartile fallback spans the complete q range directly.
    """
    qz = _validated_qz(qz_a_inv)
    if np.ptp(qz) == 0.0:
        return np.zeros(qz.size, dtype=int)
    intervals = tuple(critical_candidates) + tuple(bragg_candidates)
    if not intervals:
        edges = np.linspace(qz.min(), qz.max(), 5)
        return np.searchsorted(edges[1:-1], qz, side="right").astype(int)
    labels = np.full(qz.size, -1, dtype=int)
    next_label = 0
    # First-match ownership makes overlapping feature intervals deterministic.
    # Empty intervals do not consume a label, so region IDs remain contiguous.
    # Source q order is never sorted or otherwise changed during assignment.
    for lower, upper in intervals:
        valid = all((np.isfinite(lower), np.isfinite(upper), lower < upper))
        if not valid:
            raise ValueError("fit-region intervals must be finite and increasing")
        selected = (qz >= lower) & (qz <= upper) & (labels < 0)
        labels[selected] = next_label
        next_label += int(np.any(selected))
    remaining = labels < 0
    if np.any(remaining):
        subset = qz[remaining]
        edges = np.linspace(subset.min(), subset.max(), min(4, subset.size) + 1)
        labels[remaining] = next_label + np.searchsorted(edges[1:-1], subset, side="right").astype(int)
    return labels


def log_residuals(
    model: np.ndarray,
    observed: np.ndarray,
    r_floor: float,
) -> np.ndarray:
    """Return unweighted fitted residuals in base-10 reflectivity decades.

    The same positive floor is added to model and observation before either
    logarithm. Weighting remains outside this function so optimizers, profile
    likelihoods, and diagnostics can share one unweighted residual definition.
    """
    model, observed = _float_vectors(
        model,
        observed,
        "model and observed must have equal shapes",
    )
    if not all((isfinite(r_floor), r_floor > 0.0)):
        raise ValueError("r_floor must be positive and finite")
    if not all((np.all(np.isfinite(model)), np.all(np.isfinite(observed)))):
        raise ValueError("model and observed must be finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        shifted_model = model + r_floor
        shifted_observed = observed + r_floor
    if not all((np.all(shifted_model > 0.0), np.all(shifted_observed > 0.0))):
        raise ValueError("reflectivity plus floor must be positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        model_log = np.log10(shifted_model)
        observed_log = np.log10(shifted_observed)
        model_overflow = np.isposinf(shifted_model)
        observed_overflow = np.isposinf(shifted_observed)
        if np.any(model_overflow):
            model_log[model_overflow] = np.logaddexp(
                np.log(model[model_overflow]),
                log(r_floor),
            ) / log(10.0)
        if np.any(observed_overflow):
            observed_log[observed_overflow] = np.logaddexp(
                np.log(observed[observed_overflow]),
                log(r_floor),
            ) / log(10.0)
        result = model_log - observed_log
    different = model != observed
    comparison_scale = np.maximum.reduce((np.abs(model), np.abs(observed), np.full(model.shape, r_floor)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        close = different & (np.abs(model / comparison_scale - observed / comparison_scale) < 1e-4)
    repair = ((result == 0.0) & different) | close
    if np.any(repair):
        scale = np.maximum.reduce(
            (
                np.abs(model[repair]),
                np.abs(observed[repair]),
                np.full(np.count_nonzero(repair), r_floor),
            )
        )
        normalized_difference = (model[repair] - observed[repair]) / scale
        normalized_observed = observed[repair] / scale + r_floor / scale
        result[repair] = np.log1p(normalized_difference / normalized_observed) / np.log(10.0)
    return result


def robust_log_cost(
    delta: np.ndarray,
    weights: np.ndarray,
    c: float = 0.05,
) -> float:
    """Average the weighted soft-L1 loss with weights outside the loss.

    The frozen policy squares region weights after evaluating the soft-L1
    expression. Invalid numeric inputs produce infinity because this scalar is
    a search-policy value, not a structural decoding boundary.
    """
    delta, weights = _float_vectors(
        delta,
        weights,
        "delta and weights must have equal shapes",
    )
    valid = all(
        (
            np.isfinite(c),
            c > 0.0,
            delta.size != 0,
            np.all(np.isfinite(delta)),
            np.all(np.isfinite(weights)),
            np.all(weights > 0.0),
        )
    )
    if not valid:
        return float("inf")
    # This algebraic form is stable near zero and retains the frozen factor two.
    # Region weights sit outside the robust loss and are squared exactly once.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        scaled = delta / c
        loss = 2.0 * c**2 * (np.sqrt(1.0 + scaled**2) - 1.0)
    repair = ~np.isfinite(loss) | ((delta != 0.0) & (np.abs(scaled) < 1e-4))
    if np.any(repair):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            magnitude = np.abs(delta[repair])
            radius = np.hypot(c, magnitude)
            loss[repair] = 2.0 * c * magnitude * (magnitude / (radius + c))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        result = np.mean(weights**2 * loss)
    return float(result)


def scale_prior_penalty(
    scale: float,
    scale_hat: float | None,
    tau_s: float,
    n: int,
) -> float:
    """Return the per-point weak prior on logarithmic instrument scale.

    A missing plateau estimate disables the prior exactly. Active evidence uses
    the fitted-point count in the denominator, keeping its contribution stable
    relative to the mean data loss as a fit mask changes size.
    """
    if scale_hat is None:
        return 0.0
    valid_type = (
        isinstance(n, (int, np.integer)),
        isinstance(n, bool),
    ) == (True, False)
    if not valid_type:
        raise ValueError("scale prior requires positive and finite scale, Ŝ, τ_S and N")
    values = np.asarray((scale, scale_hat, tau_s), dtype=float)
    valid = all((n > 0, np.all(np.isfinite(values) & (values > 0.0))))
    if not valid:
        raise ValueError("scale prior requires positive and finite scale, Ŝ, τ_S and N")
    # Work in the same base-10 coordinate as fitted log residuals.
    # Dividing by N makes this additive term compatible with a mean data loss.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        standardized = _log10_ratio(scale, scale_hat) / tau_s
        penalty = np.multiply(standardized, np.divide(standardized, n))
    return float(penalty)
