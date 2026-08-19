"""Internal priors implementation."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from math import exp, isfinite, log

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
from xrr_fitter.evaluation_model import evaluate_model
from xrr_fitter.evaluation_parameters import EvaluationConstraintError, _unit_derivative, _validated_unit
from xrr_fitter.evaluation_solver import _scale_prior_residual
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.parameters import (
    ParameterDefinition,
    PriorSpec,
    _interval_half_width,
    _interval_midpoint,
    unit_to_physical,
)


def _normal_log_density(parameters: tuple[float, ...], x: float) -> float:
    mean, std = parameters
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        standardized = np.divide(np.subtract(x, mean), std)
        result = -0.5 * np.square(standardized)
    return float(result)


def _lognormal_log_density(parameters: tuple[float, ...], x: float) -> float:
    # Parameters live in log space; the 1/x Jacobian makes this a density in x.
    log_mean, log_std = parameters
    if x <= 0.0:
        return -np.inf
    log_x = log(x)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        standardized = np.divide(np.subtract(log_x, log_mean), log_std)
        result = -0.5 * np.square(standardized) - log_x
    return float(result)


def _soft_range_log_density(parameters: tuple[float, ...], x: float) -> float:
    # Flat inside [low, high], Gaussian shoulders of scale std outside it.
    low, high, std = parameters
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        distance = np.maximum(np.maximum(np.subtract(low, x), np.subtract(x, high)), 0.0)
        result = -0.5 * np.square(np.divide(distance, std))
    return float(result)


def _uniform_log_density(parameters: tuple[float, ...], x: float) -> float:
    return 0.0


PRIOR_LOG_DENSITY: dict[str, Callable[[tuple[float, ...], float], float]] = {
    "uniform": _uniform_log_density,
    "normal": _normal_log_density,
    "lognormal": _lognormal_log_density,
    "soft_range": _soft_range_log_density,
}


def _unnormalized_density(kind: str, parameters: tuple[float, ...], x: float) -> float:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return float(np.exp(PRIOR_LOG_DENSITY[kind](parameters, x)))


PRIOR_QUADRATURE_NODES = 2049


def _normal_grid_nodes(values: tuple[float, ...]) -> np.ndarray:
    mean, std = values
    fine = mean + std * np.linspace(-12.0, 12.0, 1025)
    # Exact-zero density anchors separate the refined kernel from arbitrarily
    # distant hard bounds, preventing a giant tail trapezoid.
    return np.concatenate(((mean - 40.0 * std,), fine, (mean + 40.0 * std,)))


def _lognormal_grid_nodes(values: tuple[float, ...]) -> np.ndarray:
    log_mean, log_std = values
    coordinates = np.concatenate(((-40.0,), np.linspace(-12.0, 12.0, 1025), (40.0,)))
    return np.exp(log_mean + log_std * coordinates)


def _soft_range_grid_nodes(values: tuple[float, ...]) -> np.ndarray:
    low, high, std = values
    offsets = std * np.linspace(-12.0, 12.0, 513)
    return np.concatenate(((low - 40.0 * std,), low + offsets, high + offsets, (high + 40.0 * std,)))


PRIOR_GRID_NODES: dict[str, Callable[[tuple[float, ...]], np.ndarray]] = {
    "normal": _normal_grid_nodes,
    "lognormal": _lognormal_grid_nodes,
    "soft_range": _soft_range_grid_nodes,
}


def _prior_base_grid(lower: float, upper: float) -> np.ndarray:
    """Construct the hard-support grid without overflowing its finite span."""
    lower, upper = float(lower), float(upper)
    span = upper - lower
    if isfinite(span):
        return np.linspace(lower, upper, PRIOR_QUADRATURE_NODES)
    scale = max(abs(lower), abs(upper))
    normalized = np.linspace(lower / scale, upper / scale, PRIOR_QUADRATURE_NODES)
    result = scale * normalized
    result[0], result[-1] = lower, upper
    return result


def _lognormal_support_grid(lower: float, upper: float) -> np.ndarray:
    """Resolve positive hard support uniformly in the prior's log coordinate."""
    if lower <= 0.0 or upper <= lower:
        return np.empty(0, dtype=float)
    log_grid = np.linspace(log(lower), log(upper), PRIOR_QUADRATURE_NODES)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.exp(log_grid)
    result[0], result[-1] = lower, upper
    return result[np.isfinite(result) & (result >= lower) & (result <= upper)]


def _prior_grid(
    kind: str,
    parameters: tuple[float, ...],
    lower: float,
    upper: float,
) -> np.ndarray:
    """Return a deterministic grid refined around every narrow feature.

    The original 2049 evenly spaced nodes remain the common integration grid.
    They preserve the established uniform-prior path and provide complete hard
    support coverage for every distribution.

    Normal and lognormal kernels can be much narrower than one base interval.
    Each therefore contributes 1025 nodes spanning twelve standard deviations
    on either side of its center.  Twelve sigmas include essentially all mass
    while retaining deterministic, finite work independent of the hard-box
    width.

    Lognormal nodes are spaced in the distribution's native log coordinate and
    exponentiated only after refinement.  This prevents a narrow positive-space
    peak from disappearing when the physical hard box spans several decades.

    Soft ranges have two independently sharp shoulders.  Both edges receive
    513 nodes across the same twelve-sigma window, so each transition is
    resolved without biasing the flat interior toward either edge.

    Nodes outside the declared box and non-finite exponentiation results are
    removed before the union.  The hard endpoints are inserted explicitly;
    sorting and duplicate removal are delegated to ``np.unique`` so cumulative
    integration always sees a strictly increasing coordinate vector.

    Uniform priors have no local feature and use the base grid unchanged.  The
    dispatch table keeps distribution-specific geometry out of this shared
    assembly path and avoids a branch ladder as future prior kinds are added.

    Refinement changes only numerical resolution.  It does not widen support,
    alter log kernels, change the cached normalization key, or affect problems
    with no parameter priors.

    The generated nodes are deterministic functions of immutable prior
    parameters.  No random generator, mutable cache entry, fitted vector, or
    platform path participates in their construction.

    The normal window is affine in physical coordinates.  Translating its mean
    translates every local node by the same amount; scaling its sigma scales
    every offset without moving the declared hard endpoints.

    The lognormal window applies that same affine construction before the
    exponential map.  Positive support follows from exponentiation, while an
    overflowed tail is discarded by the common finite-node filter.

    The soft-range window treats its lower and upper shoulder symmetrically.
    Concatenation order is irrelevant after sorting, but constructing both with
    one shared offset vector guarantees matching resolution.

    Base and local nodes may coincide at centers, shoulders, or hard endpoints.
    Duplicate removal prevents zero-width trapezoids and therefore avoids a
    distribution-specific special case in cumulative integration.

    Clipping happens before the union rather than after integration.  Thus no
    probability mass is sampled outside the conditioned interval and every CDF
    entry corresponds to a coordinate that is legal for the parameter.

    Keeping the base grid for narrow priors is intentional.  Local nodes resolve
    the peak while base nodes retain tails, broad shoulders, and both support
    endpoints in the same cumulative array.

    Returning the untouched base array for a uniform prior preserves its exact
    historical spacing and cumulative interpolation behavior.

    The cached normalization remains keyed by kind, parameters, and bounds, so
    repeated density and quantile calls reuse this complete refined grid.

    CDF and inverse-CDF interpolation consume the identical returned axes.
    Refinement therefore improves both directions together rather than letting
    density, probability, and quantile calculations drift onto separate grids.

    Endpoint insertion is retained even though the base grid already contains
    both bounds.  It makes the local-grid contract explicit and robust if the
    base construction is later replaced by another deterministic mesh.

    A missing dispatch entry means that the prior has no localized feature.
    Falling back to the base grid is preferable to fabricating arbitrary anchor
    points or silently changing that prior's kernel.

    Twelve-sigma windows are deliberately much wider than conventional display
    intervals.  The extra tail nodes cost little after caching and prevent the
    truncated normalization from depending on one coarse interval around a very
    narrow center.

    All arrays use NumPy's default floating dtype, matching the surrounding
    evaluation boundary and avoiding hidden precision conversions between the
    grid, density, trapezoid, and interpolation stages.

    The helper returns a newly owned array for every uncached construction.
    Callers normalize and retain it without mutating shared module constants.

    Sorted finite nodes also make monotonic CDF construction an invariant rather
    than an assumption left to each prior-specific node generator.

    This single assembly function is the only place that combines global hard
    support with local distribution geometry.

    Its output is ready for direct trapezoidal integration.
    """
    base = _prior_base_grid(lower, upper)
    nodes = PRIOR_GRID_NODES.get(kind)
    if nodes is None:
        return base
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        local = nodes(parameters)
    local = local[np.isfinite(local) & (local >= lower) & (local <= upper)]
    support = _lognormal_support_grid(lower, upper) if kind == "lognormal" else np.empty(0, dtype=float)
    return np.unique(np.concatenate((base, support, local, (lower, upper))))


@lru_cache(maxsize=256)
def _prior_norm(
    kind: str,
    parameters: tuple[float, ...],
    lower: float,
    upper: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Cache the log truncation constant, grid, and cumulative prior mass.

    Composite trapezoid on a fixed grid keeps every kind on one code path and
    makes ``prior_cdf`` and ``prior_inverse_cdf`` exact inverses by construction.
    Keys are immutable scalars only, so frozen ``PriorSpec`` values are never
    retained and repeated evaluation costs one dict lookup.
    """
    grid = _prior_grid(kind, parameters, lower, upper)
    density = np.array([_unnormalized_density(kind, parameters, float(x)) for x in grid])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        increments = np.diff(grid) * (density[:-1] + density[1:]) / 2.0
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cumulative[-1])
    if isfinite(total) and total > 0.0:
        return log(total), grid, cumulative / total

    log_density = np.array(
        [PRIOR_LOG_DENSITY[kind](parameters, float(x)) for x in grid],
        dtype=float,
    )
    log_width = np.array(
        [_uniform_log_width(float(left), float(right)) for left, right in zip(grid[:-1], grid[1:], strict=True)],
        dtype=float,
    )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        log_increments = log_width + np.logaddexp(log_density[:-1], log_density[1:]) - log(2.0)
    finite = np.isfinite(log_increments)
    if not np.any(finite):
        raise ValueError(f"{kind} prior has no mass inside the declared bounds")
    anchor = float(np.max(log_increments[finite]))
    scaled_increments = np.zeros_like(log_increments)
    scaled_increments[finite] = np.exp(log_increments[finite] - anchor)
    scaled_cumulative = np.concatenate(([0.0], np.cumsum(scaled_increments)))
    scaled_total = float(scaled_cumulative[-1])
    if not (isfinite(scaled_total) and scaled_total > 0.0):
        raise ValueError(f"{kind} prior has no mass inside the declared bounds")
    return anchor + log(scaled_total), grid, scaled_cumulative / scaled_total


def _uniform_log_width(lower: float, upper: float) -> float:
    lower, upper = float(lower), float(upper)
    span = upper - lower
    if isfinite(span) and span > 0.0:
        return log(span)
    scale = max(abs(lower), abs(upper))
    if not isfinite(scale) or scale == 0.0:
        raise ValueError("uniform prior has no mass inside the declared bounds")
    normalized_span = upper / scale - lower / scale
    if not isfinite(normalized_span) or normalized_span <= 0.0:
        raise ValueError("uniform prior has no mass inside the declared bounds")
    return log(scale) + log(normalized_span)


def _uniform_cdf(x: float, lower: float, upper: float) -> float:
    if x <= lower:
        return 0.0
    if x >= upper:
        return 1.0
    span = upper - lower
    if isfinite(span) and span > 0.0:
        return float((x - lower) / span)
    scale = max(abs(lower), abs(upper))
    normalized_span = upper / scale - lower / scale
    return float((x / scale - lower / scale) / normalized_span)


def _uniform_quantile(level: float, lower: float, upper: float) -> float:
    if level == 0.0:
        return float(lower)
    if level == 1.0:
        return float(upper)
    span = upper - lower
    if isfinite(span) and span > 0.0:
        return float(lower + level * span)
    scale = max(abs(lower), abs(upper))
    normalized = (lower / scale) * (1.0 - level) + (upper / scale) * level
    return float(scale * normalized)


def prior_log_density(spec: PriorSpec, x: float, lower: float, upper: float) -> float:
    """Return the log density of ``spec`` truncated and renormalized on the bounds."""
    lower, upper = float(lower), float(upper)
    if not lower <= x <= upper:
        return -np.inf
    if spec.kind == "uniform" and not isfinite(upper - lower):
        return -_uniform_log_width(lower, upper)
    log_total, _, _ = _prior_norm(spec.kind, spec.parameters, lower, upper)
    return PRIOR_LOG_DENSITY[spec.kind](spec.parameters, x) - log_total


def prior_cdf(spec: PriorSpec, x: float, lower: float, upper: float) -> float:
    """Return the truncated cumulative probability of ``spec`` at ``x``."""
    lower, upper = float(lower), float(upper)
    if x <= lower:
        return 0.0
    if x >= upper:
        return 1.0
    if spec.kind == "uniform" and not isfinite(upper - lower):
        _uniform_log_width(lower, upper)
        return _uniform_cdf(x, lower, upper)
    _, grid, masses = _prior_norm(spec.kind, spec.parameters, lower, upper)
    upper_index = int(np.searchsorted(grid, x, side="left"))
    if grid[upper_index] == x:
        return float(masses[upper_index])
    lower_index = upper_index - 1
    fraction = _uniform_cdf(
        x,
        float(grid[lower_index]),
        float(grid[upper_index]),
    )
    return float(masses[lower_index] * (1.0 - fraction) + masses[upper_index] * fraction)


def prior_inverse_cdf(spec: PriorSpec, level: float, lower: float, upper: float) -> float:
    """Return the truncated quantile of ``spec`` at probability ``level``."""
    lower, upper = float(lower), float(upper)
    if not 0.0 <= level <= 1.0:
        raise ValueError("prior cdf level must be within [0, 1]")
    if spec.kind == "uniform" and not isfinite(upper - lower):
        _uniform_log_width(lower, upper)
        return _uniform_quantile(level, lower, upper)
    _, grid, masses = _prior_norm(spec.kind, spec.parameters, lower, upper)
    # The declared truncation endpoints are exact quantiles even when a very
    # narrow prior leaves long zero-mass plateaus on the numerical grid.
    if level == 0.0:
        return float(lower)
    if level == 1.0:
        return float(upper)
    # ``np.interp`` requires increasing x-coordinates.  A narrow prior can
    # produce repeated cumulative masses, so retain the first occurrence of
    # each mass while preserving the endpoint contract above.
    increasing = np.r_[True, np.diff(masses) > 0.0]
    selected_masses = masses[increasing]
    selected_grid = grid[increasing]
    upper_index = int(np.searchsorted(selected_masses, level, side="left"))
    if selected_masses[upper_index] == level:
        return float(selected_grid[upper_index])
    lower_index = upper_index - 1
    fraction = (level - selected_masses[lower_index]) / (selected_masses[upper_index] - selected_masses[lower_index])
    return _uniform_quantile(
        float(fraction),
        float(selected_grid[lower_index]),
        float(selected_grid[upper_index]),
    )


def prior_bounds(spec: PriorSpec, lower: float, upper: float) -> tuple[float, float]:
    """Return the support of ``spec``, which truncation pins to the declared bounds."""
    if spec.kind not in PRIOR_LOG_DENSITY:
        raise ValueError(f"unsupported prior kind: {spec.kind}")
    return float(lower), float(upper)


def prior_center_and_spread(spec: PriorSpec) -> tuple[float, float] | None:
    """Return the physical center and spread of ``spec``, or ``None`` when flat.

    ``uniform`` carries no location, so it never takes part in conflict scoring.
    ``lognormal`` parameters are in log space; its physical spread is the
    first-order image of the log-space sigma.
    """
    if spec.kind == "normal":
        mean, std = spec.parameters
        return float(mean), float(std)
    if spec.kind == "lognormal":
        log_mean, log_std = spec.parameters
        try:
            center = exp(log_mean)
        except OverflowError:
            return float("inf"), float("inf")
        return float(center), float(center * log_std)
    if spec.kind == "soft_range":
        low, high, std = spec.parameters
        return _interval_midpoint(low, high), float(_interval_half_width(low, high) + std)
    return None


def _prior_coordinate(
    definition: ParameterDefinition,
    unit_value: float,
) -> tuple[float, float, float]:
    """Return the value and bounds a prior is declared against for one definition.

    ``roughness_fraction`` priors live on the dimensionless unit fraction in
    ``[0, 1]`` because the physical roughness depends on a geometry snapshot this
    scalar path does not own.
    """
    if definition.transform == "roughness_fraction":
        return float(unit_value), 0.0, 1.0
    value = float(unit_to_physical(definition, unit_value))
    return value, float(definition.lower), float(definition.upper)


def _parameter_prior_log_density(
    problem: FitEvaluationContext,
    unit: np.ndarray,
) -> float:
    """Return the summed parameter prior log density, exactly ``0.0`` when unused.

    Returning the literal zero keeps ``problem_log_probability`` bitwise
    identical for priorless problems, since IEEE 754 addition of ``0.0`` is exact.
    """
    total = 0.0
    for index, variable in enumerate(problem.variables):
        definition = problem.parameter_definitions[variable.parameter_index]
        if definition.prior is None:
            continue
        value, lower, upper = _prior_coordinate(definition, float(unit[index]))
        total += prior_log_density(definition.prior, value, lower, upper)
        if definition.transform != "roughness_fraction":
            total += log(_unit_derivative(definition, value))
    return total


def problem_log_probability(
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
) -> float:
    """Return the robust pseudo-posterior density used by uncertainty MCMC.

    This is a deterministic analysis density, not a normalized probability
    distribution. The data term is the same weighted soft-L1 loss used by fit,
    converted from its mean form back to a sum and scaled by ``2*c**2``.

    Invalid unit coordinates, expected physical constraints, and invalid model
    evaluations have negative-infinite density. Unsupported layouts and other
    programming errors are not swallowed. An active scale prior contributes
    ``-0.5 * standardized**2`` independently of fitted-point count.

    Returning ``-inf`` at declared domain boundaries gives ensemble samplers a
    conventional rejection signal without inventing a finite penalty magnitude.
    The accepted path uses the same weighted soft-L1 expression as deterministic
    fitting while retaining the sampler's frozen pointwise summation order. That
    explicit grouping is required for bitwise replay of stored log probabilities.

    This density omits normalization constants because downstream analysis uses
    only relative log probability. It also performs no mutation, caching, or
    random work, making repeated evaluation deterministic for a fixed context
    and unit vector.
    """
    try:
        unit = _validated_unit(problem, unit_vector)
    except ValueError:
        return -np.inf
    try:
        observed = evaluate_model(problem, unit)
    except EvaluationConstraintError:
        return -np.inf
    if not all((observed.valid, isfinite(observed.objective))):
        return -np.inf
    residual = np.asarray(observed.fit_log_residuals_decades, dtype=float)
    weights = np.asarray(problem.weights[problem.data.fit_mask], dtype=float)
    c_decades = problem.config.c_decades
    # Preserve the frozen sampler grouping: sum point losses before dividing by
    # the robust scale. Replacing this with mean * count changes retained log
    # probabilities by a few ULPs even though the expressions are algebraically
    # equivalent, which breaks deterministic checkpoint and reference replay.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        scaled_residual = residual / c_decades
        point_loss = np.sqrt(1.0 + scaled_residual**2) - 1.0
        data_loss = np.sum(weights**2 * 2.0 * c_decades**2 * point_loss)
    repair = ~np.isfinite(point_loss) | ((residual != 0.0) & (np.abs(scaled_residual) < 1e-4))
    if np.isfinite(data_loss) and c_decades**2 > 0.0 and not np.any(repair):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            log_probability = -float(data_loss) / (2.0 * c_decades**2)
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            magnitude = np.abs(residual[repair])
            radius = np.hypot(c_decades, magnitude)
            point_loss[repair] = (magnitude / c_decades) * (magnitude / (radius + c_decades))
            log_probability = -float(np.sum(weights**2 * point_loss))
    if problem.scale_prior_center is not None:
        prior = _scale_prior_residual(problem, observed)
        assert prior is not None
        log_probability -= 0.5 * prior**2
    log_probability += _parameter_prior_log_density(problem, unit)
    return float(log_probability)
