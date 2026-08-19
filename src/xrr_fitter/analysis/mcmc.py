"""Deterministic affine-invariant sampling and problem-bound MCMC.

The sampler uses the fixed two-half Goodman-Weare update schedule. Initial
walkers, retained draws, acceptance counters, and child seeds are all explicit
inputs or outputs, so repeated analysis never depends on ambient RNG state.
Problem-bound helpers map retained unit coordinates through the shared
evaluation boundary and attach convergence and boundary diagnostics without
importing fitting orchestration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite, log

import numpy as np

from xrr_fitter.evaluation import (
    _gradient_slab_counts,
    _prior_coordinate,
    _validated_unit,
    prior_center_and_spread,
    problem_log_probability,
    values_by_name,
)
from xrr_fitter.model.analysis import EnsembleSamples, McmcConfig, McmcReport
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.parameters import ParameterDefinition, ParameterPrior, PriorSpec


def _poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("cancelled")


def _prior_overlay(priors: Sequence[ParameterPrior]) -> dict[str, PriorSpec]:
    values = tuple(priors)
    if any(not isinstance(value, ParameterPrior) for value in values):
        raise TypeError("priors must contain ParameterPrior values")
    overlay = {value.name: value.prior for value in values}
    if len(overlay) != len(values):
        raise ValueError("parameter prior names must be unique")
    return overlay


def _overlay_definitions(
    problem: FitEvaluationContext,
    overlay: dict[str, PriorSpec],
) -> tuple[ParameterDefinition, ...]:
    names = {definition.name for definition in problem.parameter_definitions}
    unknown = set(overlay).difference(names)
    if unknown:
        raise ValueError(f"unknown parameter name: {min(unknown)}")
    return tuple(
        replace(definition, prior=overlay.get(definition.name, definition.prior))
        for definition in problem.parameter_definitions
    )


def with_parameter_priors(
    problem: FitEvaluationContext,
    priors: Sequence[ParameterPrior],
) -> FitEvaluationContext:
    """Return an analysis context with sidecar priors overlaid on definitions.

    The fitted ``problem`` remains untouched, so search provenance, checkpoint
    identity, and candidate coordinates continue to describe the prior-free fit.
    Empty priors preserve object identity as well as numerical behavior.
    """
    if not priors:
        return problem
    definitions = _overlay_definitions(problem, _prior_overlay(priors))
    return (
        problem
        if definitions == problem.parameter_definitions
        else replace(
            problem,
            parameter_definitions=definitions,
        )
    )


def split_rhat(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] == 0:
        raise ValueError("split-Rhat samples must have draws, walkers and parameters")
    half = values.shape[0] // 2
    if half < 2:
        return np.full(values.shape[2], np.inf, dtype=float)
    chains = np.concatenate(
        (
            np.transpose(values[:half], (1, 0, 2)),
            np.transpose(values[-half:], (1, 0, 2)),
        ),
        axis=0,
    )
    within = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(chains, axis=1), axis=0, ddof=1)
    variance = ((half - 1.0) / half) * within + between / half
    ratio = np.divide(
        variance,
        within,
        out=np.full_like(variance, np.inf),
        where=within > 0.0,
    )
    ratio[(within == 0.0) & (between == 0.0)] = 1.0
    return np.sqrt(np.maximum(ratio, 0.0))


def _mean_autocorrelation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    draws = values.shape[0]
    by_walker = np.transpose(values, (1, 0, 2))
    centered = by_walker - np.mean(by_walker, axis=1, keepdims=True)
    fft_size = 1 << (2 * draws - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size, axis=1)
    autocovariance = np.fft.irfft(
        transformed * np.conjugate(transformed),
        n=fft_size,
        axis=1,
    )[:, :draws, :]
    autocovariance /= np.arange(draws, 0, -1, dtype=float)[None, :, None]
    variance = autocovariance[:, :1, :]
    autocorrelation = np.divide(
        autocovariance,
        variance,
        out=np.zeros_like(autocovariance),
        where=variance > 0.0,
    )
    return np.mean(autocorrelation, axis=0), np.any(variance[:, 0, :] > 0.0, axis=0)


def _positive_pair_sum(correlation: np.ndarray) -> float:
    total = 0.0
    for lag in range(1, correlation.size, 2):
        pair = float(correlation[lag])
        if lag + 1 < correlation.size:
            pair += float(correlation[lag + 1])
        if pair <= 0.0:
            break
        total += pair
    return total


def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] == 0:
        raise ValueError("ESS samples must have draws, walkers and parameters")
    draws, walkers, dimension = values.shape
    correlations, nonconstant = _mean_autocorrelation(values)
    total = float(draws * walkers)
    result = []
    for index in range(dimension):
        if not nonconstant[index]:
            result.append(1.0)
            continue
        denominator = max(1.0, 1.0 + 2.0 * _positive_pair_sum(correlations[:, index]))
        result.append(float(np.clip(total / denominator, 1.0, total)))
    return np.asarray(result, dtype=float)


def _stretch_factor(rng: np.random.Generator, scale: float) -> float:
    return float(((scale - 1.0) * rng.random() + 1.0) ** 2 / scale)


def _valid_step(value: object, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer)) and value >= minimum


def _initial_walker_matrix(initial_walkers: np.ndarray) -> np.ndarray:
    walkers = np.array(initial_walkers, dtype=float, copy=True)
    if walkers.ndim != 2 or walkers.shape[1] == 0 or np.any(~np.isfinite(walkers)):
        raise ValueError("initial walkers must be a finite matrix")
    return walkers


def _validate_walker_geometry(walkers: np.ndarray, configured_count: object) -> None:
    walker_count, dimension = walkers.shape
    valid = walker_count == configured_count and walker_count % 2 == 0 and walker_count >= 2 * dimension + 2
    if not valid:
        raise ValueError("walkers must be even and at least 2*nfree+2")


def _validate_step_configuration(config: object) -> None:
    valid = (
        _valid_step(config.burn_in, 0)
        and _valid_step(config.production_steps, 4)
        and _valid_step(config.thin, 1)
        and config.thin < config.production_steps
    )
    if not valid:
        raise ValueError("invalid MCMC step configuration")


def _validate_stretch_scale(scale: object) -> None:
    if not isfinite(scale) or not 1.0 < scale <= 10.0:
        raise ValueError("stretch_scale must be in (1,10]")


def _initial_log_probability(
    log_probability: Callable[[np.ndarray], float],
    walkers: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        [float(log_probability(walker.copy())) for walker in walkers],
        dtype=float,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("initial walkers must have finite log probability")
    return values


def _validated_initial_state(
    log_probability: Callable[[np.ndarray], float],
    initial_walkers: np.ndarray,
    config: object,
    cancelled: Callable[[], bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    walkers = _initial_walker_matrix(initial_walkers)
    _validate_walker_geometry(walkers, config.walkers)
    _validate_step_configuration(config)
    _validate_stretch_scale(config.stretch_scale)
    _poll(cancelled)
    return walkers, _initial_log_probability(log_probability, walkers)


@dataclass(slots=True)
class _SamplerState:
    walkers: np.ndarray
    log_probability: np.ndarray
    accepted: np.ndarray
    attempted: np.ndarray
    retained: list[np.ndarray]
    retained_log_probability: list[np.ndarray]


def _update_group(
    state: _SamplerState,
    active: np.ndarray,
    complement: np.ndarray,
    dimension: int,
    config: object,
    rng: np.random.Generator,
    log_probability: Callable[[np.ndarray], float],
) -> None:
    for index_value in active:
        index = int(index_value)
        partner = int(rng.choice(complement))
        stretch = _stretch_factor(rng, config.stretch_scale)
        proposal = state.walkers[partner] + stretch * (state.walkers[index] - state.walkers[partner])
        proposal_logp = float(log_probability(proposal.copy()))
        state.attempted[index] += 1
        if not isfinite(proposal_logp):
            continue
        # Keep the subtraction in Python scalar arithmetic.  NumPy scalars
        # emit overflow warnings for mathematically valid +/-infinity limits.
        log_acceptance = (
            float(dimension - 1) * log(float(stretch)) + proposal_logp - float(state.log_probability[index])
        )
        if not np.isnan(log_acceptance) and log(float(rng.random())) < log_acceptance:
            state.walkers[index] = proposal
            state.log_probability[index] = proposal_logp
            state.accepted[index] += 1


def run_affine_invariant(
    log_probability: Callable[[np.ndarray], float],
    initial_walkers: np.ndarray,
    config: McmcConfig,
    child_seed: int,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> EnsembleSamples:
    """Run the fixed two-half Goodman-Weare update schedule."""
    walkers, log_values = _validated_initial_state(log_probability, initial_walkers, config, cancelled)
    state = _SamplerState(
        walkers,
        log_values,
        np.zeros(walkers.shape[0], dtype=np.int64),
        np.zeros(walkers.shape[0], dtype=np.int64),
        [],
        [],
    )
    rng = np.random.default_rng(child_seed)
    total_steps = int(config.burn_in + config.production_steps)
    dimension = walkers.shape[1]
    half = walkers.shape[0] // 2
    first = np.arange(half)
    second = np.arange(half, walkers.shape[0])
    for step in range(total_steps):
        _poll(cancelled)
        _update_group(state, first, second, dimension, config, rng, log_probability)
        _update_group(state, second, first, dimension, config, rng, log_probability)
        production_step = step - config.burn_in
        if step >= config.burn_in and production_step % config.thin == 0:
            state.retained.append(state.walkers.copy())
            state.retained_log_probability.append(state.log_probability.copy())
        if progress is not None:
            progress(step + 1, total_steps)
    samples = np.asarray(state.retained, dtype=float)
    return EnsembleSamples(
        samples,
        np.asarray(state.retained_log_probability, dtype=float),
        state.accepted / state.attempted,
        split_rhat(samples),
        effective_sample_size(samples),
    )


def _validated_candidate(problem: object, candidate: object) -> np.ndarray:
    unit = np.asarray(candidate.unit_vector, dtype=float)
    valid = (
        unit.shape == (len(problem.variables),)
        and np.all(np.isfinite(unit))
        and np.all((unit >= 0.0) & (unit <= 1.0))
        and candidate.valid
        and isfinite(candidate.objective)
    )
    if not valid:
        raise ValueError("MCMC requires a valid converged candidate")
    return unit


def _problem_seeds(child_seed: int) -> tuple[int, int]:
    spawned = np.random.SeedSequence(child_seed).spawn(2)
    return tuple(int(item.generate_state(1, dtype=np.uint64)[0]) for item in spawned)


def _problem_walkers(
    problem: object,
    center: np.ndarray,
    config: McmcConfig,
    seed: int,
    cancelled: Callable[[], bool] | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    walkers: list[np.ndarray] = []
    attempts = 0
    maximum = max(1000, int(config.walkers) * 1000)
    while len(walkers) < config.walkers and attempts < maximum:
        _poll(cancelled)
        proposal = rng.normal(center, 0.01, size=center.size)
        attempts += 1
        if np.any((proposal < 0.0) | (proposal > 1.0)):
            continue
        if isfinite(problem_log_probability(problem, proposal)):
            walkers.append(proposal)
    if len(walkers) != config.walkers:
        raise RuntimeError("could not initialize finite MCMC walkers")
    return np.vstack(walkers)


def map_problem_samples(
    problem: FitEvaluationContext,
    ensemble: EnsembleSamples,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    flat_unit, physical, _derived_names, derived = _map_problem_sample_arrays(
        problem,
        ensemble,
        cancelled,
    )
    del _derived_names, derived
    return flat_unit, physical


def _map_problem_sample_arrays(
    problem: FitEvaluationContext,
    ensemble: EnsembleSamples,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray | None]:
    names = tuple(variable.name for variable in problem.variables)
    derived_names = tuple(rule.target.parameter_name for rule in problem.constraint_rules)
    flat_unit = ensemble.samples_unit.reshape(-1, len(names))
    physical = np.empty_like(flat_unit)
    derived = np.empty((flat_unit.shape[0], len(derived_names)), dtype=float) if derived_names else None
    for index, unit in enumerate(flat_unit):
        _poll(cancelled)
        mapped = values_by_name(problem, unit)
        physical[index] = [mapped[name] for name in names]
        if derived is not None:
            derived[index] = [mapped[name] for name in derived_names]
    return flat_unit, physical, derived_names, derived


def _fixed_parameter_values(
    problem: FitEvaluationContext,
    sampled_names: tuple[str, ...],
    derived_names: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    dynamic = set((*sampled_names, *derived_names))
    return tuple(
        (definition.name, float(definition.initial))
        for definition in problem.parameter_definitions
        if definition.name not in dynamic
    )


def problem_mcmc_warnings(
    ensemble: EnsembleSamples,
    names: tuple[str, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    acceptance = np.flatnonzero((ensemble.acceptance_fraction <= 0.10) | (ensemble.acceptance_fraction >= 0.80))
    if acceptance.size:
        warnings.append(
            "mcmc_acceptance_outside_0.10_0.80:walkers=" + ",".join(str(int(index)) for index in acceptance)
        )
    rhat = np.flatnonzero(ensemble.split_rhat >= 1.10)
    if rhat.size:
        warnings.append("mcmc_split_rhat_at_least_1.10:parameters=" + ",".join(names[int(index)] for index in rhat))
    ess = np.flatnonzero(ensemble.effective_sample_size < 100.0)
    if ess.size:
        warnings.append(
            "mcmc_effective_sample_size_below_100:parameters=" + ",".join(names[int(index)] for index in ess)
        )
    return tuple(warnings)


def _near_boundary(
    definition: object,
    unit: np.ndarray,
    _physical: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if definition.transform not in {"linear", "log", "roughness_fraction"}:
        raise ValueError(f"unknown transform: {definition.transform}")
    # Boundary evidence is defined in the solver's normalized coordinate.  A
    # linear physical-distance test is wrong for log axes: it labels the
    # geometric midpoint of a wide interval as close to the lower bound and
    # can overflow when the physical span is not representable.
    return (unit <= fraction) | (unit >= 1.0 - fraction)


def mcmc_boundary_hits(
    problem: FitEvaluationContext,
    flat_unit: np.ndarray,
    physical: np.ndarray,
) -> tuple[str, ...]:
    fraction = float(problem.config.confidence.boundary_fraction)
    hits = []
    for index, variable in enumerate(problem.variables):
        definition = problem.parameter_definitions[variable.parameter_index]
        if np.any(_near_boundary(definition, flat_unit[:, index], physical[:, index], fraction)):
            hits.append(variable.name)
    return tuple(hits)


def prior_conflicts(
    problem: FitEvaluationContext,
    representative_unit: np.ndarray,
) -> tuple[str, ...]:
    """Flag free parameters whose representative estimate departs from its prior.

    The estimate and the prior center are compared in the coordinate space the
    prior is declared against (see ``_prior_coordinate``), so a
    ``roughness_fraction`` prior on the unit fraction never conflicts against a
    physical roughness value. Flat (``uniform``) and priorless parameters carry
    no location and are skipped. The tolerance is ``prior_conflict_sigmas``
    spreads about the center.
    """
    unit = _validated_unit(problem, representative_unit)
    estimates = np.asarray(
        [
            _prior_coordinate(
                problem.parameter_definitions[variable.parameter_index],
                float(unit[index]),
            )[0]
            for index, variable in enumerate(problem.variables)
        ],
        dtype=float,
    )
    return _prior_conflict_names(problem, estimates)


def _prior_conflict_names(
    problem: FitEvaluationContext,
    estimates: np.ndarray,
) -> tuple[str, ...]:
    sigmas = problem.config.confidence.prior_conflict_sigmas
    conflicts: list[str] = []
    for index, variable in enumerate(problem.variables):
        definition = problem.parameter_definitions[variable.parameter_index]
        if definition.prior is None:
            continue
        location = prior_center_and_spread(definition.prior)
        if location is None:
            continue
        center, spread = location
        if abs(float(estimates[index]) - center) > sigmas * spread:
            conflicts.append(variable.name)
    return tuple(conflicts)


def mcmc_prior_conflicts(
    problem: FitEvaluationContext,
    flat_unit: np.ndarray,
    physical: np.ndarray,
) -> tuple[str, ...]:
    """Flag posterior parameters whose retained median departs from its prior.

    Physical priors use the median of the mapped physical samples. Fractional
    roughness priors remain in their declared unit coordinate. This distinction
    matters for an even retained sample count because NumPy averages the two
    central values and nonlinear transforms do not preserve that average.
    """
    estimates = _stable_column_median(physical)
    unit_medians = _stable_column_median(flat_unit)
    for index, variable in enumerate(problem.variables):
        definition = problem.parameter_definitions[variable.parameter_index]
        if definition.transform == "roughness_fraction":
            estimates[index] = unit_medians[index]
    return _prior_conflict_names(problem, estimates)


def _stable_column_median(values: np.ndarray) -> np.ndarray:
    """Compute column medians without overflowing an even central pair."""
    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.median(array, axis=0), dtype=float)
    unstable = np.flatnonzero(~np.isfinite(result))
    if not unstable.size:
        return result
    ordered = np.sort(array, axis=0)
    midpoint = ordered.shape[0] // 2
    for index in unstable:
        left = float(ordered[midpoint - 1, index])
        right = float(ordered[midpoint, index])
        span = right - left
        if isfinite(span):
            result[index] = left + span / 2.0
            continue
        scale = max(abs(left), abs(right))
        result[index] = 0.0 if scale == 0.0 else scale * ((left / scale) * 0.5 + (right / scale) * 0.5)
    return result


def run_problem_mcmc(
    problem: FitEvaluationContext,
    candidate: object,
    config: McmcConfig,
    *,
    child_seed: int,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> McmcReport:
    """Initialize, sample, and map one converged problem candidate."""
    center = _validated_candidate(problem, candidate)
    initialization_seed, sampler_seed = _problem_seeds(child_seed)
    walkers = _problem_walkers(problem, center, config, initialization_seed, cancelled)
    ensemble = run_affine_invariant(
        lambda unit: problem_log_probability(problem, unit),
        walkers,
        config,
        sampler_seed,
        progress=progress,
        cancelled=cancelled,
    )
    names = tuple(variable.name for variable in problem.variables)
    flat_unit, physical, derived_names, derived = _map_problem_sample_arrays(
        problem,
        ensemble,
        cancelled,
    )
    return McmcReport(
        config=config,
        child_seed=int(child_seed),
        parameter_names=names,
        samples_physical=physical,
        log_probability=ensemble.log_probability.reshape(-1),
        acceptance_fraction=ensemble.acceptance_fraction,
        split_rhat=ensemble.split_rhat,
        effective_sample_size=ensemble.effective_sample_size,
        boundary_hits=mcmc_boundary_hits(problem, flat_unit, physical),
        warnings=problem_mcmc_warnings(ensemble, names),
        candidate_id=getattr(candidate, "candidate_id", None),
        prior_conflicts=mcmc_prior_conflicts(problem, flat_unit, physical),
        derived_parameter_names=derived_names,
        derived_samples_physical=derived,
        fixed_parameter_values=_fixed_parameter_values(problem, names, derived_names),
        gradient_slab_counts=tuple(_gradient_slab_counts(problem).items()),
    )
