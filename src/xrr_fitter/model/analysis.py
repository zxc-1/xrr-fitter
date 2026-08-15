"""Immutable confidence, profile, bootstrap, MCMC, and analysis results.

Every published NumPy value is copied and made read-only. Result identities
remain explicit so later project serialization can validate ownership graphs.

This module contains data contracts only. It does not evaluate reflectivity,
run optimizers, inspect files, or schedule workers. Constructors normalize
sequences and restore read-only NumPy ownership after every pickle round trip.

Profile values retain the sampled coordinate, objective trace, and boundary
evidence needed to explain one-dimensional confidence limits. A profile-basin
decision is intentionally smaller than a fitting continuation: it carries a
parameter identity and an immutable unit-space center, while fit independently
revalidates the center before changing search state.

Bootstrap results distinguish generic local aggregation from problem-owned
evidence. Owner fields are paired, parameter names define sample columns, and
published intervals must follow exactly the same ordered axis. Failure rate is
kept even when the interval gate suppresses all bounds.

Empty bootstrap intervals are therefore meaningful rather than incomplete:
they record a performed analysis whose failure-rate gate withheld confidence
bounds. Nonempty rows are normalized to floats only after their names and
ordering have been checked, so serialization cannot silently permute columns.

Ensemble and MCMC values preserve sampling geometry separately from summaries.
Their axes align draws, walkers, parameters, log probability, acceptance,
convergence, and effective sample size without retaining runtime callbacks or
random-number generators.

Uncertainty reports combine those immutable values with covariance, residual,
diagnostic, boundary, and correlation evidence for one candidate identity.
Classification stays explicit rather than being inferred from display text.
SLD uncertainty bands are defined in ``model/sld_bands`` and re-exported here,
so importers still reach every analysis value through this one module.

``FitResult`` flattens the fitting-only search schema for the supported public
result while attaching uncertainty and classification evidence. It reconstructs
the fitting value during validation so candidate lineage, eligible winner,
region arrays, and stage references obey one contract across both result types.

All validation occurs at construction. Invalid serialized state therefore
fails at the model boundary instead of surfacing later during report rendering,
project persistence, export, or service orchestration.

These values remain presentation-neutral: Chinese confidence labels are stable
persisted categories, while numerical evidence keeps its original precision.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from math import isfinite

import numpy as np

from xrr_fitter.model.fitting import (
    FitCandidate,
    FitSearchResult,
    FitStageSummary,
    PhysicsDiagnostic,
)
from xrr_fitter.model.mcmc_samples import (
    validate_derived_samples,
    validate_fixed_parameter_values,
)
from xrr_fitter.model.parameters import ParameterDefinition

# Re-exported so analysis values keep one import entry point; the band lives in
# its own module only to keep both files inside the maintainability gate.
from xrr_fitter.model.sld_bands import SldUncertaintyBands as SldUncertaintyBands


def _readonly(value: object, dtype: type, field: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _pickle_values(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, item.name) for item in fields(value))


def _positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _nonempty_unique_names(values: object, field: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not value.strip() for value in names):
        raise ValueError(f"{field} must be nonempty")
    if len(names) != len(set(names)):
        raise ValueError(f"{field} must be unique")
    return names


def _set_array_fields(value: object, names: tuple[str, ...], arrays: tuple[np.ndarray, ...]) -> None:
    for name, array in zip(names, arrays, strict=True):
        object.__setattr__(value, name, array)


def _mcmc_arrays(value: McmcReport) -> tuple[np.ndarray, ...]:
    return (
        _readonly(value.samples_physical, float, "samples_physical", 2),
        _readonly(value.log_probability, float, "log_probability", 1),
        _readonly(value.acceptance_fraction, float, "acceptance_fraction", 1),
        _readonly(value.split_rhat, float, "split_rhat", 1),
        _readonly(value.effective_sample_size, float, "effective_sample_size", 1),
    )


def _validate_mcmc_identity(value: McmcReport) -> None:
    if not isinstance(value.config, McmcConfig):
        raise TypeError("config must be McmcConfig")
    if not isinstance(value.child_seed, int) or isinstance(value.child_seed, bool) or value.child_seed < 0:
        raise ValueError("child_seed must be a nonnegative integer")
    if not value.label.strip():
        raise ValueError("label must not be empty")
    if value.candidate_id is not None and not value.candidate_id.strip():
        raise ValueError("candidate_id must be nonempty or None")


def _validate_mcmc_axes(
    config: McmcConfig,
    names: tuple[str, ...],
    arrays: tuple[np.ndarray, ...],
) -> None:
    """Keep flattened retained draws aligned with walker and parameter axes."""
    samples, probability, acceptance, rhat, effective = arrays
    parameter_axes = samples.shape[1] == len(names) == rhat.size == effective.size
    if not parameter_axes:
        raise ValueError("MCMC parameter axes do not match parameter names")
    if samples.shape[0] != probability.size:
        raise ValueError("MCMC sample and probability counts must match")
    if acceptance.size != config.walkers:
        raise ValueError("MCMC acceptance_fraction must match walker count")


def _validate_mcmc_values(arrays: tuple[np.ndarray, ...]) -> None:
    samples, probability, _acceptance, _rhat, _effective = arrays
    if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(probability)):
        raise ValueError("MCMC report arrays contain nonfinite values")


def _evidence_count(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _positive_values(values: object, field: str) -> tuple[float, ...]:
    result = tuple(values)
    if any(not isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{field} must contain positive finite values")
    return result


def _bootstrap_names(values: object) -> tuple[str, ...]:
    names = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in names):
        raise ValueError("bootstrap parameter names must contain nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError("bootstrap parameter names must be unique")
    return names


def _bootstrap_intervals(
    names: tuple[str, ...],
    values: object,
) -> tuple[tuple[str, float, float], ...]:
    """Normalize optional bounds without erasing a gated empty result."""
    intervals = tuple(values)
    if not intervals:
        return ()
    valid_rows = all(isinstance(value, (tuple, list)) and len(value) == 3 for value in intervals)
    if not valid_rows:
        raise ValueError("bootstrap intervals must contain name, lower, and upper")
    interval_names = tuple(value[0] for value in intervals)
    if interval_names != names:
        raise ValueError("bootstrap interval names must match parameter names in order")
    bounds = _bootstrap_interval_bounds(intervals)
    return tuple((name, lower, upper) for name, (lower, upper) in zip(interval_names, bounds, strict=True))


def _bootstrap_interval_bounds(
    intervals: tuple[object, ...],
) -> tuple[tuple[float, float], ...]:
    try:
        bounds = tuple((float(value[1]), float(value[2])) for value in intervals)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("bootstrap interval bounds must be finite numbers") from error
    invalid = any(not isfinite(lower) or not isfinite(upper) or lower > upper for lower, upper in bounds)
    if invalid:
        raise ValueError("bootstrap interval bounds must be finite and ordered")
    return bounds


def _bootstrap_samples(names: tuple[str, ...], values: object) -> np.ndarray:
    samples = _readonly(values, float, "bootstrap samples", 2)
    if samples.shape[1] != len(names):
        raise ValueError("bootstrap samples do not match parameter names")
    if np.any(~np.isfinite(samples)):
        raise ValueError("bootstrap samples must be finite")
    return samples


def _validate_bootstrap_failure_rate(value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("failure_rate must be in [0, 1]")


def _validate_bootstrap_owner(
    candidate_id: str | None,
    provenance_sha256: str | None,
) -> None:
    """Require a complete candidate/provenance pair when ownership is sealed."""
    if (candidate_id is None) != (provenance_sha256 is None):
        raise ValueError("bootstrap candidate_id and provenance_sha256 must be paired")
    if candidate_id is None:
        return
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("bootstrap candidate_id must be nonempty or None")
    if not isinstance(provenance_sha256, str):
        raise ValueError("bootstrap provenance_sha256 must be a lowercase SHA-256")
    valid = len(provenance_sha256) == 64 and all(value in "0123456789abcdef" for value in provenance_sha256)
    if not valid:
        raise ValueError("bootstrap provenance_sha256 must be a lowercase SHA-256")


def _validate_optional_sld_bands(value: SldUncertaintyBands | None) -> None:
    if value is not None and not isinstance(value, SldUncertaintyBands):
        raise TypeError("sld_bands must be a SldUncertaintyBands")


class ConfidenceClass(StrEnum):
    """Persisted user-facing confidence categories."""

    TRUSTED = "可信"
    CORRELATED = "可用但相关"
    MULTIPLE = "多解"
    UNTRUSTED = "不可信"


@dataclass(frozen=True, slots=True)
class ProfileBasinDecision:
    """Analysis evidence requesting fit-owned basin reconvergence.

    Analysis owns the decision but never executes the fitting continuation.
    The unit-space center and parameter identity cross that domain boundary as
    immutable evidence; fit independently re-evaluates the center before using
    it and does not trust the reported objective as search state.
    """

    parameter_name: str
    unit_vector: np.ndarray
    objective: float
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parameter_name.strip():
            raise ValueError("parameter_name must not be empty")
        unit = _readonly(self.unit_vector, float, "unit vector", 1)
        if np.any(~np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("unit vector must be finite and within [0, 1]")
        if not isfinite(self.objective):
            raise ValueError("objective must be finite")
        evidence = tuple(self.evidence)
        if not evidence or any(not isinstance(value, str) or not value for value in evidence):
            raise ValueError("evidence must contain nonempty strings")
        object.__setattr__(self, "unit_vector", unit)
        object.__setattr__(self, "evidence", evidence)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), (
            self.parameter_name,
            self.unit_vector,
            self.objective,
            self.evidence,
        )


@dataclass(frozen=True, slots=True)
class McmcConfig:
    """Affine-invariant sampling geometry and deterministic work budget."""

    walkers: int
    burn_in: int
    production_steps: int
    thin: int = 1
    stretch_scale: float = 2.0

    def __post_init__(self) -> None:
        for field in ("walkers", "production_steps", "thin"):
            _positive_integer(getattr(self, field), field)
        if not isinstance(self.burn_in, int) or isinstance(self.burn_in, bool) or self.burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer")
        if self.walkers % 2:
            raise ValueError("walkers must be even")
        if not isfinite(self.stretch_scale) or self.stretch_scale <= 1.0:
            raise ValueError("stretch_scale must be finite and greater than one")

    @classmethod
    def standard(cls, free_parameter_count: int) -> McmcConfig:
        if (
            not isinstance(free_parameter_count, int)
            or isinstance(free_parameter_count, bool)
            or free_parameter_count < 0
        ):
            raise ValueError("free_parameter_count must be a nonnegative integer")
        walkers = max(32, 2 * free_parameter_count + 2)
        if walkers % 2:
            walkers += 1
        return cls(walkers=walkers, burn_in=1000, production_steps=4000)


@dataclass(frozen=True, slots=True)
class ParameterProfile:
    """Profile coordinates, objective evidence, and closure flags."""

    name: str
    values: np.ndarray
    objectives: np.ndarray
    lower_closed: bool
    upper_closed: bool

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name must not be empty")
        values = _readonly(self.values, float, "profile values", 1)
        objectives = _readonly(self.objectives, float, "profile objectives", 1)
        if values.shape != objectives.shape:
            raise ValueError("profile values and objectives must have the same shape")
        if values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError("profile arrays contain invalid values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "objectives", objectives)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Successful physical bootstrap samples and interval summary."""

    parameter_names: tuple[str, ...]
    samples: np.ndarray
    intervals: tuple[tuple[str, float, float], ...]
    failure_rate: float
    candidate_id: str | None = None
    provenance_sha256: str | None = None

    def __post_init__(self) -> None:
        names = _bootstrap_names(self.parameter_names)
        samples = _bootstrap_samples(names, self.samples)
        _validate_bootstrap_failure_rate(self.failure_rate)
        intervals = _bootstrap_intervals(names, self.intervals)
        _validate_bootstrap_owner(self.candidate_id, self.provenance_sha256)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "intervals", intervals)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


def _ensemble_arrays(value: EnsembleSamples) -> tuple[np.ndarray, ...]:
    return (
        _readonly(value.samples_unit, float, "samples_unit", 3),
        _readonly(value.log_probability, float, "log_probability", 2),
        _readonly(value.acceptance_fraction, float, "acceptance_fraction", 1),
        _readonly(value.split_rhat, float, "split_rhat", 1),
        _readonly(value.effective_sample_size, float, "effective_sample_size", 1),
    )


def _validate_ensemble_axes(arrays: tuple[np.ndarray, ...]) -> None:
    draws, walkers, dimensions = arrays[0].shape
    valid = (
        draws > 0
        and dimensions > 0
        and arrays[1].shape == (draws, walkers)
        and arrays[2].shape == (walkers,)
        and arrays[3].shape == (dimensions,)
        and arrays[4].shape == (dimensions,)
    )
    if not valid:
        raise ValueError("ensemble diagnostic arrays have incompatible shapes")


@dataclass(frozen=True, slots=True)
class EnsembleSamples:
    """Retained unit-space ensemble and aligned convergence diagnostics."""

    samples_unit: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: np.ndarray
    split_rhat: np.ndarray
    effective_sample_size: np.ndarray

    def __post_init__(self) -> None:
        arrays = _ensemble_arrays(self)
        _validate_ensemble_axes(arrays)
        names = (
            "samples_unit",
            "log_probability",
            "acceptance_fraction",
            "split_rhat",
            "effective_sample_size",
        )
        _set_array_fields(self, names, arrays)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


@dataclass(frozen=True, slots=True)
class McmcReport:
    """Physical retained samples bound to config, seed, and candidate identity.

    Sample rows and log probabilities share the flattened retained-draw axis;
    acceptance fractions remain walker-owned, while R-hat and effective sample
    size remain parameter-owned. The optional candidate ID records which fit
    state supplied the sampling center without retaining that mutable runtime.

    Constraint targets are deterministic transforms of sampled coordinates,
    while locked coordinates are constant physical values. Both stay separate
    from sampled axes so R-hat and ESS describe free dimensions only.
    """

    config: McmcConfig
    child_seed: int
    parameter_names: tuple[str, ...]
    samples_physical: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: np.ndarray
    split_rhat: np.ndarray
    effective_sample_size: np.ndarray
    boundary_hits: tuple[str, ...]
    label: str = "目标函数伪后验"
    warnings: tuple[str, ...] = ()
    candidate_id: str | None = None
    prior_conflicts: tuple[str, ...] = ()
    derived_parameter_names: tuple[str, ...] = ()
    derived_samples_physical: np.ndarray | None = None
    fixed_parameter_values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _validate_mcmc_identity(self)
        parameter_names = _nonempty_unique_names(self.parameter_names, "parameter names")
        arrays = _mcmc_arrays(self)
        _validate_mcmc_axes(self.config, parameter_names, arrays)
        _validate_mcmc_values(arrays)
        object.__setattr__(self, "parameter_names", parameter_names)
        array_fields = (
            "samples_physical",
            "log_probability",
            "acceptance_fraction",
            "split_rhat",
            "effective_sample_size",
        )
        _set_array_fields(self, array_fields, arrays)
        object.__setattr__(self, "boundary_hits", tuple(self.boundary_hits))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "prior_conflicts", tuple(self.prior_conflicts))
        derived_names, derived = validate_derived_samples(
            parameter_names,
            arrays[0].shape[0],
            self.derived_parameter_names,
            self.derived_samples_physical,
        )
        object.__setattr__(self, "derived_parameter_names", derived_names)
        object.__setattr__(self, "derived_samples_physical", derived)
        object.__setattr__(
            self,
            "fixed_parameter_values",
            validate_fixed_parameter_values(
                parameter_names,
                derived_names,
                self.fixed_parameter_values,
            ),
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


@dataclass(frozen=True, slots=True)
class UncertaintyReport:
    """Combined covariance, profile, bootstrap, residual, and MCMC evidence."""

    correlation_names: tuple[str, ...]
    correlation_matrix: np.ndarray
    profiles: tuple[ParameterProfile, ...]
    bootstrap_intervals: tuple[tuple[str, float, float], ...]
    bootstrap_failure_rate: float
    boundary_hits: tuple[str, ...]
    strong_correlations: tuple[tuple[str, str, float], ...]
    systematic_residual: bool
    diagnostics: tuple[PhysicsDiagnostic, ...]
    residual_autocorrelation: bool = False
    mcmc: McmcReport | None = None
    candidate_id: str | None = None
    bootstrap_performed: bool = True
    sld_bands: SldUncertaintyBands | None = None
    prior_conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.correlation_names)
        matrix = _readonly(self.correlation_matrix, float, "correlation_matrix", 2)
        if matrix.shape != (len(names), len(names)):
            raise ValueError("correlation matrix shape must match correlation names")
        if not isfinite(self.bootstrap_failure_rate) or not 0.0 <= self.bootstrap_failure_rate <= 1.0:
            raise ValueError("bootstrap_failure_rate must be in [0, 1]")
        object.__setattr__(self, "correlation_names", names)
        object.__setattr__(self, "correlation_matrix", matrix)
        object.__setattr__(self, "profiles", tuple(self.profiles))
        object.__setattr__(self, "bootstrap_intervals", tuple(self.bootstrap_intervals))
        object.__setattr__(self, "boundary_hits", tuple(self.boundary_hits))
        object.__setattr__(self, "strong_correlations", tuple(self.strong_correlations))
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(value, PhysicsDiagnostic) for value in diagnostics):
            raise TypeError("diagnostics must contain PhysicsDiagnostic values")
        if self.candidate_id is not None and not self.candidate_id:
            raise ValueError("candidate_id must be a nonempty string or None")
        if not isinstance(self.bootstrap_performed, bool):
            raise TypeError("bootstrap_performed must be bool")
        _validate_optional_sld_bands(self.sld_bands)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "prior_conflicts", tuple(self.prior_conflicts))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


def _search_result_fields() -> tuple[str, ...]:
    return (
        "parameter_definitions",
        "candidates",
        "best_index",
        "warnings",
        "child_seeds",
        "stage_summaries",
        "region_labels",
        "region_weights",
    )


@dataclass(frozen=True, slots=True)
class FitResult:
    """Complete public fit result with search and analysis evidence."""

    parameter_definitions: tuple[ParameterDefinition, ...]
    candidates: tuple[FitCandidate, ...]
    best_index: int | None
    confidence: ConfidenceClass
    warnings: tuple[str, ...]
    child_seeds: tuple[int, ...]
    stage_summaries: tuple[FitStageSummary, ...]
    region_labels: np.ndarray
    region_weights: np.ndarray
    uncertainty: UncertaintyReport | None
    classification_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Reconstruct the fitting-only value so the flat public schema shares
        # exactly the same candidate graph and immutable-array validation.
        search_result = FitSearchResult(
            parameter_definitions=self.parameter_definitions,
            candidates=self.candidates,
            best_index=self.best_index,
            warnings=self.warnings,
            child_seeds=self.child_seeds,
            stage_summaries=self.stage_summaries,
            region_labels=self.region_labels,
            region_weights=self.region_weights,
        )
        if not isinstance(self.confidence, ConfidenceClass):
            raise TypeError("confidence must be ConfidenceClass")
        if self.uncertainty is not None and not isinstance(self.uncertainty, UncertaintyReport):
            raise TypeError("uncertainty must be UncertaintyReport or None")
        evidence = tuple(self.classification_evidence)
        if any(not isinstance(value, str) or not value for value in evidence):
            raise ValueError("classification_evidence must contain nonempty strings")
        for field in _search_result_fields():
            object.__setattr__(self, field, getattr(search_result, field))
        object.__setattr__(self, "classification_evidence", evidence)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)

    @classmethod
    def from_search(
        cls,
        search_result: FitSearchResult,
        *,
        confidence: ConfidenceClass,
        uncertainty: UncertaintyReport | None,
        classification_evidence: tuple[str, ...] = (),
    ) -> FitResult:
        if not isinstance(search_result, FitSearchResult):
            raise TypeError("search_result must be FitSearchResult")
        values = {field: getattr(search_result, field) for field in _search_result_fields()}
        return cls(
            **values,
            confidence=confidence,
            uncertainty=uncertainty,
            classification_evidence=classification_evidence,
        )

    @property
    def best_candidate(self) -> FitCandidate | None:
        return None if self.best_index is None else self.candidates[self.best_index]


@dataclass(frozen=True, slots=True)
class StructureEvidence:
    """Observed and modeled fringe counts plus observed peak positions."""

    m_data: int
    m_model: int
    warning: str | None
    peak_positions_a: tuple[float, ...]

    def __post_init__(self) -> None:
        for field in ("m_data", "m_model"):
            _evidence_count(getattr(self, field), field)
        peaks = _positive_values(self.peak_positions_a, "peak_positions_a")
        if len(peaks) != self.m_data:
            raise ValueError("peak_positions_a length must match m_data")
        if self.warning is not None and not isinstance(self.warning, str):
            raise TypeError("warning must be str or None")
        object.__setattr__(self, "peak_positions_a", peaks)
