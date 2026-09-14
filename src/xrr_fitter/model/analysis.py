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
from itertools import chain
from math import isfinite

import numpy as np

from xrr_fitter.model.bootstrap import MIN_BOOTSTRAP_SUCCESS as MIN_BOOTSTRAP_SUCCESS
from xrr_fitter.model.bootstrap import BootstrapResult as BootstrapResult
from xrr_fitter.model.fitting import (
    FitCandidate,
    FitSearchResult,
    FitStageSummary,
    PhysicsDiagnostic,
)
from xrr_fitter.model.inference import CovarianceEvidence as CovarianceEvidence
from xrr_fitter.model.inference import ResidualEvidence as ResidualEvidence
from xrr_fitter.model.joint_bootstrap_provenance import validate_joint_bootstrap_report
from xrr_fitter.model.mcmc_samples import McmcReport as McmcReport
from xrr_fitter.model.mcmc_sampling import EnsembleSamples as EnsembleSamples
from xrr_fitter.model.mcmc_sampling import McmcConfig as McmcConfig
from xrr_fitter.model.parameters import ParameterDefinition, ParameterReference
from xrr_fitter.model.profile import ParameterProfile as ParameterProfile

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


def _evidence_count(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _positive_values(values: object, field: str) -> tuple[float, ...]:
    result = tuple(values)
    if any(not isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{field} must contain positive finite values")
    return result


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


def _parameter_sigma(
    value: np.ndarray | None,
    dimension: int,
) -> np.ndarray | None:
    if value is None:
        return None
    sigma = _readonly(value, float, "parameter_sigma", 1)
    if sigma.shape != (dimension,):
        raise ValueError("parameter_sigma length must match correlation names")
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("parameter_sigma must be finite and nonnegative")
    return sigma


def _parameter_members(value: object, dimension: int) -> tuple[tuple[ParameterReference, ...], ...] | None:
    if value is None:
        return None
    members = tuple(tuple(group) for group in value)
    if len(members) != dimension:
        raise ValueError("parameter_members must match the correlation axis")
    if any(not group for group in members):
        raise ValueError("parameter_members groups must not be empty")
    references = tuple(chain.from_iterable(members))
    if any(not isinstance(reference, ParameterReference) for reference in references):
        raise TypeError("parameter_members must contain ParameterReference values")
    if len(set(references)) != len(references):
        raise ValueError("parameter_members references must be globally unique")
    return members


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
    systematic_residual: bool | None
    diagnostics: tuple[PhysicsDiagnostic, ...]
    residual_autocorrelation: bool | None = None
    mcmc: McmcReport | None = None
    candidate_id: str | None = None
    bootstrap_performed: bool = False
    sld_bands: SldUncertaintyBands | None = None
    prior_conflicts: tuple[str, ...] = ()
    parameter_sigma: np.ndarray | None = None
    covariance_evidence: CovarianceEvidence | None = None
    member_residuals: tuple[ResidualEvidence, ...] = ()
    search_parameter_spread: np.ndarray | None = None
    bootstrap_evidence: BootstrapResult | None = None
    # None is a local-name axis; joint publication binds every ordered global
    # axis to the exact dataset/parameter references used by the compiled fit.
    parameter_members: tuple[tuple[ParameterReference, ...], ...] | None = None

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
        sigma = _parameter_sigma(self.parameter_sigma, len(names))
        if sigma is not None:
            object.__setattr__(self, "parameter_sigma", sigma)
        object.__setattr__(self, "parameter_members", _parameter_members(self.parameter_members, len(names)))
        self._validate_inference(names)
        self._validate_bootstrap_summary()
        self._validate_owned_bootstrap()

    def _validate_bootstrap_summary(self) -> None:
        evidence = self.bootstrap_evidence
        if evidence is None:
            if self.bootstrap_performed or self.bootstrap_intervals:
                raise ValueError("performed bootstrap requires sampling evidence")
            return
        if not isinstance(evidence, BootstrapResult) or evidence.parameter_names != self.correlation_names:
            raise ValueError("bootstrap evidence must match the report parameter axis")
        if (
            not self.bootstrap_performed
            or self.bootstrap_intervals != evidence.intervals
            or self.bootstrap_failure_rate != evidence.failure_rate
        ):
            raise ValueError("bootstrap summary must agree with sampling evidence")

    def _validate_owned_bootstrap(self) -> None:
        evidence = self.bootstrap_evidence
        if evidence is not None and (self.parameter_members is not None or evidence.joint_owner_sha256 is not None):
            validate_joint_bootstrap_report(self)

    def _validate_inference(self, names: tuple[str, ...]) -> None:
        evidence = self.covariance_evidence
        if evidence is not None:
            if not isinstance(evidence, CovarianceEvidence) or evidence.names != names:
                raise ValueError("covariance evidence names must match correlation names")
            if evidence.matrix is None and self.parameter_sigma is not None:
                raise ValueError("unavailable covariance cannot publish parameter sigma")
            if evidence.matrix is not None:
                self._validate_covariance_summary(evidence.matrix)
        residuals = tuple(self.member_residuals)
        if any(not isinstance(value, ResidualEvidence) for value in residuals):
            raise TypeError("member_residuals must contain ResidualEvidence values")
        object.__setattr__(self, "member_residuals", residuals)
        self._validate_residual_summary(residuals)
        spread = _parameter_sigma(self.search_parameter_spread, len(names))
        object.__setattr__(self, "search_parameter_spread", spread)

    def _validate_covariance_summary(self, covariance: np.ndarray) -> None:
        sigma = self.parameter_sigma
        expected = np.sqrt(np.diag(covariance))
        if sigma is None or not np.allclose(sigma, expected, rtol=1e-10, atol=0.0):
            raise ValueError("parameter sigma must agree with covariance evidence")
        correlation = covariance / sigma[:, None] / sigma[None, :]
        if not np.allclose(self.correlation_matrix, correlation, rtol=1e-10, atol=1e-14):
            raise ValueError("correlation matrix must agree with covariance evidence")

    def _validate_residual_summary(self, residuals: tuple[ResidualEvidence, ...]) -> None:
        for summary, field in (
            (self.systematic_residual, "systematic"),
            (self.residual_autocorrelation, "autocorrelation"),
        ):
            if summary is not None and not isinstance(summary, bool):
                raise TypeError("residual summary must be boolean or unknown")
            if not residuals:
                continue
            values = tuple(getattr(item, field) for item in residuals)
            expected = True if True in values else (False if all(value is False for value in values) else None)
            if summary is not expected:
                raise ValueError("residual summary must agree with member evidence")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)

    @property
    def covariance(self) -> np.ndarray | None:
        """Return only the matrix with an explicit statistical provenance."""
        return None if self.covariance_evidence is None else self.covariance_evidence.matrix


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
