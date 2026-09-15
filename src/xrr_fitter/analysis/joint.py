"""Global uncertainty and confidence for one aligned joint ensemble."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from xrr_fitter.analysis.classification import (
    classify_candidate_evidence_with_reasons,
    cluster_unit_vectors,
)
from xrr_fitter.analysis.covariance import covariance_method, covariance_summary, joint_covariance
from xrr_fitter.analysis.diagnostics import aggregate_residual_flag, build_residual_evidence
from xrr_fitter.analysis.residual_calibration import validate_residual_owner
from xrr_fitter.evaluation import EvaluationConstraintError
from xrr_fitter.model.analysis import ConfidenceClass, CovarianceEvidence, ResidualEvidence, UncertaintyReport
from xrr_fitter.model.fitting import ConfidenceThresholds
from xrr_fitter.model.joint_bootstrap_provenance import validate_joint_bootstrap
from xrr_fitter.model.provenance import joint_residual_owner_sha256


@dataclass(frozen=True, slots=True)
class _JointEnsemble:
    names: tuple[str, ...]
    identifiers: tuple[str, ...]
    costs: np.ndarray
    validity: np.ndarray
    diagnostics: tuple[tuple[object, ...], ...]
    vectors: np.ndarray
    physical: np.ndarray

    @property
    def count(self) -> int:
        return len(self.identifiers)

    @property
    def width(self) -> int:
        return len(self.names)


def _matrix(rows: object, count: int, width: int, field: str) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.shape != (count, width):
        raise ValueError(f"joint {field} must match the candidate and variable axes")
    return values


def _scaled_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        scales = np.max(np.abs(values), axis=0)
        normalized = np.divide(
            values,
            scales,
            out=np.zeros_like(values),
            where=scales > 0.0,
        )
    return normalized, scales


def _parameter_spread(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        spread = np.std(values, axis=0, ddof=1)
    if np.all(np.isfinite(spread)):
        return spread
    normalized, scales = _scaled_columns(values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        spread = np.std(normalized, axis=0, ddof=1) * scales
    if np.any(~np.isfinite(spread)):
        raise ValueError("joint parameter spread is not finite")
    return spread


def _strong_correlations(
    names: tuple[str, ...],
    correlation: np.ndarray,
    threshold: float,
) -> tuple[tuple[str, str, float], ...]:
    return tuple(
        (first, names[second_index], float(correlation[first_index, second_index]))
        for first_index, first in enumerate(names)
        for second_index in range(first_index + 1, len(names))
        if abs(float(correlation[first_index, second_index])) >= threshold
    )


def _validated_ensemble(
    variable_names: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    unit_vectors: object,
    physical_values: object,
    objectives: tuple[float, ...],
    valid: tuple[bool, ...],
    diagnostics: tuple[tuple[object, ...], ...],
) -> _JointEnsemble:
    names = tuple(variable_names)
    identifiers = tuple(candidate_ids)
    costs = np.asarray(objectives, dtype=float)
    validity = np.asarray(valid, dtype=bool)
    diagnostic_rows = tuple(tuple(row) for row in diagnostics)
    count = len(identifiers)
    width = len(names)
    if costs.shape != (count,) or validity.shape != (count,):
        raise ValueError("joint objective and validity axes must match candidates")
    if len(diagnostic_rows) != count:
        raise ValueError("joint diagnostic rows must match candidates")
    return _JointEnsemble(
        names,
        identifiers,
        costs,
        validity,
        diagnostic_rows,
        _matrix(unit_vectors, count, width, "unit vectors"),
        _matrix(physical_values, count, width, "physical values"),
    )


def _eligible_indices(ensemble: _JointEnsemble) -> tuple[int, ...]:
    return tuple(
        index for index in range(ensemble.count) if ensemble.validity[index] and isfinite(float(ensemble.costs[index]))
    )


def _boundary_hits(
    names: tuple[str, ...],
    vector: np.ndarray | None,
    fraction: float,
) -> tuple[str, ...]:
    if vector is None:
        return ()
    return tuple(
        name for name, value in zip(names, vector, strict=True) if value <= fraction or value >= 1.0 - fraction
    )


def _joint_calibration(names: tuple[str, ...], best_vector: np.ndarray | None, point_evidence: Callable | None):
    if best_vector is not None and point_evidence is not None:
        return point_evidence(best_vector)
    return CovarianceEvidence(names, None, "not_evaluated", 0, (), "joint_numerical_evidence_missing"), ()


def _best_joint_diagnostics(ensemble: _JointEnsemble, best_index: int | None, residuals: tuple[ResidualEvidence, ...]):
    if residuals:
        return tuple(item for member in residuals for item in member.diagnostics)
    return () if best_index is None else ensemble.diagnostics[best_index]


def _uncertainty_report(
    ensemble: _JointEnsemble,
    thresholds: ConfidenceThresholds,
    point_evidence: Callable | None,
) -> UncertaintyReport:
    eligible = _eligible_indices(ensemble)
    values = ensemble.physical[np.asarray(eligible, dtype=int)] if eligible else ensemble.physical[:0]
    best_index = min(
        eligible,
        key=lambda index: float(ensemble.costs[index]),
        default=None,
    )
    best_vector = None if best_index is None else ensemble.vectors[best_index]
    spread = _parameter_spread(values) if values.shape[0] >= 2 else None
    covariance, residuals = _joint_calibration(ensemble.names, best_vector, point_evidence)
    correlation, sigma = covariance_summary(covariance)
    return UncertaintyReport(
        correlation_names=ensemble.names,
        correlation_matrix=correlation,
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0 if eligible else 1.0,
        boundary_hits=_boundary_hits(
            ensemble.names,
            best_vector,
            thresholds.boundary_fraction,
        ),
        strong_correlations=_strong_correlations(
            ensemble.names,
            correlation,
            thresholds.strong_correlation,
        ),
        systematic_residual=aggregate_residual_flag(residuals, "systematic"),
        diagnostics=_best_joint_diagnostics(ensemble, best_index, residuals),
        residual_autocorrelation=aggregate_residual_flag(residuals, "autocorrelation"),
        candidate_id=None if best_index is None else ensemble.identifiers[best_index],
        bootstrap_performed=False,
        prior_conflicts=(),
        parameter_sigma=sigma,
        covariance_evidence=covariance,
        member_residuals=residuals,
        search_parameter_spread=spread,
    )


def _with_bootstrap(report, vectors, candidate_ids, bootstrap, bootstrap_owner):
    if bootstrap is None or report.candidate_id is None or not report.correlation_names:
        return report
    if bootstrap_owner is None:
        raise ValueError("joint bootstrap requires an independent numerical owner check")
    vector = vectors[candidate_ids.index(report.candidate_id)]
    expected = bootstrap_owner(report.candidate_id, vector)
    sampling = bootstrap(report.candidate_id, vector)
    validate_joint_bootstrap(sampling, report.candidate_id, expected, report.correlation_names)
    return replace(
        report,
        bootstrap_evidence=sampling,
        bootstrap_intervals=sampling.intervals,
        bootstrap_failure_rate=sampling.failure_rate,
        bootstrap_performed=True,
    )


def analyze_joint_ensemble(
    *,
    variable_names: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    unit_vectors: object,
    physical_values: object,
    objectives: tuple[float, ...],
    valid: tuple[bool, ...],
    diagnostics: tuple[tuple[object, ...], ...],
    thresholds: ConfidenceThresholds,
    point_evidence: Callable | None = None,
    bootstrap: Callable | None = None,
    bootstrap_owner: Callable | None = None,
) -> tuple[UncertaintyReport, ConfidenceClass, tuple[str, ...]]:
    """Build and classify one global Stage-E candidate ensemble."""
    ensemble = _validated_ensemble(
        variable_names,
        candidate_ids,
        unit_vectors,
        physical_values,
        objectives,
        valid,
        diagnostics,
    )
    report = _uncertainty_report(ensemble, thresholds, point_evidence)
    if ensemble.count == 0:
        return report, ConfidenceClass.UNTRUSTED, ("no_active_candidates",)
    report = _with_bootstrap(report, ensemble.vectors, ensemble.identifiers, bootstrap, bootstrap_owner)
    if report.bootstrap_performed and report.bootstrap_failure_rate > 0.20:
        return report, ConfidenceClass.UNTRUSTED, ("bootstrap_failure_rate",)
    clusters = (
        (tuple(range(ensemble.count)),)
        if ensemble.width == 0
        else cluster_unit_vectors(ensemble.vectors, thresholds.cluster_join_distance)
    )
    confidence, evidence = classify_candidate_evidence_with_reasons(
        ensemble.vectors,
        ensemble.costs,
        clusters,
        valid=ensemble.validity,
        boundary_hits=report.boundary_hits,
        strong_correlations=report.strong_correlations,
        systematic_residual=report.systematic_residual,
        covariance_available=report.covariance is not None,
        diagnostics=report.diagnostics,
        distinct_cluster_distance=thresholds.distinct_cluster_distance,
        equivalent_cost_fraction=thresholds.equivalent_cost_fraction,
        equivalent_cost_floor=thresholds.equivalent_cost_floor,
    )
    return report, confidence, evidence


def _joint_residual_evidence(
    dataset_ids,
    problems,
    unit_vector,
    evaluations,
    residual_evidence,
    layout_fingerprint,
) -> tuple[ResidualEvidence, ...]:
    if residual_evidence is not None:
        evidence = tuple(residual_evidence)
        if layout_fingerprint is None or len(evidence) != len(dataset_ids):
            raise ValueError("joint residual evidence requires an aligned member axis and layout identity")
        owner = joint_residual_owner_sha256(problems, dataset_ids, unit_vector, evaluations, layout_fingerprint)
        for dataset_id, member in zip(dataset_ids, evidence, strict=True):
            validate_residual_owner(member, owner, dataset_id)
        return evidence
    residuals = []
    for dataset_id, problem, evaluation in zip(dataset_ids, problems, evaluations, strict=True):
        values = np.full(problem.data.fit_mask.shape, np.nan)
        if evaluation.valid:
            values[problem.data.fit_mask] = evaluation.fit_residuals
        residuals.append(build_residual_evidence(problem, values, evaluation.diagnostics, dataset_id=dataset_id))
    return tuple(residuals)


def analyze_joint_point(
    names: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    problems: tuple,
    unit_vector: np.ndarray,
    local_evaluations: tuple,
    inference_layout: Callable,
    *,
    residual_evidence: tuple[ResidualEvidence, ...] | None = None,
    layout_fingerprint: str | None = None,
) -> tuple[CovarianceEvidence, tuple[ResidualEvidence, ...]]:
    """Use one owned joint family for covariance; low-level calls never invent calibration."""
    evidence = _joint_residual_evidence(
        dataset_ids,
        problems,
        unit_vector,
        local_evaluations,
        residual_evidence,
        layout_fingerprint,
    )
    try:
        covariance = joint_covariance(names, unit_vector, problems, inference_layout(), evidence)
    except (EvaluationConstraintError, FloatingPointError, np.linalg.LinAlgError) as error:
        method = covariance_method(tuple(problem.config.noise_model for problem in problems))
        covariance = CovarianceEvidence(names, None, method, 0, (), f"{type(error).__name__}:{error}")
    return covariance, evidence


__all__ = ["analyze_joint_ensemble", "analyze_joint_point"]
