"""Global uncertainty and confidence for one aligned joint ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from xrr_fitter.analysis.classification import (
    classify_candidate_evidence_with_reasons,
    cluster_unit_vectors,
)
from xrr_fitter.model.analysis import ConfidenceClass, UncertaintyReport
from xrr_fitter.model.fitting import ConfidenceThresholds


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


def _correlation(values: np.ndarray, dimension: int) -> np.ndarray:
    if dimension == 0:
        return np.empty((0, 0), dtype=float)
    if values.shape[0] < 2:
        return np.eye(dimension, dtype=float)
    centered = values - np.mean(values, axis=0)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    denominator = np.outer(norms, norms)
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((dimension, dimension), dtype=float),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    return correlation


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
        index
        for index in range(ensemble.count)
        if ensemble.validity[index] and isfinite(float(ensemble.costs[index]))
    )


def _boundary_hits(
    names: tuple[str, ...],
    vector: np.ndarray | None,
    fraction: float,
) -> tuple[str, ...]:
    if vector is None:
        return ()
    return tuple(
        name
        for name, value in zip(names, vector, strict=True)
        if value <= fraction or value >= 1.0 - fraction
    )


def _uncertainty_report(
    ensemble: _JointEnsemble,
    thresholds: ConfidenceThresholds,
) -> UncertaintyReport:
    eligible = _eligible_indices(ensemble)
    values = (
        ensemble.physical[np.asarray(eligible, dtype=int)]
        if eligible
        else ensemble.physical[:0]
    )
    correlation = _correlation(values, ensemble.width)
    best_index = min(
        eligible,
        key=lambda index: float(ensemble.costs[index]),
        default=None,
    )
    best_vector = None if best_index is None else ensemble.vectors[best_index]
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
        systematic_residual=False,
        diagnostics=() if best_index is None else ensemble.diagnostics[best_index],
        residual_autocorrelation=False,
        candidate_id=None if best_index is None else ensemble.identifiers[best_index],
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
    report = _uncertainty_report(ensemble, thresholds)
    if ensemble.count == 0:
        return report, ConfidenceClass.UNTRUSTED, ("no_active_candidates",)
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
        diagnostics=report.diagnostics,
        distinct_cluster_distance=thresholds.distinct_cluster_distance,
        equivalent_cost_fraction=thresholds.equivalent_cost_fraction,
        equivalent_cost_floor=thresholds.equivalent_cost_floor,
    )
    return report, confidence, evidence


__all__ = ["analyze_joint_ensemble"]
