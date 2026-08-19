"""Deterministic candidate clustering and confidence classification."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import numpy as np

from xrr_fitter.model.analysis import ConfidenceClass
from xrr_fitter.model.fitting import (
    ConfidenceThresholds,
    FitEvaluationContext,
    candidate_selection_objective,
)
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _validated_vectors(vectors: object) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or np.any(~np.isfinite(values)):
        raise ValueError("vectors must be a nonempty finite matrix")
    return values


def _rms_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def cluster_unit_vectors(
    vectors: np.ndarray,
    join_distance: float = 0.05,
) -> tuple[tuple[int, ...], ...]:
    values = _validated_vectors(vectors)
    if not isfinite(join_distance) or join_distance <= 0.0:
        raise ValueError("join_distance must be positive and finite")
    remaining = set(range(values.shape[0]))
    clusters: list[tuple[int, ...]] = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        frontier = [first]
        cluster = {first}
        while frontier:
            current = frontier.pop()
            linked = {index for index in remaining if _rms_distance(values[current], values[index]) < join_distance}
            remaining.difference_update(linked)
            cluster.update(linked)
            frontier.extend(sorted(linked))
        clusters.append(tuple(sorted(cluster)))
    return tuple(clusters)


def cluster_candidates(
    candidates: tuple[object, ...],
    join_distance: float = 0.05,
) -> tuple[tuple[int, ...], ...]:
    values = tuple(candidates)
    if not values:
        raise ValueError("candidates must be nonempty")
    vectors = tuple(np.asarray(candidate.unit_vector, dtype=float) for candidate in values)
    if any(vector.ndim != 1 or vector.shape != vectors[0].shape for vector in vectors):
        raise ValueError("candidate unit vectors must be matching finite vectors")
    return cluster_unit_vectors(np.vstack(vectors), join_distance)


def _candidate_arrays(
    vectors: np.ndarray,
    costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=float)
    objective = np.asarray(costs, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or objective.shape != (values.shape[0],):
        raise ValueError("candidate vectors and costs have incompatible shapes")
    return values, objective


def _validate_cluster_partition(
    clusters: tuple[tuple[int, ...], ...],
    candidate_count: int,
) -> None:
    flattened = tuple(index for cluster in clusters for index in cluster)
    if sorted(flattened) != list(range(candidate_count)) or len(flattened) != len(set(flattened)):
        raise ValueError("clusters must partition candidate indices")


def _validity_mask(valid: np.ndarray | None, candidate_count: int) -> np.ndarray:
    mask = np.ones(candidate_count, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != (candidate_count,):
        raise ValueError("valid mask has incompatible shape")
    return mask


def _validated_evidence(
    vectors: np.ndarray,
    costs: np.ndarray,
    clusters: tuple[tuple[int, ...], ...],
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    values, objective = _candidate_arrays(vectors, costs)
    _validate_cluster_partition(clusters, values.shape[0])
    mask = _validity_mask(valid, values.shape[0])
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(objective)) or not np.all(mask):
        return None
    return values, objective


def _cluster_representatives(
    clusters: tuple[tuple[int, ...], ...],
    costs: np.ndarray,
) -> tuple[tuple[int, ...], int, tuple[int, ...], float, float]:
    representatives = tuple(min(cluster, key=lambda index: (costs[index], index)) for cluster in clusters)
    best_position = min(
        range(len(clusters)),
        key=lambda position: (costs[representatives[position]], representatives[position]),
    )
    best_cluster = clusters[best_position]
    best_objective = float(costs[representatives[best_position]])
    return representatives, best_position, best_cluster, best_objective, representatives[best_position]


def _multiple_reason(
    vectors: np.ndarray,
    costs: np.ndarray,
    clusters: tuple[tuple[int, ...], ...],
    *,
    distinct_cluster_distance: float,
    equivalent_delta: float,
    profile_path_merge: Callable[[np.ndarray, np.ndarray, float], bool] | None,
) -> str | None:
    representatives, best_position, _cluster, best_objective, best_index = _cluster_representatives(clusters, costs)
    limit = best_objective + equivalent_delta
    for position, other_index in enumerate(representatives):
        if position == best_position or costs[other_index] > limit:
            continue
        if _rms_distance(vectors[best_index], vectors[other_index]) >= distinct_cluster_distance:
            return "distinct_equivalent_clusters"
        if profile_path_merge is not None and not profile_path_merge(vectors[best_index], vectors[other_index], limit):
            return "profile_path_merge_failed"
    return None


def _correlated_reasons(
    best_cluster: tuple[int, ...],
    boundary_hits: tuple[str, ...],
    strong_correlations: tuple[tuple[str, str, float], ...],
    profiles_closed: bool,
    systematic_residual: bool,
    diagnostics: tuple[PhysicsDiagnostic, ...],
) -> tuple[str, ...]:
    conditions = (
        (len(best_cluster) == 2, "two_seed_cluster_support"),
        (bool(boundary_hits), "boundary_hit"),
        (bool(strong_correlations), "strong_correlation"),
        (not profiles_closed, "profile_interval_open"),
        (systematic_residual, "systematic_residual"),
        (
            any(value.code == "nevot_croce_applicability_exceeded" for value in diagnostics),
            "nevot_croce_applicability_exceeded",
        ),
    )
    return tuple(reason for active, reason in conditions if active)


def classify_candidate_evidence_with_reasons(
    vectors: np.ndarray,
    costs: np.ndarray,
    clusters: tuple[tuple[int, ...], ...],
    *,
    profile_path_merge: Callable[[np.ndarray, np.ndarray, float], bool] | None = None,
    valid: np.ndarray | None = None,
    boundary_hits: tuple[str, ...] = (),
    strong_correlations: tuple[tuple[str, str, float], ...] = (),
    profiles_closed: bool = True,
    fully_open_primary_profile: bool = False,
    systematic_residual: bool = False,
    diagnostics: tuple[PhysicsDiagnostic, ...] = (),
    distinct_cluster_distance: float = 0.10,
    equivalent_cost_fraction: float = 0.02,
    equivalent_cost_floor: float = 1e-5,
) -> tuple[ConfidenceClass, tuple[str, ...]]:
    if not clusters:
        return ConfidenceClass.UNTRUSTED, ("missing_candidate_clusters",)
    evidence = _validated_evidence(vectors, costs, clusters, valid)
    if evidence is None:
        return ConfidenceClass.UNTRUSTED, ("invalid_candidate_evidence",)
    values, objective = evidence
    representatives, best_position, best_cluster, best_objective, _best_index = _cluster_representatives(
        clusters, objective
    )
    del representatives, best_position
    delta = max(equivalent_cost_fraction * abs(best_objective), equivalent_cost_floor)
    multiple = _multiple_reason(
        values,
        objective,
        clusters,
        distinct_cluster_distance=distinct_cluster_distance,
        equivalent_delta=delta,
        profile_path_merge=None if len(best_cluster) < 2 else profile_path_merge,
    )
    if multiple is not None:
        return ConfidenceClass.MULTIPLE, (multiple,)
    if len(best_cluster) < 2:
        return ConfidenceClass.UNTRUSTED, ("insufficient_cluster_support",)
    if fully_open_primary_profile:
        return ConfidenceClass.MULTIPLE, ("primary_profile_open",)
    reasons = _correlated_reasons(
        best_cluster,
        tuple(boundary_hits),
        tuple(strong_correlations),
        profiles_closed,
        systematic_residual,
        tuple(diagnostics),
    )
    confidence = ConfidenceClass.CORRELATED if reasons else ConfidenceClass.TRUSTED
    return confidence, reasons


def classify_candidate_evidence(*args, **kwargs) -> ConfidenceClass:
    return classify_candidate_evidence_with_reasons(*args, **kwargs)[0]


def _thresholds(problem: object) -> ConfidenceThresholds:
    config = getattr(problem, "config", None)
    return getattr(config, "confidence", ConfidenceThresholds())


def _profiles_closed(profiles: tuple[object, ...]) -> bool:
    return all(profile.lower_closed and profile.upper_closed for profile in profiles)


def _fully_open_primary(profiles: tuple[object, ...]) -> bool:
    return any(
        ("thickness_a" in profile.name or "period_a" in profile.name)
        and not profile.lower_closed
        and not profile.upper_closed
        for profile in profiles
    )


def classify_result_with_evidence(
    problem: FitEvaluationContext,
    candidates: tuple[object, ...],
    report: object,
    *,
    profile_path_merge: Callable[[np.ndarray, np.ndarray, float], bool] | None = None,
) -> tuple[ConfidenceClass, tuple[str, ...]]:
    if report.bootstrap_failure_rate > 0.20:
        return ConfidenceClass.UNTRUSTED, ("bootstrap_failure_rate",)
    active = tuple(
        candidate for candidate in candidates if getattr(candidate, "stop_reason", None) != "early_eliminated"
    )
    if not active:
        return ConfidenceClass.UNTRUSTED, ("no_active_candidates",)
    vectors = np.vstack([candidate.unit_vector for candidate in active])
    costs = np.asarray(
        [candidate_selection_objective(candidate) for candidate in active],
        dtype=float,
    )
    valid = np.asarray([candidate.valid for candidate in active], dtype=bool)
    thresholds = _thresholds(problem)
    clusters = cluster_unit_vectors(vectors, thresholds.cluster_join_distance)
    merge = profile_path_merge
    if merge is None and hasattr(problem, "variables"):
        from xrr_fitter.analysis.profiles import default_profile_path_merge

        def merge(first, second, limit):
            return default_profile_path_merge(problem, first, second, limit)

    profiles = tuple(report.profiles)
    return classify_candidate_evidence_with_reasons(
        vectors,
        costs,
        clusters,
        profile_path_merge=merge,
        valid=valid,
        boundary_hits=tuple(report.boundary_hits),
        strong_correlations=tuple(report.strong_correlations),
        profiles_closed=_profiles_closed(profiles),
        fully_open_primary_profile=_fully_open_primary(profiles),
        systematic_residual=bool(report.systematic_residual),
        diagnostics=tuple(report.diagnostics),
        distinct_cluster_distance=thresholds.distinct_cluster_distance,
        equivalent_cost_fraction=thresholds.equivalent_cost_fraction,
        equivalent_cost_floor=thresholds.equivalent_cost_floor,
    )


def classify_result(*args, **kwargs) -> ConfidenceClass:
    return classify_result_with_evidence(*args, **kwargs)[0]
