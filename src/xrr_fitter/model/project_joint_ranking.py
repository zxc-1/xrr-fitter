"""Reconstruct persisted V2 joint ranks without invoking the fit layer.

Masks count the full fitted observations, not source-array lengths or search
grids. Likelihood contributions add by point count; robust data retain equal
member mass. A saved scale prior contributes once under the total denominator,
even when members have different masks. Project validation establishes aligned
candidate identity before calling this arithmetic boundary.
"""

from __future__ import annotations

from math import isclose, isfinite

from xrr_fitter.model.parameters import _log10_ratio


def _validate_invalid_joint_rank(candidates: tuple[object, ...]) -> bool:
    if candidates[0].valid:
        return False
    if any(candidate.ranking_objective is not None for candidate in candidates):
        raise ValueError("joint candidates have inconsistent invalid ranking")
    return True


def _joint_objectives(candidates: tuple[object, ...]) -> tuple[float, ...]:
    objectives = tuple(candidate.objective for candidate in candidates)
    if any(not isfinite(value) for value in objectives):
        raise ValueError("joint candidates have nonfinite objective")
    return objectives


def _stable_objective_mean(objectives: tuple[float, ...]) -> float:
    expected = sum(objectives) / len(objectives)
    if not isfinite(expected):
        return sum(value / len(objectives) for value in objectives)
    return expected


def _scale_prior_penalty(candidate: object, prior: object) -> float:
    scale = next((value.value for value in candidate.parameters if value.name == "instrument.scale"), None)
    if scale is None or not isfinite(scale) or scale <= 0.0:
        raise ValueError("joint scale prior requires a positive instrument.scale parameter")
    residual = _log10_ratio(scale, prior.s_hat) / prior.tau_s_decades
    return residual**2


def _joint_prior_correction(
    dataset: object,
    candidate: object,
    point_count: int,
    total_points: int,
    member_count: int,
) -> float:
    prior = dataset.scale_prior
    if candidate.noise_model != "robust_log" or not prior.enabled:
        return 0.0
    correction = 1.0 / total_points - 1.0 / (member_count * point_count)
    if correction == 0.0:
        return 0.0
    return _scale_prior_penalty(candidate, prior) * correction


def _expected_joint_rank(candidates: tuple[object, ...], datasets: tuple[object, ...]) -> float:
    point_counts = tuple(sum(dataset.fit_mask) for dataset in datasets)
    if any(count < 1 for count in point_counts):
        raise ValueError("valid joint candidates require fitted points in every dataset")
    total_points = sum(point_counts)
    member_count = len(candidates)
    objectives = _joint_objectives(candidates)
    if all(candidate.noise_model == "robust_log" for candidate in candidates):
        expected = _stable_objective_mean(objectives)
    else:
        expected = sum(
            objective * (1.0 / member_count if candidate.noise_model == "robust_log" else count / total_points)
            for objective, candidate, count in zip(objectives, candidates, point_counts, strict=True)
        )
    # Local J includes z²/N_i. Only robust data are member-balanced; move each
    # saved scale prior from that local mean to its single total-N contribution.
    for dataset, candidate, count in zip(datasets, candidates, point_counts, strict=True):
        expected += _joint_prior_correction(dataset, candidate, count, total_points, member_count)
    return expected


def validate_joint_rank(candidates: tuple[object, ...], datasets: tuple[object, ...]) -> None:
    """Check one aligned global candidate using its saved objective evidence."""
    if _validate_invalid_joint_rank(candidates):
        return
    ranking = candidates[0].ranking_objective
    expected = _expected_joint_rank(candidates, datasets)
    if ranking is None or not isclose(ranking, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("joint candidate global ranking does not match saved objective")
