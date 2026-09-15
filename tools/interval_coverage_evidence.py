"""Read saved profile/bootstrap eligibility and support without refitting data."""

from __future__ import annotations

from math import isfinite, sqrt

import numpy as np

import xrr_fitter.api as api

NOMINAL_COVERAGE = 0.95


def _nullable(values: object) -> list:
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        return [_nullable(row) for row in array]
    return [float(value) if isfinite(value) else None for value in array]


def _interval(source: str, name: str, reason: str, **metadata) -> dict:
    return {
        "source": source,
        "parameter_name": name,
        "available": False,
        "bounds": None,
        "support_intervals": [],
        "kind": "unavailable",
        "confidence_level": None,
        "method": None,
        "unavailable_reason": reason,
        "details": {},
    } | metadata


def _profile_details(profile: api.ParameterProfile) -> dict:
    return {
        "values": _nullable(profile.values),
        "objectives": _nullable(profile.objectives),
        "nonfinite_objective_indices": np.flatnonzero(~np.isfinite(profile.objectives)).tolist(),
        "lower_closed": profile.lower_closed,
        "upper_closed": profile.upper_closed,
        "delta_total": profile.delta_total,
        "objective_point_count": profile.objective_point_count,
        "delta_per_point": profile.objective_delta,
        "interpolation": "piecewise_linear_sqrt_delta_J",
        "baseline": "minimum recorded J; threshold is saved delta_total / objective_point_count",
    }


def _profile_reason(profile: api.ParameterProfile) -> str | None:
    if profile.unavailable_reason is not None:
        return profile.unavailable_reason
    if profile.interval_kind != "likelihood_ratio" or profile.confidence_level != NOMINAL_COVERAGE:
        return "not_a_95_percent_likelihood_ratio_interval"
    delta = profile.objective_delta
    if delta is None or not isfinite(delta) or delta <= 0.0:
        return "invalid_profile_threshold"
    return _profile_trace_reason(profile)


def _profile_trace_reason(profile: api.ParameterProfile) -> str | None:
    if not np.all(np.isfinite(profile.objectives)):
        return "nonfinite_profile_trace"
    if profile.values.size < 2 or np.any(np.diff(profile.values) <= 0.0):
        return "nonmonotonic_profile_coordinates"
    if not profile.lower_closed or not profile.upper_closed:
        return "open_profile_support"
    return None


def _support_segment(left: float, right: float, a: float, b: float, threshold: float) -> list[float] | None:
    if a > threshold and b > threshold:
        return None
    if a <= threshold and b <= threshold:
        return [left, right]
    crossing = left + (right - left) * (threshold - a) / (b - a)
    return [left, crossing] if a <= threshold else [crossing, right]


def _profile_support(profile: api.ParameterProfile) -> list[list[float]]:
    values = np.asarray(profile.values, dtype=float)
    delta = np.sqrt(np.maximum(0.0, profile.objectives - np.min(profile.objectives)))
    threshold = sqrt(profile.objective_delta)
    support = []
    for left, right, a, b in zip(values[:-1], values[1:], delta[:-1], delta[1:], strict=True):
        segment = _support_segment(float(left), float(right), float(a), float(b), threshold)
        if segment is None:
            continue
        if support and segment[0] == support[-1][1]:
            support[-1][1] = segment[1]
        else:
            support.append(segment)
    return support


def _support_reason(profile: api.ParameterProfile, support: list) -> str | None:
    if not support:
        return "empty_profile_support"
    if len(support) != 1:
        return "disjoint_profile_support"
    if support[0][0] <= profile.values[0] or support[0][1] >= profile.values[-1]:
        return "unbracketed_profile_support"
    return None


def profile_interval(profile: api.ParameterProfile) -> dict:
    """Read a single closed LR interval, preserving unusable/disjoint traces.

    Linear interpolation is applied to sqrt(J - min(recorded J)), matching the
    existing profile support convention, not to J or to an invented covariance.
    No interpolation is attempted across a nonfinite observation or an excluded
    gap between disconnected support components.
    """
    reason = _profile_reason(profile)
    support = [] if reason is not None else _profile_support(profile)
    reason = reason or _support_reason(profile, support)
    return _interval(
        "profile",
        profile.name,
        reason,
        available=reason is None,
        bounds=support[0] if reason is None else None,
        support_intervals=support,
        kind=profile.interval_kind,
        confidence_level=profile.confidence_level,
        method=profile.method,
        details=_profile_details(profile),
    )


def _bootstrap_details(evidence) -> dict:
    return {
        "parameter_names": list(evidence.parameter_names),
        "samples": _nullable(evidence.samples),
        "successful_samples": evidence.successful_samples,
        "attempted_count": evidence.attempted_count,
        "failure_rate": evidence.failure_rate,
        "failure_reasons": [list(row) for row in evidence.failure_reasons],
        "candidate_id": evidence.candidate_id,
        "provenance_sha256": evidence.provenance_sha256,
    }


def _bootstrap_reason(evidence, name: str) -> str | None:
    if evidence.unavailable_reason is not None:
        return evidence.unavailable_reason
    if evidence.successful_samples < 200:
        return "insufficient_successful_samples"
    if evidence.interval_kind != "percentile_bootstrap" or evidence.confidence_level != NOMINAL_COVERAGE:
        return "not_a_95_percent_percentile_bootstrap"
    if name not in {row[0] for row in evidence.intervals}:
        return "bootstrap_parameter_missing"
    return None


def bootstrap_interval(evidence, name: str) -> dict:
    """Consume saved bootstrap bounds; never recalculate or top up replicates."""
    if evidence is None:
        return _interval("bootstrap", name, "bootstrap_not_performed")
    reason = _bootstrap_reason(evidence, name)
    bounds = None if reason else list(next(row[1:] for row in evidence.intervals if row[0] == name))
    return _interval(
        "bootstrap",
        name,
        reason,
        available=reason is None,
        bounds=bounds,
        support_intervals=[] if bounds is None else [bounds],
        kind=evidence.interval_kind,
        confidence_level=evidence.confidence_level,
        method=evidence.method,
        details=_bootstrap_details(evidence),
    )
