from __future__ import annotations

from collections import defaultdict

import numpy as np

from tests.support.synthetic_recovery_model import (
    DESIGN_THRESHOLDS,
    EXTRA_THRESHOLDS,
    RecoveryMetric,
    SyntheticCase,
)
from xrr_fitter.model.analysis import ConfidenceClass, ParameterProfile
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.parameters import ParameterDefinition as FitParameterDefinition
from xrr_fitter.model.parameters import physical_to_unit


def _best_candidate(result):
    assert result.best_index is not None, {"warnings": result.warnings}
    return result.candidates[result.best_index]


def _values_by_name(result) -> dict[str, float]:
    best = _best_candidate(result)
    return {parameter.name: parameter.value for parameter in best.parameters}


def _bounds_by_name(result) -> dict[str, tuple[float, float]]:
    best = _best_candidate(result)
    return {parameter.name: (parameter.lower, parameter.upper) for parameter in best.parameters}


def _profiles_by_name(result) -> dict[str, ParameterProfile]:
    report = result.uncertainty
    if report is None:
        return {}
    return {profile.name: profile for profile in report.profiles}


def _valid_truth_bounds(lower: float, upper: float, truth: float) -> bool:
    return bool(np.isfinite([lower, upper, truth]).all() and upper > lower)


def _parameter_geometry(
    name: str,
    truth: float,
    bounds: dict[str, tuple[float, float]],
    definitions: dict[str, FitParameterDefinition],
) -> tuple[FitParameterDefinition, float] | None:
    definition = definitions.get(name)
    if definition is None:
        return None
    parameter_bounds = bounds.get(name)
    if parameter_bounds is None:
        return None
    lower, upper = parameter_bounds
    if not _valid_truth_bounds(lower, upper, truth):
        return None
    if lower != definition.lower:
        return None
    return definition, upper


def _background_boundary_coordinate(
    definition: FitParameterDefinition,
    truth: float,
    upper: float,
    r_floor: float,
) -> float | None:
    # Match the protected log objective's resolvable scale while preserving
    # zero as the exact physical boundary.
    if definition.transform != "linear":
        return None
    if definition.lower != 0.0:
        return None
    if not np.isfinite(r_floor):
        return None
    if r_floor <= 0.0:
        return None
    return float(np.log1p(truth / r_floor) / np.log1p(upper / r_floor))


def _parameter_boundary_coordinate(
    name: str,
    truth: float,
    bounds: dict[str, tuple[float, float]],
    definitions: dict[str, FitParameterDefinition],
    r_floor: float,
) -> float | None:
    geometry = _parameter_geometry(name, truth, bounds, definitions)
    if geometry is None:
        return None
    definition, upper = geometry
    try:
        coordinate = physical_to_unit(definition, truth, dynamic_upper=upper)
    except ValueError:
        return None
    if definition.name == "instrument.background":
        return _background_boundary_coordinate(
            definition,
            truth,
            upper,
            r_floor,
        )
    return float(coordinate)


def _truth_inside_inner_bounds(
    parameter_truths: tuple[tuple[str, float], ...],
    bounds: dict[str, tuple[float, float]],
    definitions: dict[str, FitParameterDefinition],
    r_floor: float,
    fraction: float = 0.05,
) -> bool:
    for name, truth in parameter_truths:
        coordinate = _parameter_boundary_coordinate(
            name,
            truth,
            bounds,
            definitions,
            r_floor,
        )
        if coordinate is None:
            return False
        if coordinate <= fraction:
            return False
        if coordinate >= 1.0 - fraction:
            return False
    return True


def _metric_error(metric: RecoveryMetric, values: dict[str, float]) -> float:
    fitted = float(metric.value(values))
    threshold = {**DESIGN_THRESHOLDS, **EXTRA_THRESHOLDS}[metric.family]
    if threshold.error_kind == "relative":
        return abs(fitted / metric.truth - 1.0)
    if threshold.error_kind == "absolute":
        return abs(fitted - metric.truth)
    raise AssertionError(f"unknown error kind: {threshold.error_kind}")


def _threshold_crossing(
    values: np.ndarray,
    objectives: np.ndarray,
    index: int,
    best: float,
    threshold: float,
) -> float:
    left_height = np.sqrt(max(0.0, float(objectives[index]) - best))
    right_height = np.sqrt(max(0.0, float(objectives[index + 1]) - best))
    threshold_height = np.sqrt(threshold - best)
    fraction = np.clip(
        (threshold_height - left_height) / (right_height - left_height),
        0.0,
        1.0,
    )
    left = float(values[index])
    right = float(values[index + 1])
    return left + float(fraction) * (right - left)


def _supported_interval(
    values: np.ndarray,
    objectives: np.ndarray,
    supported: np.ndarray,
    index: int,
    best: float,
    threshold: float,
) -> tuple[float, float] | None:
    left_supported = bool(supported[index])
    right_supported = bool(supported[index + 1])
    if not left_supported and not right_supported:
        return None
    left = float(values[index])
    right = float(values[index + 1])
    if left_supported and right_supported:
        return left, right
    crossing = _threshold_crossing(values, objectives, index, best, threshold)
    return (left, crossing) if left_supported else (crossing, right)


def _profile_covers_truth(profile: ParameterProfile, truth: float) -> bool:
    values = np.asarray(profile.values, dtype=float)
    objectives = np.asarray(profile.objectives, dtype=float)
    finite = np.isfinite(values) & np.isfinite(objectives)
    if not np.isfinite(truth) or not np.any(finite):
        return False
    order = np.argsort(values[finite], kind="stable")
    values = values[finite][order]
    objectives = objectives[finite][order]
    best = float(np.min(objectives))
    threshold = best + max(0.02 * abs(best), 1e-5)
    supported = objectives <= threshold
    if np.any(np.isclose(values[supported], truth, rtol=0.0, atol=1e-12)):
        return True
    intervals = (
        _supported_interval(values, objectives, supported, index, best, threshold) for index in range(values.size - 1)
    )
    return any(interval is not None and interval[0] <= truth <= interval[1] for interval in intervals)


def _metric_truth_is_eligible(metric: RecoveryMetric) -> bool:
    threshold = {**DESIGN_THRESHOLDS, **EXTRA_THRESHOLDS}[metric.family]
    if threshold.minimum_truth is None:
        return True
    return metric.truth >= threshold.minimum_truth


def _metric_profile_requests(
    metric: RecoveryMetric,
) -> tuple[tuple[str, float], ...]:
    if len(metric.parameter_truths) > 1:
        return ((metric.label, metric.truth),)
    return metric.parameter_truths


def _required_metric_profile_truths(
    case: SyntheticCase,
    metric: RecoveryMetric,
    profiles: dict[str, ParameterProfile],
) -> tuple[tuple[ParameterProfile, float], ...]:
    requests = _metric_profile_requests(metric)
    missing = tuple(name for name, _truth in requests if name not in profiles)
    if missing:
        raise AssertionError(
            {
                "case_id": case.case_id,
                "metric": metric.label,
                "missing_profiles": missing,
            }
        )
    return tuple((profiles[name], truth) for name, truth in requests)


def _metric_profiles_are_closed(
    profile_truths: tuple[tuple[ParameterProfile, float], ...],
) -> bool:
    return all(profile.lower_closed and profile.upper_closed for profile, _truth in profile_truths)


class _MetricAccumulator:
    def __init__(self) -> None:
        self.errors_by_family: defaultdict[str, list[float]] = defaultdict(list)
        self.closed_included = 0
        self.open_interval_covered = 0
        self.open_interval_total = 0

    def _record_closed(
        self,
        metric: RecoveryMetric,
        values: dict[str, float],
        in_inner_bounds: bool,
    ) -> None:
        if not in_inner_bounds:
            return
        self.errors_by_family[metric.family].append(_metric_error(metric, values))
        self.closed_included += 1

    def _record_open(
        self,
        metric: RecoveryMetric,
        result,
        profile_truths: tuple[tuple[ParameterProfile, float], ...],
    ) -> None:
        self.open_interval_total += 1
        for profile, truth in profile_truths:
            assert _profile_covers_truth(profile, truth), {
                "metric": metric.label,
                "truth": truth,
                "profile_values": profile.values,
            }
        assert result.confidence is not ConfidenceClass.TRUSTED, {
            "metric": metric.label,
            "confidence": result.confidence.value,
        }
        self.open_interval_covered += 1

    def add_case(self, case: SyntheticCase, result, data: PreparedData) -> None:
        values = _values_by_name(result)
        bounds = _bounds_by_name(result)
        definitions = {definition.name: definition for definition in result.parameter_definitions}
        profiles = _profiles_by_name(result)
        for metric in case.metrics:
            if not _metric_truth_is_eligible(metric):
                continue
            profile_truths = _required_metric_profile_truths(
                case,
                metric,
                profiles,
            )
            in_inner_bounds = _truth_inside_inner_bounds(
                metric.parameter_truths,
                bounds,
                definitions,
                data.r_floor,
            )
            if _metric_profiles_are_closed(profile_truths):
                self._record_closed(metric, values, in_inner_bounds)
                continue
            self._record_open(metric, result, profile_truths)

    def assert_thresholds(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        thresholds = {**DESIGN_THRESHOLDS, **EXTRA_THRESHOLDS}
        for family, threshold in thresholds.items():
            values = np.array(self.errors_by_family.get(family, ()), dtype=float)
            assert values.size > 0, f"no closed, in-bound profile samples for {family}"
            median = float(np.median(values))
            p95 = float(np.percentile(values, 95, method="linear"))
            summary[family] = {"count": float(values.size), "median": median, "p95": p95}
            assert median <= threshold.median, {"family": family, "median": median, "limit": threshold.median}
            if threshold.p95 is not None:
                assert p95 <= threshold.p95, {"family": family, "p95": p95, "limit": threshold.p95}
        return summary
