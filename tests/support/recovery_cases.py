from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10

import numpy as np


INNER_BOUND_FRACTION = 0.05
METRIC_THRESHOLDS = {
    "backing.roughness_a": 0.15,
    "component.thickness_a": 0.10,
    "component.period_a": 0.08,
    "instrument.background": 0.20,
    "instrument.footprint_spill_angle_deg": 0.02,
    "instrument.scale": 0.05,
}


def _validate_metric_identity(name: str, family: str) -> None:
    if not name or not family:
        raise ValueError("recovery metric identity must not be empty")


def _validate_metric_transform(transform: str, lower: float, floor: float) -> None:
    if transform not in {"linear", "log", "log_floor"}:
        raise ValueError("unsupported recovery metric transform")
    if transform == "log" and lower <= 0.0:
        raise ValueError("log recovery metric requires positive bounds")
    if transform == "log_floor" and floor <= 0.0:
        raise ValueError("log-floor recovery metric requires a positive objective floor")


def _validate_metric_values(values: tuple[float, ...], lower: float, upper: float) -> None:
    if not all(isfinite(value) for value in values) or lower >= upper:
        raise ValueError("recovery metric values and bounds must be finite and ordered")


@dataclass(frozen=True, slots=True)
class RecoveryMetric:
    name: str
    family: str
    transform: str
    target: float
    fitted: float
    lower: float
    upper: float
    objective_floor: float = 0.0

    def __post_init__(self) -> None:
        values = (self.target, self.fitted, self.lower, self.upper, self.objective_floor)
        _validate_metric_identity(self.name, self.family)
        _validate_metric_transform(self.transform, self.lower, self.objective_floor)
        _validate_metric_values(values, self.lower, self.upper)


def parameter_family(name: str) -> str:
    if name == "instrument.scale":
        return "instrument.scale"
    if "background" in name:
        return "instrument.background"
    if name == "instrument.footprint_spill_angle_deg":
        return "instrument.footprint_spill_angle_deg"
    if name.endswith("period_a"):
        return "component.period_a"
    if name.endswith("thickness_a"):
        return "component.thickness_a"
    if name.endswith("roughness_a"):
        return "backing.roughness_a"
    raise ValueError(f"unsupported recovery metric parameter: {name}")


def metric_coordinate(metric: RecoveryMetric, value: float) -> float:
    if metric.transform == "linear":
        return value
    if metric.transform == "log":
        if value <= 0.0:
            raise ValueError("log recovery metric value must be positive")
        return log10(value)
    shifted = value + metric.objective_floor
    if shifted <= 0.0:
        raise ValueError("log-floor recovery metric value is outside its domain")
    return log10(shifted)


def inner_coordinate_bounds(
    metric: RecoveryMetric,
    *,
    fraction: float = INNER_BOUND_FRACTION,
) -> tuple[float, float]:
    if not 0.0 <= fraction < 0.5:
        raise ValueError("inner-bound fraction must be in [0, 0.5)")
    lower = metric_coordinate(metric, metric.lower)
    upper = metric_coordinate(metric, metric.upper)
    margin = fraction * (upper - lower)
    return lower + margin, upper - margin


def is_inside_inner_bounds(metric: RecoveryMetric) -> bool:
    lower, upper = inner_coordinate_bounds(metric)
    fitted = metric_coordinate(metric, metric.fitted)
    return lower <= fitted <= upper


def target_is_supported(metric: RecoveryMetric) -> bool:
    lower, upper = inner_coordinate_bounds(metric)
    target = metric_coordinate(metric, metric.target)
    return lower <= target <= upper


def normalized_coordinate_error(metric: RecoveryMetric) -> float:
    lower = metric_coordinate(metric, metric.lower)
    upper = metric_coordinate(metric, metric.upper)
    target = metric_coordinate(metric, metric.target)
    fitted = metric_coordinate(metric, metric.fitted)
    return abs(fitted - target) / (upper - lower)


def open_metric_names(metrics: tuple[RecoveryMetric, ...]) -> tuple[str, ...]:
    return tuple(
        metric.name
        for metric in metrics
        if target_is_supported(metric)
        and normalized_coordinate_error(metric) > METRIC_THRESHOLDS[metric.family]
    )


def fitted_model_error(
    observed: np.ndarray,
    model: np.ndarray,
    fit_mask: np.ndarray,
) -> float:
    observed_values = np.asarray(observed, dtype=float)
    model_values = np.asarray(model, dtype=float)
    mask = np.asarray(fit_mask, dtype=bool)
    if observed_values.shape != model_values.shape or mask.shape != observed_values.shape:
        raise ValueError("model-error arrays must have equal shape")
    selected_observed = observed_values[mask]
    selected_model = model_values[mask]
    if selected_observed.size == 0 or not np.all(np.isfinite(selected_observed)):
        raise ValueError("fitted observations must be finite and nonempty")
    if not np.all(np.isfinite(selected_model)):
        raise ValueError("model must be finite inside the fit mask")
    return float(np.sqrt(np.mean((selected_model - selected_observed) ** 2)))


def deterministic_metric_cases() -> tuple[RecoveryMetric, ...]:
    return (
        RecoveryMetric(
            "component.0.thickness_a",
            parameter_family("component.0.thickness_a"),
            "log",
            204.0,
            210.0,
            3.7,
            7715.0,
        ),
        RecoveryMetric(
            "instrument.scale",
            parameter_family("instrument.scale"),
            "log",
            1.0,
            1.02,
            0.001,
            1000.0,
        ),
        RecoveryMetric(
            "instrument.background",
            parameter_family("instrument.background"),
            "log_floor",
            0.0,
            1e-9,
            0.0,
            1e-4,
            objective_floor=1e-8,
        ),
    )
