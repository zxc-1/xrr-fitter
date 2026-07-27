from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.support.recovery_cases import (
    METRIC_THRESHOLDS,
    RecoveryMetric,
    deterministic_metric_cases,
    fitted_model_error,
    inner_coordinate_bounds,
    is_inside_inner_bounds,
    metric_coordinate,
    normalized_coordinate_error,
    open_metric_names,
    parameter_family,
)


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param(
            RecoveryMetric(
                "component.0.thickness_a",
                "component.thickness_a",
                "log",
                204.0,
                204.0,
                3.7,
                7715.0,
            ),
            id="thickness-period",
        ),
        pytest.param(
            RecoveryMetric(
                "instrument.scale",
                "instrument.scale",
                "log",
                1.0,
                1.0,
                0.001,
                1000.0,
            ),
            id="instrument-scale",
        ),
    ],
)
def test_closed_log_metric_uses_transform_coordinate_for_inner_bounds(
    metric: RecoveryMetric,
) -> None:
    lower, upper = inner_coordinate_bounds(metric)
    raw_lower_fraction = (metric.lower + 0.05 * (metric.upper - metric.lower))

    assert lower == pytest.approx(
        metric_coordinate(metric, metric.lower)
        + 0.05
        * (metric_coordinate(metric, metric.upper) - metric_coordinate(metric, metric.lower))
    )
    assert upper > lower
    assert metric_coordinate(metric, raw_lower_fraction) != pytest.approx(lower)


def test_closed_background_uses_objective_floor_coordinate_for_inner_bounds() -> None:
    metric = RecoveryMetric(
        "instrument.background",
        "instrument.background",
        "log_floor",
        0.0,
        1e-9,
        0.0,
        1e-4,
        objective_floor=1e-8,
    )

    lower, _upper = inner_coordinate_bounds(metric)

    assert lower > np.log10(metric.objective_floor)
    assert metric_coordinate(metric, 0.0) == pytest.approx(np.log10(metric.objective_floor))
    assert normalized_coordinate_error(metric) < 0.05


def test_closed_profile_excluded_by_inner_bounds_is_not_counted_as_open() -> None:
    profile = RecoveryMetric(
        "component.0.period_a",
        "component.period_a",
        "log",
        10.0,
        900.0,
        10.0,
        1000.0,
    )

    assert not is_inside_inner_bounds(replace(profile, fitted=profile.target))
    assert open_metric_names((profile,)) == ()


def test_fit_output_contract_allows_undefined_model_outside_fit_mask() -> None:
    observed = np.asarray([1.0, 0.8, 0.6, 0.4])
    model = np.asarray([np.nan, 0.75, 0.55, np.nan])
    fit_mask = np.asarray([False, True, True, False])

    error = fitted_model_error(observed, model, fit_mask)

    assert error == pytest.approx(0.05)


def test_instrument_scale_and_background_have_independent_metric_families() -> None:
    assert parameter_family("instrument.scale") == "instrument.scale"
    assert parameter_family("instrument.background") == "instrument.background"
    assert parameter_family("instrument.linear_background_per_a_inv") == "instrument.background"
    assert METRIC_THRESHOLDS["instrument.scale"] != METRIC_THRESHOLDS["instrument.background"]


def test_joint_period_inner_bounds_use_component_log_coordinates() -> None:
    metric = RecoveryMetric(
        "component.0.period_a",
        parameter_family("component.0.period_a"),
        "log",
        75.0,
        80.0,
        20.0,
        300.0,
    )

    lower, upper = inner_coordinate_bounds(metric)

    assert lower == pytest.approx(np.log10(20.0) + 0.05 * np.log10(300.0 / 20.0))
    assert upper == pytest.approx(np.log10(300.0) - 0.05 * np.log10(300.0 / 20.0))


def test_metric_thresholds_include_released_footprint_angle() -> None:
    assert parameter_family("instrument.footprint_spill_angle_deg") in METRIC_THRESHOLDS
    assert METRIC_THRESHOLDS["instrument.footprint_spill_angle_deg"] == pytest.approx(0.02)


def test_recovery_metric_cases_are_deterministic_and_family_complete() -> None:
    first = deterministic_metric_cases()
    second = deterministic_metric_cases()

    assert first == second
    assert tuple(metric.family for metric in first) == (
        "component.thickness_a",
        "instrument.scale",
        "instrument.background",
    )
