from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support import synthetic_recovery_runs
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


def test_statistical_recovery_requests_only_metric_profile_evidence() -> None:
    from tests.support.synthetic_recovery import build_corpus

    cases = build_corpus()
    by_category = {case.category: case for case in cases}

    assert synthetic_recovery_runs._case_profile_names(
        by_category["single_layer"]
    ) == (
        "component.0.thickness_a",
        "component.0.density_scale",
        "component.0.roughness_a",
    )
    assert synthetic_recovery_runs._case_profile_names(
        by_category["periodic_mosi"]
    ) == (
        "component.0.period_a",
        "component.0.layer.0.fraction",
        "component.0.layer.0.roughness_a",
        "component.0.layer.1.roughness_a",
    )
    assert synthetic_recovery_runs._case_profile_names(
        by_category["instrument_effects"]
    ) == (
        "component.0.thickness_a",
        "component.0.density_scale",
        "component.0.roughness_a",
        "instrument.angle_offset_deg",
        "instrument.scale",
        "instrument.background",
        "instrument.relative_sigma",
    )
    assert synthetic_recovery_runs._case_profile_names(
        by_category["ambiguous"]
    ) is None
    assert synthetic_recovery_runs._case_profile_names(
        by_category["model_error"]
    ) is None


def test_corpus_fit_dispatch_is_spawned_bounded_and_ordered(monkeypatch) -> None:
    from tests.support.synthetic_recovery import build_corpus

    observed: dict[str, object] = {}

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context, initializer, initargs) -> None:
            observed["setup"] = (max_workers, mp_context, initializer, initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def map(self, function, case_ids, *, chunksize):
            observed["map"] = (function, tuple(case_ids), chunksize)
            return tuple(
                f"outcome:{case_id}"
                for case_id in observed["map"][1]
            )

    monkeypatch.setattr(synthetic_recovery_runs, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(synthetic_recovery_runs.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(
        synthetic_recovery_runs.multiprocessing,
        "get_context",
        lambda name: f"{name}-context",
    )
    corpus = build_corpus()
    single = next(case for case in corpus if case.case_id == "single-11000")
    periodic_10 = next(
        case for case in corpus if case.case_id == "periodic-13000-n10"
    )
    periodic_100 = next(
        case for case in corpus if case.case_id == "periodic-13019-n100"
    )
    cases = (single, periodic_10, periodic_100)

    outcomes = synthetic_recovery_runs._parallel_case_outcomes(cases)

    assert outcomes == tuple(f"outcome:{case.case_id}" for case in cases)
    assert observed["setup"] == (
        5,
        "spawn-context",
        synthetic_recovery_runs._initialize_worker_cases,
        (2,),
    )
    assert observed["map"] == (
        synthetic_recovery_runs._fit_worker_case,
        (
            periodic_100.case_id,
            periodic_10.case_id,
            single.case_id,
        ),
        1,
    )


def test_statistical_corpus_reuses_one_parallel_dispatch_across_partitions(
    monkeypatch,
) -> None:
    from tests.support import synthetic_recovery

    cases = synthetic_recovery.build_corpus()
    outcomes = tuple(
        synthetic_recovery_runs._CaseOutcome(case.case_id)
        for case in cases
    )
    dispatches: list[tuple[str, ...]] = []
    partitions: dict[str, tuple[tuple[str, ...], tuple[str, ...] | None]] = {}

    def parallel(requested):
        dispatches.append(tuple(case.case_id for case in requested))
        return outcomes

    def observe_partition(name):
        def observed(requested, requested_outcomes=None) -> None:
            partitions[name] = (
                tuple(case.case_id for case in requested),
                (
                    None
                    if requested_outcomes is None
                    else tuple(outcome.case_id for outcome in requested_outcomes)
                ),
            )

        return observed

    monkeypatch.setattr(
        synthetic_recovery,
        "_parallel_case_outcomes",
        parallel,
        raising=False,
    )
    monkeypatch.setattr(
        synthetic_recovery,
        "_run_slow_statistical_recovery_corpus",
        observe_partition("recovery"),
    )
    monkeypatch.setattr(
        synthetic_recovery,
        "_run_slow_ambiguous_corpus",
        observe_partition("ambiguous"),
    )
    monkeypatch.setattr(
        synthetic_recovery,
        "_run_slow_model_error_corpus",
        observe_partition("model_error"),
    )

    report = synthetic_recovery.run_corpus(cases)

    assert report.status == "PASS"
    assert dispatches == [tuple(case.case_id for case in cases)]
    for case_ids, outcome_ids in partitions.values():
        assert outcome_ids == case_ids
