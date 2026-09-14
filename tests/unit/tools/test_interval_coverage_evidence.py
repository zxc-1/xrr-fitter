"""Saved-evidence eligibility, trace topology, and bootstrap sample gates."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import xrr_fitter.api as api
from xrr_fitter.model.bootstrap import BootstrapResult

TARGET = "component.0.thickness_a"


def _profile(**changes):
    fields = {
        "name": TARGET,
        "values": np.array([80.0, 90.0, 100.0, 110.0, 120.0]),
        "objectives": np.array([1.16, 1.04, 1.0, 1.04, 1.16]),
        "lower_closed": True,
        "upper_closed": True,
        "interval_kind": "likelihood_ratio",
        "confidence_level": 0.95,
        "method": "chi_square_1df_asymptotic",
        "delta_total": 4.0,
        "objective_point_count": 100,
    }
    return api.ParameterProfile(**(fields | changes))


def _bootstrap(successful=200, *, method="joint_gaussian_parametric"):
    failed = 200 - successful
    reason = "excessive_fit_failures" if failed > 40 else "insufficient_successful_samples"
    return BootstrapResult(
        (TARGET,),
        np.full((successful, 1), 100.0),
        ((TARGET, 97.0, 103.0),) if failed == 0 else (),
        failed / 200,
        200,
        tuple((index, "optimizer_nonconvergence: budget exhausted") for index in range(successful, 200)),
        method=method,
        unavailable_reason=None if failed == 0 else reason,
    )


@pytest.mark.parametrize(("total", "expected"), ((4.0, [90.0, 110.0]), (9.0, [85.0, 115.0])))
def test_profile_bounds_use_saved_total_per_point_and_sqrt_interpolation(load_tool_module, total, expected):
    module = load_tool_module("interval_coverage_evidence")
    interval = module.profile_interval(_profile(delta_total=total))

    assert interval["available"] is True
    assert interval["bounds"] == pytest.approx(expected)


def test_profile_labels_come_from_saved_calibration(load_tool_module):
    module = load_tool_module("interval_coverage_evidence")
    interval = module.profile_interval(_profile())

    assert interval["kind"] == "likelihood_ratio"
    assert interval["confidence_level"] == 0.95
    assert interval["method"] == "chi_square_1df_asymptotic"


@pytest.mark.parametrize("total", (4.0, 9.0))
def test_profile_threshold_metadata_is_not_replaced_by_an_implicit_default(load_tool_module, total):
    module = load_tool_module("interval_coverage_evidence")
    interval = module.profile_interval(_profile(delta_total=total))

    assert interval["details"]["delta_total"] == total
    assert interval["details"]["objective_point_count"] == 100
    assert interval["details"]["delta_per_point"] == total / 100
    assert interval["details"]["interpolation"] == "piecewise_linear_sqrt_delta_J"


def test_underflowed_per_point_threshold_cannot_publish_a_zero_width_interval(load_tool_module):
    module = load_tool_module("interval_coverage_evidence")
    interval = module.profile_interval(_profile(delta_total=float(np.nextafter(0.0, 1.0))))

    assert interval["available"] is False
    assert interval["unavailable_reason"] == "invalid_profile_threshold"


@pytest.mark.parametrize(
    ("changes", "reason"),
    (
        ({"lower_closed": False}, "open_profile_support"),
        ({"upper_closed": False}, "open_profile_support"),
        ({"objectives": np.array([np.inf, 1.04, 1.0, 1.04, 1.16])}, "nonfinite_profile_trace"),
        ({"objectives": np.full(5, np.nan)}, "nonfinite_profile_trace"),
        ({"values": np.array([80, 90, 90, 110, 120])}, "nonmonotonic_profile_coordinates"),
        ({"values": np.array([80, 100, 90, 110, 120])}, "nonmonotonic_profile_coordinates"),
        ({"objectives": np.array([1.01, 1.0, 1.0, 1.0, 1.01])}, "unbracketed_profile_support"),
    ),
)
def test_irregular_profile_is_unavailable_without_erasing_diagnostics(load_tool_module, changes, reason):
    module = load_tool_module("interval_coverage_evidence")
    profile = _profile(**changes)
    interval = module.profile_interval(profile)

    assert interval["available"] is False
    assert interval["bounds"] is None
    assert interval["unavailable_reason"] == reason
    assert interval["details"]["lower_closed"] is profile.lower_closed
    assert interval["details"]["upper_closed"] is profile.upper_closed
    assert len(interval["details"]["objectives"]) == profile.objectives.size
    json.dumps(interval, allow_nan=False)


def test_disjoint_profile_support_is_not_joined_across_excluded_gaps(load_tool_module):
    module = load_tool_module("interval_coverage_evidence")
    profile = _profile(values=np.arange(7.0), objectives=np.array([1.16, 1, 1.16, 1, 1.16, 1, 1.16]))
    interval = module.profile_interval(profile)

    assert interval["available"] is False
    assert interval["bounds"] is None
    assert interval["unavailable_reason"] == "disjoint_profile_support"
    assert np.asarray(interval["support_intervals"]) == pytest.approx(np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]))


@pytest.mark.parametrize("kind", ("loss_support", "unavailable"))
def test_loss_support_never_masquerades_as_nominal_95_interval(load_tool_module, kind):
    module = load_tool_module("interval_coverage_evidence")
    interval = module.profile_interval(
        _profile(interval_kind=kind, confidence_level=None, unavailable_reason="robust_objective_is_not_a_likelihood")
    )

    assert interval["available"] is False
    assert interval["kind"] == kind
    assert interval["confidence_level"] is None
    assert interval["unavailable_reason"] == "robust_objective_is_not_a_likelihood"


def test_bootstrap_uses_saved_bounds_instead_of_recomputing_from_samples(load_tool_module):
    module = load_tool_module("interval_coverage_evidence")
    interval = module.bootstrap_interval(_bootstrap(), TARGET)

    assert interval["available"] is True
    assert interval["bounds"] == [97.0, 103.0]
    assert interval["kind"] == "percentile_bootstrap"
    assert interval["confidence_level"] == 0.95
    assert interval["method"] == "joint_gaussian_parametric"


def test_bootstrap_keeps_actual_sampling_evidence(load_tool_module):
    module = load_tool_module("interval_coverage_evidence")
    interval = module.bootstrap_interval(_bootstrap(), TARGET)

    assert interval["details"]["attempted_count"] == 200
    assert interval["details"]["successful_samples"] == 200
    assert interval["details"]["samples"] == [[100.0]] * 200


@pytest.mark.parametrize("successful", (199, 150, 0))
def test_insufficient_bootstrap_success_keeps_failure_indices_and_is_unavailable(load_tool_module, successful):
    module = load_tool_module("interval_coverage_evidence")
    evidence = _bootstrap(successful)
    interval = module.bootstrap_interval(evidence, TARGET)

    assert interval["available"] is False
    assert interval["bounds"] is None
    assert interval["confidence_level"] is None
    assert interval["unavailable_reason"] == evidence.unavailable_reason
    assert interval["details"]["successful_samples"] == successful
    assert interval["details"]["failure_reasons"] == [list(row) for row in evidence.failure_reasons]
    assert interval["details"]["attempted_count"] == 200


def test_absent_or_wrong_parameter_bootstrap_is_unavailable(load_tool_module):
    module = load_tool_module("interval_coverage_evidence")

    assert module.bootstrap_interval(None, TARGET)["unavailable_reason"] == "bootstrap_not_performed"
    assert module.bootstrap_interval(_bootstrap(), "other")["unavailable_reason"] == "bootstrap_parameter_missing"


def test_bootstrap_metadata_cannot_override_insufficient_actual_success(load_tool_module):
    module = load_tool_module("interval_coverage_evidence")
    evidence = _bootstrap(199)
    invalid = SimpleNamespace(
        **{name: getattr(evidence, name) for name in evidence.__dataclass_fields__},
        successful_samples=199,
        interval_kind="percentile_bootstrap",
        confidence_level=0.95,
    )
    invalid.intervals = ((TARGET, 97.0, 103.0),)
    invalid.unavailable_reason = None

    assert module.bootstrap_interval(invalid, TARGET)["available"] is False
