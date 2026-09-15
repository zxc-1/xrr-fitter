"""Finite-MC joint Poisson intervals have explicit, separately named levels."""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import beta

from xrr_fitter.analysis.bootstrap_samples import bootstrap_result_from_fits
from xrr_fitter.io.codec_inference import bootstrap_from_dict, bootstrap_to_dict
from xrr_fitter.model.joint_bootstrap_provenance import seal_joint_bootstrap


def _result(values, method="joint_poisson_parametric"):
    return bootstrap_result_from_fits(
        ("x",), (np.array([value]) if value is not None else None for value in values), len(values), None, method=method
    )


def test_b200_uses_the_shortest_symmetric_96_percent_content_95_percent_assurance_interval():
    result = _result(list(range(200)))
    assert result.intervals == (("x", 1.0, 198.0),), "finite B must not use inward Type7 endpoints"
    assert result.interval_method == "poisson_joint_content_tolerance_v1"
    assert result.interval_ranks == (2, 199)


def test_nominal_coverage_content_target_and_monte_carlo_assurance_are_separate():
    result = _result(list(range(200)))
    assert result.bootstrap_content_target == 0.96
    assert result.monte_carlo_assurance == 0.95
    assert result.diagnostic_error_budget == 0.01
    assert result.confidence_level == 0.95


@pytest.mark.parametrize("count", (200, 250, 999))
def test_ranks_meet_fixed_beta_content_assurance_and_cannot_be_narrowed(count):
    result = _result(list(range(count)))
    ranks = getattr(result, "interval_ranks", None)
    assert ranks is not None, "finite MC interval must record one-based endpoint ranks"
    lower, upper = ranks
    assert upper == count + 1 - lower
    assert beta.cdf(0.96, count + 1 - 2 * lower, 2 * lower) <= 0.05
    assert beta.cdf(0.96, count - 1 - 2 * lower, 2 * (lower + 1)) > 0.05
    assert result.intervals == (("x", float(lower - 1), float(upper - 1)),)


def test_joint_policy_is_monotone_transformation_equivariant_and_tie_safe():
    values = np.linspace(1.0, 3.0, 200)
    base = _result(values)
    transformed = _result(values**3)
    assert transformed.intervals[0][1:] == tuple(bound**3 for bound in base.intervals[0][1:])
    constant = _result(np.ones(200))
    assert constant.intervals == (("x", 1.0, 1.0),)
    assert constant.interval_ranks == (2, 199)


def test_joint_policy_selects_extreme_finite_values_without_overflow_or_jitter():
    limit = np.finfo(float).max
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _result([-limit] * 5 + [limit] * 195)
    assert result.intervals == (("x", -limit, limit),)
    assert not any(item.category is RuntimeWarning for item in caught)


@pytest.mark.parametrize("values", ([1.0] * 199, [None] + [1.0] * 199, [None] * 51 + [1.0] * 199))
def test_existing_sample_and_failure_gates_are_not_relaxed(values):
    result = _result(values)
    assert result.intervals == ()
    assert result.confidence_level is None
    assert getattr(result, "interval_method", None) == "poisson_joint_content_tolerance_v1"
    assert result.interval_ranks is None


@pytest.mark.parametrize(
    "method",
    (
        "custom_resampling",
        "poisson_parametric",
        "joint_gaussian_parametric",
        "joint_mixed:poisson_parametric,gaussian_parametric",
    ),
)
def test_other_sampling_methods_do_not_silently_acquire_poisson_gate_budget(method):
    result = _result(list(range(200)), method)
    assert result.intervals[0][1:] == pytest.approx((4.975, 194.025))
    assert getattr(result, "interval_method", None) == "percentile_linear_v1"
    assert result.diagnostic_error_budget == 0.0
    assert result.monte_carlo_assurance is None
    assert result.interval_ranks is None


def test_codec_metadata_distinguishes_nominal_content_and_monte_carlo_assurance():
    sampling = _result(list(range(200)))
    assert getattr(sampling, "interval_method", None) == "poisson_joint_content_tolerance_v1"
    sealed = seal_joint_bootstrap(sampling, "E-0", "a" * 64)
    payload = bootstrap_to_dict(sealed)
    assert payload["interval_ranks"] == [2, 199]
    assert payload["bootstrap_content_target"] == 0.96
    assert payload["monte_carlo_assurance"] == 0.95
    assert bootstrap_from_dict(payload).interval_ranks == (2, 199)
    payload["bootstrap_content_target"] = 0.95
    with pytest.raises(ValueError, match="metadata"):
        bootstrap_from_dict(payload)
    with pytest.raises(ValueError, match="policy|method"):
        replace(sampling, interval_method="percentile_linear_v1", interval_ranks=None)
