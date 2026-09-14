"""Continuous diagnostic scores and the inclusive symmetric max family."""

from __future__ import annotations

from importlib import import_module, util
from types import SimpleNamespace

import numpy as np
import pytest

from xrr_fitter.model.instrument import InstrumentSpec


def _statistics():
    name = "xrr_fitter.analysis.residual_statistics"
    assert util.find_spec(name) is not None, "continuous residual statistics are required"
    return import_module(name)


def _problem(size=400, *, footprint="none", background="constant", qmax=1.0):
    return SimpleNamespace(
        data=SimpleNamespace(
            qz_a_inv=np.linspace(0.01, qmax, size),
            two_theta_deg=np.linspace(0.1, 5.0, size),
            fit_mask=np.ones(size, dtype=bool),
        ),
        instrument=InstrumentSpec(footprint_mode=footprint, background_kind=background),
    )


def test_rms_ties_sparse_column_maxima_instead_of_ranking_raw_units():
    values = np.zeros((200, 4))
    values[0, 0], values[1, 1], values[2, 2], values[3, 3] = 100, 4, 5, 6
    result = _statistics().symmetric_max(values)
    assert result.tail_count == result.tie_count == 4
    assert result.p_value == 0.020
    assert result.adjusted_p_values == (0.020, 1.0, 1.0, 1.0)


def test_constant_zero_family_uses_inclusive_ties_without_jitter():
    result = _statistics().symmetric_max(np.zeros((100, 4)))
    assert result.tail_count == result.tie_count == 100
    assert result.p_value == 1.0
    assert result.observed_score == 0.0
    assert result.adjusted_p_values == (1.0,) * 4
    np.testing.assert_array_equal(result.scales, np.zeros(4))


def test_symmetric_scaling_uses_observed_row_and_is_permutation_equivariant():
    values = np.array([[0, 3], [1, 2], [2, 1], [20, -5]], dtype=float)
    first = _statistics().symmetric_max(values)
    order = np.array([3, 0, 2, 1])
    second = _statistics().symmetric_max(values[order])
    np.testing.assert_array_equal(first.centers, [1.5, 1.5])
    np.testing.assert_allclose(first.scales, np.sqrt([86.25, 11.25]))
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.scales, second.scales)
    np.testing.assert_array_equal(first.scores[order], second.scores)
    assert min(first.adjusted_p_values) == first.p_value


@pytest.mark.parametrize("values", [np.zeros(4), np.empty((4, 0)), np.zeros((1, 3))])
def test_invalid_family_axes_are_rejected(values):
    with pytest.raises(ValueError, match="matrix"):
        _statistics().symmetric_max(values)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_statistic_is_not_silently_zeroed(bad):
    values = np.zeros((100, 2))
    values[20, 1] = bad
    with pytest.raises(FloatingPointError, match="finite"):
        _statistics().symmetric_max(values)


def test_all_applicable_columns_exist_even_when_observed_screen_is_negative():
    result = _statistics().residual_statistics(_problem(), np.zeros(400), dataset_id="curve")
    assert tuple(item.kind for item in result) == ("footprint", "background", "surface", "acf")
    assert all(item.dataset_id == "curve" for item in result)
    assert all(item.observed == 0.0 for item in result)
    assert all(item.adjusted_p_value is None for item in result)


def test_instrument_and_geometry_declare_family_before_looking_at_residuals():
    problem = _problem(80, footprint="fit", background="linear", qmax=0.05)
    for residuals in (np.zeros(80), np.linspace(-5, 5, 80)):
        result = _statistics().residual_statistics(problem, residuals)
        assert tuple(item.kind for item in result) == ("acf",)


def test_continuous_trend_keeps_amplitude_without_boolean_clipping():
    residuals = np.zeros(400)
    residuals[:60] = np.linspace(3, 0, 60)
    residuals[-80:] = np.linspace(6, 0, 80)
    stats = {item.kind: item.observed for item in _statistics().residual_statistics(_problem(), residuals)}
    assert stats["footprint"] == pytest.approx(2.033898305084746 / (0.75 * 0.05))
    assert stats["background"] == pytest.approx(4.10126582278481 / (0.70 * 0.05))
    assert stats["footprint"] > 1.0 and stats["background"] > 1.0


def test_acf_score_is_the_second_absolute_lag_not_lag_one():
    values = np.random.default_rng(119).normal(size=400)
    values[2:] += 0.9 * values[:-2]
    centered = values - values.mean()
    lags = sorted(abs(float(centered[:-k] @ centered[k:] / (centered @ centered))) for k in range(1, 21))
    result = _statistics().residual_statistics(_problem(), values)
    acf = next(item for item in result if item.kind == "acf")
    assert acf.observed == pytest.approx(np.sqrt(400) / 3 * lags[-2])


def test_masked_nan_is_allowed_but_fitted_nan_is_unavailable():
    problem = _problem(80)
    problem.data.fit_mask[0] = False
    residuals = np.zeros(80)
    residuals[0] = np.nan
    assert _statistics().residual_statistics(problem, residuals)
    residuals[1] = np.nan
    with pytest.raises(FloatingPointError, match="finite"):
        _statistics().residual_statistics(problem, residuals)


def test_statistics_follow_stable_q_sort_with_mask_preserved():
    problem = _problem(80)
    values = np.sin(np.linspace(0, 3, 80))
    baseline = _statistics().residual_statistics(problem, values)
    order = np.random.default_rng(71).permutation(80)
    for name in ("qz_a_inv", "two_theta_deg", "fit_mask"):
        setattr(problem.data, name, getattr(problem.data, name)[order])
    assert _statistics().residual_statistics(problem, values[order]) == baseline
