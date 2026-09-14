"""The v4 RMS contract, not an independent Poisson acceptance experiment."""

from itertools import permutations
from math import sqrt

import numpy as np
import pytest

from xrr_fitter.analysis.residual_statistics import symmetric_max


def _continuous_family(rows, columns):
    matrix = np.column_stack([np.roll(np.linspace(-1, 1, rows), j + 2) for j in range(columns)])
    matrix[0, 0] = 1e6
    return matrix


def test_fixed_unit_floor_no_longer_drowns_small_unit_signal():
    values = np.zeros((1000, 2))
    values[1:30, 0] = 5.0
    values[:, 1] = np.linspace(-0.05, 0.05, 1000)
    values[0, 1] = 3.6
    result = symmetric_max(values)
    assert result.tail_count == result.tie_count == 1
    assert result.p_value == 0.001
    assert result.adjusted_p_values == (1.0, 0.001)
    assert 0 < result.scales[1] < result.scales[0] < 1


def test_power_of_two_column_units_preserve_scores_and_probabilities():
    values = np.array([[0, 3, 1], [1, 2, -3], [2, 1, 2], [20, -5, 0]], dtype=float)
    factors = np.array([0.125, 4, 16])
    offsets = np.array([16, -32, 64])
    first = symmetric_max(values)
    second = symmetric_max(values * factors + offsets)
    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.centers * factors + offsets, second.centers)
    np.testing.assert_array_equal(first.scales * factors, second.scales)
    assert first.adjusted_p_values == second.adjusted_p_values
    assert (first.tail_count, first.tie_count) == (second.tail_count, second.tie_count)


@pytest.mark.parametrize("rows,columns", [(100, 4), (1000, 12)])
def test_continuous_family_does_not_force_every_column_maximum_to_tie(rows, columns):
    result = symmetric_max(_continuous_family(rows, columns))
    assert result.tail_count == result.tie_count == 1
    assert result.p_value == 1 / rows
    assert result.adjusted_p_values[0] == result.p_value


def test_sparse_peaks_tie_exactly_even_with_extreme_different_units():
    values = np.zeros((8, 4))
    values[np.arange(4), np.arange(4)] = [100, 1, 1e-100, 1e100]
    result = symmetric_max(values)
    np.testing.assert_array_equal(result.scores[:4], np.repeat(result.scores[0], 4))
    assert result.tail_count == result.tie_count == 4
    assert result.p_value == 0.5


@pytest.mark.parametrize("value", [0.0, -0.0, 7.0, np.finfo(float).max, -np.finfo(float).max])
def test_exact_constant_column_has_explicit_zero_scale(value):
    result = symmetric_max(np.full((100, 3), value))
    np.testing.assert_array_equal(result.centers, [value] * 3)
    np.testing.assert_array_equal(result.scales, np.zeros(3))
    np.testing.assert_array_equal(result.scores, np.zeros(100))
    assert not np.signbit(result.scales).any()
    assert not np.signbit(result.scores).any()
    if value == 0:
        assert not np.signbit(result.centers).any()
    assert result.adjusted_p_values == (1.0,) * 3
    assert result.tail_count == result.tie_count == 100


@pytest.mark.parametrize("amplitude", [1e-200, 1e200])
def test_peak_normalization_avoids_square_underflow_and_overflow(amplitude):
    result = symmetric_max(np.array([[amplitude], [0], [0]]))
    assert result.centers[0] == 0
    assert result.scales[0] == pytest.approx(amplitude / sqrt(3), rel=1e-15, abs=0)
    assert result.observed_score == pytest.approx(sqrt(3))
    assert result.tail_count == result.tie_count == 1


def test_even_median_avoids_same_sign_sum_overflow():
    largest = np.finfo(float).max
    result = symmetric_max(np.array([[largest / 2], [largest * 0.75], [largest], [largest]]))
    assert result.centers[0] == largest * 0.875
    assert 0 < result.scales[0] < largest
    assert np.isfinite(result.scores).all()


def test_centered_difference_overflow_is_not_a_pass():
    largest = np.finfo(float).max
    with pytest.raises(FloatingPointError):
        symmetric_max(np.array([[-largest], [largest], [largest]]))


def test_nonconstant_scale_underflow_is_not_a_constant_column():
    smallest = np.nextafter(0.0, 1.0)
    with pytest.raises(FloatingPointError, match="scale"):
        symmetric_max(np.array([[smallest], [0], [0], [0]]))


def test_lost_nonzero_normalization_value_is_unavailable():
    values = np.array([[np.finfo(float).max], [np.nextafter(0.0, 1.0)], [0], [0], [0]])
    with pytest.raises(FloatingPointError, match="normaliz"):
        symmetric_max(values)


def test_squared_relative_underflow_does_not_erase_a_representable_score():
    result = symmetric_max(np.array([[1e-200], [1.0], [0], [0], [0]]))
    assert result.observed_score == pytest.approx(sqrt(5) * 1e-200, rel=1e-15, abs=0)
    assert result.p_value == 2 / 5


def test_factorized_score_is_not_recomputed_from_rounded_subnormal_scale():
    smallest = np.nextafter(0.0, 1.0)
    result = symmetric_max(np.array([[5 * smallest], [0], [0], [0]]))
    assert result.scales[0] == 2 * smallest
    assert result.observed_score == 2.0
    assert (5 * smallest) / result.scales[0] == 2.5


def test_sorted_accurate_sum_keeps_representable_small_square_contributions():
    small = 2.0**-27
    values = np.array([1, *([small] * 4), *([-small] * 4), 0, 0], dtype=float)[:, None]
    result = symmetric_max(values)
    assert result.centers[0] == 0
    assert result.scales[0] == sqrt((1 + 2.0**-51) / 11)


def test_negative_single_spike_is_not_positive_tail_evidence():
    result = symmetric_max(np.array([[-0.005], [0], [0], [0]]))
    assert result.scales[0] == 0.0025
    assert result.observed_score == 0
    assert result.p_value == 1


def test_all_row_and_column_permutations_have_identical_numeric_mapping():
    values = np.array([[8, 3, 0], [1, 2, -2], [-1, 1, 2], [2, 0, 0], [0, -3, 1], [-2, 4, -1.0]])
    first = symmetric_max(values)
    for row_order in permutations(range(6)):
        order = list(row_order)
        second = symmetric_max(values[order])
        np.testing.assert_array_equal(second.centers, first.centers)
        np.testing.assert_array_equal(second.scales, first.scales)
        np.testing.assert_array_equal(second.scores, first.scores[order])
        assert second.tail_count == np.count_nonzero(first.scores >= first.scores[order[0]])
    for column_order in permutations(range(3)):
        order = list(column_order)
        second = symmetric_max(values[:, order])
        np.testing.assert_array_equal(second.scores, first.scores)
        np.testing.assert_array_equal(second.scales, first.scales[order])
        assert second.adjusted_p_values == tuple(first.adjusted_p_values[i] for i in order)


def test_layout_input_immutability_and_readonly_outputs():
    values = _continuous_family(100, 4)
    expected = symmetric_max(values)
    backing = np.zeros((200, 8))
    backing[::2, ::2] = values
    for matrix in (values.copy(order="C"), values.copy(order="F"), backing[::2, ::2]):
        original = matrix.copy()
        matrix.setflags(write=False)
        result = symmetric_max(matrix)
        np.testing.assert_array_equal(matrix, original)
        for name in ("centers", "scales", "scores"):
            np.testing.assert_array_equal(getattr(result, name), getattr(expected, name))
            assert not getattr(result, name).flags.writeable
        assert min(result.adjusted_p_values) == result.p_value
