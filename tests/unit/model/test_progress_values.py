from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.progress import downsampled_preview, freeze_preview_axes


def _curve(size: int) -> tuple[np.ndarray, np.ndarray]:
    qz = np.linspace(0.01, 0.4, size)
    return qz, np.exp(-qz)


def test_preview_under_the_bound_is_returned_whole_and_readonly() -> None:
    qz, model = _curve(50)
    kept_qz, kept_model = downsampled_preview(qz, model, max_points=200)
    np.testing.assert_allclose(kept_qz, qz)
    np.testing.assert_allclose(kept_model, model)
    assert not kept_qz.flags.writeable
    assert not kept_model.flags.writeable


def test_preview_over_the_bound_keeps_endpoints_and_sample_order() -> None:
    qz, model = _curve(5000)
    kept_qz, kept_model = downsampled_preview(qz, model, max_points=200)
    assert kept_qz.size <= 200
    assert kept_qz.size == kept_model.size
    assert kept_qz[0] == qz[0]
    assert kept_qz[-1] == qz[-1]
    assert np.all(np.diff(kept_qz) > 0)
    assert not kept_qz.flags.writeable
    assert not kept_model.flags.writeable


def test_preview_pairs_every_kept_abscissa_with_its_own_ordinate() -> None:
    qz, model = _curve(1000)
    kept_qz, kept_model = downsampled_preview(qz, model, max_points=64)
    expected = np.exp(-kept_qz)
    np.testing.assert_allclose(kept_model, expected)


@pytest.mark.parametrize("max_points", [2, 3, 7, 199])
def test_preview_respects_every_bound_at_least_two(max_points: int) -> None:
    qz, model = _curve(2048)
    kept_qz, _kept_model = downsampled_preview(qz, model, max_points=max_points)
    assert 2 <= kept_qz.size <= max_points
    assert kept_qz[0] == qz[0]
    assert kept_qz[-1] == qz[-1]


@pytest.mark.parametrize("max_points", [True, False, 1, 0, -3, 2.0, "200", None])
def test_preview_rejects_a_bound_that_is_not_an_integer_of_at_least_two(
    max_points: object,
) -> None:
    qz, model = _curve(10)
    with pytest.raises(ValueError, match="max_points"):
        downsampled_preview(qz, model, max_points=max_points)


def test_absent_preview_axes_stay_absent() -> None:
    assert freeze_preview_axes(None, None) == (None, None)


@pytest.mark.parametrize(
    ("qz", "model", "message"),
    [
        ([0.1, 0.2], None, "together"),
        (None, [0.1, 0.2], "together"),
        ([0.1, 0.2, 0.3], [0.1, 0.2], "equal lengths"),
        ([], [], "must not be empty"),
        ([[0.1, 0.2]], [[0.1, 0.2]], "1-dimensional"),
    ],
)
def test_preview_axes_reject_unusable_pairs(
    qz: object, model: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        freeze_preview_axes(qz, model)


def test_frozen_preview_axes_own_their_values_independently_of_the_caller() -> None:
    source_qz = np.array([0.1, 0.2, 0.3])
    source_model = np.array([1.0, 0.5, 0.25])
    qz, model = freeze_preview_axes(source_qz, source_model)
    assert qz is not None and model is not None
    source_qz[0] = 9.0
    source_model[0] = 9.0
    assert qz[0] == 0.1
    assert model[0] == 1.0
    assert not qz.flags.writeable
    assert not model.flags.writeable
