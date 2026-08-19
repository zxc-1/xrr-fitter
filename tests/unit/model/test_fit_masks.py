from __future__ import annotations

import warnings

import numpy as np
import pytest
from tests.support.model_cases import prepared_data

from xrr_fitter.model.data import log_domain_mask, with_fit_mask


def test_fit_mask_is_immutable_overlay_and_cannot_enable_invalid_point() -> None:
    valid = np.ones(32, dtype=bool)
    valid[-1] = False
    data = prepared_data(validation_mask=valid, fit_mask=valid)
    requested = valid.copy()
    requested[0] = False

    updated = with_fit_mask(data, requested)
    requested[1] = False

    assert updated.fit_mask[1]
    assert updated.fit_mask.flags.writeable is False
    invalid = updated.fit_mask.copy()
    invalid[-1] = True
    with pytest.raises(ValueError, match="cannot enable invalid"):
        with_fit_mask(updated, invalid)


def test_floor_fallback_and_log_domain_mask() -> None:
    mask = log_domain_mask(np.array([1.0, 0.0, -1e-9, -1.0, np.nan]), 1e-8)

    assert mask.tolist() == [True, True, True, False, False]
    assert mask.flags.writeable is False


def test_log_domain_mask_handles_large_finite_sum_without_warning() -> None:
    maximum = np.finfo(float).max

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = log_domain_mask(np.array([maximum, -maximum]), maximum)

    assert mask.tolist() == [True, False]
    assert not any(item.category is RuntimeWarning for item in caught)


def test_mask_recomputes_fit_ready() -> None:
    data = prepared_data(size=32)
    sparse = np.zeros(32, dtype=bool)
    sparse[:29] = True

    updated = with_fit_mask(data, sparse)

    assert data.fit_ready is True
    assert updated.fit_ready is False
