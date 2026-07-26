from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.support.model_cases import prepared_data
from xrr_fitter.model.data import PreparedData, fit_ready


def test_prepared_arrays_are_copied_and_read_only() -> None:
    angles = np.linspace(0.1, 3.2, 32)
    data = prepared_data(two_theta_deg=angles)

    angles[0] = 99.0

    assert data.two_theta_deg[0] == pytest.approx(0.1)
    for field in (
        "two_theta_deg",
        "intensity_raw",
        "qz_a_inv",
        "intensity_normalized",
        "validation_mask",
        "fit_mask",
    ):
        value = getattr(data, field)
        assert value.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            value[0] = value[0]


def test_fit_ready_requires_thirty_unique_points_and_nonzero_span() -> None:
    angles = np.linspace(0.1, 3.2, 32)
    qz = np.linspace(0.01, 0.2, 32)
    selected = np.ones(32, dtype=bool)

    assert fit_ready(angles, qz, selected)
    assert not fit_ready(angles[:29], qz[:29], selected[:29])
    assert not fit_ready(np.ones(32), qz, selected)
    assert not fit_ready(angles, np.ones(32), selected)


def test_nonfinite_angle_is_preserved_as_invalid_without_runtime_warning() -> None:
    angles = np.linspace(0.1, 3.2, 32)
    angles[-1] = np.nan
    valid = np.ones(32, dtype=bool)
    valid[-1] = False

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        data = prepared_data(two_theta_deg=angles, validation_mask=valid, fit_mask=valid)

    assert np.isnan(data.two_theta_deg[-1])
    assert not data.validation_mask[-1]
    assert not data.fit_mask[-1]


def test_qz_normalization_noise_floor_and_zero_intensity() -> None:
    intensity = np.linspace(1000.0, 0.0, 32)
    data = prepared_data(intensity_raw=intensity)

    assert data.normalization == pytest.approx(1000.0)
    assert data.r_floor > 0.0
    assert data.intensity_normalized[-1] == 0.0


def test_trailing_nonfinite_numeric_row_remains_derived_invalid_data() -> None:
    intensity = np.linspace(1000.0, 10.0, 32)
    intensity[-1] = np.nan
    valid = np.ones(32, dtype=bool)
    valid[-1] = False

    data = prepared_data(intensity_raw=intensity, validation_mask=valid, fit_mask=valid)

    assert np.isnan(data.intensity_raw[-1])
    assert not data.validation_mask[-1]


def test_prepared_data_rejects_misaligned_derived_arrays() -> None:
    data = prepared_data()
    values = {field: getattr(data, field) for field in data.__dataclass_fields__}
    values["qz_a_inv"] = np.zeros(31)

    with pytest.raises(ValueError, match="qz_a_inv.*length"):
        PreparedData(**values)
