from __future__ import annotations

import warnings

import numpy as np
import pytest
from tests.support.model_cases import prepared_data

from xrr_fitter.model.data import (
    AngleConventionEvidence,
    PreparedData,
    fit_ready,
    qz_from_two_theta,
)


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


def test_fit_ready_rejects_nonmonotonic_selected_qz() -> None:
    angles = np.linspace(0.1, 3.2, 32)
    qz = np.linspace(0.01, 0.2, 32)
    qz[15], qz[16] = qz[16], qz[15]
    selected = np.ones(32, dtype=bool)

    assert not fit_ready(angles, qz, selected)


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


def test_qz_overflow_is_preserved_as_an_invalid_point_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qz, valid = qz_from_two_theta(np.array([2.0]), 1e-310, 0.0)

    assert np.isnan(qz[0])
    assert valid.tolist() == [False]
    assert not any(item.category is RuntimeWarning for item in caught)


def test_qz_angle_overflow_is_preserved_as_an_invalid_point_without_warning() -> None:
    maximum = np.finfo(float).max

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qz, valid = qz_from_two_theta(np.array([maximum]), 1.0, maximum)

    assert np.isnan(qz[0])
    assert valid.tolist() == [False]
    assert not any(item.category is RuntimeWarning for item in caught)


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


def test_declared_angle_convention_evidence_carries_the_line_it_read() -> None:
    """有声明就必须能出示原文，而且不再带「说不清」的理由。

    ``declared`` 是给回填用的开关：它为真，导入对话框就会把轴那两个单选拨过去。
    所以它不能只是一个布尔——判定依据（那一行表头）必须同时在手，否则用户看见轴
    被改了却无从核对。同理，能定论就不该再留 warning：两者并存读起来是「我确定，
    但我不确定」。
    """
    evidence = AngleConventionEvidence(
        convention="theta",
        declared=True,
        header_line="# Omega Intensity",
        warning=None,
    )

    assert evidence.convention == "theta"
    assert evidence.header_line == "# Omega Intensity"
    assert evidence.warning is None
    with pytest.raises(ValueError, match="header_line"):
        AngleConventionEvidence(convention="theta", declared=True, header_line=None, warning=None)
    with pytest.raises(ValueError, match="warning"):
        AngleConventionEvidence(
            convention="theta",
            declared=True,
            header_line="# Omega Intensity",
            warning="也许是入射角",
        )


def test_undeclared_angle_convention_evidence_must_say_why() -> None:
    """没定论必须给理由，并且只回落到 ``two_theta``。

    回落方向只有这一个是安全的：猜成 ``"theta"`` 会让角度整段翻倍，而
    ``"two_theta"`` 正是既有默认，猜错也只是维持现状。
    """
    evidence = AngleConventionEvidence(
        convention="two_theta",
        declared=False,
        header_line=None,
        warning="表头没有写角度轴",
    )

    assert evidence.convention == "two_theta"
    assert evidence.declared is False
    with pytest.raises(ValueError, match="warning"):
        AngleConventionEvidence(convention="two_theta", declared=False, header_line=None, warning=None)


def test_angle_convention_evidence_rejects_an_axis_the_model_cannot_read() -> None:
    with pytest.raises(ValueError, match="angle_convention"):
        AngleConventionEvidence(convention="omega", declared=False, header_line=None, warning="?")
