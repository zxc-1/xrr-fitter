from __future__ import annotations

import warnings
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from xrr_fitter.io.xy import _merged_intensity, read_xy, resolution_to_sigma_q, xy_bytes
from xrr_fitter.model.data import BeamSpec, DataColumnMapping, with_fit_mask

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/source/header_and_duplicates.xy"
MONO = BeamSpec(kind="monochromatic")


def test_xy_bytes_uses_the_canonical_export_format() -> None:
    angles = np.array([0.08, 1.234567890123])
    intensities = np.array([0.5, 1.2345678901234567e-8])

    content = xy_bytes(angles, intensities)

    assert content == (
        b"# 2theta_deg intensity\n0.0800000000 5.0000000000000000e-01\n1.2345678901 1.2345678901234567e-08\n"
    )


@pytest.mark.parametrize(
    ("angles", "intensities"),
    (
        (np.array([[0.1]]), np.array([1.0])),
        (np.array([0.1, 0.2]), np.array([1.0])),
        (np.array([0.1]), np.array([np.nan])),
        (np.array([]), np.array([])),
    ),
)
def test_xy_bytes_rejects_invalid_curve_axes(
    angles: np.ndarray,
    intensities: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="curve axes"):
        xy_bytes(angles, intensities)


def _write_numeric_curve(
    path: Path,
    angles: np.ndarray,
    intensities: np.ndarray,
) -> None:
    rows = ["header"] + [f"{angle:.17g} {intensity:.17g}" for angle, intensity in zip(angles, intensities, strict=True)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_r22_parser_fixture_has_exact_frozen_bytes() -> None:
    content = FIXTURE.read_bytes()

    assert len(content) == 103
    assert sha256(content).hexdigest() == ("621f5b7c5234f5f5781d819b98722efdfc7ac5e8a517670364ae99aba46f9136")


def test_read_xy_requires_an_explicit_beam() -> None:
    with pytest.raises(TypeError):
        read_xy(FIXTURE)


def test_read_xy_preserves_hash_and_merges_duplicate_angles() -> None:
    data = read_xy(FIXTURE, beam=MONO)

    assert data.beam == MONO
    assert len(data.raw_rows) == 9
    np.testing.assert_allclose(data.two_theta_deg, [0.10, 0.20, 0.30, 0.40])
    np.testing.assert_allclose(data.intensity_raw, [1000, 600, -1, 100])
    assert data.fit_mask.tolist() == [True, True, False, True]
    assert data.source_sha256 == sha256(FIXTURE.read_bytes()).hexdigest()
    assert data.raw_parse_status == (
        "header",
        "header",
        "header",
        "data",
        "data",
        "data",
        "malformed",
        "data",
        "data",
    )
    assert data.source_row_groups == ((3,), (4, 5), (7,), (8,))


def test_duplicate_angle_inverse_variance_merge_scales_extreme_small_sigmas() -> None:
    group = [
        (0.1, 100.0, 1e-300, float("nan"), 0),
        (0.1, 200.0, 1e-300, float("nan"), 1),
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intensity, sigma = _merged_intensity(group, has_sigma=True)

    assert intensity == pytest.approx(150.0)
    assert sigma == pytest.approx(1e-300 / np.sqrt(2.0))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_duplicate_angle_median_avoids_finite_extreme_sum_overflow() -> None:
    maximum = np.finfo(float).max
    group = [
        (0.1, maximum, float("nan"), float("nan"), 0),
        (0.1, maximum, float("nan"), float("nan"), 1),
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intensity, sigma = _merged_intensity(group, has_sigma=False)

    assert intensity == maximum
    assert np.isnan(sigma)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_duplicate_angle_weighted_mean_avoids_finite_extreme_sum_overflow() -> None:
    maximum = np.finfo(float).max
    group = [
        (0.1, maximum, 1.0, float("nan"), 0),
        (0.1, maximum, 1.0, float("nan"), 1),
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intensity, sigma = _merged_intensity(group, has_sigma=True)

    assert intensity == maximum
    assert sigma == pytest.approx(1.0 / np.sqrt(2.0))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_duplicate_angle_resolution_median_avoids_finite_extreme_sum_overflow(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resolution-median-overflow.xy"
    maximum = np.finfo(float).max
    rows = [
        f"0.2 100 {maximum:.17g}",
        f"0.2 200 {maximum:.17g}",
        *[f"{0.2 + 0.02 * index:.17g} {1000.0 / index:.17g} 0.01" for index in range(1, 40)],
    ]
    path.write_text("header\n" + "\n".join(rows) + "\n", encoding="utf-8")
    mapping = DataColumnMapping(
        two_theta=0,
        intensity=1,
        resolution=2,
        resolution_kind="sigma_q_a_inv",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = read_xy(path, beam=MONO, column_mapping=mapping)

    assert data.resolution_raw is not None
    assert data.resolution_raw[0] == maximum
    assert data.sigma_q_a_inv is not None
    assert data.sigma_q_a_inv[0] == maximum
    assert not any(item.category is RuntimeWarning for item in caught)


def test_imported_fit_mask_is_immutable_overlay(tmp_path: Path) -> None:
    path = tmp_path / "mask.xy"
    angles = np.linspace(0.1, 4.0, 40)
    intensities = np.geomspace(1.0, 1e-7, 40)
    intensities[7] = -10.0
    _write_numeric_curve(path, angles, intensities)
    data = read_xy(path, beam=MONO)
    requested = data.fit_mask.copy()
    requested[8] = False

    updated = with_fit_mask(data, requested)

    assert updated.source_sha256 == data.source_sha256
    assert not updated.fit_mask[8]
    assert updated.fit_mask.flags.writeable is False
    invalid = updated.fit_mask.copy()
    invalid[7] = True
    with pytest.raises(ValueError, match="cannot enable invalid"):
        with_fit_mask(updated, invalid)


def test_import_marks_overflowed_normalized_intensity_invalid_without_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "normalization-overflow.xy"
    angles = np.linspace(0.1, 4.0, 40)
    intensities = np.r_[
        np.full(20, 1e-300),
        np.full(20, np.finfo(float).max),
    ]
    _write_numeric_curve(path, angles, intensities)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = read_xy(path, beam=MONO)

    assert np.isposinf(data.intensity_normalized[-1])
    assert not np.any(data.validation_mask[20:])
    assert not any(item.category is RuntimeWarning for item in caught)


def test_import_uses_explicit_floor_fallback_when_high_angle_noise_overflows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "noise-overflow.xy"
    angles = np.linspace(0.1, 10.0, 100)
    intensities = np.r_[
        np.full(20, 1e-300),
        np.full(80, np.finfo(float).max),
    ]
    _write_numeric_curve(path, angles, intensities)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = read_xy(path, beam=MONO)

    assert data.r_floor == pytest.approx(1e-8)
    assert "高角归一化强度不可表示，R_floor 使用 1e-8" in data.warnings
    assert not any(item.category is RuntimeWarning for item in caught)


def test_import_rejects_unrepresentable_normalized_intensity_uncertainty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sigma-normalization-overflow.xy"
    angles = np.linspace(0.1, 4.0, 40)
    intensities = np.r_[np.full(20, 1e-300), np.full(20, 1e-200)]
    rows = [
        f"{angle:.17g} {intensity:.17g} {np.finfo(float).max:.17g}"
        for angle, intensity in zip(angles, intensities, strict=True)
    ]
    path.write_text("header\n" + "\n".join(rows) + "\n", encoding="utf-8")
    mapping = DataColumnMapping(two_theta=0, intensity=1, intensity_sigma=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="normalized intensity uncertainty"):
            read_xy(path, beam=MONO, column_mapping=mapping)

    assert not any(item.category is RuntimeWarning for item in caught)


def test_explicit_column_mapping_uses_inverse_variance_and_converts_fwhm_angle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mapped.xy"
    rows = [
        "header",
        "0.20 100 10 0.020",
        "0.20 200 20 0.020",
        *[f"{0.02 * index + 0.2:.4f} {1000 / index:.8g} 5 0.030" for index in range(1, 40)],
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    mapping = DataColumnMapping(
        two_theta=0,
        intensity=1,
        intensity_sigma=2,
        resolution=3,
        resolution_kind="fwhm_two_theta_deg",
    )

    data = read_xy(path, beam=MONO, column_mapping=mapping)

    assert data.intensity_raw[0] == pytest.approx(120.0)
    assert data.intensity_sigma_raw is not None
    assert data.intensity_sigma_raw[0] == pytest.approx(np.sqrt(1 / (1 / 10**2 + 1 / 20**2)))
    assert data.resolution_raw is not None
    expected = resolution_to_sigma_q(
        data.two_theta_deg,
        data.resolution_raw,
        "fwhm_two_theta_deg",
        wavelength_a=data.beam.effective_wavelength_a,
        angle_offset_deg=data.import_angle_offset_deg,
    )
    np.testing.assert_allclose(data.sigma_q_a_inv, expected)
    np.testing.assert_allclose(
        data.intensity_sigma_normalized,
        data.intensity_sigma_raw / data.normalization,
    )


def test_unmapped_extra_columns_are_not_guessed(tmp_path: Path) -> None:
    path = tmp_path / "extra.xy"
    rows = [f"{0.02 * index:.4f} {1000 / index:.8g} 999 0.01" for index in range(1, 41)]
    path.write_text("header\n" + "\n".join(rows), encoding="utf-8")

    data = read_xy(path, beam=MONO)

    assert data.intensity_sigma_raw is None
    assert data.resolution_raw is None
    assert data.sigma_q_a_inv is None


def test_import_rejects_incident_angles_above_ninety_degrees(tmp_path: Path) -> None:
    path = tmp_path / "out-of-domain-angle.xy"
    angles = np.r_[np.linspace(0.1, 3.9, 40), 200.0]
    intensities = np.geomspace(1.0, 1e-7, angles.size)
    _write_numeric_curve(path, angles, intensities)

    data = read_xy(path, beam=MONO)

    assert data.fit_ready is True
    assert data.validation_mask[-1] is np.False_
    assert data.fit_mask[-1] is np.False_


@pytest.mark.parametrize(
    "mapping",
    (
        {"two_theta": 0, "intensity": 0},
        {"two_theta": -1, "intensity": 1},
        {
            "two_theta": 0,
            "intensity": 1,
            "resolution": 2,
            "resolution_kind": "unknown",
        },
    ),
)
def test_column_mapping_rejects_invalid_indices_and_kind(
    mapping: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        DataColumnMapping(**mapping)
