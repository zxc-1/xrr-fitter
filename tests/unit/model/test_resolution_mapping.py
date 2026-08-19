from __future__ import annotations

import warnings

import numpy as np
import pytest

from xrr_fitter.model.instrument import resolution_to_sigma_q


def test_angle_resolution_handles_nonfinite_angle_without_runtime_warning() -> None:
    angles = np.array([0.1, np.nan, 0.3])
    resolution = np.array([0.02, 0.02, 0.03])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sigma_q = resolution_to_sigma_q(
            angles,
            resolution,
            "fwhm_two_theta_deg",
            1.5406,
            0.0,
        )

    assert np.isfinite(sigma_q[[0, 2]]).all()
    assert np.isnan(sigma_q[1])
    assert sigma_q.flags.writeable is False


def test_angle_resolution_rejects_finite_inputs_that_overflow_q_conversion() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="finite"):
            resolution_to_sigma_q(
                np.array([1.0]),
                np.array([1e308]),
                "sigma_two_theta_deg",
                1e-300,
                0.0,
            )

    assert not any(item.category is RuntimeWarning for item in caught)


def test_angle_resolution_avoids_intermediate_inverse_wavelength_overflow() -> None:
    two_theta = np.array([2.0])
    sigma_two_theta = np.array([1e-10])
    wavelength = 1e-308

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = resolution_to_sigma_q(
            two_theta,
            sigma_two_theta,
            "sigma_two_theta_deg",
            wavelength,
            0.0,
        )

    expected = (
        4.0 * np.pi * np.abs(np.cos(np.deg2rad(two_theta / 2.0))) * np.deg2rad(sigma_two_theta / 2.0) / wavelength
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-15)
    assert np.all(np.isfinite(observed))
    assert not any(item.category is RuntimeWarning for item in caught)
