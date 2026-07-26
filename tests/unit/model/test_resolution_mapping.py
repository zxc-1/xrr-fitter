from __future__ import annotations

import warnings

import numpy as np

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
