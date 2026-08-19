from __future__ import annotations

from math import pi

import numpy as np
import pytest

from xrr_fitter.model.data import BeamSpec, qz_from_two_theta
from xrr_fitter.model.instrument import InstrumentSpec


def test_beam_values_validate_domain_and_effective_wavelength() -> None:
    mono = BeamSpec("monochromatic", wavelength_a=1.6)
    mixed = BeamSpec("mixed_kalpha", wavelength_1_a=1.5, wavelength_2_a=1.6)

    assert mono.effective_wavelength_a == 1.6
    assert mixed.effective_wavelength_a == 1.5
    with pytest.raises(ValueError, match="beam kind"):
        BeamSpec("unknown")
    with pytest.raises(ValueError, match="wavelength_a"):
        BeamSpec("monochromatic", wavelength_a=0.0)


def test_mixed_kalpha_dataset_builds_qz_grid_from_kalpha1() -> None:
    beam = BeamSpec("mixed_kalpha", wavelength_1_a=1.2, wavelength_2_a=1.4)
    angles = np.array([1.0, 2.0])

    qz, _ = qz_from_two_theta(angles, beam.effective_wavelength_a, 0.0)

    expected = 4.0 * pi * np.sin(np.deg2rad(angles / 2.0)) / 1.2
    np.testing.assert_allclose(qz, expected)


def test_qz_is_derived_from_stored_angles_without_grid_reconstruction() -> None:
    angles = np.array([0.1, 0.14, 0.31, 0.9])

    qz, positive = qz_from_two_theta(angles, 1.5406, 0.02)

    theta = angles / 2.0 + 0.02
    np.testing.assert_allclose(qz, 4.0 * pi * np.sin(np.deg2rad(theta)) / 1.5406)
    assert positive.tolist() == [True, True, True, True]


def test_qz_marks_angles_beyond_the_grazing_incidence_domain_invalid() -> None:
    angles = np.array([0.0, 0.2, 180.0, 180.2, 360.0])

    _qz, positive = qz_from_two_theta(angles, 1.5406, 0.0)

    assert positive.tolist() == [False, True, True, False, False]


def test_instrument_values_enforce_mode_specific_geometry() -> None:
    angle = float(np.rad2deg(np.arcsin(0.1)))
    value = InstrumentSpec(
        footprint_mode="geometry",
        footprint_spill_angle_deg=angle,
        sample_length_mm=10.0,
        beam_width_mm=1.0,
    )

    assert value.footprint_mode == "geometry"
    with pytest.raises(ValueError, match="dimensions"):
        InstrumentSpec(footprint_mode="fit", sample_length_mm=10.0)
    with pytest.raises(ValueError, match="zero spill"):
        InstrumentSpec(footprint_mode="none", footprint_spill_angle_deg=0.1)
    with pytest.raises(ValueError, match="footprint_spill_angle_deg"):
        InstrumentSpec(footprint_mode="fit", footprint_spill_angle_deg=90.1)
