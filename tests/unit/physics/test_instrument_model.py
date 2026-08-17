from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.model.structure import (
    LayerSpec,
    MaterialSpec,
    StructureSpec,
)
from xrr_fitter.physics.footprint import footprint_factor
from xrr_fitter.physics.parratt import parratt_reflectivity
from xrr_fitter.physics.reflectivity import instrument_reflectivity, qz_from_theta_deg
from xrr_fitter.physics.stack import expand_structure

MONO = BeamSpec(kind="monochromatic")


def _stack(sld: complex = 30e-6 + 0.2e-6j) -> SlabStack:
    return SlabStack([0, 120, 0], [0j, sld, 20e-6 + 0.1e-6j], [3, 4])


def test_footprint_factor_truncates_below_spill_angle_and_is_unity_above() -> None:
    theta = np.array([0.05, 0.15, 0.30, 0.31, 1])
    actual = footprint_factor(theta, 0.30)
    np.testing.assert_allclose(actual[:2], np.sin(np.deg2rad(theta[:2])) / np.sin(np.deg2rad(0.30)))
    np.testing.assert_array_equal(actual[2:], 1)


def test_footprint_factor_is_identically_one_when_spill_angle_is_zero() -> None:
    np.testing.assert_array_equal(footprint_factor(np.linspace(0.01, 2, 50), 0), np.ones(50))


def test_expert_linear_background_is_applied_in_q_space() -> None:
    theta = np.linspace(0.05, 2, 150)
    stack = _stack()
    q = qz_from_theta_deg(theta, MONO.wavelength_a)
    actual = instrument_reflectivity(theta, stack, MONO, background=3e-8, linear_background_per_a_inv=2e-8)
    np.testing.assert_allclose(actual, parratt_reflectivity(q, stack) + 3e-8 + 2e-8 * q, rtol=1e-13)


def test_expert_powerlaw_background_adds_diffuse_tail_in_q_space() -> None:
    theta = np.linspace(0.2, 2, 120)
    stack = _stack()
    q = qz_from_theta_deg(theta, MONO.wavelength_a)
    actual = instrument_reflectivity(
        theta, stack, MONO, background=3e-8, powerlaw_background_amplitude=1e-10, powerlaw_background_exponent=2.5
    )
    np.testing.assert_allclose(actual, parratt_reflectivity(q, stack) + 3e-8 + 1e-10 * q**-2.5, rtol=1e-13)


@pytest.mark.parametrize(
    ("amplitude", "exponent", "message"),
    [
        (-1e-10, 3.0, "powerlaw_background_amplitude"),
        (1e-10, 0.99, "powerlaw_background_exponent"),
        (1e-10, 4.01, "powerlaw_background_exponent"),
    ],
)
def test_powerlaw_background_rejects_invalid_parameters(amplitude: float, exponent: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        instrument_reflectivity(
            np.array([0.2]),
            _stack(),
            MONO,
            powerlaw_background_amplitude=amplitude,
            powerlaw_background_exponent=exponent,
        )


def test_mixed_kalpha_is_ratio_weighted_sum_of_full_single_wavelength_models() -> None:
    theta = np.linspace(0.05, 2, 120)
    beam = BeamSpec(kind="mixed_kalpha", intensity_ratio_21=0.37)
    first, second = _stack(), _stack(29e-6 + 0.25e-6j)
    mixed = instrument_reflectivity(theta, first, beam, second, relative_sigma=0.002)
    one = instrument_reflectivity(
        theta, first, BeamSpec(kind="monochromatic", wavelength_a=beam.wavelength_1_a), relative_sigma=0.002
    )
    two = instrument_reflectivity(
        theta, second, BeamSpec(kind="monochromatic", wavelength_a=beam.wavelength_2_a), relative_sigma=0.002
    )
    np.testing.assert_allclose(mixed, (one + beam.intensity_ratio_21 * two) / (1 + beam.intensity_ratio_21), rtol=1e-13)


def test_mixed_kalpha_requires_secondary_stack_and_mono_rejects_secondary_stack() -> None:
    with pytest.raises(ValueError, match="secondary_stack"):
        instrument_reflectivity(np.array([0.2]), _stack(), BeamSpec(kind="mixed_kalpha"))
    with pytest.raises(ValueError, match="secondary_stack"):
        instrument_reflectivity(np.array([0.2]), _stack(), MONO, _stack())


def test_mixed_kalpha_applies_scale_once_and_samples_background_at_primary_q() -> None:
    theta = np.linspace(0.2, 2, 100)
    beam = BeamSpec(kind="mixed_kalpha", intensity_ratio_21=0.37)
    first, second = _stack(), _stack(29e-6 + 0.25e-6j)
    first_curve = instrument_reflectivity(
        theta, first, BeamSpec(kind="monochromatic", wavelength_a=beam.wavelength_1_a), relative_sigma=0.002
    )
    second_curve = instrument_reflectivity(
        theta, second, BeamSpec(kind="monochromatic", wavelength_a=beam.wavelength_2_a), relative_sigma=0.002
    )
    q_primary = qz_from_theta_deg(theta, beam.wavelength_1_a)
    expected_signal = (first_curve + beam.intensity_ratio_21 * second_curve) / (1 + beam.intensity_ratio_21)
    actual = instrument_reflectivity(
        theta,
        first,
        beam,
        second,
        scale=2.3,
        background=3e-8,
        linear_background_per_a_inv=2e-8,
        relative_sigma=0.002,
    )
    np.testing.assert_allclose(actual, 2.3 * expected_signal + 3e-8 + 2e-8 * q_primary, rtol=1e-13)


def test_formula_material_expands_to_different_sld_for_mixed_kalpha_wavelengths() -> None:
    air = MaterialSpec("Air", None, None, 0j)
    si = MaterialSpec("Si", "Si", 2.329)
    structure = StructureSpec(air, (LayerSpec("Si", si, 20),), si)
    beam = BeamSpec(kind="mixed_kalpha")
    assert not np.array_equal(
        expand_structure(structure, beam.wavelength_1_a).sld_a2, expand_structure(structure, beam.wavelength_2_a).sld_a2
    )


def test_footprint_multiplies_smeared_signal_before_background() -> None:
    theta = np.linspace(0.05, 1.5, 200)
    stack = _stack()
    plain = instrument_reflectivity(theta, stack, MONO, scale=2, background=3e-8, relative_sigma=0.002)
    actual = instrument_reflectivity(
        theta, stack, MONO, scale=2, background=3e-8, footprint_spill_angle_deg=0.35, relative_sigma=0.002
    )
    factor = footprint_factor(theta, 0.35)
    np.testing.assert_allclose(actual, factor * (plain - 3e-8) + 3e-8, rtol=1e-13)


def test_theta_domain_resolution_rejects_q_domain_widths() -> None:
    theta = np.linspace(0.1, 1.5, 80)
    with pytest.raises(ValueError, match="q-domain and theta-domain resolution cannot be combined"):
        instrument_reflectivity(
            theta, _stack(), MONO, resolution_domain="theta", relative_sigma=0.01, sigma_theta_deg=0.004
        )
    with pytest.raises(ValueError, match="resolution_domain must be q or theta"):
        instrument_reflectivity(theta, _stack(), MONO, resolution_domain="q", sigma_theta_deg=0.004)


def test_theta_domain_resolution_matches_gaussian_smear_in_theta_space() -> None:
    from xrr_fitter.physics.resolution import gaussian_smear

    theta = np.linspace(0.1, 1.5, 80)
    stack = _stack()
    actual = instrument_reflectivity(theta, stack, MONO, resolution_domain="theta", sigma_theta_deg=0.001)
    expected = gaussian_smear(
        theta,
        lambda query: parratt_reflectivity(qz_from_theta_deg(query, MONO.wavelength_a), stack),
        absolute_sigma_a_inv=0.001,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
