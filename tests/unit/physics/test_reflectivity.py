from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.structure import SlabStack
from xrr_fitter.physics.parratt import fresnel_interfaces, layer_kz, parratt_reflectivity


def test_bare_substrate_matches_fresnel_amplitude() -> None:
    q = np.geomspace(1e-4, 1, 401)
    sld = 20e-6 + 0.2e-6j
    kz = np.sqrt((q / 2) ** 2 - 4 * np.pi * sld)
    expected = np.abs((q / 2 - kz) / (q / 2 + kz)) ** 2
    np.testing.assert_allclose(parratt_reflectivity(q, SlabStack([0, 0], [0j, sld], [0])), expected, rtol=5e-12, atol=1e-13)


def test_non_air_fronting_preserves_complete_complex_sld_difference() -> None:
    q = np.array([0.003, 0.02, 0.2])
    front, back = 6e-6 + 2e-6j, 20e-6 + 0.5e-6j
    kz = np.sqrt(((q / 2) ** 2 - 4 * np.pi * (back - front)).astype(complex))
    kz = np.where((kz.imag > 0) | ((kz.imag == 0) & (kz.real < 0)), -kz, kz)
    expected = np.abs((q / 2 - kz) / (q / 2 + kz)) ** 2
    np.testing.assert_allclose(parratt_reflectivity(q, SlabStack([0, 0], [front, back], [0])), expected, rtol=5e-12, atol=1e-13)


def test_absorbing_layer_never_produces_nonfinite_reflectivity() -> None:
    result = parratt_reflectivity(np.geomspace(1e-4, 1, 800), SlabStack([0, 400, 0], [0j, 40e-6 + 15e-6j, 20e-6 + 1e-6j], [3, 5]))
    assert np.all(np.isfinite(result)) and np.all(result >= 0)


def test_equal_sld_internal_layer_disappears() -> None:
    q = np.linspace(0.005, 0.8, 500)
    sld = 12e-6 + 0.1e-6j
    np.testing.assert_allclose(parratt_reflectivity(q, SlabStack([0, 73, 0], [0j, sld, sld], [0, 0])), parratt_reflectivity(q, SlabStack([0, 0], [0j, sld], [0])), rtol=1e-10, atol=1e-13)


def test_single_film_returns_finite_interference_curve() -> None:
    result = parratt_reflectivity(np.linspace(0.005, 0.8, 500), SlabStack([0, 137, 0], [0j, 35e-6 + 0.8e-6j, 20e-6 + 0.2e-6j], [4, 6]))
    assert np.all(np.isfinite(result))
    assert np.ptp(result) > 0


def test_zero_thickness_test_stack_matches_layer_deletion() -> None:
    q = np.linspace(0.005, 0.8, 500)
    with_zero = SlabStack([0, 90, 0, 70, 0], [0j, 12e-6, 30e-6, 12e-6, 20e-6], [0, 0, 0, 0])
    deleted = SlabStack([0, 90, 70, 0], [0j, 12e-6, 12e-6, 20e-6], [0, 0, 0])
    np.testing.assert_allclose(parratt_reflectivity(q, with_zero), parratt_reflectivity(q, deleted), rtol=1e-10, atol=1e-13)


def test_roughness_reduces_high_q_envelope_and_zero_restores_ideal() -> None:
    q = np.linspace(0.4, 1, 600)
    ideal = SlabStack([0, 0], [0j, 30e-6 + 0.2e-6j], [0])
    assert np.all(parratt_reflectivity(q, SlabStack([0, 0], ideal.sld_a2, [8])) <= parratt_reflectivity(q, ideal) * (1 + 1e-12))


def test_single_layer_kiessig_spacing_is_two_pi_over_thickness() -> None:
    from scipy.signal import find_peaks
    thickness = 250.0
    q = np.linspace(0.1, 0.8, 12000)
    peaks, _ = find_peaks(parratt_reflectivity(q, SlabStack([0, thickness, 0], [0j, 35e-6, 20e-6], [0, 0])))
    assert np.median(np.diff(q[peaks][-12:])) == pytest.approx(2 * np.pi / thickness, rel=0.02)


def test_layer_wavevectors_use_decaying_branch_and_nonnegative_real_tie() -> None:
    from xrr_fitter.physics.parratt import select_decaying_branch

    roots = np.array([-2 + 0j, 3 + 0j, 1 + 2j, 1 - 2j])
    selected = select_decaying_branch(roots)
    np.testing.assert_array_equal(selected, np.array([2 + 0j, 3 + 0j, -1 - 2j, 1 - 2j]))
    kz = layer_kz(np.array([0.0, 0.2]), np.array([0j, 20e-6, -5e-6]))
    assert np.all(kz.imag <= 0)


def test_exact_fresnel_denominator_is_a_hard_failure() -> None:
    with pytest.raises(FloatingPointError, match="zero Fresnel denominator"):
        fresnel_interfaces(np.array([[1 + 0j, -1 + 0j]]), np.array([0.0]))
