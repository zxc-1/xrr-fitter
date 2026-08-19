from __future__ import annotations

import warnings
from time import perf_counter

import numpy as np
import pytest

from xrr_fitter.model.slab_stack import PeriodicSpan, SlabStack
from xrr_fitter.physics.parratt import normalize_mobius, parratt_reflectivity


def _overflow_stack(*, periodic: bool = False) -> SlabStack:
    """Return a valid low-level stack whose Nevot-Croce exponent overflows."""
    media = [
        0j,
        0.09009274 + 5.25772738e-05j,
        0.08972989 + 3.62374192e-04j,
    ]
    if periodic:
        return SlabStack(
            [0, 1, 1, 1, 1, 0],
            [media[0], media[1], media[2], media[1], media[2], media[0]],
            [0, 30, 0, 30, 0],
            (PeriodicSpan(1, 2, 2),),
        )
    return SlabStack([0, 1, 1, 0], [*media, media[0]], [0, 30, 0])


@pytest.mark.parametrize("periodic", [False, True])
def test_parratt_rejects_nonfinite_nevot_croce_path_without_warning(periodic: bool) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="finite|Nevot-Croce"):
            parratt_reflectivity(np.asarray([0.011774154985086273]), _overflow_stack(periodic=periodic))

    assert caught == []


def _periodic(repeats: int = 20) -> tuple[SlabStack, SlabStack]:
    thickness = np.array([0] + [28, 42] * repeats + [0], float)
    sld = np.array([0j] + [55e-6 + 1.4e-6j, 20e-6 + 0.2e-6j] * repeats + [24e-6 + 0.3e-6j])
    roughness = np.array([2.5, 4] + [3, 4] * (repeats - 1) + [5], float)
    return SlabStack(thickness, sld, roughness, (PeriodicSpan(1, 2, repeats),)), SlabStack(thickness, sld, roughness)


def test_periodic_mobius_parratt_matches_expanded_recurrence() -> None:
    periodic, expanded = _periodic()
    q = np.r_[np.geomspace(1e-4, 0.03, 180), np.linspace(0.03, 1, 420)]
    np.testing.assert_allclose(
        parratt_reflectivity(q, periodic), parratt_reflectivity(q, expanded), rtol=5e-12, atol=2e-14
    )


def test_periodic_bragg_spacing_is_two_pi_over_bilayer_period() -> None:
    from scipy.signal import find_peaks

    q = np.linspace(0.08, 0.9, 16000)
    periodic, _ = _periodic()
    expected = 2 * np.pi / 70
    peaks, properties = find_peaks(
        parratt_reflectivity(q, periodic), prominence=1e-10, distance=int(0.5 * expected / (q[1] - q[0]))
    )
    strongest = np.sort(q[peaks[np.argsort(properties["prominences"])[-8:]]])
    assert np.median(np.diff(strongest)) == pytest.approx(expected, rel=0.03)


def test_high_repeat_periodic_span_matches_independent_collapsed_layer_in_bounded_time() -> None:
    repeats = 100_000
    thickness = np.r_[0.0, np.full(repeats, 2.0), 0.0]
    sld = np.r_[0j, np.full(repeats, 25e-6 + 0.3e-6j), 20e-6 + 0.1e-6j]
    roughness = np.r_[2.5, np.zeros(repeats - 1), 5.0]
    periodic = SlabStack(thickness, sld, roughness, (PeriodicSpan(1, 1, repeats),))
    collapsed = SlabStack([0, 2.0 * repeats, 0], [0j, 25e-6 + 0.3e-6j, 20e-6 + 0.1e-6j], [2.5, 5])
    q = np.linspace(0.01, 0.4, 64)
    started = perf_counter()
    actual = parratt_reflectivity(q, periodic)
    elapsed = perf_counter() - started
    np.testing.assert_allclose(actual, parratt_reflectivity(q, collapsed), rtol=5e-12, atol=2e-14)
    assert elapsed < 2.0


@pytest.mark.parametrize("matrix", [np.zeros((1, 2, 2), complex), np.full((1, 2, 2), np.inf + 0j)])
def test_invalid_mobius_normalization_is_a_hard_failure(matrix: np.ndarray) -> None:
    with pytest.raises(FloatingPointError, match="invalid periodic Parratt transform"):
        normalize_mobius(matrix)
