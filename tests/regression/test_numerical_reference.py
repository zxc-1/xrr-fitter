from __future__ import annotations

import numpy as np

from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.physics.parratt import parratt_reflectivity


def test_single_layer_matches_pinned_refnx() -> None:
    from refnx.reflect import abeles

    q = np.r_[np.geomspace(1e-4, 0.03, 200), np.linspace(0.03, 1, 500)]
    layers = np.array([[0, 0, 0, 0], [137, 35, 0.8, 4], [0, 20, 0.2, 6]], float)
    expected = abeles(q, layers)
    actual = parratt_reflectivity(q, SlabStack(layers[:, 0], (layers[:, 1] + 1j * layers[:, 2]) * 1e-6, layers[1:, 3]))
    tolerance = np.where(expected >= 1e-12, 1e-10 + 5e-7 * expected, 1e-12)
    assert np.all(np.abs(actual - expected) <= tolerance)


def test_randomized_air_fronting_stacks_match_pinned_refnx() -> None:
    from refnx.reflect import abeles as pinned_abeles
    from refnx.reflect.reflect_model import get_reflect_backend

    pinned_parratt = get_reflect_backend("py_parratt")
    generator = np.random.default_rng(20260715)
    q = np.r_[np.geomspace(1e-4, 0.03, 200), np.linspace(0.03, 1, 500)[1:]]
    abeles_point_count = 0
    fallback_point_count = 0
    for structure_index in range(500):
        finite_layer_count = int(generator.integers(0, 21))
        finite_thickness = generator.uniform(2.0, 5000.0, finite_layer_count)
        thickness = np.r_[0.0, finite_thickness, 0.0]
        media_sld = generator.uniform(0.0, 150e-6, finite_layer_count + 1) + 1j * generator.uniform(
            0.0, 20e-6, finite_layer_count + 1
        )
        sld = np.r_[0j, media_sld]
        if finite_layer_count == 0:
            roughness_limits = np.array([50.0])
        else:
            effective_thickness = np.empty(finite_layer_count + 1)
            effective_thickness[0] = finite_thickness[0]
            effective_thickness[-1] = finite_thickness[-1]
            if finite_layer_count > 1:
                effective_thickness[1:-1] = np.minimum(finite_thickness[:-1], finite_thickness[1:])
            roughness_limits = np.minimum(50.0, 0.30 * effective_thickness)
        roughness = generator.uniform(0.0, roughness_limits)
        layers = np.column_stack((thickness, sld.real * 1e6, sld.imag * 1e6, np.r_[0.0, roughness]))
        expected_abeles = pinned_abeles(q, layers, threads=1)
        expected_parratt = pinned_parratt(q, layers, threads=1)
        assert np.all(np.isfinite(expected_parratt)), (
            f"pinned refnx Parratt nonfinite at structure_index={structure_index}"
        )
        oracle_tolerance = np.where(expected_parratt >= 1e-12, 1e-10 + 5e-7 * expected_parratt, 1e-12)
        use_abeles = np.isfinite(expected_abeles) & (np.abs(expected_abeles - expected_parratt) <= oracle_tolerance)
        expected = np.where(use_abeles, expected_abeles, expected_parratt)
        abeles_point_count += int(np.count_nonzero(use_abeles))
        fallback_point_count += int(np.count_nonzero(~use_abeles))
        actual = parratt_reflectivity(q, SlabStack(thickness, sld, roughness))
        tolerance = np.where(expected >= 1e-12, 1e-10 + 5e-7 * expected, 1e-12)
        absolute_error = np.abs(actual - expected)
        assert np.all(absolute_error <= tolerance), (
            f"structure_index={structure_index}, finite_layers={finite_layer_count}, "
            f"max_abs_error={np.max(absolute_error):.17g}"
        )
    assert abeles_point_count > 0
    assert fallback_point_count > 0
