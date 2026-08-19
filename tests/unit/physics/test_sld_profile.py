from __future__ import annotations

import warnings

import numpy as np
import pytest

from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.physics.sld_profile import sld_depth_profile

MAX_EXPECTED_SLD_PROFILE_POINTS = 1_000_000


def test_zero_roughness_sld_profile_is_piecewise_constant() -> None:
    depth, sld = sld_depth_profile(SlabStack([0, 20, 0], [0j, 10e-6 + 1e-6j, 20e-6 + 2e-6j], [0, 0]), step_a=1)
    assert depth[0] == -10 and depth[-1] == 30
    np.testing.assert_allclose(sld[depth == 0], 5e-6 + 0.5e-6j, rtol=0, atol=1e-20)
    np.testing.assert_allclose(sld[depth == 20], 15e-6 + 1.5e-6j, rtol=0, atol=1e-20)
    assert not depth.flags.writeable and not sld.flags.writeable


def test_zero_thickness_media_are_collapsed_at_sharp_interfaces() -> None:
    stack = SlabStack(
        [0, 0, 20, 0],
        [0j, 10e-6, 20e-6, 30e-6],
        [0, 0, 0],
    )

    depth, sld = sld_depth_profile(stack, step_a=1.0)

    # The zero-thickness medium has no depth interval and must not decide the
    # value at the coincident interface.
    np.testing.assert_allclose(sld[depth == 0], 10e-6, atol=1e-18)
    np.testing.assert_allclose(sld[depth == 20], 25e-6, atol=1e-18)


def test_rough_interface_midpoint_is_average_sld() -> None:
    depth, sld = sld_depth_profile(SlabStack([0, 20, 0], [0j, 10e-6, 20e-6], [2, 3]), step_a=0.5)
    np.testing.assert_allclose(sld[depth == 0], 5e-6, atol=1e-18)
    np.testing.assert_allclose(sld[depth == 20], 15e-6, atol=1e-18)


def test_sharp_interface_midpoint_avoids_finite_extreme_sum_overflow() -> None:
    maximum = np.finfo(float).max
    extreme = complex(maximum, maximum)
    stack = SlabStack([0.0, 0.0], [extreme, extreme], [0.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        depth, sld = sld_depth_profile(stack, step_a=1.0)

    assert sld[depth == 0.0] == extreme
    assert np.all(np.isfinite(sld.real))
    assert np.all(np.isfinite(sld.imag))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_passive_absorbing_sld_profile_keeps_nonnegative_absorption_with_overlapping_roughness() -> None:
    _, sld = sld_depth_profile(
        SlabStack([0, 10, 8, 0], [0j, 20e-6 + 2e-6j, 5e-6 + 0.2e-6j, 30e-6 + 3e-6j], [4, 3, 4]), step_a=0.25
    )
    assert np.all(sld.imag >= 0)


def test_profile_depth_bounds_scale_with_outer_interface_roughness() -> None:
    depth, _ = sld_depth_profile(SlabStack([0, 20, 0], [0j, 10e-6, 20e-6], [3, 4]), step_a=1)
    assert depth[0] == -15
    assert depth[-1] == 40


def test_profile_depth_bounds_include_internal_interface_roughness_tails() -> None:
    stack = SlabStack(
        [0, 20, 20, 0],
        [0j, 10e-6, 15e-6, 20e-6],
        [0, 10, 0],
    )

    depth, _ = sld_depth_profile(stack, step_a=1.0)

    assert depth[0] == -30.0
    assert depth[-1] == 70.0


def test_profile_rejects_oversized_depth_grid_before_allocation() -> None:
    stack = SlabStack(
        [0, float(MAX_EXPECTED_SLD_PROFILE_POINTS), 0],
        [0j, 10e-6, 20e-6],
        [0, 0],
    )

    with pytest.raises(ValueError, match="profile grid.*exceeds"):
        sld_depth_profile(stack, step_a=1.0)


def test_profile_rejects_finite_roughness_that_overflows_tail_extent() -> None:
    stack = SlabStack([0, 20, 0], [0j, 10e-6, 20e-6], [1e308, 0.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="roughness.*finite|depth span"):
            sld_depth_profile(stack, step_a=1.0)

    assert not any(item.category is RuntimeWarning for item in caught)


def test_profile_returns_owned_read_only_arrays() -> None:
    depth, sld = sld_depth_profile(SlabStack([0, 20, 0], [0j, 10e-6, 20e-6], [2, 3]), step_a=0.5)
    assert depth.flags.owndata and sld.flags.owndata
    assert not depth.flags.writeable and not sld.flags.writeable


def test_rough_profile_uses_a_continuous_error_function_transition() -> None:
    depth, sld = sld_depth_profile(SlabStack([0, 0], [0j, 20e-6 + 2e-6j], [2]), step_a=0.5)
    transition = sld[(depth >= -2) & (depth <= 2)]
    assert np.all(np.diff(transition.real) > 0)
    assert np.all(np.diff(transition.imag) > 0)
    np.testing.assert_allclose(sld[depth == 0], 10e-6 + 1e-6j, atol=1e-18)
