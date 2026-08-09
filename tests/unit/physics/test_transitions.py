from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pytest

from xrr_fitter.model.structure import InterfaceTransition, TransitionBranch
from xrr_fitter.physics.transitions import (
    ERF_HALF_WIDTH_SIGMAS,
    TRANSITION_KINDS,
    transition_fractions,
    transition_profile,
    transition_slab_count,
)


@pytest.mark.parametrize("kind", sorted(TRANSITION_KINDS))
def test_every_kind_maps_zero_to_zero_and_one_to_one(kind: str) -> None:
    values = transition_profile(kind, np.array([0.0, 1.0]))
    assert values[0] == 0.0
    assert values[1] == 1.0


@pytest.mark.parametrize("kind", sorted(TRANSITION_KINDS))
def test_every_kind_is_monotone_nondecreasing(kind: str) -> None:
    t = np.linspace(0.0, 1.0, 257)
    values = transition_profile(kind, t)
    assert np.all(np.diff(values) >= 0.0)


def test_step_kind_is_a_sharp_half_width_jump() -> None:
    values = transition_profile("step", np.array([0.0, 0.49, 0.5, 1.0]))
    np.testing.assert_array_equal(values, np.array([0.0, 0.0, 1.0, 1.0]))


def test_linear_kind_is_the_identity() -> None:
    t = np.linspace(0.0, 1.0, 129)
    np.testing.assert_array_equal(transition_profile("linear", t), t)


def test_erf_kind_matches_the_documented_half_width_constant() -> None:
    c = ERF_HALF_WIDTH_SIGMAS
    t = 0.25
    expected = 0.5 * (1.0 + erf(c * (2.0 * t - 1.0) / sqrt(2.0)) / erf(c / sqrt(2.0)))
    value = transition_profile("erf", np.array([t]))
    assert value[0] == pytest.approx(expected, rel=0, abs=1e-15)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="gaussian"):
        transition_profile("gaussian", np.linspace(0.0, 1.0, 5))


def test_slab_count_is_at_least_one_and_ceils() -> None:
    assert transition_slab_count(10.0, 3.0) == 4
    assert transition_slab_count(2.0, 2.0) == 1
    assert transition_slab_count(1.0, 4.0) == 1


def test_kind_sets_agree_across_model_and_physics() -> None:
    from xrr_fitter.model.structure import TRANSITION_KINDS as model_kinds
    from xrr_fitter.physics.transitions import TRANSITION_KINDS as physics_kinds

    assert model_kinds == physics_kinds


@pytest.mark.parametrize(
    ("width_a", "microslab_max_a"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
    ],
)
def test_slab_count_rejects_nonpositive_inputs(width_a: float, microslab_max_a: float) -> None:
    with pytest.raises(ValueError):
        transition_slab_count(width_a, microslab_max_a)


def test_transition_fractions_are_slab_centered_and_monotone() -> None:
    transition = InterfaceTransition(
        branches=(TransitionBranch(kind="linear", weight=1.0, thickness_a=10.0),),
        microslab_max_a=2.5,
    )
    fractions = transition_fractions(transition, 4)
    np.testing.assert_allclose(fractions, [0.125, 0.375, 0.625, 0.875])
    assert np.all(np.diff(fractions) > 0.0)


def test_transition_fractions_weight_branches_of_different_widths() -> None:
    transition = InterfaceTransition(
        branches=(
            TransitionBranch(kind="linear", weight=0.5, thickness_a=10.0),
            TransitionBranch(kind="linear", weight=0.5, thickness_a=5.0),
        ),
        microslab_max_a=2.5,
    )
    fractions = transition_fractions(transition, 4)
    np.testing.assert_allclose(fractions, [0.1875, 0.5625, 0.8125, 0.9375])


def test_transition_fractions_end_below_one_for_interior_centers() -> None:
    transition = InterfaceTransition(
        branches=(TransitionBranch(kind="erf", weight=1.0, thickness_a=8.0),),
        microslab_max_a=1.0,
    )
    fractions = transition_fractions(transition, 8)
    assert fractions[0] > 0.0
    assert fractions[-1] < 1.0
    assert np.all(np.diff(fractions) >= 0.0)
