"""SLD band replay: alignment, real/imaginary separation, thinning, failures."""

from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.analysis.sld_bands import (
    MAX_REPLAY_SAMPLES,
    QUANTILE_LEVELS,
    _common_grid,
    sld_uncertainty_bands,
)
from xrr_fitter.model.analysis import McmcConfig, McmcReport
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec
from xrr_fitter.physics.sld_profile import sld_depth_profile
from xrr_fitter.physics.stack import expand_structure

WAVELENGTH_A = 1.5406
AIR = MaterialSpec("Air", None, None, 0j)
SI = MaterialSpec("Si", None, None, 20e-6 + 2e-6j)
MO = MaterialSpec("Mo", None, None, 55e-6 + 1.0e-6j)
VACUUM = MaterialSpec("vacuum", None, None, 0j)


def _structure(thickness_a: float = 40.0) -> StructureSpec:
    return StructureSpec(
        AIR,
        (LayerSpec("Mo", MO, thickness_a, roughness_a=3.0),),
        SI,
        backing_roughness_a=4.0,
    )


def _vacuum_spacer(thickness_a: float = 40.0) -> StructureSpec:
    return StructureSpec(
        AIR,
        (LayerSpec("vacuum spacer", VACUUM, thickness_a),),
        SI,
        backing_roughness_a=0.0,
    )


def _report(values: np.ndarray, names: tuple[str, ...] = ("component.0.thickness_a",)) -> McmcReport:
    samples = np.asarray(values, dtype=float).reshape(-1, len(names))
    walkers = 4
    steps = samples.shape[0]
    return McmcReport(
        config=McmcConfig(walkers=walkers, burn_in=0, production_steps=steps),
        child_seed=7,
        parameter_names=names,
        samples_physical=samples,
        log_probability=np.zeros(steps),
        acceptance_fraction=np.full(walkers, 0.4),
        split_rhat=np.ones(len(names)),
        effective_sample_size=np.full(len(names), float(steps)),
        boundary_hits=(),
    )


def test_identical_samples_collapse_every_quantile_onto_one_profile() -> None:
    report = _report(np.full((8, 1), 40.0))
    bands = sld_uncertainty_bands(_structure(), report, wavelength_a=WAVELENGTH_A)
    assert bands.quantiles == QUANTILE_LEVELS
    for row in range(1, len(QUANTILE_LEVELS)):
        np.testing.assert_allclose(bands.real[row], bands.real[0], rtol=0, atol=0)
        np.testing.assert_allclose(bands.imaginary[row], bands.imaginary[0], rtol=0, atol=0)


def test_zero_variance_median_is_the_direct_profile_on_the_same_backing_axis() -> None:
    structure = _structure()
    step_a = 0.5
    bands = sld_uncertainty_bands(
        structure,
        _report(np.full((8, 1), 40.0)),
        wavelength_a=WAVELENGTH_A,
        step_a=step_a,
        align="backing",
    )
    stack = expand_structure(structure, WAVELENGTH_A)
    depth, profile = sld_depth_profile(stack, step_a=step_a)
    backing_depth = depth - float(np.sum(stack.thickness_a[1:-1]))
    expected = np.interp(bands.depth_a, backing_depth, profile.real)
    expected_imaginary = np.interp(bands.depth_a, backing_depth, profile.imag)

    np.testing.assert_array_equal(bands.real[2], expected)
    np.testing.assert_array_equal(bands.imaginary[2], expected_imaginary)


def test_backing_alignment_removes_a_vacuum_spacer_translation() -> None:
    samples = np.tile((40.0, 60.0), 16)
    bands = sld_uncertainty_bands(
        _vacuum_spacer(),
        _report(samples),
        wavelength_a=WAVELENGTH_A,
        align="backing",
    )

    np.testing.assert_array_equal(bands.real[0], bands.real[-1])
    np.testing.assert_array_equal(bands.imaginary[0], bands.imaginary[-1])


def test_surface_alignment_keeps_a_vacuum_spacer_translation_visible() -> None:
    samples = np.tile((40.0, 60.0), 16)
    bands = sld_uncertainty_bands(
        _vacuum_spacer(),
        _report(samples),
        wavelength_a=WAVELENGTH_A,
        align="surface",
    )

    assert np.any(bands.real[-1] > bands.real[0])
    assert np.any(bands.imaginary[-1] > bands.imaginary[0])


def test_imaginary_sld_samples_widen_only_the_imaginary_envelope() -> None:
    samples = np.linspace(0.25e-6, 3.0e-6, 32)
    bands = sld_uncertainty_bands(
        _structure(),
        _report(samples, names=("component.0.sld_imag_a2",)),
        wavelength_a=WAVELENGTH_A,
    )

    np.testing.assert_array_equal(bands.real[0], bands.real[-1])
    assert np.any(bands.imaginary[-1] > bands.imaginary[0])


def test_wider_thickness_spread_widens_the_band_between_the_outer_quantiles() -> None:
    narrow = sld_uncertainty_bands(_structure(), _report(np.linspace(39.5, 40.5, 12)), wavelength_a=WAVELENGTH_A)
    wide = sld_uncertainty_bands(_structure(), _report(np.linspace(30.0, 50.0, 12)), wavelength_a=WAVELENGTH_A)
    narrow_width = float(np.max(narrow.real[-1] - narrow.real[0]))
    wide_width = float(np.max(wide.real[-1] - wide.real[0]))
    assert wide_width > narrow_width


def test_real_and_imaginary_envelopes_stay_separate_values() -> None:
    bands = sld_uncertainty_bands(_structure(), _report(np.linspace(35.0, 45.0, 10)), wavelength_a=WAVELENGTH_A)
    assert bands.real.shape == bands.imaginary.shape
    assert not np.allclose(bands.real, bands.imaginary)
    # Mo carries roughly fifty times more real SLD than absorption, so a
    # collapsed modulus would erase the absorption envelope entirely.
    assert float(np.max(bands.real)) > 10.0 * float(np.max(bands.imaginary))


def test_backing_and_surface_alignment_place_their_interface_at_the_same_depth() -> None:
    samples = np.linspace(30.0, 50.0, 12)
    backing = sld_uncertainty_bands(_structure(), _report(samples), wavelength_a=WAVELENGTH_A, align="backing")
    surface = sld_uncertainty_bands(_structure(), _report(samples), wavelength_a=WAVELENGTH_A, align="surface")
    assert backing.align_label != surface.align_label
    # Aligning on a different interface must move the depth window, otherwise
    # the offset was never applied to the replayed profiles.
    assert not np.allclose(backing.depth_a[[0, -1]], surface.depth_a[[0, -1]])


def test_thinning_is_deterministic_and_caps_the_replay_count() -> None:
    total = MAX_REPLAY_SAMPLES + 40
    samples = np.linspace(30.0, 50.0, total)
    first = sld_uncertainty_bands(_structure(), _report(samples), wavelength_a=WAVELENGTH_A)
    second = sld_uncertainty_bands(_structure(), _report(samples), wavelength_a=WAVELENGTH_A)
    assert first.sample_count == MAX_REPLAY_SAMPLES
    assert first.total_samples == total
    np.testing.assert_array_equal(first.real, second.real)
    np.testing.assert_array_equal(first.depth_a, second.depth_a)


def test_instrument_parameters_in_the_report_are_ignored_by_the_replay() -> None:
    # A real fit samples instrument.scale and friends alongside the structure
    # coordinates; they carry no structural meaning, so the replay must skip
    # them instead of refusing the whole report.
    thickness = np.linspace(30.0, 50.0, 12)
    with_instrument = np.column_stack([thickness, np.linspace(0.9, 1.1, 12)])
    mixed = sld_uncertainty_bands(
        _structure(),
        _report(with_instrument, names=("component.0.thickness_a", "instrument.scale")),
        wavelength_a=WAVELENGTH_A,
    )
    structure_only = sld_uncertainty_bands(_structure(), _report(thickness), wavelength_a=WAVELENGTH_A)
    np.testing.assert_array_equal(mixed.real, structure_only.real)
    np.testing.assert_array_equal(mixed.depth_a, structure_only.depth_a)


def test_unknown_sample_parameter_name_fails_instead_of_being_ignored() -> None:
    report = _report(np.full((8, 1), 40.0), names=("component.9.thickness_a",))
    with pytest.raises(ValueError, match="component.9.thickness_a"):
        sld_uncertainty_bands(_structure(), report, wavelength_a=WAVELENGTH_A)


def test_unknown_alignment_choice_is_rejected() -> None:
    with pytest.raises(ValueError, match="align"):
        sld_uncertainty_bands(_structure(), _report(np.full((8, 1), 40.0)), wavelength_a=WAVELENGTH_A, align="middle")


@pytest.mark.parametrize("step_a", (0.0, -0.5, float("nan"), float("inf"), float("-inf")))
def test_nonpositive_or_nonfinite_profile_step_is_rejected_at_the_api(step_a: float) -> None:
    with pytest.raises(ValueError, match="step_a must be finite and positive"):
        sld_uncertainty_bands(
            _structure(),
            _report(np.full((8, 1), 40.0)),
            wavelength_a=WAVELENGTH_A,
            step_a=step_a,
        )


@pytest.mark.parametrize("max_samples", (0, -1, 1.5, True))
def test_nonpositive_or_noninteger_replay_limit_is_rejected_at_the_api(max_samples: object) -> None:
    with pytest.raises(ValueError, match="max_samples must be a positive integer"):
        sld_uncertainty_bands(
            _structure(),
            _report(np.full((8, 1), 40.0)),
            wavelength_a=WAVELENGTH_A,
            max_samples=max_samples,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("wavelength_a", (0.0, -1.0, float("nan"), float("inf"), float("-inf")))
def test_nonpositive_or_nonfinite_wavelength_is_rejected_at_the_api(wavelength_a: float) -> None:
    with pytest.raises(ValueError, match="wavelength_a must be finite and positive"):
        sld_uncertainty_bands(
            _structure(),
            _report(np.full((8, 1), 40.0)),
            wavelength_a=wavelength_a,
        )


def test_exactly_five_percent_failed_replays_are_accepted_and_reported() -> None:
    roughness = np.full(20, 3.0)
    roughness[0] = 25.0

    bands = sld_uncertainty_bands(
        _structure(),
        _report(roughness, names=("component.0.roughness_a",)),
        wavelength_a=WAVELENGTH_A,
    )

    assert bands.sample_count == 19
    assert bands.failure_rate == pytest.approx(0.05)


def test_more_than_five_percent_failed_replays_are_rejected() -> None:
    roughness = np.full(20, 3.0)
    roughness[:2] = 25.0

    with pytest.raises(ValueError, match=r"failure rate 0\.100 exceeds 0\.050"):
        sld_uncertainty_bands(
            _structure(),
            _report(roughness, names=("component.0.roughness_a",)),
            wavelength_a=WAVELENGTH_A,
        )


def test_aligned_profiles_with_an_empty_depth_intersection_are_rejected() -> None:
    profiles = (
        (np.array([0.0, 1.0]), np.zeros(2, dtype=complex)),
        (np.array([2.0, 3.0]), np.zeros(2, dtype=complex)),
    )

    with pytest.raises(ValueError, match="share no overlapping depth range"):
        _common_grid(profiles, step_a=0.5)
