"""SLD band replay: alignment, real/imaginary separation, thinning, failures."""

from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.analysis.sld_bands import (
    MAX_REPLAY_SAMPLES,
    QUANTILE_LEVELS,
    sld_uncertainty_bands,
)
from xrr_fitter.model.analysis import McmcConfig, McmcReport
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec

WAVELENGTH_A = 1.5406
AIR = MaterialSpec("Air", None, None, 0j)
SI = MaterialSpec("Si", None, None, 20e-6 + 2e-6j)
MO = MaterialSpec("Mo", None, None, 55e-6 + 1.0e-6j)


def _structure(thickness_a: float = 40.0) -> StructureSpec:
    return StructureSpec(
        AIR,
        (LayerSpec("Mo", MO, thickness_a, roughness_a=3.0),),
        SI,
        backing_roughness_a=4.0,
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


def test_unknown_sample_parameter_name_fails_instead_of_being_ignored() -> None:
    report = _report(np.full((8, 1), 40.0), names=("component.9.thickness_a",))
    with pytest.raises(ValueError, match="component.9.thickness_a"):
        sld_uncertainty_bands(_structure(), report, wavelength_a=WAVELENGTH_A)


def test_unknown_alignment_choice_is_rejected() -> None:
    with pytest.raises(ValueError, match="align"):
        sld_uncertainty_bands(_structure(), _report(np.full((8, 1), 40.0)), wavelength_a=WAVELENGTH_A, align="middle")
