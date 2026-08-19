"""SLD band replay: alignment, real/imaginary separation, thinning, failures."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from tests.support.drift_cases import drift_structure, drift_values
from tests.support.model_cases import prepared_data

import xrr_fitter.analysis.sld_bands as sld_bands
from xrr_fitter.analysis.sld_bands import (
    MAX_REPLAY_SAMPLES,
    QUANTILE_LEVELS,
    _common_grid,
    _interpolated,
    sld_uncertainty_bands,
)
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import McmcConfig, McmcReport
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    DriftSpec,
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.sld_profile import sld_depth_profile
from xrr_fitter.physics.stack import expand_structure, rebuild_structure

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


def _periodic_structure(top_roughness_a: float | None = None) -> StructureSpec:
    return StructureSpec(
        AIR,
        (
            PeriodicBlock(
                "Mo/Si",
                (
                    LayerSpec("Mo", MO, 24.0, roughness_a=3.0),
                    LayerSpec("Si", SI, 32.0, roughness_a=2.0),
                ),
                repeats=2,
                top_roughness_a=top_roughness_a,
            ),
        ),
        SI,
        backing_roughness_a=4.0,
    )


def _roughness_drift_with_explicit_top() -> StructureSpec:
    block = PeriodicBlock(
        "p",
        (
            LayerSpec("a", MO, 20.0, roughness_a=2.0),
            LayerSpec("b", SI, 500.0, roughness_a=3.0),
        ),
        repeats=3,
        top_roughness_a=1.0,
        drift=DriftSpec(kind="linear", target="roughness", amount=0.1),
    )
    return StructureSpec(AIR, (block,), SI, backing_roughness_a=3.0)


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


def _compiled_drift_report(structure: StructureSpec, samples: int = 8) -> tuple[McmcReport, dict[str, float]]:
    problem = compile_fit_problem(
        prepared_data(size=40),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(707), scale_prior_enabled=False),
    )
    values = {definition.name: definition.initial for definition in problem.parameter_definitions}
    values.update(drift_values(structure))
    names = tuple(variable.name for variable in problem.variables)
    derived_names = tuple(rule.target.parameter_name for rule in problem.constraint_rules)
    row = np.asarray([values[name] for name in names], dtype=float)
    derived_row = np.asarray([values[name] for name in derived_names], dtype=float)
    return (
        McmcReport(
            config=McmcConfig(walkers=4, burn_in=0, production_steps=samples),
            child_seed=71,
            parameter_names=names,
            samples_physical=np.repeat(row[None, :], samples, axis=0),
            log_probability=np.zeros(samples),
            acceptance_fraction=np.full(4, 0.4),
            split_rhat=np.ones(len(names)),
            effective_sample_size=np.full(len(names), float(samples)),
            boundary_hits=(),
            derived_parameter_names=derived_names,
            derived_samples_physical=np.repeat(derived_row[None, :], samples, axis=0),
        ),
        values,
    )


def _assert_median_matches_rebuilt_profile(
    structure: StructureSpec,
    report: McmcReport,
    values: dict[str, float],
) -> None:
    step_a = 0.5
    bands = sld_uncertainty_bands(
        structure,
        report,
        wavelength_a=WAVELENGTH_A,
        step_a=step_a,
    )
    stack = expand_structure(rebuild_structure(structure, values), WAVELENGTH_A)
    depth, profile = sld_depth_profile(stack, step_a=step_a)
    backing_depth = depth - float(np.sum(stack.thickness_a[1:-1]))

    assert bands.sample_count == report.samples_physical.shape[0]
    np.testing.assert_array_equal(bands.real[0], bands.real[-1])
    np.testing.assert_array_equal(bands.imaginary[0], bands.imaginary[-1])
    np.testing.assert_allclose(
        bands.real[2],
        np.interp(bands.depth_a, backing_depth, profile.real),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        bands.imaginary[2],
        np.interp(bands.depth_a, backing_depth, profile.imag),
        rtol=0.0,
        atol=0.0,
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


def test_gradient_replay_uses_the_topology_recorded_by_the_fitted_problem() -> None:
    structure = StructureSpec(
        AIR,
        (
            GradientLayerSpec(
                "gradient",
                upper_sld_a2=10e-6 + 0.5e-6j,
                lower_sld_a2=50e-6 + 2.0e-6j,
                thickness_a=20.0,
                roughness_a=0.0,
                microslab_max_a=10.0,
            ),
        ),
        SI,
        backing_roughness_a=0.0,
    )
    count = 8
    report = replace(
        _report(np.full(8, 20.0)),
        gradient_slab_counts=(("component.0", count),),
    )
    step_a = 0.5

    bands = sld_uncertainty_bands(
        structure,
        report,
        wavelength_a=WAVELENGTH_A,
        step_a=step_a,
    )
    stack = expand_structure(
        structure,
        WAVELENGTH_A,
        {"component.0": count},
    )
    depth, profile = sld_depth_profile(stack, step_a=step_a)
    backing_depth = depth - float(np.sum(stack.thickness_a[1:-1]))

    np.testing.assert_array_equal(
        bands.real[2],
        np.interp(bands.depth_a, backing_depth, profile.real),
    )
    np.testing.assert_array_equal(
        bands.imaginary[2],
        np.interp(bands.depth_a, backing_depth, profile.imag),
    )


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


def test_quantiles_interpolate_finite_opposite_extreme_sld_samples_stably() -> None:
    maximum = np.finfo(float).max
    material = MaterialSpec("extreme", None, None, 0j)
    structure = StructureSpec(
        AIR,
        (LayerSpec("extreme", material, 20.0, roughness_a=0.0),),
        AIR,
        backing_roughness_a=0.0,
    )
    report = _report(
        np.array([-maximum, maximum]),
        names=("component.0.sld_real_a2",),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bands = sld_uncertainty_bands(
            structure,
            report,
            wavelength_a=WAVELENGTH_A,
            step_a=1.0,
        )

    interior = int(np.flatnonzero(bands.depth_a == -10.0)[0])
    expected = maximum * np.array((-0.95, -0.68, 0.0, 0.68, 0.95))
    assert np.all(np.isfinite(bands.real))
    assert np.all(np.diff(bands.real, axis=0) >= 0.0)
    np.testing.assert_allclose(bands.real[:, interior], expected, rtol=1e-15)
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


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


def test_derived_mcmc_samples_are_used_for_sld_replay() -> None:
    derived = _report(
        np.ones(12),
        names=("instrument.scale",),
    )
    derived = replace(
        derived,
        derived_parameter_names=("component.0.thickness_a",),
        derived_samples_physical=np.linspace(30.0, 50.0, 12).reshape(-1, 1),
    )
    direct = _report(np.linspace(30.0, 50.0, 12))

    from_derived = sld_uncertainty_bands(_structure(), derived, wavelength_a=WAVELENGTH_A)
    from_direct = sld_uncertainty_bands(_structure(), direct, wavelength_a=WAVELENGTH_A)

    np.testing.assert_array_equal(from_derived.depth_a, from_direct.depth_a)
    np.testing.assert_array_equal(from_derived.real, from_direct.real)
    np.testing.assert_array_equal(from_derived.imaginary, from_direct.imaginary)


def test_locked_structure_values_are_used_for_sld_replay() -> None:
    report = replace(
        _report(np.ones(12), names=("component.0.density_scale",)),
        fixed_parameter_values=(("component.0.thickness_a", 70.0),),
    )
    structure = _structure()
    expected_structure = replace(
        structure,
        components=(replace(structure.components[0], thickness_a=70.0),),
    )
    expected = sld_uncertainty_bands(
        expected_structure,
        _report(np.ones(12), names=("component.0.density_scale",)),
        wavelength_a=WAVELENGTH_A,
    )

    actual = sld_uncertainty_bands(structure, report, wavelength_a=WAVELENGTH_A)

    np.testing.assert_array_equal(actual.depth_a, expected.depth_a)
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imaginary, expected.imaginary)


def test_inherited_periodic_top_roughness_fixed_value_does_not_break_sld_replay() -> None:
    base_report = _report(np.ones(12), names=("component.0.layer.0.density_scale",))
    report = replace(
        base_report,
        fixed_parameter_values=(("component.0.top_roughness_a", 3.0),),
    )
    structure = _periodic_structure(top_roughness_a=None)

    expected = sld_uncertainty_bands(structure, base_report, wavelength_a=WAVELENGTH_A)
    actual = sld_uncertainty_bands(structure, report, wavelength_a=WAVELENGTH_A)

    np.testing.assert_array_equal(actual.depth_a, expected.depth_a)
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imaginary, expected.imaginary)

    explicit_structure = _periodic_structure(top_roughness_a=3.0)
    explicit_report = replace(
        base_report,
        fixed_parameter_values=(("component.0.top_roughness_a", 3.0),),
    )
    explicit_expected = sld_uncertainty_bands(
        explicit_structure,
        base_report,
        wavelength_a=WAVELENGTH_A,
    )
    explicit_actual = sld_uncertainty_bands(
        explicit_structure,
        explicit_report,
        wavelength_a=WAVELENGTH_A,
    )

    np.testing.assert_array_equal(explicit_actual.depth_a, explicit_expected.depth_a)
    np.testing.assert_array_equal(explicit_actual.real, explicit_expected.real)
    np.testing.assert_array_equal(explicit_actual.imaginary, explicit_expected.imaginary)


def test_inherited_periodic_top_fixed_value_is_ignored_during_drift_sld_replay() -> None:
    structure = drift_structure()
    report, _values = _compiled_drift_report(structure)
    fixed = replace(
        report,
        fixed_parameter_values=(("component.0.top_roughness_a", 3.0),),
    )

    expected = sld_uncertainty_bands(structure, report, wavelength_a=WAVELENGTH_A)
    actual = sld_uncertainty_bands(structure, fixed, wavelength_a=WAVELENGTH_A)

    np.testing.assert_array_equal(actual.depth_a, expected.depth_a)
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imaginary, expected.imaginary)


@pytest.mark.parametrize(
    "structure",
    (
        pytest.param(drift_structure(), id="thickness-drift-inherited-top"),
        pytest.param(_roughness_drift_with_explicit_top(), id="roughness-drift-explicit-top"),
    ),
)
def test_drift_mcmc_reports_replay_sampled_scale_and_derived_repeat_coordinates(
    structure: StructureSpec,
) -> None:
    report, values = _compiled_drift_report(structure)

    _assert_median_matches_rebuilt_profile(structure, report, values)


def test_unknown_sample_parameter_name_fails_instead_of_being_ignored() -> None:
    report = _report(np.full((8, 1), 40.0), names=("component.9.thickness_a",))
    with pytest.raises(ValueError, match="component.9.thickness_a"):
        sld_uncertainty_bands(_structure(), report, wavelength_a=WAVELENGTH_A)


def test_unknown_fixed_parameter_name_fails_instead_of_being_ignored() -> None:
    report = replace(
        _report(np.full((8, 1), 40.0)),
        fixed_parameter_values=(("component.9.roughness_a", 3.0),),
    )
    with pytest.raises(ValueError, match="component.9.roughness_a"):
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


def test_interpolation_rejects_a_matrix_over_the_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = tuple((np.arange(2.0), np.zeros(2, dtype=complex)) for _ in range(2))
    grid = np.arange(6.0)
    monkeypatch.setattr(sld_bands, "MAX_INTERPOLATION_CELLS", 10, raising=False)

    with pytest.raises(ValueError, match="interpolation matrix exceeds"):
        _interpolated(profiles, grid)


def test_replay_collection_rejects_aggregate_profile_points_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = (
        (np.arange(2.0), np.zeros(2, dtype=complex)),
        (np.arange(2.0), np.zeros(2, dtype=complex)),
    )
    monkeypatch.setattr(sld_bands, "MAX_REPLAY_PROFILE_POINTS", 3, raising=False)

    assert hasattr(sld_bands, "_collect_profiles")
    with pytest.raises(ValueError, match="replayed SLD profiles exceed"):
        sld_bands._collect_profiles(iter(profiles))
