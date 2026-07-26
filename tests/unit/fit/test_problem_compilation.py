"""Fit-problem compilation contracts.

Coverage keeps parameter layout, stage locks, geometry-dependent bounds, and
analytic evaluation tied to one immutable compiled snapshot.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import encode_physical_vector, unit_to_physical, values_by_name
from xrr_fitter.fit.candidates import bounded_perturbations
from xrr_fitter.fit.initialization import estimate_initial_candidates
from xrr_fitter.fit.objective import evaluate_jacobian, evaluate_vector
from xrr_fitter.fit.problem import (
    compile_fit_problem,
    compile_fixed_parameter_problem,
    compile_stage_problem,
)
from xrr_fitter.model.data import BeamSpec, DataColumnMapping
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec, resolution_to_sigma_q
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)


def _config(seed: int = 11) -> FitConfig:
    return replace(FitConfig.fast(master_seed=seed), scale_prior_enabled=False)


def _problem(
    *,
    data=None,
    structure=None,
    instrument=None,
    settings: tuple[ParameterSetting, ...] = (),
    seed: int = 11,
):
    return compile_fit_problem(
        data or prepared_data(size=72),
        structure or simple_structure(),
        instrument or InstrumentSpec(footprint_mode="fit"),
        _config(seed),
        settings,
    )


def _initial_values(problem) -> dict[str, float]:
    return {
        definition.name: definition.initial for definition in problem.parameter_definitions
    }


def _richardson(problem, unit: np.ndarray) -> np.ndarray:
    def residual(value: np.ndarray) -> np.ndarray:
        result = evaluate_vector(problem, value)
        assert result.valid
        return result.fit_log_residuals_decades

    output = np.empty((np.count_nonzero(problem.data.fit_mask), unit.size))
    step = 5e-5
    for index in range(unit.size):
        coarse_plus = unit.copy()
        coarse_minus = unit.copy()
        fine_plus = unit.copy()
        fine_minus = unit.copy()
        coarse_plus[index] += step
        coarse_minus[index] -= step
        fine_plus[index] += step / 2.0
        fine_minus[index] -= step / 2.0
        coarse = (residual(coarse_plus) - residual(coarse_minus)) / (2.0 * step)
        fine = (residual(fine_plus) - residual(fine_minus)) / step
        output[:, index] = (4.0 * fine - coarse) / 3.0
    return output


def _periodic_structure() -> StructureSpec:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="a", thickness_a=22.0),
            replace(film, name="b", thickness_a=38.0),
        ),
        repeats=5,
        top_roughness_a=1.0,
    )
    return StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0)


def _theta_resolution_problem():
    return _problem(
        instrument=InstrumentSpec(footprint_mode="none", resolution_domain="theta"),
        settings=(
            ParameterSetting(
                "instrument.sigma_theta_deg", 0.01, 0.0, 0.05, locked=False
            ),
        ),
        seed=18,
    )


def _periodic_jacobian_problem():
    return _problem(
        structure=_periodic_structure(),
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=19,
    )


def _mixed_kalpha_problem():
    data = prepared_data(size=72, beam=BeamSpec(kind="mixed_kalpha"))
    return _problem(
        data=data,
        instrument=InstrumentSpec(footprint_mode="none"),
        seed=24,
    )


def _linear_background_jacobian_problem():
    return _problem(
        instrument=InstrumentSpec(footprint_mode="none", background_kind="linear"),
        settings=(
            ParameterSetting(
                "instrument.linear_background_per_a_inv", 0.0, -0.01, 0.01
            ),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=20,
    )


def _powerlaw_background_jacobian_problem():
    return _problem(
        instrument=InstrumentSpec(footprint_mode="none", background_kind="powerlaw"),
        settings=(
            ParameterSetting(
                "instrument.powerlaw_background_amplitude", 1e-7, 0.0, 1e-6
            ),
            ParameterSetting(
                "instrument.powerlaw_background_exponent", 2.5, 1.0, 4.0
            ),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=21,
    )


def _direct_sld_jacobian_problem():
    base = simple_structure()
    direct = MaterialSpec("direct film", None, None, 60e-6 + 2e-6j)
    structure = StructureSpec(
        base.fronting,
        (LayerSpec("direct film", direct, 140.0, roughness_a=3.0),),
        base.backing,
        backing_roughness_a=2.0,
    )
    return _problem(
        structure=structure,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("component.0.sld_real_a2", 60e-6, 30e-6, 90e-6),
            ParameterSetting("component.0.sld_imag_a2", 2e-6, 0.5e-6, 4e-6),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=22,
    )


def _gradient_jacobian_problem():
    base = simple_structure()
    structure = StructureSpec(
        base.fronting,
        (
            GradientLayerSpec(
                "gradient",
                upper_sld_a2=25e-6 + 0.5e-6j,
                lower_sld_a2=55e-6 + 2e-6j,
                thickness_a=100.0,
                roughness_a=2.0,
                microslab_max_a=20.0,
            ),
        ),
        base.backing,
        backing_roughness_a=2.0,
    )
    return _problem(
        structure=structure,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("component.0.upper_sld_real_a2", 25e-6, 10e-6, 40e-6),
            ParameterSetting("component.0.upper_sld_imag_a2", 0.5e-6, 0.1e-6, 2e-6),
            ParameterSetting("component.0.lower_sld_real_a2", 55e-6, 40e-6, 70e-6),
            ParameterSetting("component.0.lower_sld_imag_a2", 2e-6, 0.5e-6, 4e-6),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=23,
    )


def _angular_point_resolution_jacobian_problem():
    data = prepared_data(size=72)
    raw = np.full(data.qz_a_inv.size, 0.01)
    sigma = resolution_to_sigma_q(
        data.two_theta_deg,
        raw,
        "sigma_two_theta_deg",
        data.beam.effective_wavelength_a,
        data.import_angle_offset_deg,
    )
    data = replace(
        data,
        resolution_raw=raw,
        sigma_q_a_inv=sigma,
        column_mapping=DataColumnMapping(
            two_theta=0,
            intensity=1,
            resolution=2,
            resolution_kind="sigma_two_theta_deg",
        ),
    )
    return _problem(
        data=data,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("instrument.absolute_sigma_a_inv", 0.001, 0.0, 0.005),
        ),
        seed=25,
    )


def _direct_q_point_resolution_jacobian_problem():
    data = prepared_data(size=72)
    sigma = np.linspace(1e-4, 2e-4, data.qz_a_inv.size)
    data = replace(
        data,
        resolution_raw=sigma,
        sigma_q_a_inv=sigma,
        column_mapping=DataColumnMapping(
            two_theta=0,
            intensity=1,
            resolution=2,
            resolution_kind="sigma_q_a_inv",
        ),
    )
    return _problem(
        data=data,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("instrument.absolute_sigma_a_inv", 0.001, 0.0, 0.005),
        ),
        seed=26,
    )


def test_bounded_perturbations_are_seeded_counted_and_clipped() -> None:
    center = np.array([0.0, 0.5, 1.0])

    first = bounded_perturbations(center, 7, seed=42, sigma=0.3)
    second = bounded_perturbations(center, 7, seed=42, sigma=0.3)

    assert len(first) == 7
    assert all(not vector.flags.writeable for vector in first)
    assert all(np.all((0.0 <= vector) & (vector <= 1.0)) for vector in first)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_bounded_perturbations_return_no_starts_for_empty_stage() -> None:
    assert bounded_perturbations(np.empty(0), 4, seed=7) == ()


def test_stage_e_incumbent_perturbations_use_two_thousandths_unit_sigma() -> None:
    center = np.full(3, 0.5)
    expected = np.clip(
        center + np.random.default_rng(91).normal(0.0, 0.002, size=(5, 3)),
        0.0,
        1.0,
    )

    actual = bounded_perturbations(center, 5, seed=91, sigma=0.002)

    np.testing.assert_array_equal(np.stack(actual), expected)


def test_compile_rejects_nonfinite_objective_threshold() -> None:
    config = _config(36)
    object.__setattr__(config, "c_decades", np.nan)

    with pytest.raises(ValueError, match="fit configuration|c_decades"):
        compile_fit_problem(
            prepared_data(size=72),
            simple_structure(),
            InstrumentSpec(footprint_mode="fit"),
            config,
        )


def test_compile_stage_problem_releases_the_exact_stage_parameter_groups() -> None:
    problem = _problem()
    values = _initial_values(problem)
    stage_names = {
        stage: {coordinate.name for coordinate in compile_stage_problem(problem, stage, values).variables}
        for stage in ("B", "C", "D", "E")
    }

    assert stage_names["B"] == {
        "component.0.thickness_a",
        "instrument.angle_offset_deg",
        "instrument.scale",
        "instrument.background",
        "instrument.footprint_spill_angle_deg",
    }
    assert stage_names["C"] == {"component.0.density_scale"}
    assert stage_names["D"] == {
        "component.0.roughness_a",
        "backing.roughness_a",
        "instrument.relative_sigma",
    }
    assert stage_names["E"] == {coordinate.name for coordinate in problem.variables}


def test_compile_stage_problem_respects_theta_resolution_and_disabled_footprint() -> None:
    problem = _theta_resolution_problem()
    values = _initial_values(problem)

    stage_b = compile_stage_problem(problem, "B", values)
    stage_d = compile_stage_problem(problem, "D", values)

    names_b = {coordinate.name for coordinate in stage_b.variables}
    names_d = {coordinate.name for coordinate in stage_d.variables}
    assert "instrument.footprint_spill_angle_deg" not in names_b
    assert "instrument.relative_sigma" not in names_d
    assert "instrument.sigma_theta_deg" in names_d


@pytest.mark.parametrize(
    ("background_kind", "active_names"),
    [
        pytest.param("linear", {"instrument.linear_background_per_a_inv"}, id="linear-active_names0"),
        pytest.param("powerlaw", {"instrument.powerlaw_background_amplitude"}, id="powerlaw-active_names1"),
    ],
)
def test_compile_stage_problem_releases_only_active_background_modes(
    background_kind: str,
    active_names: set[str],
) -> None:
    problem = _problem(
        instrument=InstrumentSpec(footprint_mode="none", background_kind=background_kind)
    )

    stage_b = compile_stage_problem(problem, "B", _initial_values(problem))

    names = {coordinate.name for coordinate in stage_b.variables}
    assert active_names <= names
    inactive = {
        "instrument.linear_background_per_a_inv",
        "instrument.powerlaw_background_amplitude",
    } - active_names
    assert names.isdisjoint(inactive)


def test_compiled_and_evaluated_arrays_are_read_only() -> None:
    problem = _periodic_jacobian_problem()
    unit = encode_physical_vector(problem, {})
    evaluation = evaluate_vector(problem, unit)

    arrays = (
        problem.region_labels,
        problem.weights,
        unit,
        evaluation.qz_a_inv,
        evaluation.model_normalized,
        evaluation.fit_log_residuals_decades,
        evaluation.fit_weighted_residuals,
    )
    assert all(not array.flags.writeable for array in arrays)


def test_declared_two_angstrom_layer_remains_inside_compiled_bounds() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    structure = replace(
        base,
        components=(replace(film, thickness_a=2.0, roughness_a=0.2),),
        backing_roughness_a=0.2,
    )

    problem = _problem(structure=structure)

    definition = next(
        item for item in problem.parameter_definitions if item.name == "component.0.thickness_a"
    )
    assert definition.lower == 2.0
    encode_physical_vector(problem, _initial_values(problem))


@pytest.mark.parametrize(
    "problem_factory",
    [
        pytest.param(_linear_background_jacobian_problem, id="linear_background_jacobian_problem"),
        pytest.param(_powerlaw_background_jacobian_problem, id="powerlaw_background_jacobian_problem"),
        pytest.param(_direct_sld_jacobian_problem, id="direct_sld_jacobian_problem"),
        pytest.param(_gradient_jacobian_problem, id="gradient_jacobian_problem"),
    ],
)
def test_evaluate_jacobian_covers_expert_background_and_sld_paths(problem_factory) -> None:
    problem = problem_factory()
    unit = np.full(len(problem.variables), 0.46)

    analytic = evaluate_jacobian(problem, unit)
    reference = _richardson(problem, unit)

    np.testing.assert_allclose(analytic, reference, rtol=1e-6, atol=5e-11)


@pytest.mark.parametrize(
    ("problem_factory", "unit_value"),
    [
        pytest.param(_periodic_jacobian_problem, 0.47, id="periodic_jacobian_problem-0.47"),
        pytest.param(_mixed_kalpha_problem, 0.45, id="mixed_kalpha_problem-0.45"),
        pytest.param(_theta_resolution_problem, 0.45, id="theta_resolution_problem-0.45"),
        pytest.param(
            _angular_point_resolution_jacobian_problem,
            0.45,
            id="angular_point_resolution_jacobian_problem-0.45",
        ),
        pytest.param(
            _direct_q_point_resolution_jacobian_problem,
            0.45,
            id="direct_q_point_resolution_jacobian_problem-0.45",
        ),
    ],
)
def test_evaluate_jacobian_covers_periodic_mixed_and_theta_paths(
    problem_factory,
    unit_value: float,
) -> None:
    problem = problem_factory()
    unit = np.full(len(problem.variables), unit_value)

    analytic = evaluate_jacobian(problem, unit)
    reference = _richardson(problem, unit)

    np.testing.assert_allclose(analytic, reference, rtol=1e-6, atol=5e-11)


def test_expert_density_outside_standard_bounds_requires_explicit_setting() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    structure = replace(base, components=(replace(film, density_scale=1.25),))

    with pytest.raises(ValueError, match="initial outside compiled bounds"):
        _problem(structure=structure)

    problem = _problem(
        structure=structure,
        settings=(ParameterSetting("component.0.density_scale", 1.25, 0.5, 1.5),),
    )
    encode_physical_vector(problem, _initial_values(problem))


def test_fit_dataset_passes_the_same_instrument_to_compile_and_initialization() -> None:
    instrument = InstrumentSpec(footprint_mode="fit")
    data = prepared_data(size=72)
    structure = simple_structure()

    problem = _problem(data=data, structure=structure, instrument=instrument)
    initial = estimate_initial_candidates(
        data, structure, problem.instrument, np.random.default_rng(12)
    )

    assert problem.instrument is instrument
    assert 0.0 in initial.footprint_angles_deg


def test_fit_dataset_preserves_and_deduplicates_input_and_problem_warnings() -> None:
    data = replace(prepared_data(size=72), warnings=("input-warning", "input-warning"))
    config = FitConfig.fast(master_seed=13)

    problem = compile_fit_problem(
        data,
        simple_structure(),
        InstrumentSpec(footprint_mode="fit"),
        config,
    )
    warnings = tuple(dict.fromkeys((*data.warnings, *problem.warnings)))

    assert warnings.count("input-warning") == 1
    assert all(warnings.count(value) == 1 for value in problem.warnings)


def test_fit_dataset_supports_stages_with_no_free_parameters() -> None:
    problem = _problem()
    values = _initial_values(problem)
    locked = tuple(
        ParameterSetting(name, value, value, value, locked=True)
        for name, value in values.items()
    )
    no_free = compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        locked,
    )

    assert no_free.variables == ()
    np.testing.assert_array_equal(encode_physical_vector(no_free, {}), np.empty(0))


def test_fixed_density_subproblem_locks_density_and_keeps_other_parameters_free() -> None:
    problem = _problem()

    fixed = compile_fixed_parameter_problem(problem, "component.0.density_scale", 0.91)

    names = {coordinate.name for coordinate in fixed.variables}
    definition = next(
        item for item in fixed.parameter_definitions if item.name == "component.0.density_scale"
    )
    assert definition.locked
    assert definition.initial == definition.lower == definition.upper == 0.91
    assert "component.0.density_scale" not in names
    assert "component.0.thickness_a" in names


def test_footprint_parameter_bounds_and_locking() -> None:
    active = _problem(instrument=InstrumentSpec(footprint_mode="fit"))
    definition = next(
        item
        for item in active.parameter_definitions
        if item.name == "instrument.footprint_spill_angle_deg"
    )
    disabled = _problem(instrument=InstrumentSpec(footprint_mode="none"))

    assert definition.lower == 0.0
    assert definition.upper > 0.0
    assert "instrument.footprint_spill_angle_deg" not in {
        coordinate.name for coordinate in disabled.variables
    }


def test_log_unit_bounds_decode_to_exact_physical_bounds() -> None:
    problem = _problem()
    definition = next(
        item for item in problem.parameter_definitions if item.transform == "log"
    )
    definition = replace(
        definition,
        initial=100.0,
        lower=58.46351284627307,
        upper=34727.25865026945,
    )

    assert unit_to_physical(definition, 0.0) == definition.lower
    assert unit_to_physical(definition, 1.0) == definition.upper


def test_parameter_settings_cannot_unlock_inactive_instrument_modes() -> None:
    setting = ParameterSetting("instrument.relative_sigma", 0.01, 0.0, 0.1)

    with pytest.raises(ValueError, match="theta-domain mode requires"):
        _problem(
            instrument=InstrumentSpec(resolution_domain="theta"),
            settings=(setting,),
        )


def test_periodic_block_compiles_shared_variables() -> None:
    problem = _problem(structure=_periodic_structure())
    names = tuple(coordinate.name for coordinate in problem.variables)

    assert names.count("component.0.layer.0.thickness_a") == 1
    assert names.count("component.0.layer.1.thickness_a") == 1
    assert not any("repeat.1" in name for name in names)


def test_plateau_free_problem_records_one_dedicated_inactive_reason() -> None:
    problem = compile_fit_problem(
        prepared_data(size=72),
        simple_structure(),
        InstrumentSpec(footprint_mode="fit"),
        FitConfig.fast(master_seed=15),
    )

    assert problem.scale_prior_center is None
    assert problem.scale_prior_reason
    assert problem.warnings.count(problem.scale_prior_reason) == 1


def test_unit_upper_bound_decodes_to_a_strictly_legal_roughness() -> None:
    problem = _problem()
    unit = np.array(
        [
            1.0 if coordinate.transform == "roughness_fraction" else 0.5
            for coordinate in problem.variables
        ]
    )

    values = values_by_name(problem, unit)

    film_thickness = values["component.0.thickness_a"]
    assert values["component.0.roughness_a"] < 0.49 * film_thickness
