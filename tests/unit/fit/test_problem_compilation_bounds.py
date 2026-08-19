from __future__ import annotations

from tests.unit.fit.problem_compilation_cases import *


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
    locked = tuple(ParameterSetting(name, value, value, value, locked=True) for name, value in values.items())
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
    definition = next(item for item in fixed.parameter_definitions if item.name == "component.0.density_scale")
    assert definition.locked
    assert definition.initial == definition.lower == definition.upper == 0.91
    assert "component.0.density_scale" not in names
    assert "component.0.thickness_a" in names


def test_footprint_parameter_bounds_and_locking() -> None:
    active = _problem(instrument=InstrumentSpec(footprint_mode="fit"))
    definition = next(
        item for item in active.parameter_definitions if item.name == "instrument.footprint_spill_angle_deg"
    )
    disabled = _problem(instrument=InstrumentSpec(footprint_mode="none"))

    assert definition.lower == 0.0
    assert definition.upper > 0.0
    assert "instrument.footprint_spill_angle_deg" not in {coordinate.name for coordinate in disabled.variables}


def test_fit_footprint_bound_stays_inside_physical_angle_domain(monkeypatch) -> None:
    monkeypatch.setattr("xrr_fitter.fit.parameters._footprint_upper_deg", lambda _data: 120.0)

    problem = _problem(instrument=InstrumentSpec(footprint_mode="fit"))
    definition = next(
        item for item in problem.parameter_definitions if item.name == "instrument.footprint_spill_angle_deg"
    )

    assert definition.upper == 90.0


def test_fit_footprint_bound_retains_declared_initial_above_data_estimate() -> None:
    problem = _problem(
        instrument=InstrumentSpec(
            footprint_mode="fit",
            footprint_spill_angle_deg=10.0,
        )
    )
    definition = next(
        item for item in problem.parameter_definitions if item.name == "instrument.footprint_spill_angle_deg"
    )

    assert definition.initial == 10.0
    assert definition.upper >= definition.initial


def test_log_unit_bounds_decode_to_exact_physical_bounds() -> None:
    problem = _problem()
    definition = next(item for item in problem.parameter_definitions if item.transform == "log")
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
    unit = np.array([1.0 if coordinate.transform == "roughness_fraction" else 0.5 for coordinate in problem.variables])

    values = values_by_name(problem, unit)

    film_thickness = values["component.0.thickness_a"]
    assert values["component.0.roughness_a"] < 0.49 * film_thickness
