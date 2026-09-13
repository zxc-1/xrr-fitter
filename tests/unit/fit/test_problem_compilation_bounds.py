from __future__ import annotations

from tests.unit.fit.problem_compilation_cases import *

from xrr_fitter.model.parameters import ParameterDefinition


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
        ParameterSetting(name, value, value, value, freedom=ParameterFreedom.FIXED) for name, value in values.items()
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


def _folded_gear(name: str, setting: ParameterSetting) -> tuple[ParameterDefinition, bool]:
    """把一条 setting 折进编译结果，取回那个量的定义和它是否还在变量表里。"""
    problem = _problem(settings=(setting,))
    definition = next(item for item in problem.parameter_definitions if item.name == name)
    return definition, any(coordinate.name == name for coordinate in problem.variables)


def test_range_only_keeps_the_declared_box_while_the_other_two_gears_replace_it() -> None:
    """三档折进编译结果的方式各不相同，而「仅范围」是唯一不动搜索盒的那一档。

    区间要是也当成盒子，服务层为它派生的那条 ``soft_range`` 先验就没有惩罚区了——先验一律
    被截断到定义的上下限，区间和盒子重合时截断掉的正是「越界渐进受罚」那一段，剩下的就是盒
    内等权，跟「自由」分毫不差。所以这一档只挪初值、保留声明的硬边界，让区间只以先验的形式
    起作用；另外两档的区间本来就是硬边界，直接换掉上下限。
    """
    name = "component.0.thickness_a"
    declared = next(item for item in _problem().parameter_definitions if item.name == name)
    lower = declared.lower + 1.0
    upper = lower + 30.0
    initial = lower + 10.0

    folded = {
        freedom: _folded_gear(name, ParameterSetting(name, initial, lower, upper, freedom))
        for freedom in ParameterFreedom
    }

    ranged, ranged_fitted = folded[ParameterFreedom.RANGE_ONLY]
    freed, freed_fitted = folded[ParameterFreedom.FREE]
    pinned, pinned_fitted = folded[ParameterFreedom.FIXED]
    assert (ranged.lower, ranged.upper) == (declared.lower, declared.upper)
    assert (freed.lower, freed.upper) == (lower, upper)
    assert (pinned.lower, pinned.upper) == (lower, upper)
    assert (ranged.initial, freed.initial, pinned.initial) == (initial, initial, initial)
    # 「仅范围」和「自由」一样参与拟合，只有「固定」退出变量表。
    assert (ranged.locked, freed.locked, pinned.locked) == (False, False, True)
    assert (ranged_fitted, freed_fitted, pinned_fitted) == (True, True, False)


def test_unit_upper_bound_decodes_to_a_strictly_legal_roughness() -> None:
    problem = _problem()
    unit = np.array([1.0 if coordinate.transform == "roughness_fraction" else 0.5 for coordinate in problem.variables])

    values = values_by_name(problem, unit)

    film_thickness = values["component.0.thickness_a"]
    assert values["component.0.roughness_a"] < 0.49 * film_thickness
