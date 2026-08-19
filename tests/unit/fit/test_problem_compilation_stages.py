from __future__ import annotations

from tests.unit.fit.problem_compilation_cases import *


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


def test_compile_stage_problem_locks_current_values_and_preserves_user_locks() -> None:
    problem = _problem(settings=(ParameterSetting("instrument.scale", 1.25, 1.25, 1.25, locked=True),))
    values = _initial_values(problem)
    values.update(
        {
            "component.0.thickness_a": values["component.0.thickness_a"] * 1.05,
            "component.0.density_scale": 0.9,
            "component.0.roughness_a": 3.5,
            "instrument.scale": 2.0,
        }
    )

    stage_b = compile_stage_problem(problem, "B", values)
    definitions = {value.name: value for value in stage_b.parameter_definitions}

    assert definitions["component.0.thickness_a"].initial == values["component.0.thickness_a"]
    assert not definitions["component.0.thickness_a"].locked
    for name in ("component.0.density_scale", "component.0.roughness_a"):
        assert definitions[name].initial == values[name]
        assert definitions[name].locked
        assert definitions[name].lower == definitions[name].upper == values[name]
    assert definitions["instrument.scale"].initial == 1.25
    assert definitions["instrument.scale"].locked
