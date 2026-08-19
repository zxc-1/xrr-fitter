from __future__ import annotations

from tests.unit.fit.problem_compilation_cases import *


def test_compile_stage_problem_rejects_incomplete_current_candidate_values() -> None:
    problem = _problem()
    values = _initial_values(problem)
    del values["component.0.density_scale"]

    with pytest.raises(ValueError, match="missing current stage values.*density_scale"):
        compile_stage_problem(problem, "B", values)


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


def test_theta_domain_does_not_silently_ignore_point_resolution_columns() -> None:
    data = _angular_point_resolution_jacobian_problem().data

    with pytest.raises(ValueError, match="per-point resolution.*theta-domain"):
        compile_fit_problem(
            data,
            simple_structure(),
            InstrumentSpec(footprint_mode="none", resolution_domain="theta"),
            _config(27),
        )


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
    problem = _problem(instrument=InstrumentSpec(footprint_mode="none", background_kind=background_kind))

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

    definition = next(item for item in problem.parameter_definitions if item.name == "component.0.thickness_a")
    assert definition.lower == 2.0
    encode_physical_vector(problem, _initial_values(problem))
