from __future__ import annotations

from tests.unit.fit.problem_compilation_cases import *


def test_direct_backing_sld_changes_the_evaluated_model() -> None:
    problem = _direct_backing_jacobian_problem()
    lower = evaluate_vector(
        problem,
        encode_physical_vector(problem, {"backing.sld_real_a2": 20e-6}),
    )
    upper = evaluate_vector(
        problem,
        encode_physical_vector(problem, {"backing.sld_real_a2": 80e-6}),
    )

    assert lower.valid and upper.valid
    assert not np.array_equal(lower.model_normalized, upper.model_normalized)


@pytest.mark.parametrize(
    "problem_factory",
    [
        pytest.param(_linear_background_jacobian_problem, id="linear_background_jacobian_problem"),
        pytest.param(_powerlaw_background_jacobian_problem, id="powerlaw_background_jacobian_problem"),
        pytest.param(_direct_sld_jacobian_problem, id="direct_sld_jacobian_problem"),
        pytest.param(_direct_backing_jacobian_problem, id="direct_backing_jacobian_problem"),
        pytest.param(_gradient_jacobian_problem, id="gradient_jacobian_problem"),
    ],
)
def test_evaluate_jacobian_covers_expert_background_and_sld_paths(problem_factory) -> None:
    problem = problem_factory()
    unit = np.full(len(problem.variables), 0.46)

    analytic = evaluate_jacobian(problem, unit)
    reference = _richardson(problem, unit)

    np.testing.assert_allclose(analytic, reference, rtol=1e-6, atol=5e-11)


def test_negative_sampled_background_is_invalid_for_primal_and_analytic_solver() -> None:
    problem = _linear_background_jacobian_problem()
    unit = np.array(encode_physical_vector(problem, {}), copy=True)
    names = tuple(variable.name for variable in problem.variables)
    unit[names.index("instrument.background")] = 0.0
    unit[names.index("instrument.linear_background_per_a_inv")] = 0.499999

    primal = evaluate_vector(problem, unit)
    residual, jacobian = least_squares_system(problem, unit)

    assert primal.valid is False
    assert primal.reason == "constraint_violation:ValueError"
    np.testing.assert_array_equal(residual, np.full(residual.shape, 1e6))
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))


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
    initial = estimate_initial_candidates(data, structure, problem.instrument, np.random.default_rng(12))

    assert problem.instrument is instrument
    assert 0.0 in initial.footprint_angles_deg
