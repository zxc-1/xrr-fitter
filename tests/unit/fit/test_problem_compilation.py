from __future__ import annotations

from tests.unit.fit.problem_compilation_cases import *


def test_compilation_returns_the_shared_typed_evaluation_context() -> None:
    fitting = import_module("xrr_fitter.model.fitting")
    problem_module = import_module("xrr_fitter.fit.problem")
    problem = _problem()

    assert isinstance(problem, fitting.FitEvaluationContext)
    assert not hasattr(problem_module, "CompiledFitProblem")
    assert problem.region_labels.flags.writeable is False
    assert problem.weights.flags.writeable is False

    restored = pickle.loads(pickle.dumps(problem))

    assert isinstance(restored, fitting.FitEvaluationContext)
    assert restored.region_labels.flags.writeable is False
    assert restored.weights.flags.writeable is False
    np.testing.assert_array_equal(restored.region_labels, problem.region_labels)
    np.testing.assert_array_equal(restored.weights, problem.weights)


def test_extreme_finite_high_angle_tail_keeps_background_bounds_finite() -> None:
    data = _extreme_high_angle_tail_data()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem = _problem(data=data)

    background = next(
        definition for definition in problem.parameter_definitions if definition.name == "instrument.background"
    )
    assert 0.1 <= background.upper < np.finfo(float).max
    assert not any(item.category is RuntimeWarning for item in caught)


def test_extreme_finite_high_angle_tail_keeps_background_starts_finite() -> None:
    data = _extreme_high_angle_tail_data()
    problem = _problem(data=data)
    background = next(
        definition for definition in problem.parameter_definitions if definition.name == "instrument.background"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        initial = estimate_initial_candidates(
            data,
            simple_structure(),
            InstrumentSpec(footprint_mode="fit"),
            np.random.default_rng(12),
        )

    assert initial.backgrounds[-1] == background.upper
    assert all(np.isfinite(value) for value in initial.backgrounds)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_extreme_finite_high_angle_tail_keeps_linear_background_jacobian_finite() -> None:
    problem = _problem(
        data=_extreme_high_angle_tail_data(),
        instrument=InstrumentSpec(
            footprint_mode="fit",
            background_kind="linear",
        ),
    )
    unit = encode_physical_vector(problem, {})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidate = evaluate_vector(problem, unit)
        jacobian = evaluate_jacobian(problem, unit)

    assert candidate.valid
    assert np.all(np.isfinite(jacobian))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_gradient_microslab_topology_parameter_cannot_be_unlocked() -> None:
    initial = _gradient_jacobian_problem()
    name = "component.0.microslab_max_a"

    with pytest.raises(ValueError, match="microslab topology"):
        compile_fit_problem(
            initial.data,
            initial.structure,
            initial.instrument,
            initial.config,
            (ParameterSetting(name, 20.0, 10.0, 30.0, locked=False),),
        )


def test_gradient_thickness_lower_bound_respects_microslab_maximum() -> None:
    problem = _gradient_jacobian_problem()
    definitions = {definition.name: definition for definition in problem.parameter_definitions}

    assert definitions["component.0.thickness_a"].lower >= 20.0
