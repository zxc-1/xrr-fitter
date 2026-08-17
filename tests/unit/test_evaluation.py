from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from typing import get_type_hints

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ParameterDefinition,
    ParameterSetting,
)
from xrr_fitter.model.structure import (
    DriftSpec,
    GradientLayerSpec,
    InterfaceTransition,
    LayerSpec,
    PeriodicBlock,
    StructureSpec,
    TransitionBranch,
)
from xrr_fitter.physics.geometry import expand_geometry, expand_structure_with_jacobian
from xrr_fitter.physics.resolution import GaussHermiteConvergenceWarning
from xrr_fitter.physics.stack import expand_structure, rebuild_structure


def test_model_evaluation_recomputes_qz_and_shared_periodic_layers() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="first", thickness_a=24.0),
            replace(film, name="second", thickness_a=36.0),
        ),
        repeats=4,
        top_roughness_a=1.5,
    )
    problem = compile_fit_problem(
        prepared_data(size=80),
        StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=17), scale_prior_enabled=False),
    )
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical.update(
        {
            "component.0.layer.0.thickness_a": 27.0,
            "component.0.layer.1.thickness_a": 43.0,
            "instrument.angle_offset_deg": 0.025,
        }
    )
    unit = encode_physical_vector(problem, physical)

    evaluation = evaluate_vector(problem, unit)

    expected_qz = (
        4.0
        * np.pi
        * np.sin(np.deg2rad(problem.data.two_theta_deg / 2.0 + physical["instrument.angle_offset_deg"]))
        / problem.data.beam.effective_wavelength_a
    )
    assert evaluation.valid
    np.testing.assert_allclose(evaluation.qz_a_inv, expected_qz)
    assert evaluation.expanded_stack is not None
    np.testing.assert_allclose(
        evaluation.expanded_stack.thickness_a[1:-1],
        np.tile([27.0, 43.0], 4),
    )


def test_gradient_candidate_over_expanded_slab_budget_is_invalid() -> None:
    base = simple_structure()
    structure = replace(
        base,
        components=(
            GradientLayerSpec(
                "ramp",
                1e-5 + 1e-7j,
                3e-5 + 3e-7j,
                20.0,
                microslab_max_a=0.1,
            ),
        ),
        backing_roughness_a=0.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=40),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(18), scale_prior_enabled=False),
    )
    unit = np.array(encode_physical_vector(problem, {}), copy=True)
    thickness_index = next(
        index for index, coordinate in enumerate(problem.variables) if coordinate.name == "component.0.thickness_a"
    )
    unit[thickness_index] = 1.0

    result = evaluate_vector(problem, unit)
    residual, jacobian = evaluation.least_squares_system(problem, unit)

    assert result.valid is False
    assert result.reason == "constraint_violation:ExpandedSlabLimitError"
    np.testing.assert_array_equal(residual, np.full(residual.shape, 1e6))
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))


def test_gradient_roughness_uses_parent_thickness_limit_and_tangent() -> None:
    base = simple_structure()
    structure = replace(
        base,
        components=(
            GradientLayerSpec(
                "ramp",
                1e-5 + 1e-7j,
                3e-5 + 3e-7j,
                20.0,
                roughness_a=3.0,
                microslab_max_a=2.0,
            ),
        ),
        backing_roughness_a=3.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=48),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=18), scale_prior_enabled=False),
    )

    unit = np.array(encode_physical_vector(problem, {}), copy=True)
    dynamic = evaluation.roughness_dynamic_uppers(problem, unit)
    values, jacobians = evaluation.values_and_jacobians(problem, unit)
    expected_upper = np.nextafter(0.49 * 20.0, 0.0)

    assert dynamic["component.0.roughness_a"] == pytest.approx(expected_upper)
    assert dynamic["backing.roughness_a"] == pytest.approx(expected_upper)
    assert values["component.0.roughness_a"] == pytest.approx(3.0)
    assert values["backing.roughness_a"] == pytest.approx(3.0)

    thickness_index = next(
        index for index, coordinate in enumerate(problem.variables) if coordinate.name == "component.0.thickness_a"
    )
    step = 1e-6
    forward = unit.copy()
    backward = unit.copy()
    forward[thickness_index] += step
    backward[thickness_index] -= step
    high = evaluation.values_by_name(problem, forward)
    low = evaluation.values_by_name(problem, backward)
    for name in ("component.0.roughness_a", "backing.roughness_a"):
        finite_difference = (high[name] - low[name]) / (2.0 * step)
        assert jacobians[name][thickness_index] == pytest.approx(finite_difference)


def test_dynamic_roughness_decode_does_not_expand_material_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=21), scale_prior_enabled=False),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dynamic roughness decoding expanded a material stack")

    monkeypatch.setattr(evaluation, "expand_structure", forbidden)

    values = evaluation.values_by_name(problem, np.full(len(problem.variables), 0.5))

    assert values["component.0.roughness_a"] > 0.0


def test_single_repeat_periodic_latent_layer_roughness_decodes_for_primal_and_jacobian() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "single",
        (replace(film, name="only", roughness_a=3.0),),
        repeats=1,
        top_roughness_a=2.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=64),
        StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=4.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=211), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.5)
    latent = "component.0.layer.0.roughness_a"

    values = evaluation.values_by_name(problem, unit)
    jacobian_values, value_jacobians = evaluation.values_and_jacobians(problem, unit)
    residual_jacobian = evaluation.evaluate_model_jacobian(problem, unit)

    assert values[latent] == pytest.approx(jacobian_values[latent])
    latent_index = next(index for index, coordinate in enumerate(problem.variables) if coordinate.name == latent)
    assert value_jacobians[latent][latent_index] > 0.0
    assert np.all(np.isfinite(residual_jacobian))
    np.testing.assert_allclose(residual_jacobian[:, latent_index], 0.0, atol=1e-12)


def test_missing_roughness_cap_mapping_is_rejected_for_public_coordinates() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=22), scale_prior_enabled=False),
    )

    with pytest.raises(ValueError, match="roughness coordinate"):
        evaluation._fill_missing_roughness_caps(problem, {})


def test_roughness_drift_without_explicit_top_does_not_allow_missing_base_coordinate() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (replace(film, name="film", roughness_a=1.0),),
        repeats=3,
        drift=DriftSpec(kind="linear", target="roughness", amount=0.1),
    )
    problem = compile_fit_problem(
        prepared_data(size=48),
        StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=2201), scale_prior_enabled=False),
    )
    definitions = {
        definition.name: definition
        for definition in problem.parameter_definitions
        if definition.transform == "roughness_fraction"
    }
    missing = "component.0.layer.0.roughness_a"
    dynamic = {name: definition.upper for name, definition in definitions.items() if name != missing}

    with pytest.raises(ValueError, match="roughness coordinate mapping missing"):
        evaluation._fill_missing_roughness_caps(problem, dynamic)


def test_shared_solver_primitives_match_the_compiled_objective() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=18), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.55)
    residual = evaluation.least_squares_residual(problem, unit)
    jacobian = evaluation.least_squares_residual_jacobian(problem, unit)
    rho = evaluation.least_squares_loss(problem)(residual**2)

    assert jacobian.shape == (residual.size, unit.size)
    optimizer_objective = 0.5 * float(np.sum(rho[0])) / residual.size
    assert optimizer_objective == pytest.approx(
        evaluate_vector(problem, unit).objective,
        rel=1e-12,
        abs=1e-14,
    )


def test_fit_evaluation_keeps_unconverged_resolution_as_structured_diagnostic() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1802), scale_prior_enabled=False),
    )
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical["instrument.relative_sigma"] = 0.08
    unit = encode_physical_vector(problem, physical)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GaussHermiteConvergenceWarning)
        result = evaluation.evaluate_model(problem, unit)

    assert result.valid
    assert not any(item.category is GaussHermiteConvergenceWarning for item in caught)
    assert any(diagnostic.code == "gauss_hermite_unconverged" for diagnostic in result.diagnostics)


def test_joint_solver_system_matches_separate_residual_and_jacobian_paths() -> None:
    problem = compile_fit_problem(
        prepared_data(size=72),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1801), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.55)

    residual, jacobian = evaluation.least_squares_system(problem, unit)

    np.testing.assert_array_equal(
        residual,
        evaluation.least_squares_residual(problem, unit),
    )
    np.testing.assert_allclose(
        jacobian,
        evaluation.least_squares_residual_jacobian(problem, unit),
        rtol=0.0,
        atol=0.0,
    )


def test_cached_solver_callbacks_evaluate_one_parameter_vector_once() -> None:
    calls: list[np.ndarray] = []

    def system(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(np.array(unit, copy=True))
        return unit + 1.0, np.diag(unit + 2.0)

    residual, jacobian = evaluation.cached_least_squares_callbacks(system)
    first = np.asarray([0.2, 0.4])
    second = np.asarray([0.3, 0.5])

    np.testing.assert_array_equal(residual(first), first + 1.0)
    np.testing.assert_array_equal(jacobian(first.copy()), np.diag(first + 2.0))
    np.testing.assert_array_equal(residual(second), second + 1.0)
    np.testing.assert_array_equal(jacobian(second.copy()), np.diag(second + 2.0))

    assert len(calls) == 2
    np.testing.assert_array_equal(calls, (first, second))


def test_cached_solver_callbacks_isolate_concurrent_optimizer_threads() -> None:
    calls: list[np.ndarray] = []
    barrier = Barrier(2)

    def system(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(np.array(unit, copy=True))
        return unit + 1.0, np.diag(unit + 2.0)

    residual, jacobian = evaluation.cached_least_squares_callbacks(system)

    def evaluate(unit: np.ndarray) -> None:
        np.testing.assert_array_equal(residual(unit), unit + 1.0)
        barrier.wait()
        np.testing.assert_array_equal(jacobian(unit), np.diag(unit + 2.0))

    units = (np.asarray([0.2, 0.4]), np.asarray([0.3, 0.5]))
    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(evaluate, units))

    assert len(calls) == 2


def test_shared_problem_log_probability_uses_the_soft_l1_data_likelihood() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=19), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.45)
    residual = evaluation.least_squares_residual(problem, unit)
    weights = problem.weights[problem.data.fit_mask]
    c = problem.config.c_decades
    expected = -float(np.sum(weights**2 * 2.0 * c**2 * (np.sqrt(1.0 + (residual / c) ** 2) - 1.0))) / (2.0 * c**2)

    assert evaluation.problem_log_probability(problem, unit) == expected


@pytest.mark.parametrize(
    "name",
    (
        "values_and_jacobians",
        "values_by_name",
        "encode_physical_vector",
        "evaluate_model",
        "expanded_structure_jacobian",
        "evaluate_model_jacobian",
        "least_squares_residual",
        "least_squares_residual_jacobian",
        "least_squares_loss",
        "problem_log_probability",
    ),
)
def test_public_evaluation_entries_require_the_typed_context(name: str) -> None:
    hints = get_type_hints(getattr(evaluation, name))

    assert hints["problem"] is FitEvaluationContext


def test_analytic_stack_roughness_failure_is_a_candidate_constraint() -> None:
    structure = simple_structure()
    layer = replace(
        structure.components[0],
        thickness_a=2.0,
        roughness_a=1.0,
    )
    structure = replace(structure, components=(layer,))
    config = replace(FitConfig.fast(20), scale_prior_enabled=False)
    initial = compile_fit_problem(
        prepared_data(size=40),
        structure,
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    locked = {"component.0.thickness_a", "component.0.roughness_a"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial if definition.name in locked else definition.lower,
            definition.initial if definition.name in locked else definition.upper,
            locked=definition.name in locked or definition.locked,
        )
        for definition in initial.parameter_definitions
    )
    problem = compile_fit_problem(
        initial.data,
        structure,
        initial.instrument,
        config,
        settings,
    )
    unit = np.full(len(problem.variables), 0.5)

    with pytest.raises(EvaluationConstraintError, match="constraint_violation"):
        evaluation.evaluate_model_jacobian(problem, unit)

    jacobian = evaluation.least_squares_residual_jacobian(problem, unit)
    np.testing.assert_array_equal(
        jacobian,
        np.zeros((np.count_nonzero(problem.data.fit_mask), len(problem.variables))),
    )


def transition_problem() -> FitEvaluationContext:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    layer = replace(
        film,
        thickness_a=30.0,
        roughness_a=0.0,
        transition=InterfaceTransition(
            branches=(TransitionBranch(kind="erf", weight=1.0, thickness_a=12.0),),
            microslab_max_a=3.0,
        ),
    )
    structure = StructureSpec(
        base.fronting,
        (layer,),
        base.backing,
        backing_roughness_a=3.0,
    )
    config = replace(FitConfig.fast(master_seed=91), scale_prior_enabled=False)
    initial = compile_fit_problem(
        prepared_data(size=48),
        structure,
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    bounds = {"component.0.thickness_a": (20.0, 45.0)}
    settings = tuple(_transition_setting(definition, bounds) for definition in initial.parameter_definitions)
    return compile_fit_problem(
        initial.data,
        structure,
        initial.instrument,
        config,
        settings,
    )


def _transition_setting(
    definition: ParameterDefinition,
    bounds: dict[str, tuple[float, float]],
) -> ParameterSetting:
    """Lock every roughness to zero and pin thickness to a log-symmetric range."""
    if definition.name.endswith("roughness_a"):
        return ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial,
            definition.initial,
            locked=True,
        )
    lower, upper = bounds.get(definition.name, (definition.lower, definition.upper))
    return ParameterSetting(
        definition.name,
        definition.initial,
        lower,
        upper,
        locked=definition.locked,
    )


@pytest.mark.parametrize(
    "name",
    ["component.0.thickness_a", "component.0.density_scale"],
)
def test_transition_jacobian_matches_finite_differences(name: str) -> None:
    problem = transition_problem()
    index = next(position for position, coordinate in enumerate(problem.variables) if coordinate.name == name)
    unit = np.full(len(problem.variables), 0.5)
    stack = evaluation.expanded_structure_jacobian(problem, unit)
    assert stack.stack.thickness_a.size == 7

    step = 1e-6
    forward = unit.copy()
    forward[index] += step
    backward = unit.copy()
    backward[index] -= step
    high = evaluation.expanded_structure_jacobian(problem, forward).stack
    low = evaluation.expanded_structure_jacobian(problem, backward).stack

    np.testing.assert_allclose(
        stack.thickness_jacobian[:, index],
        (high.thickness_a - low.thickness_a) / (2.0 * step),
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        stack.sld_jacobian[:, index],
        (high.sld_a2 - low.sld_a2) / (2.0 * step),
        rtol=1e-6,
        atol=1e-16,
    )


def test_transition_expansion_aligns_across_all_three_paths() -> None:
    problem = transition_problem()
    unit = np.full(len(problem.variables), 0.5)
    expected_media = 7  # fronting + four transition slabs + layer body + backing
    values, value_jacobians = evaluation.values_and_jacobians(problem, unit)
    rebuilt = rebuild_structure(problem.structure, values)
    wavelength = problem.data.beam.effective_wavelength_a

    primal = expand_structure(rebuilt, wavelength)
    differentiable = expand_structure_with_jacobian(
        problem.structure,
        values,
        value_jacobians,
        wavelength,
        len(problem.variables),
    )
    geometry = expand_geometry(rebuilt, len(problem.variables), value_jacobians)

    assert primal.thickness_a.size == expected_media
    np.testing.assert_array_equal(primal.thickness_a, differentiable.stack.thickness_a)
    np.testing.assert_array_equal(primal.thickness_a, geometry.thickness_a)
    assert differentiable.thickness_jacobian.shape[0] == expected_media
    assert geometry.thickness_jacobian is not None
    assert geometry.thickness_jacobian.shape[0] == expected_media
    assert len(geometry.interface_names) == primal.thickness_a.size - 1
