from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import log
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
    PriorSpec,
    _prior_center,
)
from xrr_fitter.model.structure import (
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


def _prior_problem() -> FitEvaluationContext:
    return compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=19), scale_prior_enabled=False),
    )


def test_problem_log_probability_is_bitwise_unchanged_without_priors() -> None:
    problem = _prior_problem()
    unit = np.full(len(problem.variables), 0.45)
    residual = evaluation.least_squares_residual(problem, unit)
    weights = problem.weights[problem.data.fit_mask]
    c = problem.config.c_decades
    baseline = -float(np.sum(weights**2 * 2.0 * c**2 * (np.sqrt(1.0 + (residual / c) ** 2) - 1.0))) / (2.0 * c**2)

    assert all(definition.prior is None for definition in problem.parameter_definitions)
    assert evaluation.problem_log_probability(problem, unit) == baseline


def test_prior_log_density_matches_closed_form() -> None:
    uniform = PriorSpec("uniform")
    normal = PriorSpec("normal", (20.0, 2.0))
    lognormal = PriorSpec("lognormal", (log(20.0), 0.25))
    soft_range = PriorSpec("soft_range", (10.0, 30.0, 2.0))

    def unnormalized(spec: PriorSpec, x: float) -> float:
        return evaluation.prior_log_density(spec, x, 2.0, 200.0) - evaluation.prior_log_density(spec, 20.0, 2.0, 200.0)

    assert unnormalized(uniform, 100.0) == pytest.approx(0.0)
    assert unnormalized(normal, 22.0) == pytest.approx(-0.5)
    assert unnormalized(normal, 16.0) == pytest.approx(-2.0)
    assert unnormalized(lognormal, 20.0 * np.exp(0.25)) == pytest.approx(-0.5 - 0.25)
    assert unnormalized(soft_range, 25.0) == pytest.approx(0.0)
    assert unnormalized(soft_range, 34.0) == pytest.approx(-2.0)


def test_prior_cdf_is_monotone_and_spans_zero_to_one() -> None:
    lower, upper = 2.0, 200.0
    grid = np.linspace(lower, upper, 257)
    for spec in (
        PriorSpec("uniform"),
        PriorSpec("normal", (20.0, 2.0)),
        PriorSpec("lognormal", (log(20.0), 0.25)),
        PriorSpec("soft_range", (10.0, 30.0, 2.0)),
    ):
        values = np.array([evaluation.prior_cdf(spec, float(x), lower, upper) for x in grid])

        assert np.all(np.diff(values) >= -1e-12)
        assert values[0] == pytest.approx(0.0, abs=1e-12)
        assert values[-1] == pytest.approx(1.0, abs=1e-9)


def test_prior_inverse_cdf_round_trips() -> None:
    lower, upper = 2.0, 200.0
    for spec in (
        PriorSpec("uniform"),
        PriorSpec("normal", (20.0, 2.0)),
        PriorSpec("lognormal", (log(20.0), 0.25)),
        PriorSpec("soft_range", (10.0, 30.0, 2.0)),
    ):
        # Sample inside the strictly increasing part of the cdf: a truncated
        # prior is flat far out in its tails, where the inverse is not unique.
        for level in (0.1, 0.25, 0.5, 0.75, 0.9):
            x = evaluation.prior_inverse_cdf(spec, level, lower, upper)

            assert evaluation.prior_cdf(spec, x, lower, upper) == pytest.approx(level, rel=1e-10)
            assert evaluation.prior_inverse_cdf(spec, evaluation.prior_cdf(spec, x, lower, upper), lower, upper) == (
                pytest.approx(x, rel=1e-10)
            )


def test_prior_bounds_are_respected() -> None:
    assert evaluation.prior_bounds(PriorSpec("uniform"), 2.0, 200.0) == (2.0, 200.0)
    assert evaluation.prior_bounds(PriorSpec("normal", (20.0, 2.0)), 2.0, 200.0) == (2.0, 200.0)
    assert evaluation.prior_bounds(PriorSpec("soft_range", (10.0, 30.0, 2.0)), 2.0, 200.0) == (2.0, 200.0)


def test_prior_center_and_spread_maps_each_kind() -> None:
    assert evaluation.prior_center_and_spread(PriorSpec("uniform")) is None
    assert evaluation.prior_center_and_spread(PriorSpec("normal", (20.0, 2.0))) == (20.0, 2.0)

    center, spread = evaluation.prior_center_and_spread(PriorSpec("lognormal", (log(20.0), 0.25)))

    assert (center, spread) == pytest.approx((20.0, 20.0 * 0.25))
    assert evaluation.prior_center_and_spread(PriorSpec("soft_range", (10.0, 30.0, 2.0))) == (20.0, 12.0)


def test_prior_center_matches_the_model_side_lightweight_mapping() -> None:
    for spec in (
        PriorSpec("uniform"),
        PriorSpec("normal", (20.0, 2.0)),
        PriorSpec("lognormal", (log(20.0), 0.25)),
        PriorSpec("soft_range", (10.0, 30.0, 2.0)),
    ):
        expected = evaluation.prior_center_and_spread(spec)

        assert _prior_center(spec) == (None if expected is None else pytest.approx(expected[0]))


def test_prior_log_density_normalization_constant_is_cached() -> None:
    spec = PriorSpec("normal", (20.0, 2.0))
    evaluation.prior_log_density(spec, 20.0, 2.0, 200.0)
    before = evaluation._prior_norm.cache_info()
    for _ in range(8):
        evaluation.prior_log_density(spec, 21.0, 2.0, 200.0)
    after = evaluation._prior_norm.cache_info()

    assert after.hits > before.hits
    assert after.misses == before.misses


def test_extremely_narrow_normal_prior_retains_finite_mass_and_quantiles() -> None:
    spec = PriorSpec("normal", (100.03, 1e-4))

    density = evaluation.prior_log_density(spec, 100.03, 0.0, 200.0)
    median = evaluation.prior_inverse_cdf(spec, 0.5, 0.0, 200.0)

    assert np.isfinite(density)
    assert median == pytest.approx(100.03, abs=1e-7)
    assert evaluation.prior_cdf(spec, median, 0.0, 200.0) == pytest.approx(0.5)


def test_extremely_narrow_lognormal_prior_retains_finite_mass_and_quantiles() -> None:
    center = 75.07
    spec = PriorSpec("lognormal", (log(center), 1e-5))

    density = evaluation.prior_log_density(spec, center, 1.0, 200.0)
    median = evaluation.prior_inverse_cdf(spec, 0.5, 1.0, 200.0)

    assert np.isfinite(density)
    assert median == pytest.approx(center, rel=1e-7)
    assert evaluation.prior_cdf(spec, median, 1.0, 200.0) == pytest.approx(0.5)
