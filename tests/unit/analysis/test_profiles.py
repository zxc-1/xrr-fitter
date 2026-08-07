from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting


def _api():
    return import_module("xrr_fitter.analysis.profiles")


def profile_parameter(*args, **kwargs):
    return _api().profile_parameter(*args, **kwargs)


def profile_parameter_with_decision(*args, **kwargs):
    return _api().profile_parameter_with_decision(*args, **kwargs)


def profile_covers_value(*args, **kwargs):
    return _api().profile_covers_value(*args, **kwargs)


def select_profile_names(*args, **kwargs):
    return _api().select_profile_names(*args, **kwargs)


def default_profile_path_merge(*args, **kwargs):
    return _api().default_profile_path_merge(*args, **kwargs)


def build_problem_profile(*args, **kwargs):
    return _api().build_problem_profile(*args, **kwargs)


def recover_profile_basin(*args, **kwargs):
    return _api().recover_profile_basin(*args, **kwargs)


def _problem(*targets: str):
    base = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(901), scale_prior_enabled=False),
    )
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in targets,
        )
        for definition in base.parameter_definitions
    )
    return compile_fit_problem(
        base.data, base.structure, base.instrument, base.config, settings
    )


def test_quadratic_profile_closes_before_bounds_and_flat_profile_stays_open() -> None:
    center = np.asarray([0.50, 0.35])
    closed = profile_parameter(
        lambda unit: float(np.sum((unit - center) ** 2)),
        center,
        parameter_index=0,
        steps=101,
    )
    opened = profile_parameter(
        lambda _unit: 0.0,
        center,
        parameter_index=0,
        steps=101,
    )

    assert closed.lower_closed and closed.upper_closed
    assert not opened.lower_closed and not opened.upper_closed


def test_profile_boundary_value_can_prove_crossing_near_bound() -> None:
    center = np.asarray([0.001])
    profile = profile_parameter(
        lambda unit: float((unit[0] - center[0]) ** 2),
        center,
        parameter_index=0,
        steps=5,
        objective_delta=1e-8,
    )

    assert profile.lower_closed


def test_profile_value_mapper_receives_joint_optimized_vector() -> None:
    center = np.asarray([0.5, 0.25])

    profile = profile_parameter(
        lambda unit: float((unit[0] - 0.5) ** 2 + 10.0 * (unit[1] - unit[0] / 2.0) ** 2),
        center,
        parameter_index=0,
        steps=11,
        value_mapper=lambda unit: float(unit.sum()),
    )

    assert np.all(np.diff(profile.values) > 0.0)
    assert np.any(np.isclose(profile.values, 0.3))


def test_profile_reoptimizes_coupled_nuisance_parameter_along_valley() -> None:
    center = np.asarray([0.5, 0.5])

    profile = profile_parameter(
        lambda unit: float((unit[1] - unit[0]) ** 2 + 0.02 * (unit[0] - 0.5) ** 2),
        center,
        parameter_index=0,
        steps=17,
    )

    assert np.nanmax(profile.objectives) < 0.01


def test_profile_uses_supplied_analytic_gradient_for_nuisance_optimization() -> None:
    calls = 0
    center = np.asarray([0.5, 0.5])

    def gradient(unit: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray(
            [2.0 * (unit[0] - unit[1]), 2.0 * (unit[1] - unit[0])]
        )

    profile_parameter(
        lambda unit: float((unit[1] - unit[0]) ** 2),
        center,
        parameter_index=0,
        steps=9,
        gradient=gradient,
    )

    assert calls > 0


def test_profile_uses_supplied_residual_jacobian_for_nuisance_optimization() -> None:
    calls = 0
    center = np.asarray([0.5, 0.5])

    def jacobian(_unit: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray([[1.0, -1.0]])

    profile_parameter(
        lambda unit: float((unit[0] - unit[1]) ** 2),
        center,
        parameter_index=0,
        steps=9,
        residual=lambda unit: np.asarray([unit[0] - unit[1]]),
        residual_jacobian=jacobian,
    )

    assert calls > 0


# Profile closure resolves objective deltas at the 1e-5 scale. Precision below
# 1e-6 did not change closure decisions but multiplied periodic nuisance work,
# so both residual least-squares paths must retain the analysis tolerance.
def test_profile_residual_solvers_use_analysis_tolerance(monkeypatch) -> None:
    module = _api()
    calls: list[tuple[tuple[int, ...], float, float, float]] = []

    def solve(_residual, start, **options):
        value = np.asarray(start, dtype=float)
        calls.append(
            (
                value.shape,
                options["ftol"],
                options["xtol"],
                options["gtol"],
            )
        )
        return SimpleNamespace(x=value)

    monkeypatch.setattr(module, "least_squares", solve)
    center = np.asarray([0.5, 0.5])
    profile_parameter(
        lambda unit: float((unit[0] - 0.8) ** 2 + (unit[1] - 0.5) ** 2),
        center,
        parameter_index=0,
        steps=9,
        residual=lambda unit: np.asarray([unit[0] - 0.8, unit[1] - 0.5]),
        residual_jacobian=lambda _unit: np.eye(2),
    )

    assert {shape for shape, *_tolerances in calls} == {(1,), (2,)}
    assert {
        (ftol, xtol, gtol)
        for _shape, ftol, xtol, gtol in calls
    } == {(1e-6, 1e-6, 1e-6)}


def test_profile_tolerates_physically_invalid_parameter_boundary() -> None:
    center = np.asarray([0.5, 0.5])

    def objective(unit: np.ndarray) -> float:
        if unit[0] in {0.0, 1.0}:
            return float("inf")
        return float(np.sum((unit - center) ** 2))

    profile = profile_parameter(objective, center, parameter_index=0, steps=9)

    assert np.isinf(profile.objectives[[0, -1]]).all()
    assert np.isfinite(profile.objectives[1:-1]).any()


def test_profile_basin_recovery_treats_physical_constraint_failures_as_invalid_probes(
    monkeypatch,
) -> None:
    problem = SimpleNamespace(
        variables=(SimpleNamespace(name="component.0.thickness_a"),),
        config=SimpleNamespace(
            budget=SimpleNamespace(
                bootstrap_samples=8,
                local_min_nfev=5,
                local_nfev_per_parameter=1,
            ),
            confidence=SimpleNamespace(
                equivalent_cost_fraction=0.01,
                equivalent_cost_floor=1e-8,
            ),
        ),
        data=SimpleNamespace(fit_mask=np.ones(3, dtype=np.bool_)),
    )
    candidate = SimpleNamespace(
        valid=True,
        objective=0.20,
        unit_vector=np.asarray([0.5]),
    )

    def evaluate(_problem, unit):
        if unit[0] in {0.0, 1.0}:
            raise EvaluationConstraintError("constraint_violation:ValueError")
        return SimpleNamespace(valid=True, objective=float((unit[0] - 0.8) ** 2))

    monkeypatch.setattr(_api(), "evaluate_model", evaluate)

    decision = recover_profile_basin(problem, candidate)

    assert decision is not None
    assert decision.parameter_name == "component.0.thickness_a"
    assert decision.unit_vector[0] > 0.7


def test_direct_problem_profile_treats_physical_constraint_failures_as_invalid_probes(
    monkeypatch,
) -> None:
    problem = SimpleNamespace(
        variables=(SimpleNamespace(name="component.0.thickness_a"),),
        config=SimpleNamespace(
            budget=SimpleNamespace(
                bootstrap_samples=8,
                local_min_nfev=5,
                local_nfev_per_parameter=1,
            ),
            c_decades=0.05,
        ),
        data=SimpleNamespace(fit_mask=np.ones(3, dtype=np.bool_)),
        weights=np.ones(3),
        scale_prior_center=None,
    )

    def evaluate(_problem, unit):
        if unit[0] in {0.0, 1.0}:
            raise EvaluationConstraintError("constraint_violation:ValueError")
        return SimpleNamespace(valid=True, objective=float((unit[0] - 0.5) ** 2))

    monkeypatch.setattr(
        import_module("xrr_fitter.analysis.binary_profiles"),
        "binary_derived_profiles",
        lambda _problem: (),
    )
    monkeypatch.setattr(_api(), "evaluate_model", evaluate)
    monkeypatch.setattr(
        _api(),
        "values_by_name",
        lambda _problem, unit: {"component.0.thickness_a": float(unit[0])},
    )

    profile = build_problem_profile(
        problem,
        np.asarray([0.5]),
        "component.0.thickness_a",
    )

    assert np.isinf(profile.objectives[[0, -1]]).all()
    assert np.isfinite(profile.objectives[1:-1]).any()


def test_profile_polls_cancellation_inside_nuisance_solver() -> None:
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls > 3

    with pytest.raises(InterruptedError, match="cancelled"):
        profile_parameter(
            lambda unit: float(np.sum((unit - 0.5) ** 2)),
            np.asarray([0.5, 0.5]),
            parameter_index=0,
            steps=21,
            cancelled=cancelled,
        )


def test_profile_adaptively_refines_an_ordinary_coarse_support_transition() -> None:
    profile = profile_parameter(
        lambda unit: 0.0 if unit[0] < 0.75 else 1.0,
        np.asarray([0.4]),
        parameter_index=0,
        upper=0.8,
        steps=5,
        objective_delta=0.1,
    )

    np.testing.assert_allclose(
        profile.values,
        np.asarray([0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8]),
        rtol=0.0,
        atol=1e-15,
    )


def test_profile_discards_transition_probes_without_new_support() -> None:
    profile = profile_parameter(
        lambda unit: 0.0 if unit[0] < 0.75 else float("inf"),
        np.asarray([0.5]),
        parameter_index=0,
        steps=5,
        objective_delta=0.1,
    )

    assert len(np.unique(profile.values)) == profile.values.size
    assert not profile.upper_closed


def test_profile_publishes_unsupported_probes_that_expand_reported_support() -> None:
    profile = profile_parameter(
        lambda unit: float((unit[0] - 0.5) ** 2) if unit[0] <= 0.8 else float("inf"),
        np.asarray([0.5]),
        parameter_index=0,
        steps=5,
        objective_delta=0.04,
    )

    assert np.isinf(profile.objectives).any()
    assert profile.values.max() == 1.0


def test_profile_recenters_a_materially_better_coarse_basin() -> None:
    center = np.asarray([0.2, 0.5])

    profile, decision = profile_parameter_with_decision(
        lambda unit: float(min((unit[0] - 0.2) ** 2 + 0.05, (unit[0] - 0.8) ** 2)),
        center,
        parameter_index=0,
        name="component.0.thickness_a",
        steps=9,
    )

    assert profile.values.size >= 9
    assert decision is not None
    assert decision.unit_vector[0] > 0.7
    assert decision.objective < 0.01


def test_profile_refines_coarse_support_boundaries_for_a_narrow_secondary_basin() -> None:
    center = np.asarray([0.25])

    profile, decision = profile_parameter_with_decision(
        lambda unit: float(
            min((unit[0] - 0.25) ** 2 + 0.02, ((unit[0] - 0.735) / 0.03) ** 2)
        ),
        center,
        parameter_index=0,
        name="component.0.thickness_a",
        steps=9,
    )

    assert profile.values.size > 9
    assert decision is not None
    assert abs(decision.unit_vector[0] - 0.735) < 0.03


def test_profile_coverage_uses_supported_objective_region() -> None:
    profile = profile_parameter(
        lambda unit: float((unit[0] - 0.5) ** 2),
        np.asarray([0.5]),
        parameter_index=0,
        steps=21,
        objective_delta=0.04,
    )

    assert profile_covers_value(profile, 0.65, objective_delta=0.04)
    assert not profile_covers_value(profile, 0.95, objective_delta=0.04)


def test_profile_coverage_interpolates_objective_threshold_crossing() -> None:
    profile = profile_parameter(
        lambda unit: float((unit[0] - 0.5) ** 2),
        np.asarray([0.5]),
        parameter_index=0,
        steps=5,
        objective_delta=0.04,
    )

    assert profile_covers_value(profile, 0.70, objective_delta=0.04)


def test_profile_selection_covers_all_small_problems_and_required_large_parameters() -> None:
    small = _problem("component.0.thickness_a", "component.0.density_scale")

    assert select_profile_names(small) == (
        "component.0.thickness_a",
        "component.0.density_scale",
    )


def test_profile_selection_treats_twelve_parameter_layout_as_evidence_focused() -> None:
    names = (
        "component.0.thickness_a",
        "component.0.density_scale",
        "component.0.roughness_a",
        "backing.roughness_a",
        "instrument.angle_offset_deg",
        "instrument.scale",
        "instrument.background",
        "instrument.relative_sigma",
        "nuisance.0",
        "nuisance.1",
        "nuisance.2",
        "nuisance.3",
    )
    problem = SimpleNamespace(
        variables=tuple(SimpleNamespace(name=name) for name in names),
    )
    preliminary = SimpleNamespace(
        boundary_hits=("instrument.scale",),
        strong_correlations=(("nuisance.0", "nuisance.1", 0.99),),
    )

    assert select_profile_names(problem, preliminary) == (
        "component.0.thickness_a",
        "component.0.density_scale",
        "component.0.roughness_a",
        "backing.roughness_a",
        "instrument.angle_offset_deg",
        "instrument.scale",
        "nuisance.0",
        "nuisance.1",
    )


def test_build_report_profiles_use_residual_least_squares_path(monkeypatch) -> None:
    problem = _problem("component.0.thickness_a", "component.0.density_scale")
    unit = encode_physical_vector(problem, {})
    monkeypatch.setattr(
        _api(),
        "minimize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("direct problem profile used scalar minimize")
        ),
    )

    profile = build_problem_profile(
        problem,
        unit,
        "component.0.thickness_a",
    )

    assert profile.name == "component.0.thickness_a"
    assert np.isfinite(profile.objectives).any()


def test_problem_profile_batch_flattens_all_direction_scans_into_one_task_batch() -> None:
    module = _api()
    problem = _problem(
        "component.0.thickness_a",
        "component.0.density_scale",
    )
    unit = encode_physical_vector(problem, {})
    names = tuple(variable.name for variable in problem.variables)
    batch_sizes: list[int] = []

    def run_reversed(tasks):
        values = tuple(tasks)
        batch_sizes.append(len(values))
        results = [None] * len(values)
        for index in reversed(range(len(values))):
            results[index] = values[index]()
        return tuple(results)

    profiles = module.build_problem_profiles(
        problem,
        unit,
        names,
        task_runner=run_reversed,
    )
    expected = tuple(_api().build_problem_profile(problem, unit, name) for name in names)

    assert batch_sizes == [2 * len(names), len(names)]
    for observed, reference in zip(profiles, expected, strict=True):
        assert observed.name == reference.name
        np.testing.assert_array_equal(observed.values, reference.values)
        np.testing.assert_array_equal(observed.objectives, reference.objectives)
        assert observed.lower_closed is reference.lower_closed
        assert observed.upper_closed is reference.upper_closed


def test_default_profile_path_detects_barrier_between_coarse_grid_points() -> None:
    class NarrowBarrier:
        def __call__(self, unit: np.ndarray) -> float:
            value = float(np.mean(unit))
            return 0.0100 + 0.10 * np.exp(-((value - 0.4385) / 0.0006) ** 2)

    assert not default_profile_path_merge(
        NarrowBarrier(), np.full(4, 0.40), np.full(4, 0.47), 0.0102
    )
