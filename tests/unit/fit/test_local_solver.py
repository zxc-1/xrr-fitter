from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting


def _api():
    return import_module("xrr_fitter.fit.local_search")


def _problem(*, seed: int = 701, scale_prior: bool = False):
    config = replace(
        FitConfig.fast(seed),
        budget=SearchBudget(0, 0, 12, 2, 1),
        local_workers=1,
        scale_prior_enabled=scale_prior,
    )
    return compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )


def _fixed_problem(*, scale_prior: bool = False):
    problem = _problem(scale_prior=scale_prior)
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial,
            definition.initial,
            locked=True,
        )
        for definition in problem.parameter_definitions
    )
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )


def test_local_solver_reduces_objective_deterministically() -> None:
    api = _api()
    problem = _problem()
    start = np.full(len(problem.variables), 0.72)

    first = api.solve_local(problem, start, max_nfev=24)
    second = api.solve_local(problem, start, max_nfev=24)

    assert first.evaluation.valid
    assert first.evaluation.objective <= api.evaluate_vector(problem, start).objective
    np.testing.assert_array_equal(first.unit_vector, second.unit_vector)
    assert first.evaluation.objective == second.evaluation.objective
    assert first.stop_reason == second.stop_reason
    assert first.nfev == second.nfev


def test_local_solver_retains_the_start_when_scipy_returns_a_worse_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    problem = _problem()
    start = np.full(len(problem.variables), 0.4)
    worse = np.full(len(problem.variables), 0.9)
    start_evaluation = api.evaluate_vector(problem, start)
    worse_evaluation = api.evaluate_vector(problem, worse)
    assert start_evaluation.valid and worse_evaluation.valid
    assert worse_evaluation.objective > start_evaluation.objective

    monkeypatch.setattr(
        api,
        "least_squares",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=worse,
            message="synthetic worse result",
            nfev=7,
        ),
    )

    result = api.solve_local(problem, start, max_nfev=8)

    np.testing.assert_array_equal(result.unit_vector, start)
    assert result.evaluation.objective == start_evaluation.objective
    assert result.stop_reason == "local_objective_increased"
    assert result.nfev == 7


def test_local_solver_passes_an_analytic_jacobian_to_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    problem = _problem()
    start = np.full(len(problem.variables), 0.55)
    jacobian_calls = 0
    original_jacobian = api.evaluate_jacobian

    def audited_jacobian(problem_value, unit_value):
        nonlocal jacobian_calls
        jacobian_calls += 1
        return original_jacobian(problem_value, unit_value)

    def fake_least_squares(fun, x0, *, jac, **kwargs):
        del kwargs
        x = np.asarray(x0, dtype=float)
        residual = np.asarray(fun(x), dtype=float)
        analytic = np.asarray(jac(x), dtype=float)
        step = 1e-6
        finite = np.column_stack(
            [
                (
                    fun(x + np.eye(x.size)[index] * step)
                    - fun(x - np.eye(x.size)[index] * step)
                )
                / (2.0 * step)
                for index in range(x.size)
            ]
        )
        np.testing.assert_allclose(analytic, finite, rtol=2e-4, atol=2e-7)
        return SimpleNamespace(
            x=x,
            fun=residual,
            message="captured analytic jacobian",
            nfev=1,
            success=True,
            status=1,
            optimality=0.0,
            active_mask=np.zeros(x.size, dtype=int),
        )

    monkeypatch.setattr(api, "evaluate_jacobian", audited_jacobian)
    monkeypatch.setattr(api, "least_squares", fake_least_squares)

    result = api.solve_local(problem, start, max_nfev=5)

    assert result.nfev == 1
    assert jacobian_calls > 0


def test_local_solver_scales_trust_region_steps_from_the_analytic_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    problem = _problem()
    start = np.full(len(problem.variables), 0.55)
    captured: dict[str, object] = {}

    def fake_least_squares(_fun, x0, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            message="captured trust-region scaling",
            nfev=1,
        )

    monkeypatch.setattr(api, "least_squares", fake_least_squares)

    api.solve_local(problem, start, max_nfev=5)

    assert captured["x_scale"] == "jac"


def test_local_residual_and_jacobian_append_the_analytic_scale_prior_row() -> None:
    api = _api()
    problem = replace(_problem(scale_prior=True), scale_prior_center=1.0)
    unit = np.full(len(problem.variables), 0.6)

    residual = api.local_residual(problem, unit)
    jacobian = api.local_jacobian(problem, unit)

    fitted_count = int(np.count_nonzero(problem.data.fit_mask))
    assert residual.shape == (fitted_count + 1,)
    assert jacobian.shape == (fitted_count + 1, len(problem.variables))
    scale_index = next(
        index for index, coordinate in enumerate(problem.variables)
        if coordinate.name == "instrument.scale"
    )
    nonzero = np.flatnonzero(np.abs(jacobian[-1]) > 1e-12)
    np.testing.assert_array_equal(nonzero, [scale_index])
    step = 1e-6
    plus = unit.copy()
    minus = unit.copy()
    plus[scale_index] += step
    minus[scale_index] -= step
    finite = (api.local_residual(problem, plus)[-1] - api.local_residual(problem, minus)[-1]) / (
        2.0 * step
    )
    assert jacobian[-1, scale_index] == pytest.approx(finite, rel=1e-5, abs=1e-8)


def test_local_solver_uses_external_soft_l1_weights_and_a_gaussian_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    problem = replace(_problem(scale_prior=True), scale_prior_center=1.0)
    start = np.full(len(problem.variables), 0.55)

    def audited_solver(fun, x0, *, loss, **kwargs):
        del kwargs
        unit = np.asarray(x0, dtype=float)
        residual = np.asarray(fun(unit), dtype=float)
        rho = np.asarray(loss(residual**2), dtype=float)
        weights = problem.weights[problem.data.fit_mask]
        c_decades = problem.config.c_decades
        data_squared = residual[:-1] ** 2
        scaled = 1.0 + data_squared / c_decades**2

        np.testing.assert_allclose(
            rho[0, :-1],
            4.0 * weights**2 * c_decades**2 * (np.sqrt(scaled) - 1.0),
        )
        np.testing.assert_allclose(rho[1, :-1], 2.0 * weights**2 / np.sqrt(scaled))
        np.testing.assert_allclose(
            rho[2, :-1],
            -(weights**2 / c_decades**2) * scaled ** (-1.5),
        )
        np.testing.assert_allclose(rho[:, -1], (2.0 * residual[-1] ** 2, 2.0, 0.0))
        optimizer_objective = 0.5 * float(np.sum(rho[0])) / data_squared.size
        assert optimizer_objective == pytest.approx(
            api.evaluate_vector(problem, unit).objective,
            rel=1e-12,
            abs=1e-14,
        )
        return SimpleNamespace(
            x=unit,
            fun=residual,
            message="captured robust loss",
            nfev=1,
            success=True,
            status=1,
            optimality=0.0,
            active_mask=np.zeros(unit.size, dtype=int),
        )

    monkeypatch.setattr(api, "least_squares", audited_solver)

    result = api.solve_local(problem, start, max_nfev=5)

    assert result.nfev == 1


def test_local_solver_polls_cancellation_inside_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    problem = _problem()
    start = np.full(len(problem.variables), 0.5)
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 3

    def exercising_solver(fun, x0, **kwargs):
        del kwargs
        for _ in range(8):
            fun(np.asarray(x0, dtype=float))
        pytest.fail("solver returned after cancellation")

    monkeypatch.setattr(api, "least_squares", exercising_solver)

    with pytest.raises(api.SearchCancelled, match="cancel"):
        api.solve_local(problem, start, max_nfev=100, cancelled=cancelled)

    assert polls == 3


@pytest.mark.parametrize(
    "start",
    [
        pytest.param(np.asarray(0.5), id="scalar"),
        pytest.param(np.zeros((1, 8)), id="matrix"),
        pytest.param(np.zeros(7), id="wrong-width"),
        pytest.param(np.full(8, np.nan), id="nonfinite"),
        pytest.param(np.full(8, -0.01), id="below-unit-bounds"),
        pytest.param(np.full(8, 1.01), id="above-unit-bounds"),
    ],
)
def test_local_solver_rejects_invalid_unit_starts(start: np.ndarray) -> None:
    api = _api()

    with pytest.raises(ValueError, match="start|unit|shape|finite|bounds"):
        api.solve_local(_problem(), start, max_nfev=5)


def test_local_solver_evaluates_a_no_free_parameter_problem_once() -> None:
    api = _api()
    problem = _fixed_problem()

    result = api.solve_local(problem, np.empty(0), max_nfev=5)

    assert result.evaluation.valid
    assert result.stop_reason == "no_free_parameters"
    assert result.nfev == 1
    np.testing.assert_array_equal(result.unit_vector, np.empty(0))


def test_fit_search_preserves_fully_locked_lineage_through_stage_e() -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")

    result = pipeline.run_fit_search(
        pipeline.FitSearchRequest("fixed", _fixed_problem())
    )

    assert tuple(summary.stage for summary in result.stage_summaries) == (
        "A",
        "B",
        "C",
        "D",
        "E",
    )
    final = tuple(
        candidate
        for candidate in result.candidates
        if candidate.candidate_id.startswith("E-")
    )
    assert tuple(candidate.candidate_id for candidate in final) == (
        "E-0",
        "E-1",
        "E-2",
        "E-3",
    )
    assert all(candidate.stop_reason == "no_free_parameters" for candidate in final)
    assert all(candidate.unit_vector.size == 0 for candidate in final)
    assert result.best_candidate in final


def test_local_solver_propagates_unexpected_evaluation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    sentinel = RuntimeError("unexpected evaluator failure")

    def fail(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(api, "evaluate_vector", fail)

    with pytest.raises(RuntimeError, match="unexpected evaluator failure") as captured:
        api.solve_local(
            _problem(),
            np.full(8, 0.5),
            max_nfev=5,
        )

    assert captured.value is sentinel


def test_invalid_local_residual_and_jacobian_are_finite_and_shape_compatible() -> None:
    api = _api()
    problem = _problem()
    invalid = np.zeros(len(problem.variables))

    residual = api.local_residual(problem, invalid)
    jacobian = api.local_jacobian(problem, invalid)

    fitted_count = int(np.count_nonzero(problem.data.fit_mask))
    assert residual.shape == (fitted_count,)
    assert jacobian.shape == (fitted_count, len(problem.variables))
    np.testing.assert_array_equal(residual, np.full(fitted_count, 1e6))
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))
    assert np.all(np.isfinite(residual))
    assert np.all(np.isfinite(jacobian))


def test_locked_scale_has_a_zero_local_prior_jacobian_row() -> None:
    api = _api()
    problem = replace(_fixed_problem(scale_prior=True), scale_prior_center=1.0)

    residual = api.local_residual(problem, np.empty(0))
    jacobian = api.local_jacobian(problem, np.empty(0))

    fitted_count = int(np.count_nonzero(problem.data.fit_mask))
    assert residual.shape == (fitted_count + 1,)
    assert jacobian.shape == (fitted_count + 1, 0)
    assert jacobian[-1].size == 0
