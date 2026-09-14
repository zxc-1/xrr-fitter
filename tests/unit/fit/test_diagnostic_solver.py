"""Bounded phase transitions are independent of SciPy's advisory work counters."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

from xrr_fitter.fit.local_search import SearchCancelled


def _api():
    name = "xrr_fitter.fit.diagnostic_solver"
    assert find_spec(name) is not None, "the fixed two-stage kernel is not implemented"
    return import_module(name)


def _system(unit):
    return unit - 0.3, np.eye(unit.size)


def _loss(squared):
    return np.vstack((2.0 * squared, np.full(squared.size, 2.0), np.zeros(squared.size)))


def _result(unit, *, success=True, status=1, message="test stop", nfev=1):
    return SimpleNamespace(x=np.array(unit, copy=True), success=success, status=status, message=message, nfev=nfev)


def _solver(api, *, validate=lambda _unit: None, system=_system, loss=_loss, budget=9, cancelled=None):
    return api.TwoStageSolver(system, loss, validate, budget=budget, poisson=True, cancelled=cancelled)


def _localize_once(fun, initial, *, callback, **_kwargs):
    fun(initial)
    return _result(initial, status=0, message="CONVERGENCE", nfev=1)


def _refine_once(fun, initial, *, jac, **_kwargs):
    fun(initial)
    jac(initial)
    return _result(initial)


@pytest.mark.parametrize("stop", ["success", "abnormal", "cap"])
def test_every_localization_exit_uses_the_fixed_accepted_handoff_and_shared_budget(monkeypatch, stop):
    api = _api()
    handoffs, refinements = [], []
    solver = _solver(api, validate=lambda unit: handoffs.append(unit.copy()))

    def localize(fun, initial, *, callback, **kwargs):
        assert kwargs == {
            "method": "L-BFGS-B",
            "jac": True,
            "bounds": [(0.0, 1.0)],
            "options": {"ftol": 1e-10, "gtol": 1e-10, "maxfun": 4, "maxiter": 4},
        }
        q, gradient = fun(initial)
        assert q == pytest.approx(0.25)
        np.testing.assert_allclose(gradient, [1.0])
        accepted = np.array([0.4])
        fun(accepted)
        callback(accepted)
        accepted[:] = 0.9
        fun(np.array([0.1]))  # Rejected trial must never become the handoff.
        if stop == "cap":
            while True:
                fun(np.array([0.1]))
        return _result(
            [0.4],
            success=stop == "success",
            status=0 if stop == "success" else 2,
            message="CONVERGENCE" if stop == "success" else "ABNORMAL",
            nfev=900,
        )

    def refine(fun, initial, *, jac, loss, **kwargs):
        refinements.append(initial.copy())
        expected_spent = 5 if stop == "cap" else 4
        assert solver.nfev == expected_spent
        assert kwargs == {
            "bounds": (0.0, 1.0),
            "method": "trf",
            "x_scale": "jac",
            "ftol": None,
            "xtol": 1e-10,
            "gtol": 1e-10,
            "max_nfev": 9 - expected_spent,
            "callback": kwargs["callback"],
        }
        np.testing.assert_allclose(fun(initial), [0.1])
        np.testing.assert_array_equal(jac(initial), [[1.0]])
        assert loss(np.asarray([0.01]))[0, 0] == 0.02
        kwargs["callback"](initial)
        return _result(initial, nfev=800)

    monkeypatch.setattr(api, "minimize", localize)
    monkeypatch.setattr(api, "least_squares", refine)
    unit, failure = solver.solve(np.array([0.8]))
    assert failure is None
    np.testing.assert_array_equal(handoffs, [[0.4]])
    np.testing.assert_array_equal(refinements, [[0.4]])
    np.testing.assert_array_equal(unit, [0.4])
    assert solver.nfev == (6 if stop == "cap" else 5)
    assert [item.phase for item in solver.exits] == ["localize", "handoff", "refine"]
    assert sum(item.nfev for item in solver.exits) == solver.nfev
    assert solver.exits[0].status == (None if stop == "cap" else (0 if stop == "success" else 2))
    assert solver.exits[0].success is (stop == "success")
    assert (
        solver.exits[0].message
        == {
            "success": "CONVERGENCE",
            "abnormal": "ABNORMAL",
            "cap": "diagnostic_localization_budget_exhausted",
        }[stop]
    )


def test_first_evaluated_declaration_is_the_initial_accepted_point_not_the_last_trial(monkeypatch):
    api = _api()
    handoffs = []
    solver = _solver(api, budget=5, validate=lambda unit: handoffs.append(unit.copy()))

    def no_step(fun, initial, **_kwargs):
        fun(initial)
        while True:
            fun(np.array([0.2]))

    monkeypatch.setattr(api, "minimize", no_step)
    monkeypatch.setattr(api, "least_squares", _refine_once)
    unit, failure = solver.solve(np.array([0.8]))
    assert failure is None
    np.testing.assert_array_equal(handoffs, [[0.8]])
    np.testing.assert_array_equal(unit, [0.8])
    assert solver.nfev == 4


@pytest.mark.parametrize("budget", [1, 2])
def test_insufficient_two_stage_budget_is_explicit_without_extra_requests(monkeypatch, budget):
    api = _api()
    monkeypatch.setattr(api, "minimize", lambda *_a, **_k: pytest.fail("must not localize"))
    monkeypatch.setattr(api, "least_squares", lambda *_a, **_k: pytest.fail("must not refine"))
    solver = _solver(api, budget=budget)
    unit, failure = solver.solve(np.array([0.8]))
    assert unit is None and "insufficient_two_stage_budget" in failure
    assert solver.nfev == 0


def test_locked_kernel_does_not_spend_a_handoff_or_optimizer_request(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "minimize", lambda *_a, **_k: pytest.fail("must not localize"))
    monkeypatch.setattr(api, "least_squares", lambda *_a, **_k: pytest.fail("must not refine"))
    solver = _solver(api, validate=lambda _x: pytest.fail("publication belongs to caller"))
    unit, failure = solver.solve(np.array([]))
    assert failure is None and unit.size == 0
    assert solver.nfev == 0 and not solver.exits


@pytest.mark.parametrize("failure", ["invalid_model", "incomplete_axes", "nonfinite_reporting_axes"])
def test_invalid_handoff_never_uses_another_valid_point(monkeypatch, failure):
    api = _api()
    monkeypatch.setattr(api, "minimize", _localize_once)
    monkeypatch.setattr(api, "least_squares", lambda *_a, **_k: pytest.fail("invalid handoff cannot refine"))
    solver = _solver(api, validate=lambda _x: failure)
    unit, reason = solver.solve(np.array([0.8]))
    assert unit is None and reason == failure
    assert solver.nfev == 2
    assert [item.phase for item in solver.exits] == ["localize", "handoff"]
    assert not solver.exits[-1].success


@pytest.mark.parametrize("component", ["residual", "jacobian", "cost", "gradient"])
def test_nonfinite_complete_score_aborts_instead_of_handing_off_a_previous_point(monkeypatch, component):
    api = _api()
    calls = []

    def system(unit):
        calls.append(unit.copy())
        residual, jacobian = _system(unit)
        if len(calls) == 2:
            if component == "residual":
                residual[:] = np.nan
            elif component == "jacobian":
                jacobian[:] = np.inf
            elif component == "gradient":
                jacobian[:] = 1e308
                residual[:] = 2.0
        return residual, jacobian

    def loss(squared):
        rho = _loss(squared)
        if component == "cost" and len(calls) == 2:
            rho[0] = np.inf
        return rho

    def localize(fun, initial, **_kwargs):
        fun(initial)
        fun(np.array([0.5]))
        pytest.fail("nonfinite score must stop")

    monkeypatch.setattr(api, "minimize", localize)
    solver = _solver(api, system=system, loss=loss, validate=lambda _x: pytest.fail("must not hand off"))
    with pytest.raises(FloatingPointError):
        solver.solve(np.array([0.8]))
    assert solver.nfev == 2
    assert len(solver.exits) == 1 and not solver.exits[0].success
    assert "FloatingPointError" in solver.exits[0].message


@pytest.mark.parametrize("phase", ["localize", "handoff", "refine"])
def test_numerical_exception_retains_the_actual_work_and_phase(monkeypatch, phase):
    api = _api()

    def error(*_args, **_kwargs):
        raise FloatingPointError("numeric phase failure")

    def localize(fun, initial, **kwargs):
        fun(initial)
        return error() if phase == "localize" else _localize_once(fun, initial, **kwargs)

    def refine(fun, initial, **_kwargs):
        fun(initial)
        error()

    monkeypatch.setattr(api, "minimize", localize)
    monkeypatch.setattr(api, "least_squares", refine)
    solver = _solver(api, validate=error if phase == "handoff" else lambda _x: None)
    with pytest.raises(FloatingPointError, match="numeric phase failure"):
        solver.solve(np.array([0.8]))
    assert solver.nfev == {"localize": 1, "handoff": 3, "refine": 4}[phase]
    assert solver.exits[-1].phase == phase and not solver.exits[-1].success
    assert sum(item.nfev for item in solver.exits) == solver.nfev


@pytest.mark.parametrize("error", [ValueError("programming"), TypeError("programming"), RuntimeError("programming")])
def test_programming_errors_propagate_unchanged(monkeypatch, error):
    api = _api()

    def broken(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(api, "minimize", _localize_once)
    solver = _solver(api, validate=broken)
    with pytest.raises(type(error)) as caught:
        solver.solve(np.array([0.8]))
    assert caught.value is error


@pytest.mark.parametrize("boundary", ["system", "accepted", "localize_return", "handoff", "refine_return"])
def test_cancellation_never_becomes_stage_success(monkeypatch, boundary):
    api = _api()
    cancelled = [False]

    def system(unit):
        if boundary == "system":
            cancelled[0] = True
        return _system(unit)

    def localize(fun, initial, *, callback, **_kwargs):
        fun(initial)
        if boundary == "accepted":
            cancelled[0] = True
            callback(initial)
        if boundary == "localize_return":
            cancelled[0] = True
        return _result(initial, status=0)

    def validate(_unit):
        if boundary == "handoff":
            cancelled[0] = True

    def refine(fun, initial, **_kwargs):
        fun(initial)
        if boundary == "refine_return":
            cancelled[0] = True
        return _result(initial)

    monkeypatch.setattr(api, "minimize", localize)
    monkeypatch.setattr(api, "least_squares", refine)
    solver = _solver(api, system=system, validate=validate, cancelled=lambda: cancelled[0])
    with pytest.raises(SearchCancelled):
        solver.solve(np.array([0.8]))


def test_final_trf_failure_is_not_relabelled_from_localization_success(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "minimize", _localize_once)

    def refine(fun, initial, **_kwargs):
        fun(initial)
        return _result(initial, success=False, status=0, message="maximum evaluations")

    monkeypatch.setattr(api, "least_squares", refine)
    solver = _solver(api)
    unit, reason = solver.solve(np.array([0.8]))
    assert unit is None and reason == "diagnostic_refit_nonconverged:maximum evaluations"
    assert solver.nfev == 3
    assert solver.exits[0].success and not solver.exits[-1].success


def test_refinement_cannot_exceed_remaining_budget_even_if_solver_ignores_its_option(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "minimize", _localize_once)

    def refine(fun, initial, **_kwargs):
        while True:
            fun(initial)

    monkeypatch.setattr(api, "least_squares", refine)
    solver = _solver(api, budget=5)
    unit, reason = solver.solve(np.array([0.8]))
    assert unit is None and "refinement_budget_exhausted" in reason
    assert solver.nfev == 5
    assert [item.nfev for item in solver.exits] == [1, 1, 3]


def test_nonfinite_jacobian_on_a_rejected_trf_trial_aborts_the_path(monkeypatch):
    """TRF may reject residuals without requesting their cached Jacobian."""
    api = _api()
    nonfinite_trials = []

    def system(unit):
        if unit[0] <= 0.4:
            return unit - 0.8, np.ones((1, 1))
        nonfinite_trials.append(unit.copy())
        return 10.0 + unit, np.full((1, 1), np.nan)

    monkeypatch.setattr(api, "minimize", _localize_once)
    solver = _solver(api, system=system, budget=80)
    with pytest.raises(FloatingPointError, match="nonfinite diagnostic Jacobian"):
        solver.solve(np.array([0.2]))
    assert len(nonfinite_trials) == 1
    assert solver.exits[-1].phase == "refine" and not solver.exits[-1].success


def test_jacobian_cache_miss_cannot_execute_uncharged_model_work(monkeypatch):
    api = _api()
    traversals = []

    def system(unit):
        traversals.append(unit.copy())
        return _system(unit)

    def refine(fun, initial, *, jac, **_kwargs):
        fun(initial)
        assert solver.nfev == 3
        jac(np.array([0.7]))
        return _result(initial)

    monkeypatch.setattr(api, "minimize", _localize_once)
    monkeypatch.setattr(api, "least_squares", refine)
    solver = _solver(api, system=system, budget=3)
    with pytest.raises(RuntimeError, match="charged residual"):
        solver.solve(np.array([0.8]))
    assert solver.nfev == 3
    assert len(traversals) == 1
