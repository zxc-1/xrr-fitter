"""Basin recovery uses the same mode-aware analytic system as reported profiles."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import scale_problem

from xrr_fitter.analysis import profiles
from xrr_fitter.evaluation import encode_physical_vector, evaluate_model, least_squares_loss
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.parameters import ParameterSetting


def _problem(mode="robust_log"):
    base = scale_problem(mode)
    settings = tuple(
        ParameterSetting(
            item.name,
            item.initial,
            item.lower,
            item.upper,
            locked=item.name not in {"component.0.thickness_a", "instrument.scale"},
        )
        for item in base.parameter_definitions
    )
    return compile_fit_problem(base.data, base.structure, base.instrument, base.config, settings)


def _candidate(problem):
    unit = encode_physical_vector(problem, {"component.0.thickness_a": 30.0, "instrument.scale": 1.0})
    return SimpleNamespace(valid=True, objective=evaluate_model(problem, unit).objective, unit_vector=unit)


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
def test_basin_recovery_uses_bounded_analytic_system_for_each_noise_model(mode, monkeypatch) -> None:
    problem = _problem(mode)
    candidate = _candidate(problem)
    expected_loss = least_squares_loss(problem)
    original = profiles.least_squares
    budgets = []

    def reject_scalar(*_args, **_kwargs):
        raise AssertionError("problem basin recovery used scalar finite differences")

    def solve(residual, start, **options):
        squared = residual(start) ** 2
        np.testing.assert_array_equal(options["loss"](squared), expected_loss(squared))
        assert callable(options["jac"])
        budgets.append(options["max_nfev"])
        return original(residual, start, **options)

    monkeypatch.setattr(profiles, "minimize", reject_scalar)
    monkeypatch.setattr(profiles, "least_squares", solve)

    decision = profiles.recover_profile_basin(problem, candidate)

    assert budgets
    assert set(budgets) == {
        max(
            problem.config.budget.local_min_nfev,
            problem.config.budget.local_nfev_per_parameter * len(problem.variables),
        )
    }
    assert decision is not None
    assert decision.objective < candidate.objective
    assert evaluate_model(problem, decision.unit_vector).objective == pytest.approx(decision.objective)


def test_basin_recovery_does_not_swallow_analytic_system_errors(monkeypatch) -> None:
    problem = _problem()

    def broken(*_args):
        raise RuntimeError("analytic system failed")

    monkeypatch.setattr(profiles, "least_squares_system", broken)

    with pytest.raises(RuntimeError, match="analytic system failed"):
        profiles.recover_profile_basin(problem, _candidate(problem))


def test_basin_recovery_keeps_cancellation_inside_nuisance_solves() -> None:
    problem = _problem()
    polls = 0

    def cancelled():
        nonlocal polls
        polls += 1
        return polls > 6

    with pytest.raises(InterruptedError, match="cancelled"):
        profiles.recover_profile_basin(problem, _candidate(problem), cancelled=cancelled)


@pytest.mark.parametrize("valid,objective", [(False, 1.0), (True, float("inf")), (True, 0.0)])
def test_basin_recovery_keeps_ineligible_candidates_out_of_the_solver(valid, objective, monkeypatch) -> None:
    problem = _problem()
    candidate = _candidate(problem)
    candidate.valid, candidate.objective = valid, objective

    def unexpected(*_args, **_kwargs):
        raise AssertionError("ineligible candidate entered profile search")

    monkeypatch.setattr(profiles, "profile_parameter_with_decision", unexpected)

    assert profiles.recover_profile_basin(problem, candidate) is None
