"""Failure accounting and numerical boundaries of the same-estimator coordinator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.diagnostic_work_cases import completed_work, failed_work, zero_work
from tests.unit.analysis.test_residual_calibration_flow import _calibrate, _engine, _fixed_refit, _locked_problem

from xrr_fitter.analysis.diagnostics import build_residual_evidence
from xrr_fitter.analysis.report import uncertainty_seed
from xrr_fitter.evaluation import evaluate_model
from xrr_fitter.fit.problem import recompile_resampled_problem
from xrr_fitter.model.diagnostic_calibration import DiagnosticRefit
from xrr_fitter.model.diagnostic_work import combine_refit_work
from xrr_fitter.model.instrument import PhysicsDiagnostic


def test_generation_failures_keep_every_attempted_index_without_replacement():
    calls = []

    def refit(problems):
        calls.append(True)
        return _fixed_refit(problems)

    def recompile(_problem, _data):
        raise FloatingPointError("generated count is unrepresentable")

    evidence = _calibrate(refit=refit, recompile=recompile)[0]
    calibration = evidence.calibration
    assert calls == [True]
    assert (calibration.attempted_count, calibration.successful_count) == (16, 0)
    assert tuple(index for index, _reason in calibration.failure_reasons) == tuple(range(16))
    assert all(
        reason.startswith("generation_failed:FloatingPointError:") for _index, reason in calibration.failure_reasons
    )
    assert calibration.p_value is None and evidence.executed is False


def test_an_observed_failure_preserves_work_and_never_generates_nulls():
    def refit(_problems):
        return DiagnosticRefit(
            None, (), 7, 2, "observed_numerical_failure", combine_refit_work(completed_work(1), failed_work(4))
        )

    def forbidden(*_args):
        pytest.fail("a failed observed refit requested synthetic data")

    evidence = _calibrate(refit=refit, recompile=forbidden)[0]
    calibration = evidence.calibration
    assert calibration.failure_reasons == ((-1, "observed_numerical_failure"),)
    assert calibration.refit_nfev == 7
    assert calibration.attempted_count == 0
    assert evidence.unavailable_reason == "observed_numerical_failure"


def test_finite_but_unrepresentable_observed_statistic_is_not_a_zero_score():
    def refit(problems):
        fitted = _fixed_refit(problems)
        value = fitted.evaluations[0]
        residuals = np.full(value.fit_residuals.size, 1e308)
        residuals[-20:] = -1e308
        value = replace(value, fit_residuals=residuals)
        return replace(fitted, evaluations=(value,))

    evidence = _calibrate(refit=refit)[0]
    assert evidence.unavailable_reason.startswith("diagnostic_statistics_unavailable:")
    assert evidence.calibration.failure_reasons[0][0] == -1
    assert evidence.calibration.attempted_count == 0
    assert evidence.calibration.p_value is None


def test_joint_members_cannot_silently_use_different_diagnostic_budgets():
    evidence = _calibrate((_locked_problem(count=99), _locked_problem(seed=18, count=199)))
    assert all(member.unavailable_reason == "joint_diagnostic_budget_mismatch" for member in evidence)
    assert evidence[0].calibration is evidence[1].calibration
    assert evidence[0].calibration.attempted_count == 0


@pytest.mark.parametrize("sampling_field", ("objective_count", "multipliers"))
def test_calibration_refuses_to_substitute_a_reduced_observation_grid(sampling_field):
    problem = _locked_problem()
    if sampling_field == "objective_count":
        problem = replace(problem, objective_point_count=problem.objective_point_count + 1)
    else:
        multipliers = problem.sampling_multipliers.copy()
        multipliers[0] = 2.0
        problem = replace(problem, sampling_multipliers=multipliers)
    evidence = _calibrate((problem,))[0]
    assert evidence.unavailable_reason == "diagnostic_requires_full_observation_grid"
    assert not evidence.executed


def test_true_physical_warning_survives_nonrejection_and_is_not_an_advisory():
    problem = _locked_problem()
    diagnostic = PhysicsDiagnostic("nevot_croce_applicability_exceeded", "roughness outside applicability")
    value = replace(evaluate_model(problem, np.empty(0)), diagnostics=(diagnostic,))
    evidence = _engine().calibrate_residuals(
        (problem,),
        ("curve",),
        np.empty(0),
        (value,),
        owner_sha256="a" * 64,
        refit=_fixed_refit,
        recompile=recompile_resampled_problem,
    )[0]
    assert evidence.calibration.rejected is False
    assert evidence.diagnostics == (diagnostic,)
    assert diagnostic not in evidence.advisories


def test_gaussian_screen_keeps_its_original_effective_conclusions():
    problem = _locked_problem(mode="gaussian")
    evaluation = evaluate_model(problem, np.empty(0))
    expected = build_residual_evidence(problem, evaluation.fit_residuals, dataset_id="curve-0")
    evidence = _calibrate((problem,))[0]
    assert (evidence.executed, evidence.systematic, evidence.autocorrelation) == (
        expected.executed,
        expected.systematic,
        expected.autocorrelation,
    )
    assert evidence.diagnostics == expected.diagnostics
    assert evidence.calibration is None


def test_diagnostic_seed_and_draws_are_separate_from_interval_bootstrap_budget():
    problem = _locked_problem()
    changed = replace(
        problem, config=replace(problem.config, budget=replace(problem.config.budget, bootstrap_samples=200))
    )
    first = _calibrate((problem,))[0].calibration
    second = _calibrate((changed,))[0].calibration
    assert first.child_seed == second.child_seed
    assert first.child_seed != uncertainty_seed(problem.config)
    assert first.null_statistics_sha256 == second.null_statistics_sha256
    assert first.statistics == second.statistics


def test_symmetric_poisson_kl_uses_count_space_and_no_factor_of_one_half():
    problem = _locked_problem()
    evaluation = evaluate_model(problem, np.empty(0))
    first = replace(evaluation, model_normalized=np.full(80, 1.0 / problem.data.normalization))
    second = replace(evaluation, model_normalized=np.full(80, 2.0 / problem.data.normalization))
    observed = DiagnosticRefit(np.empty(0), (second,), 0, 1, work=zero_work())
    distances = _engine().refit_discrepancy((problem,), np.empty(0), (first,), observed)
    assert distances == pytest.approx((0.0, 0.0, 80 * np.log(2.0)))


def test_incomplete_task_runner_is_a_programming_error_not_fake_null_success():
    with pytest.raises(RuntimeError, match="unexpected result count"):
        _calibrate(task_runner=lambda _tasks: ())


@pytest.mark.parametrize(
    ("observed", "first_null", "reason"),
    [
        (np.nextafter(0.0, 1.0), 0.0, "RMS scale is not representable"),
        (1e308, np.nextafter(0.0, 1.0), "RMS normalization lost a nonzero value"),
    ],
)
def test_rms_numeric_failure_withholds_probability_after_complete_null_work(
    monkeypatch,
    observed,
    first_null,
    reason,
):
    engine = _engine()
    original = engine._all_statistics
    calls = []

    def extreme_statistics(*args):
        values = original(*args)
        value = observed if not calls else first_null if len(calls) == 1 else 0.0
        calls.append(value)
        return tuple(replace(item, observed=value if index == 0 else 0.0) for index, item in enumerate(values))

    monkeypatch.setattr(engine, "_all_statistics", extreme_statistics)
    evidence = _calibrate()[0]
    calibration = evidence.calibration
    assert len(calls) == 100
    assert calibration.attempted_count == calibration.successful_count == 99
    assert calibration.status == "unavailable"
    assert evidence.unavailable_reason == f"diagnostic_statistics_unavailable:diagnostic {reason}"
    assert evidence.executed is False
    assert calibration.p_value is calibration.observed_score is calibration.tail_count is None
    assert all(item.scale is item.center is item.adjusted_p_value is None for item in calibration.statistics)
