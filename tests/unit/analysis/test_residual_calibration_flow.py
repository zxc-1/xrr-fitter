"""Monte Carlo diagnostics keep raw evidence, strict failure states, and one family."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module, util

import numpy as np
import pytest
from tests.support.diagnostic_work_cases import zero_work
from tests.unit.analysis.covariance_cases import scale_problem

from xrr_fitter.analysis.diagnostics import build_residual_evidence
from xrr_fitter.evaluation import evaluate_model
from xrr_fitter.fit.problem import compile_fit_problem, recompile_resampled_problem
from xrr_fitter.model.diagnostic_calibration import DiagnosticRefit
from xrr_fitter.model.parameters import ParameterFreedom, ParameterSetting


def _engine():
    name = "xrr_fitter.analysis.residual_calibration"
    assert util.find_spec(name) is not None, "Poisson residual calibration coordinator is required"
    return import_module(name)


def _locked_problem(*, seed=17, count=99, mode="poisson"):
    original = scale_problem(mode, seed=seed)
    settings = tuple(
        ParameterSetting(
            item.name,
            0.5 if item.name == "instrument.scale" else item.initial,
            item.lower,
            item.upper,
            freedom=ParameterFreedom.FIXED,
        )
        for item in original.parameter_definitions
    )
    config = replace(original.config, budget=replace(original.config.budget, diagnostic_samples=count))
    return compile_fit_problem(original.data, original.structure, original.instrument, config, settings)


def _fixed_refit(problems):
    unit = np.empty(0)
    return DiagnosticRefit(unit, tuple(evaluate_model(problem, unit) for problem in problems), 0, 1, work=zero_work())


def _calibrate(problems=None, **kwargs):
    problems = (_locked_problem(),) if problems is None else problems
    evaluations = tuple(evaluate_model(problem, np.empty(0)) for problem in problems)
    options = {"refit": _fixed_refit, "recompile": recompile_resampled_problem, **kwargs}
    return _engine().calibrate_residuals(
        problems,
        tuple(f"curve-{index}" for index in range(len(problems))),
        np.empty(0),
        evaluations,
        owner_sha256="a" * 64,
        **options,
    )


def test_low_level_positive_poisson_screen_is_unknown_without_calibration():
    problem = _locked_problem()
    evaluation = evaluate_model(problem, np.empty(0))
    residual = build_residual_evidence(problem, evaluation.fit_residuals)
    assert residual.raw_systematic is True
    assert residual.executed is False
    assert residual.systematic is residual.autocorrelation is None
    assert residual.calibration is None
    assert residual.unavailable_reason == "poisson_diagnostic_calibration_required"


def test_resampling_preserves_identical_masked_nan_coordinates():
    original = _locked_problem()
    mask = original.data.fit_mask.copy()
    qz = original.data.qz_a_inv.copy()
    mask[0], qz[0] = False, np.nan
    data = replace(original.data, qz_a_inv=qz, fit_mask=mask)
    problem = replace(original, data=data, objective_point_count=int(mask.sum()))
    restored = recompile_resampled_problem(problem, replace(data))
    np.testing.assert_array_equal(restored.data.qz_a_inv, problem.data.qz_a_inv)
    np.testing.assert_array_equal(restored.data.fit_mask, mask)


def test_known_low_count_null_keeps_advisory_but_does_not_mislabel_misspecification():
    evidence = _calibrate()[0]
    assert evidence.raw_systematic is True
    assert "suspected_diffuse_background" in {item.code for item in evidence.advisories}
    assert evidence.executed is True
    assert evidence.systematic is False
    assert evidence.autocorrelation is False
    assert evidence.diagnostics == ()


def test_completed_calibration_retains_the_full_sampling_account_and_owner():
    evidence = _calibrate()[0]
    assert evidence.calibration.status == "available"
    assert evidence.calibration.attempted_count == evidence.calibration.successful_count == 99
    assert evidence.calibration.p_value > 0.01
    assert evidence.calibration.owner_sha256 == evidence.owner_sha256


def test_bounded_batches_keep_fixed_sample_count_and_serial_parallel_identity():
    calls, batch_sizes = [], []

    def refit(problems):
        calls.append(tuple(problem.data.intensity_raw.tobytes() for problem in problems))
        return _fixed_refit(problems)

    def reverse_execution(tasks):
        batch_sizes.append(len(tasks))
        results = [task() for task in reversed(tasks)]
        return tuple(reversed(results))

    serial = _calibrate(refit=refit)[0]
    parallel = _calibrate(task_runner=reverse_execution)[0]
    assert len(calls) == 100  # one observed and all 99 predeclared null fits
    assert sum(batch_sizes) == 99 and max(batch_sizes) <= 16
    assert serial == parallel
    assert serial.calibration.refit_nfev == 0


def test_budget_shortfall_never_publishes_a_formal_diagnostic_pass():
    def forbidden(_problems):
        pytest.fail("insufficient budget must not start the refitter")

    evidence = _calibrate((_locked_problem(count=98),), refit=forbidden)[0]
    assert evidence.executed is False
    assert evidence.systematic is evidence.autocorrelation is None
    assert evidence.calibration.status == "unavailable"
    assert evidence.unavailable_reason == "diagnostic_budget_insufficient"
    assert evidence.calibration.attempted_count == 0
    assert evidence.calibration.p_value is None


@pytest.mark.parametrize(
    ("field", "reason"),
    [("refit", "diagnostic_refitter_unavailable"), ("recompile", "diagnostic_compiler_unavailable")],
)
def test_missing_runtime_capability_withholds_calibration(field, reason):
    evidence = _calibrate(**{field: None})[0]
    assert evidence.executed is False
    assert evidence.unavailable_reason == reason
    assert evidence.calibration.p_value is None


def test_a_failed_null_is_retained_without_replacement_or_partial_probability():
    calls = 0

    def refit(problems):
        nonlocal calls
        calls += 1
        if calls == 7:
            # A locked path has no optimizer; it can still fail at publication.
            return DiagnosticRefit(None, (), 0, 1, "publication_failure", zero_work())
        return _fixed_refit(problems)

    evidence = _calibrate(refit=refit)[0]
    calibration = evidence.calibration
    assert calibration.failure_reasons == ((5, "publication_failure"),)
    assert calibration.attempted_count == 16
    assert calibration.successful_count == 15
    assert calls == 17
    assert calibration.refit_nfev == 0
    assert calibration.p_value is calibration.tail_count is calibration.tie_count is None
    assert not evidence.executed


def test_programming_errors_are_not_converted_to_unavailable():
    def broken(_problems):
        raise TypeError("implementation defect")

    with pytest.raises(TypeError, match="implementation defect"):
        _calibrate(refit=broken)


def test_observed_prediction_mismatch_is_not_hidden_by_equal_objective():
    def wrong_mean(problems):
        result = _fixed_refit(problems)
        evaluation = result.evaluations[0]
        changed = replace(evaluation, model_normalized=evaluation.model_normalized * 1.1)
        return replace(result, evaluations=(changed,))

    evidence = _calibrate(refit=wrong_mean)[0]
    assert evidence.unavailable_reason == "diagnostic_refit_mismatch"
    assert evidence.calibration.attempted_count == 0
    assert evidence.calibration.failure_reasons == ((-1, "diagnostic_refit_mismatch"),)
    assert evidence.calibration.refit_discrepancy[2] > 1e-6


def test_observed_statistics_come_from_the_observed_refit_not_product_residuals():
    calls = 0
    observed = None

    def refit(problems):
        nonlocal calls, observed
        calls += 1
        result = _fixed_refit(problems)
        if calls == 1:
            value = result.evaluations[0]
            residual = value.fit_residuals + np.linspace(0, 1e-7, value.fit_residuals.size)
            observed = residual
            result = replace(result, evaluations=(replace(value, fit_residuals=residual),))
        return result

    problem = _locked_problem()
    evidence = _calibrate((problem,), refit=refit)[0]
    statistics = import_module("xrr_fitter.analysis.residual_statistics")
    expected = statistics.residual_statistics(problem, observed, dataset_id="curve-0")
    assert tuple(item.observed for item in evidence.calibration.statistics) == tuple(item.observed for item in expected)


def test_joint_members_share_one_complete_family_and_one_refit_per_replicate():
    calls = []

    def refit(problems):
        calls.append(len(problems))
        return _fixed_refit(problems)

    evidence = _calibrate((_locked_problem(), _locked_problem(seed=18)), refit=refit)
    assert calls == [2] * 100
    assert evidence[0].calibration is evidence[1].calibration
    calibration = evidence[0].calibration
    assert {item.dataset_id for item in calibration.statistics} == {"curve-0", "curve-1"}
    assert any(item.systematic for item in evidence) == calibration.rejected


def test_mixed_noise_joint_cannot_claim_a_poisson_null_calibration():
    evidence = _calibrate((_locked_problem(), _locked_problem(mode="gaussian")))
    assert evidence[0].unavailable_reason == "mixed_noise_diagnostic_calibration_unsupported"
    assert evidence[0].calibration.p_value is None
    assert evidence[1].calibration is None
    assert evidence[1].executed


def test_cancellation_is_propagated_before_any_probability_publication():
    with pytest.raises(InterruptedError, match="cancelled"):
        _calibrate(cancelled=lambda: True)
