"""Saved work must include both observed and null paths, including failures."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.diagnostic_work_cases import completed_work, failed_work
from tests.unit.analysis.test_residual_calibration_flow import _calibrate, _fixed_refit
from tests.unit.fit.test_diagnostic_refit import _context, _declared

from xrr_fitter.analysis import residual_calibration as coordinator
from xrr_fitter.evaluation import evaluate_model
from xrr_fitter.fit.diagnostic_refit import refit_diagnostic_joint, refit_diagnostic_single
from xrr_fitter.model.diagnostic_calibration import DiagnosticCalibration, DiagnosticRefit
from xrr_fitter.model.diagnostic_work import DiagnosticRefitWork, combine_refit_work


def test_real_locked_refits_keep_separate_observed_and_null_joint_histograms():
    evidence = _calibrate(refit=lambda problems: refit_diagnostic_single(problems[0]))[0]
    calibration = evidence.calibration
    assert hasattr(calibration, "observed_work"), "the coordinator must retain numerical work"
    assert calibration.status == "available"
    assert calibration.observed_work.path_count == 1
    assert calibration.null_work.path_count == 99
    assert calibration.observed_work.nfev == calibration.null_work.nfev == calibration.refit_nfev == 0
    assert calibration.observed_work.declared_paths == calibration.null_work.declared_paths == 1
    assert all(stage.kind == "zero_dimensional" for stage in calibration.null_work.paths[0].stages)


def test_escaped_numeric_error_is_not_recast_as_zero_work():
    def escaped(problems):
        _fixed_refit(problems)
        raise FloatingPointError("work already performed")

    with pytest.raises(FloatingPointError, match="work already performed"):
        _calibrate(refit=escaped)


def _pending(observed_work):
    return DiagnosticCalibration(
        "unavailable",
        99,
        5,
        "a" * 64,
        unavailable_reason="diagnostic_calibration_pending",
        observed_work=observed_work,
        refit_nfev=observed_work.nfev,
    )


def test_batch_keeps_work_before_and_after_failed_refits_and_generation_failures():
    complete = completed_work()
    prefix = combine_refit_work(completed_work(1), failed_work())
    batch = (
        coordinator._Replicate((1.0,), complete.nfev, work=complete),
        coordinator._Replicate(nfev=prefix.nfev, failure_reason="numeric failure", work=prefix),
        coordinator._Replicate(failure_reason="generation_failed:FloatingPointError"),
        coordinator._Replicate((2.0,), complete.nfev, work=complete),
    )
    calibration = coordinator._collect_batch(_pending(complete), batch)
    assert (calibration.attempted_count, calibration.successful_count) == (4, 2)
    assert calibration.failure_reasons == ((1, "numeric failure"), (2, "generation_failed:FloatingPointError"))
    assert calibration.observed_work == complete
    assert (calibration.null_work.path_count, calibration.null_work.completed_paths) == (10, 9)
    assert calibration.null_work.nfev == 28 and calibration.refit_nfev == 40


def test_generation_only_batch_keeps_nonzero_observed_work_without_null_work():
    observed = completed_work()
    batch = (coordinator._Replicate(failure_reason="generation_failed:FloatingPointError"),) * 16
    calibration = coordinator._collect_batch(_pending(observed), batch)
    assert calibration.attempted_count == 16 and calibration.successful_count == 0
    assert calibration.observed_work == observed and calibration.refit_nfev == 12
    assert calibration.null_work == DiagnosticRefitWork()


@pytest.mark.parametrize("failure", ["statistics", "family"])
def test_post_solver_statistic_failure_retains_every_completed_path(monkeypatch, failure):
    problem = _context("single")
    unit = _declared("single", problem)
    work = completed_work()
    fitted = DiagnosticRefit(unit, (evaluate_model(problem, unit),), work.nfev, 4, work=work)
    statistics = coordinator._all_statistics((problem,), ("curve",), fitted.evaluations)

    def changed_statistics(*_args):
        if failure == "statistics":
            raise FloatingPointError("unrepresentable statistic")
        return (replace(statistics[0], dataset_id="different"), *statistics[1:])

    monkeypatch.setattr(coordinator, "_all_statistics", changed_statistics)
    columns = tuple((value.dataset_id, value.kind) for value in statistics)
    result = coordinator._replicate((problem,), ("curve",), lambda _problems: fitted, None, columns)
    expected = "diagnostic_statistics_unavailable:" if failure == "statistics" else "diagnostic_family_changed"
    assert result.failure_reason.startswith(expected)
    assert result.work == work and result.nfev == work.nfev == 12
    assert result.values == ()


def test_joint_replicate_aggregates_its_global_path_work_only_once():
    problem = _context("joint")
    fitted = refit_diagnostic_joint(problem, problem.problems)
    assert fitted.failure_reason is None and fitted.nfev > 0
    statistics = coordinator._all_statistics(problem.problems, problem.dataset_ids, fitted.evaluations)
    columns = tuple((value.dataset_id, value.kind) for value in statistics)
    result = coordinator._replicate(
        problem.problems,
        problem.dataset_ids,
        lambda _problems: fitted,
        None,
        columns,
    )
    assert result.work == fitted.work and result.nfev == fitted.nfev
    calibration = coordinator._collect_batch(_pending(fitted.work), (result,))
    assert calibration.null_work.path_count == 4
    assert calibration.null_work.nfev == fitted.nfev
    assert calibration.refit_nfev == 2 * fitted.nfev
