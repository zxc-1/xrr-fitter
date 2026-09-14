"""Sampling-work provenance in composed uncertainty reports."""

from __future__ import annotations

import numpy as np
from tests.unit.analysis.test_report import _api, _candidate, _problem

from xrr_fitter.model.bootstrap import BootstrapResult


def test_build_report_reads_actual_attempts_from_bootstrap_evidence() -> None:
    """Attempts belong to this sampling record, not the configuration budget."""
    problem = _problem()
    candidate = _candidate(problem, "E-0")
    names = tuple(variable.name for variable in problem.variables)
    bootstrap = BootstrapResult(
        names,
        np.zeros((192, len(names))),
        (),
        0.04,
        200,
        tuple((index, "fixture_fit_failed") for index in range(192, 200)),
        unavailable_reason="insufficient_successful_samples",
    )

    report = _api().build_uncertainty_report(problem, (candidate,), bootstrap=bootstrap)

    assert report.bootstrap_sample_count == 200
    assert report.bootstrap_performed is True


def test_build_report_records_no_resample_count_without_a_bootstrap() -> None:
    """没跑自助抽样就不能编一个次数出来：0 读作「未记录」，与 ``bootstrap_performed=False`` 自洽。"""
    problem = _problem()
    candidate = _candidate(problem, "E-0")

    report = _api().build_uncertainty_report(problem, (candidate,), profile_names=())

    assert report.bootstrap_sample_count == 0


def test_build_report_survives_a_bootstrap_that_lost_every_sample() -> None:
    """Every failed attempt remains countable even when no sample survives."""
    problem = _problem()
    candidate = _candidate(problem, "E-0")
    names = tuple(variable.name for variable in problem.variables)
    bootstrap = BootstrapResult(
        names,
        np.zeros((0, len(names))),
        (),
        1.0,
        200,
        tuple((index, "fixture_fit_failed") for index in range(200)),
        unavailable_reason="excessive_fit_failures",
    )

    report = _api().build_uncertainty_report(problem, (candidate,), bootstrap=bootstrap)

    assert report.bootstrap_sample_count == 200
    assert report.bootstrap_performed is True
