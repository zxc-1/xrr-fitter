"""Joint confidence must consume the completed bootstrap failure evidence."""

import json
from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import joint_scale_searches

from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.io.codec_results import fit_result_from_dict, fit_result_to_dict
from xrr_fitter.model.analysis import ConfidenceClass


def _case(mode):
    problem, searches = joint_scale_searches(mode)
    members = tuple(
        replace(local, config=replace(local.config, budget=replace(local.config.budget, bootstrap_samples=200)))
        for local in problem.problems
    )
    return compile_joint_problem(problem.dataset_ids, members, problem.sharing_rules), searches


def _refit_failures(monkeypatch, fitting, failure_count):
    original = fitting.refit_resampled_joint
    attempts = []

    def refit(*args, **kwargs):
        attempts.append(len(attempts))
        if len(attempts) <= failure_count:
            return "optimizer_nonconvergence:max_nfev"
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, "refit_resampled_joint", refit)
    return attempts


def _assert_sampling(report, failure_count, attempts):
    sampling = report.bootstrap_evidence
    assert len(attempts) == sampling.attempted_count == 200
    assert sampling.successful_samples == 200 - failure_count
    assert report.bootstrap_performed is True
    assert report.bootstrap_failure_rate == failure_count / 200
    assert sampling.failure_reasons == tuple(
        (index, "optimizer_nonconvergence:max_nfev") for index in range(failure_count)
    )


@pytest.mark.parametrize("mode", ("gaussian", "poisson", "robust_log"))
@pytest.mark.parametrize("failure_count", (41, 200))
def test_joint_bootstrap_failures_downgrade_every_member_and_roundtrip(monkeypatch, mode, failure_count):
    fitting = import_module("xrr_fitter.services.fitting")
    problem, searches = _case(mode)
    baseline = fitting._analyze_joint_searches(problem, searches, ((), ()))
    assert baseline[0].confidence in (ConfidenceClass.TRUSTED, ConfidenceClass.CORRELATED)
    attempts = _refit_failures(monkeypatch, fitting, failure_count)

    results = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)

    _assert_sampling(results[0].uncertainty, failure_count, attempts)
    assert results[0].uncertainty is results[1].uncertainty
    for result in results:
        assert result.confidence is ConfidenceClass.UNTRUSTED
        assert result.classification_evidence == ("bootstrap_failure_rate",)
        payload = json.dumps(fit_result_to_dict(result), allow_nan=False)
        restored = fit_result_from_dict(json.loads(payload))
        assert json.dumps(fit_result_to_dict(restored), allow_nan=False) == payload


@pytest.mark.parametrize("mode", ("gaussian", "poisson", "robust_log"))
@pytest.mark.parametrize("failure_count", (0, 40))
def test_joint_bootstrap_at_or_below_threshold_keeps_existing_classification(monkeypatch, mode, failure_count):
    fitting = import_module("xrr_fitter.services.fitting")
    problem, searches = _case(mode)
    baseline = fitting._analyze_joint_searches(problem, searches, ((), ()))
    attempts = _refit_failures(monkeypatch, fitting, failure_count)

    results = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)

    _assert_sampling(results[0].uncertainty, failure_count, attempts)
    for before, after in zip(baseline, results, strict=True):
        assert after.confidence is before.confidence
        assert after.classification_evidence == before.classification_evidence
        assert after.uncertainty.member_residuals == before.uncertainty.member_residuals
        np.testing.assert_array_equal(after.uncertainty.covariance, before.uncertainty.covariance)


def test_disabled_joint_bootstrap_does_not_invoke_the_refitter(monkeypatch):
    fitting = import_module("xrr_fitter.services.fitting")
    problem, searches = _case("gaussian")
    attempts = _refit_failures(monkeypatch, fitting, 200)

    results = fitting._analyze_joint_searches(problem, searches, ((), ()))

    assert not attempts
    assert results[0].uncertainty.bootstrap_performed is False
    assert results[0].confidence is ConfidenceClass.TRUSTED
