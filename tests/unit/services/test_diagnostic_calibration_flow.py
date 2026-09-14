"""The service owns refits; reports, profiles, and automatic gates reuse evidence."""

from __future__ import annotations

import pickle
from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import scale_problem

from xrr_fitter.analysis.automatic import assess_automatic_quality
from xrr_fitter.analysis.report import AnalysisRequest, build_uncertainty_report
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.local_search import solve_local
from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.fitting import FitSearchResult, FitStageSummary
from xrr_fitter.model.provenance import diagnostic_calibration_sha256, fit_search_provenance_sha256


def _scale_search(count=99):
    original = scale_problem("poisson")
    budget = replace(original.config.budget, diagnostic_samples=count, local_min_nfev=80)
    problem = replace(original, config=replace(original.config, budget=budget, profile_steps=9))
    initial = encode_physical_vector(problem, {"instrument.scale": 0.5})
    solved = solve_local(problem, initial, max_nfev=80)
    candidate = candidate_from_evaluation(
        problem,
        solved.unit_vector,
        solved.evaluation,
        candidate_id="E-0",
        seed_index=0,
        stop_reason=solved.stop_reason,
        nfev=solved.nfev,
    )
    search = FitSearchResult(
        problem.parameter_definitions,
        (candidate,),
        0,
        (),
        (17,),
        (FitStageSummary("E", ("E-0",), candidate.objective, solved.nfev, (solved.stop_reason,)),),
        problem.region_labels,
        problem.weights,
    )
    return problem, replace(search, provenance_sha256=fit_search_provenance_sha256(problem, search))


def _service():
    service = import_module("xrr_fitter.services.fitting")
    assert hasattr(service, "refit_diagnostic_single"), "service must compose the diagnostic refitter"
    return service


def _result(problem, search, **kwargs):
    return _service().run_analysis(AnalysisRequest("curve", problem, search, bootstrap_enabled=False, **kwargs))


def test_preliminary_report_and_profile_share_exactly_one_calibration(monkeypatch):
    service, report_module = _service(), import_module("xrr_fitter.analysis.report")
    problem, search = _scale_search()
    refit, covariance = service.refit_diagnostic_single, report_module.problem_covariance
    calls = {"refit": 0, "covariance": 0}

    def counted_refit(*args, **kwargs):
        calls["refit"] += 1
        return refit(*args, **kwargs)

    def counted_covariance(*args, **kwargs):
        calls["covariance"] += 1
        return covariance(*args, **kwargs)

    monkeypatch.setattr(service, "refit_diagnostic_single", counted_refit)
    monkeypatch.setattr(report_module, "problem_covariance", counted_covariance)
    monkeypatch.setattr(report_module, "_evidence_focused_layout", lambda _problem: True)
    result = _result(problem, search, profile_names=None)
    member = result.uncertainty.member_residuals[0]
    assert member.raw_systematic is True and member.systematic is False
    assert member.calibration.status == "available"
    assert calls == {"refit": 100, "covariance": 1}
    assert result.uncertainty.profiles[0].confidence_level == 0.95
    assert not result.best_candidate.diagnostics


def test_cached_evidence_is_pickle_safe_and_final_report_does_not_repeat_mc(monkeypatch):
    problem, search = _scale_search()
    first = _result(problem, search, profile_names=())
    evidence = first.uncertainty.member_residuals[0]
    assert "residual_evidence" in AnalysisRequest.__dataclass_fields__, (
        "analysis requests must carry immutable residual evidence"
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("cached exact-winner analysis repeated its diagnostic refit")

    monkeypatch.setattr(_service(), "refit_diagnostic_single", forbidden)
    request = AnalysisRequest(
        "curve", problem, search, profile_names=(), bootstrap_enabled=False, residual_evidence=evidence
    )
    restored = pickle.loads(pickle.dumps(request))
    final = _service().run_analysis(restored)
    assert final.uncertainty.member_residuals[0] is restored.residual_evidence
    assert restored.residual_evidence == evidence


def test_cache_owner_rejects_changed_configuration_and_damaged_seal():
    problem, search = _scale_search()
    evidence = _result(problem, search, profile_names=()).uncertainty.member_residuals[0]
    changed = replace(
        problem, config=replace(problem.config, budget=replace(problem.config.budget, diagnostic_samples=199))
    )
    resealed = replace(search, provenance_sha256=fit_search_provenance_sha256(changed, search))
    with pytest.raises(ValueError, match="residual evidence owner"):
        AnalysisRequest("curve", changed, resealed, residual_evidence=evidence)
    damaged = replace(evidence, calibration=replace(evidence.calibration, provenance_sha256="b" * 64))
    with pytest.raises(ValueError, match="diagnostic calibration provenance"):
        AnalysisRequest("curve", problem, search, residual_evidence=damaged)


def test_unavailable_diagnostic_is_not_an_automatic_quality_pass():
    problem, search = _scale_search()
    report = build_uncertainty_report(problem, search.candidates)
    assert report.systematic_residual is None
    result = FitResult.from_search(search, confidence=ConfidenceClass.CORRELATED, uncertainty=report)
    decision = assess_automatic_quality(problem, result)
    assert decision.passed is False, "unknown residual conclusions must not turn into a truthy pass"
    assert any("unavailable" in reason for reason in decision.reasons)


def test_exhausted_diagnostic_budget_withholds_formal_profile():
    problem, search = _scale_search(count=98)
    result = _result(problem, search, profile_names=("instrument.scale",))
    member = result.uncertainty.member_residuals[0]
    assert not member.executed
    profile = result.uncertainty.profiles[0]
    assert profile.confidence_level is None
    assert profile.unavailable_reason == "diagnostic_budget_insufficient"


def test_genuine_calibrated_rejection_still_withholds_formal_profile():
    problem, search = _scale_search()
    member = _result(problem, search, profile_names=()).uncertainty.member_residuals[0]
    original = member.calibration
    statistics = tuple(
        replace(item, adjusted_p_value=0.01 if item.kind == "background" else 1.0) for item in original.statistics
    )
    calibration = replace(
        original, tail_count=1, tie_count=1, statistics=statistics, observed_score=100.0, provenance_sha256=None
    )
    calibration = replace(calibration, provenance_sha256=diagnostic_calibration_sha256(calibration))
    rejected = replace(
        member,
        systematic=True,
        autocorrelation=False,
        calibration=calibration,
        diagnostics=tuple(item for item in member.advisories if item.code == "suspected_diffuse_background"),
    )
    result = _result(problem, search, profile_names=("instrument.scale",), residual_evidence=rejected)
    profile = result.uncertainty.profiles[0]
    assert profile.interval_kind == "loss_support"
    assert profile.confidence_level is None
    assert profile.unavailable_reason == "residual_diagnostics_failed"


def test_analysis_constructor_and_unpickling_do_not_execute_calibration(monkeypatch):
    problem, search = _scale_search()
    evidence = _result(problem, search, profile_names=()).uncertainty.member_residuals[0]
    engine = import_module("xrr_fitter.analysis.residual_calibration")

    def forbidden(*_args, **_kwargs):
        pytest.fail("an immutable request performed diagnostic work")

    monkeypatch.setattr(engine, "calibrate_residuals", forbidden)
    request = AnalysisRequest("curve", problem, search, residual_evidence=evidence)
    restored = pickle.loads(pickle.dumps(request))
    assert restored.residual_evidence == evidence
    assert not restored.problem.data.intensity_raw.flags.writeable


def test_negative_poisson_screen_does_not_run_monte_carlo(monkeypatch):
    service = _service()
    problem, search = _scale_search()
    problem = replace(problem, instrument=replace(problem.instrument, footprint_mode="fit", background_kind="linear"))
    search = replace(search, provenance_sha256=fit_search_provenance_sha256(problem, search))

    def forbidden(*_args, **_kwargs):
        pytest.fail("a negative screen requested Monte Carlo")

    monkeypatch.setattr(service, "refit_diagnostic_single", forbidden)
    result = _result(problem, search, profile_names=())
    evidence = result.uncertainty.member_residuals[0]
    assert evidence.executed and not evidence.systematic
    assert evidence.raw_systematic is False
    assert evidence.calibration is None


def test_automatic_fast_and_final_reports_reuse_the_same_calibration(monkeypatch):
    from tests.unit.services.test_fitting import _automatic_prepared

    from xrr_fitter.analysis.automatic import AutomaticQualityDecision
    from xrr_fitter.fit.pipeline import FitSearchRequest
    from xrr_fitter.services.fitting_phases.automatic_dataset import fit_automatic_prepared_dataset

    service = _service()
    problem, search = _scale_search()
    calls, requests = [], []
    original = service.refit_diagnostic_single

    def refit(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    def analyze(request, **kwargs):
        requests.append(request)
        return service.run_analysis(request, **kwargs)

    def forbidden(*_args, **_kwargs):
        pytest.fail("clean diagnostic evidence requested search recovery")

    monkeypatch.setattr(service, "refit_diagnostic_single", refit)
    result = fit_automatic_prepared_dataset(
        _automatic_prepared(problem),
        local_workers=1,
        fit_search_request=FitSearchRequest,
        run_fit_search=lambda *_args, **_kwargs: search,
        analysis_request=AnalysisRequest,
        run_analysis=analyze,
        assess_automatic_quality=lambda *_args: AutomaticQualityDecision(True, False, (), ("instrument.scale",), ()),
        automatic_profile_recovery=forbidden,
        automatic_absorption_search=forbidden,
    )
    assert result.passed
    assert len(requests) == 2
    assert len(calls) == 100, "automatic finalization repeated the same winner's Monte Carlo family"
    assert requests[-1].residual_evidence is result.fit_result.uncertainty.member_residuals[0]


def test_product_bootstrap_retains_samples_but_uses_diagnostic_qualification():
    from xrr_fitter.analysis.bootstrap import bootstrap_local
    from xrr_fitter.model.provenance import bootstrap_provenance_sha256

    problem, search = _scale_search(count=98)
    original = bootstrap_local(
        lambda rng, _index: rng.normal(size=1), ("instrument.scale",), sample_count=200, child_seed=21
    )
    original = replace(original, candidate_id=search.best_candidate.candidate_id, provenance_sha256="a" * 64)
    original = replace(
        original, provenance_sha256=bootstrap_provenance_sha256(problem, search.best_candidate, original)
    )
    result = _result(problem, search, profile_names=(), bootstrap=original)
    evidence = result.uncertainty.bootstrap_evidence
    assert evidence.confidence_level is None, "sample sufficiency cannot override unavailable Poisson diagnostics"
    assert evidence.diagnostic_unavailable_reason == "diagnostic_budget_insufficient"
    assert evidence.intervals == original.intervals
    np.testing.assert_array_equal(evidence.samples, original.samples)
    assert evidence.provenance_sha256 == bootstrap_provenance_sha256(problem, search.best_candidate, evidence)


def test_cached_evidence_requires_an_actual_candidate_not_an_empty_search():
    problem, search = _scale_search(count=98)
    evidence = _result(problem, search, profile_names=()).uncertainty.member_residuals[0]
    empty = replace(search, candidates=(), best_index=None, stage_summaries=())
    empty = replace(empty, provenance_sha256=fit_search_provenance_sha256(problem, empty))
    with pytest.raises(ValueError, match="winner"):
        AnalysisRequest("curve", problem, empty, residual_evidence=evidence)
