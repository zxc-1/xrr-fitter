"""Shared-fit diagnostics are one family, not independent member refits."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from inspect import signature

import numpy as np
import pytest
from tests.support.model_cases import dataset_project
from tests.unit.analysis.covariance_cases import scale_problem

from xrr_fitter.analysis.automatic import assess_automatic_quality
from xrr_fitter.analysis.bootstrap import bootstrap_local
from xrr_fitter.analysis.bootstrap_samples import bootstrap_result_from_fits
from xrr_fitter.analysis.report import AnalysisRequest
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.joint_candidates import consensus_joint_vector
from xrr_fitter.fit.joint_pipeline import JointFitRequest, _project_candidate, _summary
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_solvers import solve_joint
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.model.fitting import FitSearchResult
from xrr_fitter.model.joint_bootstrap_provenance import joint_bootstrap_owner_sha256, seal_joint_bootstrap
from xrr_fitter.model.parameters import ParameterReference, SharingRule
from xrr_fitter.model.provenance import fit_search_provenance_sha256
from xrr_fitter.services import fitting
from xrr_fitter.services.fitting_phases import joint_selection
from xrr_fitter.services.fitting_phases.common import AutomaticPreparedResult, PreparedDatasetFit


def _joint_searches(*, count=99, modes=("poisson", "poisson")):
    members = []
    for seed, mode in zip((17, 18), modes, strict=True):
        member = scale_problem(mode, seed=seed)
        budget = replace(member.config.budget, diagnostic_samples=count, local_min_nfev=80, bootstrap_samples=200)
        members.append(replace(member, config=replace(member.config, budget=budget)))
    ids = ("small", "large")
    rule = SharingRule("shared-scale", tuple(ParameterReference(name, "instrument.scale") for name in ids))
    problem = compile_joint_problem(ids, tuple(members), (rule,))
    rows = []
    for index, scale in enumerate((0.46, 0.49, 0.51, 0.54)):
        initial = encode_physical_vector(problem.problems[0], {"instrument.scale": scale})
        solved = solve_joint(problem, initial, 80, None)
        rows.append(_project_candidate(problem, solved, f"E-{index}", index))
    summary = _summary("E", tuple(rows))
    searches = tuple(
        FitSearchResult(
            member.parameter_definitions,
            tuple(candidates),
            min(range(len(candidates)), key=lambda index: candidates[index].ranking_objective),
            (),
            (17, 18, 19, 20),
            (summary,),
            member.region_labels,
            member.weights,
        )
        for member, candidates in zip(problem.problems, zip(*rows, strict=True), strict=True)
    )
    searches = tuple(
        replace(search, provenance_sha256=fit_search_provenance_sha256(member, search))
        for member, search in zip(problem.problems, searches, strict=True)
    )
    return problem, searches


def _forbidden(*_args, **_kwargs):
    pytest.fail("a shared winner was independently refitted or recalibrated")


def _prepared(problem):
    return tuple(
        PreparedDatasetFit(
            dataset_id,
            index,
            replace(dataset_project(dataset_id), structure=member.structure, instrument=member.instrument),
            member,
        )
        for index, (dataset_id, member) in enumerate(zip(problem.dataset_ids, problem.problems, strict=True))
    )


def test_joint_report_covariance_and_bootstrap_share_one_refit_family(monkeypatch):
    problem, searches = _joint_searches()
    calls = []
    original = fitting.refit_diagnostic_joint
    joint_module = import_module("xrr_fitter.analysis.joint")
    covariance = joint_module.joint_covariance
    covariance_calls = []

    def refit(template, members, **kwargs):
        calls.append(tuple(member.config.noise_model for member in members))
        assert template is problem
        return original(template, members, **kwargs)

    def counted_covariance(*args, **kwargs):
        covariance_calls.append(True)
        return covariance(*args, **kwargs)

    def bootstrap(problem, candidates, vector, **kwargs):
        names = tuple(variable.name for variable in problem.global_variables)
        samples = bootstrap_local(
            lambda rng, _index: rng.normal(size=len(names)), names, sample_count=200, child_seed=21
        )
        return _owned_sampling(problem, candidates, vector, kwargs["child_seed"], samples.samples)

    monkeypatch.setattr(fitting, "refit_diagnostic_joint", refit)
    monkeypatch.setattr(fitting, "refit_diagnostic_single", _forbidden)
    monkeypatch.setattr(fitting, "bootstrap_joint_local", bootstrap)
    monkeypatch.setattr(joint_module, "joint_covariance", counted_covariance)
    results = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)
    report = results[0].uncertainty
    assert len(calls) == 100, "observed + 99 joint null fits must run exactly once"
    assert covariance_calls == [True]
    assert all(row == ("poisson", "poisson") for row in calls)
    assert results[1].uncertainty is report
    members = report.member_residuals
    assert any(member.raw_systematic for member in members)
    assert all(member.executed and member.systematic is False for member in members)
    assert members[0].calibration is members[1].calibration
    assert {stat.dataset_id for stat in members[0].calibration.statistics} == set(problem.dataset_ids)
    assert report.bootstrap_evidence.confidence_level == 0.95


def test_joint_budget_unavailable_qualifies_shared_bootstrap_without_discarding_samples(monkeypatch):
    problem, searches = _joint_searches(count=98)
    samples = bootstrap_local(
        lambda rng, _index: rng.normal(size=1), ("shared:shared-scale",), sample_count=200, child_seed=31
    )

    def bootstrap(problem, candidates, vector, **kwargs):
        return _owned_sampling(problem, candidates, vector, kwargs["child_seed"], samples.samples)

    monkeypatch.setattr(fitting, "refit_diagnostic_joint", _forbidden)
    monkeypatch.setattr(fitting, "bootstrap_joint_local", bootstrap)
    results = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)
    report = results[0].uncertainty
    assert report.bootstrap_evidence.confidence_level is None
    assert report.bootstrap_evidence.diagnostic_unavailable_reason == "diagnostic_budget_insufficient"
    np.testing.assert_array_equal(report.bootstrap_evidence.samples, samples.samples)
    assert all(member.unavailable_reason == "diagnostic_budget_insufficient" for member in report.member_residuals)
    assert "residual_diagnostics_not_executed" in results[0].classification_evidence


def test_mixed_joint_does_not_claim_poisson_null_calibration(monkeypatch):
    problem, searches = _joint_searches(modes=("poisson", "gaussian"))
    monkeypatch.setattr(fitting, "refit_diagnostic_joint", _forbidden)
    members = fitting._analyze_joint_searches(problem, searches, ((), ()))[0].uncertainty.member_residuals
    assert members[0].unavailable_reason == "mixed_noise_diagnostic_calibration_unsupported"
    assert not members[0].executed and members[0].calibration.status == "unavailable"
    assert members[1].executed and members[1].calibration is None


def test_operation_cache_is_exact_owner_scoped_not_a_global_fit_cache():
    assert "cache" in signature(fitting._joint_point_evidence).parameters, (
        "joint operation needs an explicit local cache"
    )
    problem, searches = _joint_searches()
    unit = searches[0].best_candidate.unit_vector
    cache = {}
    first = fitting._joint_point_evidence(problem, unit, cache=cache)
    second = fitting._joint_point_evidence(problem, unit, cache=cache)
    assert second is first
    changed = tuple(
        replace(member, config=replace(member.config, budget=replace(member.config.budget, diagnostic_samples=98)))
        for member in problem.problems
    )
    new_problem = compile_joint_problem(problem.dataset_ids, changed, problem.sharing_rules)
    different = fitting._joint_point_evidence(new_problem, unit, cache=cache)
    assert different[1][0].unavailable_reason == "diagnostic_budget_insufficient"
    assert different[1][0].owner_sha256 != first[1][0].owner_sha256
    moved = fitting._joint_point_evidence(problem, unit + 0.001, cache=cache)
    assert moved[1][0].unavailable_reason == "diagnostic_refit_mismatch"
    assert len(cache) == 3


def test_cancelling_last_joint_null_fit_cannot_publish_a_report(monkeypatch):
    problem, searches = _joint_searches()
    calls, cancelled = [], []
    original = fitting.refit_diagnostic_joint

    def refit(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 100:
            cancelled.append(True)
        return result

    monkeypatch.setattr(fitting, "refit_diagnostic_joint", refit)
    with pytest.raises((InterruptedError, SearchCancelled), match="cancel"):
        fitting._analyze_joint_searches(problem, searches, ((), ()), cancelled=lambda: bool(cancelled))


def test_automatic_shared_poisson_uses_joint_evidence_without_local_analysis():
    problem, searches = _joint_searches()
    prepared = _prepared(problem)
    _problem, results, local_results, decisions = joint_selection._run_automatic_joint_refinement(
        prepared,
        problem.sharing_rules,
        {dataset_id: search.best_candidate for dataset_id, search in zip(problem.dataset_ids, searches, strict=True)},
        progress=None,
        cancelled=None,
        checkpoint=None,
        compile_joint_problem=compile_joint_problem,
        consensus_joint_vector=consensus_joint_vector,
        joint_fit_request=JointFitRequest,
        run_joint_fit=lambda *_args, **_kwargs: searches,
        analysis_request=AnalysisRequest,
        run_analysis=_forbidden,
        assess_automatic_quality=assess_automatic_quality,
        analyze_joint_searches=fitting._analyze_joint_searches,
    )
    assert all(decision.passed for decision in decisions)
    for dataset_id, local, result in zip(problem.dataset_ids, local_results, results, strict=True):
        members = local.uncertainty.member_residuals
        assert len(members) == 1 and members[0].dataset_id == dataset_id
        assert members[0] in result.uncertainty.member_residuals
        assert members[0].calibration is result.uncertainty.member_residuals[0].calibration


def test_prefit_isolation_uses_effective_evidence_not_a_stale_raw_advisory():
    from tests.unit.services.test_diagnostic_calibration_flow import _result, _scale_search

    problem, search = _scale_search()
    result = _result(problem, search, profile_names=())
    advisory = result.uncertainty.member_residuals[0].advisories[0]
    candidate = replace(result.best_candidate, diagnostics=(*result.best_candidate.diagnostics, advisory))
    result = replace(result, candidates=(candidate,))
    prepared = PreparedDatasetFit("curve", 0, dataset_project("curve"), problem)
    prefit = AutomaticPreparedResult(prepared, result, True, None)
    assert joint_selection._prefit_isolation_reason(prefit) is None


def test_unknown_joint_member_diagnostics_require_refinement_review():
    from types import SimpleNamespace

    problem, searches = _joint_searches(count=98)
    prepared = _prepared(problem)
    results = fitting._analyze_joint_searches(problem, searches, ((), ()))
    prefits = tuple(
        AutomaticPreparedResult(item, result, True, None) for item, result in zip(prepared, results, strict=True)
    )
    local = tuple(
        SimpleNamespace(uncertainty=SimpleNamespace(systematic_residual=None, residual_autocorrelation=None))
        for _ in prepared
    )
    assert joint_selection._joint_result_conflicts(prepared, prefits, results, local) is True


@pytest.mark.parametrize("missing_report", (False, True))
def test_prefit_without_usable_diagnostics_cannot_enter_shared_fit(missing_report):
    from tests.unit.services.test_diagnostic_calibration_flow import _result, _scale_search

    problem, search = _scale_search(count=98)
    result = _result(problem, search, profile_names=())
    if missing_report:
        result = replace(result, uncertainty=None)
    prepared = PreparedDatasetFit("curve", 0, dataset_project("curve"), problem)
    prefit = AutomaticPreparedResult(prepared, result, True, None)
    assert "unavailable" in (joint_selection._prefit_isolation_reason(prefit) or "")


def test_prefit_keeps_genuine_physics_diagnostics_even_when_residuals_are_clean():
    from tests.unit.services.test_diagnostic_calibration_flow import _result, _scale_search

    from xrr_fitter.model.instrument import PhysicsDiagnostic

    problem, search = _scale_search()
    result = _result(problem, search, profile_names=())
    diagnostic = PhysicsDiagnostic("nevot_croce_applicability_exceeded", "roughness exceeds validity")
    candidate = replace(result.best_candidate, diagnostics=(diagnostic,))
    result = replace(result, candidates=(candidate,))
    prepared = PreparedDatasetFit("curve", 0, dataset_project("curve"), problem)
    reason = joint_selection._prefit_isolation_reason(AutomaticPreparedResult(prepared, result, True, None))
    assert "nevot_croce_applicability_exceeded" in reason


def test_joint_consumer_validates_layout_owner_and_seal_before_covariance():
    problem, searches = _joint_searches()
    unit = searches[0].best_candidate.unit_vector
    _covariance, members = fitting._joint_point_evidence(problem, unit)
    evaluation = fitting.evaluate_joint_vector(problem, unit)
    arguments = (
        tuple(variable.name for variable in problem.global_variables),
        problem.dataset_ids,
        problem.problems,
        unit,
        evaluation.local_evaluations,
        _forbidden,
    )
    with pytest.raises(ValueError, match="owner"):
        fitting.analyze_joint_point(*arguments, residual_evidence=members, layout_fingerprint="different-layout")
    damaged = replace(members[0], calibration=replace(members[0].calibration, provenance_sha256="b" * 64))
    with pytest.raises(ValueError, match="provenance"):
        fitting.analyze_joint_point(
            *arguments,
            residual_evidence=(damaged, members[1]),
            layout_fingerprint=problem.layout_fingerprint,
        )


def _owned_sampling(problem, candidates, vector, seed, matrix):
    names = tuple(variable.name for variable in problem.global_variables)
    result = bootstrap_result_from_fits(names, iter(matrix), matrix.shape[0], None, method="joint_poisson_parametric")
    return seal_joint_bootstrap(
        result, candidates[0].candidate_id, joint_bootstrap_owner_sha256(problem, candidates, vector, seed)
    )
