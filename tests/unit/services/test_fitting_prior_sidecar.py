from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result, project
from tests.unit.services.test_automatic_joint import FIT_GROUP_ID, _prepared
from tests.unit.services.test_fitting import (
    _automatic_problem,
    _bands_mcmc_report,
    _FittingHarness,
    _project,
    _RecordingTaskRunner,
    _stage_e_search,
)

import xrr_fitter.api as api
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.pipeline import FitSearchRequest
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.provenance import fit_search_provenance_sha256
from xrr_fitter.services import fitting
from xrr_fitter.services.fitting_phases import operations as operations_phase
from xrr_fitter.services.fitting_phases import sharing
from xrr_fitter.services.fitting_phases.base import fit_prepared_dataset


def _density_problem():
    initial = _automatic_problem()
    settings = tuple(
        api.ParameterSetting(
            definition.name,
            definition.initial,
            0.5 if definition.name == "component.0.density_scale" else definition.lower,
            1.1 if definition.name == "component.0.density_scale" else definition.upper,
            locked=False if definition.name == "component.0.density_scale" else definition.locked,
        )
        for definition in initial.parameter_definitions
    )
    return compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        settings,
    )


def _joint_stage_e_search(problem):
    search = _stage_e_search(problem)
    candidate = search.best_candidate
    assert candidate is not None
    ranked = replace(candidate, ranking_objective=candidate.objective)
    result = replace(search, candidates=(ranked,))
    return replace(
        result,
        provenance_sha256=fit_search_provenance_sha256(problem, result),
    )


def _density_prior() -> api.ParameterPrior:
    return api.ParameterPrior(
        "component.0.density_scale",
        api.PriorSpec("normal", (0.6, 0.05)),
    )


def test_fitting_forwards_persisted_parameter_priors_to_analysis_request(
    tmp_path,
) -> None:
    value = _project(tmp_path)
    prepared = fitting.prepare_dataset_fit(value, "curve", value.master_seed)
    priors = (_density_prior(),)
    prepared = replace(
        prepared,
        updated_dataset=replace(prepared.updated_dataset, parameter_priors=priors),
    )
    search = SimpleNamespace(
        best_candidate=SimpleNamespace(objective=0.25, ranking_objective=None),
    )
    analyzed = final_fit_result()
    harness = _FittingHarness(search, None, search, analyzed)

    result = fit_prepared_dataset(
        prepared,
        fit_search_request=FitSearchRequest,
        run_fit_search=harness.run_search,
        recover_profile_basin=harness.recover,
        continue_profile_basin=harness.continue_search,
        analysis_request=harness.analysis_request,
        run_analysis=harness.run_analysis,
        task_runner_factory=_RecordingTaskRunner,
    )

    assert result is analyzed
    request_call = next(call for call in harness.calls if call[0] == "analysis-request")
    assert request_call[-1] == priors


def test_joint_analysis_unions_local_prior_conflicts_as_global_names() -> None:
    problems = (_density_problem(), _density_problem())
    searches = tuple(_joint_stage_e_search(problem) for problem in problems)
    joint = compile_joint_problem(("left", "right"), problems, ())

    baseline = fitting._analyze_joint_searches(joint, searches, ((), ()))
    results = fitting._analyze_joint_searches(joint, searches, ((_density_prior(),), ()))

    assert all(result.uncertainty is not None for result in results)
    assert tuple(result.confidence for result in results) == tuple(result.confidence for result in baseline)
    assert tuple(result.classification_evidence for result in results) == tuple(
        result.classification_evidence for result in baseline
    )
    assert all(result.uncertainty.prior_conflicts == ("left:component.0.density_scale",) for result in results)


def test_joint_analysis_deduplicates_shared_roughness_conflict_as_global_name() -> None:
    prepared = (
        _prepared("left", 0, released_imag=()),
        _prepared("right", 1, released_imag=()),
    )
    problems = tuple(item.problem for item in prepared)
    searches = tuple(_joint_stage_e_search(problem) for problem in problems)
    rules = sharing.automatic_sharing_rules(
        prepared,
        FIT_GROUP_ID,
        share_roughness=True,
    )
    joint = compile_joint_problem(("left", "right"), problems, rules)
    local_name = "component.0.roughness_a"
    shared_name = next(
        rule.sharing_key for rule in rules if any(member.parameter_name == local_name for member in rule.members)
    )
    prior = api.ParameterPrior(
        local_name,
        api.PriorSpec("normal", (0.9, 0.01)),
    )

    results = fitting._analyze_joint_searches(
        joint,
        searches,
        ((prior,), (prior,)),
    )

    assert all(result.uncertainty.prior_conflicts == (shared_name,) for result in results)
    assert all(local_name not in result.uncertainty.prior_conflicts for result in results)


def test_mcmc_phase_overlays_priors_only_after_result_ownership_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _density_problem()
    search = _stage_e_search(problem)
    candidate = search.best_candidate
    assert candidate is not None
    uncertainty = api.UncertaintyReport(
        correlation_names=(),
        correlation_matrix=np.empty((0, 0)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate.candidate_id,
    )
    result = api.FitResult.from_search(
        search,
        confidence=api.ConfidenceClass.TRUSTED,
        uncertainty=uncertainty,
    )
    priors = (_density_prior(),)
    dataset = replace(
        dataset_project("curve", result=result),
        structure=problem.structure,
        instrument=problem.instrument,
        parameter_priors=priors,
    )
    value = project(dataset)
    prepared = fitting.PreparedDatasetFit("curve", 0, dataset, problem)
    observed = {}
    sentinel_report = _bands_mcmc_report(
        np.full((8, len(problem.variables)), 1.0),
        names=tuple(variable.name for variable in problem.variables),
    )

    monkeypatch.setattr(
        operations_phase,
        "inspect_sources",
        lambda _project: SimpleNamespace(valid=True, issues=(), datasets=()),
    )

    def run_problem(analysis_problem, selected, config, **_kwargs):
        observed["problem"] = analysis_problem
        observed["candidate"] = selected
        observed["config"] = config
        return sentinel_report

    updated = operations_phase._run_mcmc(
        value,
        "curve",
        candidate.candidate_id,
        api.McmcConfig(walkers=6, burn_in=0, production_steps=2),
        None,
        None,
        compile_dataset=lambda *_args, **_kwargs: prepared,
        with_parameter_priors=fitting.with_parameter_priors,
        run_problem_mcmc=run_problem,
        sld_bands=lambda _structure, report, _wavelength: (None, report),
    )

    analysis_problem = observed["problem"]
    assert prepared.problem.parameter_definitions == result.parameter_definitions
    assert analysis_problem is not prepared.problem
    assert (
        next(
            definition
            for definition in analysis_problem.parameter_definitions
            if definition.name == "component.0.density_scale"
        ).prior
        == priors[0].prior
    )
    assert updated.datasets[0].last_valid_result.uncertainty.mcmc is sentinel_report
