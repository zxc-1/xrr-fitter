from __future__ import annotations

from importlib import import_module

import pytest
from tests.unit.analysis.test_report import (
    _analysis_candidates,
    _problem,
    _search_result,
)

from xrr_fitter.model.analysis import ConfidenceClass
from xrr_fitter.model.parameters import ParameterPrior, PriorSpec


def _api():
    return import_module("xrr_fitter.analysis.report")


def _prior() -> ParameterPrior:
    return ParameterPrior(
        "component.0.density_scale",
        PriorSpec("normal", (0.6, 0.05)),
    )


def test_analysis_request_applies_prior_sidecar_after_ownership_validation() -> None:
    module = _api()
    problem = _problem()
    search = _search_result(problem, _analysis_candidates(problem))
    prior = _prior()

    request = module.AnalysisRequest(
        "curve",
        problem,
        search,
        profile_names=(),
        bootstrap_enabled=False,
        parameter_priors=(prior,),
    )
    result = module.run_analysis(request)

    assert request.problem is problem
    assert result.parameter_definitions == search.parameter_definitions
    assert result.uncertainty is not None
    assert result.uncertainty.prior_conflicts == (prior.name,)


def test_prior_sidecar_does_not_enter_profile_or_confidence_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _api()
    problem = _problem()
    search = _search_result(problem, _analysis_candidates(problem))
    observed = {}
    build = module.build_uncertainty_report

    def record_build(problem_value, candidates, **options):
        observed["build_problem"] = problem_value
        return build(problem_value, candidates, **options)

    def classify(problem_value, candidates, report):
        observed["classification_problem"] = problem_value
        observed["conflicts"] = report.prior_conflicts
        return ConfidenceClass.TRUSTED, ()

    monkeypatch.setattr(module, "build_uncertainty_report", record_build)
    monkeypatch.setattr(module, "classify_result_with_evidence", classify)

    result = module.analyze_search_result(
        problem,
        search,
        profile_names=(),
        bootstrap_enabled=False,
        parameter_priors=(_prior(),),
    )

    assert observed["build_problem"] is problem
    assert observed["classification_problem"] is problem
    assert observed["conflicts"] == ("component.0.density_scale",)
    assert result.confidence is ConfidenceClass.TRUSTED
