"""Contracts for analysis ownership, worker values, and report composition.

These tests bind search and bootstrap fixtures to exact evaluation contexts,
then prove drift is rejected without replaying candidate physics. They retain
expected invalid fitting evidence, full historical candidate graphs, immutable
pickle round trips, and deterministic winner identity.

Report tests cover covariance/profile selection, diagnostic enrichment,
classification order, progress, and uncertainty stage publication. Joint
ranking is exercised by the real joint pipeline in its focused test module.

Automatic fits deliberately disable bootstrap on their clean fast path while
retaining bounded profile recovery for ambiguous evidence. The report contract
therefore records whether work actually ran, rather than inferring it from
default field values.

Prior-conflict coverage checks that the composed uncertainty report flags each
free parameter whose representative estimate leaves its declared prior window,
while parameters that stay within tolerance and parameters without a locating
prior are never reported. The comparison is performed in the coordinate space
the prior is declared against, so a roughness-fraction prior is scored on the
unit fraction rather than the physical roughness it expands to. The low-level
joint ensemble report has no per-dataset sidecars and leaves the field empty;
the fitting service maps local conflicts onto global joint names afterward.
Reports constructed without any conflict argument default to an empty tuple. A
final contract proves that prior-conflict labels never influence which
parameters are chosen for profile refinement.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, fields, replace
from functools import partial
from importlib import import_module
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector, evaluate_model
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import BootstrapResult, ConfidenceClass, UncertaintyReport
from xrr_fitter.model.fitting import (
    FitConfig,
    FitEvaluationContext,
    FitProgress,
    FitSearchResult,
    FitStageSummary,
)
from xrr_fitter.model.instrument import InstrumentSpec, PhysicsDiagnostic
from xrr_fitter.model.parameters import ParameterSetting, PriorSpec
from xrr_fitter.model.provenance import (
    bootstrap_provenance_sha256,
    fit_search_provenance_sha256,
)


def _api():
    return import_module("xrr_fitter.analysis.report")


@pytest.mark.parametrize(
    ("module_name", "function_names"),
    (
        (
            "binary_profiles",
            ("binary_derived_profiles", "decode_binary_coordinate", "build_binary_profile"),
        ),
        ("bootstrap", ("bootstrap_problem_local",)),
        ("classification", ("classify_result_with_evidence",)),
        (
            "derivatives",
            ("objective_gradient", "objective_information", "physical_parameter_jacobian"),
        ),
        ("diagnostics", ("ordered_fit_residuals", "diagnose_residual_patterns")),
        ("mcmc", ("map_problem_samples", "mcmc_boundary_hits", "run_problem_mcmc")),
        ("profiles", ("build_problem_profile", "select_profile_names", "recover_profile_basin")),
        ("report", ("build_uncertainty_report", "analyze_search_result")),
    ),
)
def test_public_analysis_entries_require_the_typed_context(
    module_name: str,
    function_names: tuple[str, ...],
) -> None:
    module = import_module(f"xrr_fitter.analysis.{module_name}")

    for name in function_names:
        assert get_type_hints(getattr(module, name))["problem"] is FitEvaluationContext


def _problem(*, thickness_a: float = 20.0):
    structure = simple_structure()
    structure = replace(
        structure,
        components=(replace(structure.components[0], thickness_a=thickness_a),),
    )
    initial = compile_fit_problem(
        prepared_data(size=48),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(937), scale_prior_enabled=False),
    )
    targets = {"component.0.thickness_a", "component.0.density_scale"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in targets,
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


def _candidate(problem, candidate_id: str, offset: float = 0.0):
    unit = np.clip(encode_physical_vector(problem, {}) + offset, 0.0, 1.0)
    return candidate_from_evaluation(
        problem,
        unit,
        evaluate_model(problem, unit),
        candidate_id=candidate_id,
        seed_index=int(candidate_id.rsplit("-", 1)[-1]),
        stop_reason="test candidate",
        nfev=1,
    )


def _candidate_at(problem, candidate_id: str, unit: np.ndarray):
    return candidate_from_evaluation(
        problem,
        unit,
        evaluate_vector(problem, unit),
        candidate_id=candidate_id,
        seed_index=int(candidate_id.rsplit("-", 1)[-1]),
        stop_reason="test candidate",
        nfev=1,
    )


def _angle_offset_problem():
    initial = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(941), scale_prior_enabled=False),
    )
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name != "instrument.angle_offset_deg",
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


def _legacy(candidate, *, candidate_id=...):
    values = {field: getattr(candidate, field) for field in candidate.__dataclass_fields__ if field != "candidate_id"}
    if candidate_id is not ...:
        values["candidate_id"] = candidate_id
    return SimpleNamespace(**values)


def _search_result(problem, candidates) -> FitSearchResult:
    values = tuple(candidates)
    best_index = min(range(len(values)), key=lambda index: values[index].objective)
    summary = FitStageSummary(
        "E",
        tuple(candidate.candidate_id for candidate in values),
        values[best_index].objective,
        sum(candidate.nfev for candidate in values),
        tuple(candidate.stop_reason for candidate in values),
    )
    result = FitSearchResult(
        problem.parameter_definitions,
        values,
        best_index,
        (),
        (11, 12, 13, 14),
        (summary,),
        problem.region_labels,
        problem.weights,
    )
    return replace(
        result,
        provenance_sha256=fit_search_provenance_sha256(problem, result),
    )


def _record_analysis(
    observed: dict[str, object],
    result: object,
    problem: object,
    search: object,
    **options: object,
) -> object:
    observed.update(problem=problem, search=search, **options)
    return result


def _never_cancelled() -> bool:
    return False


def _ignore_progress(_value: object) -> None:
    return None


@dataclass(frozen=True)
class _UnsupportedBootstrapPayload:
    opaque: object


def test_provenance_rejects_unsupported_payload_values() -> None:
    problem = _problem()
    candidate = _candidate(problem, "E-0")

    with pytest.raises(TypeError, match="unsupported provenance value"):
        bootstrap_provenance_sha256(
            problem,
            candidate,
            _UnsupportedBootstrapPayload(object()),
        )


def _assert_analysis_pickle_contract(api, request, restored) -> None:
    assert tuple(field.name for field in fields(request)) == (
        "dataset_id",
        "problem",
        "search_result",
        "profile_names",
        "bootstrap",
        "bootstrap_enabled",
        "parameter_priors",
    )
    assert restored.dataset_id == "curve"
    assert restored.problem.data.qz_a_inv.flags.writeable is False
    assert restored.search_result.region_weights.flags.writeable is False
    assert restored.parameter_priors == ()
    assert pickle.loads(pickle.dumps(api.run_analysis)) is api.run_analysis


def _assert_analysis_handler_call(
    observed: dict[str, object],
    restored,
    result: object,
    sentinel: object,
) -> None:
    assert result is sentinel
    assert observed["problem"] is restored.problem
    assert observed["search"] is restored.search_result
    assert observed["dataset_id"] == "curve"
    assert observed["profile_names"] == ()
    assert observed["parameter_priors"] == ()
    assert observed["progress"] is _ignore_progress


def test_analysis_request_and_handler_are_pickle_safe_worker_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    problem = _problem()
    search = _search_result(problem, tuple(_candidate(problem, f"E-{index}") for index in range(4)))
    request = api.AnalysisRequest(
        "curve",
        problem,
        search,
        profile_names=(),
    )

    restored = pickle.loads(pickle.dumps(request))
    _assert_analysis_pickle_contract(api, request, restored)

    sentinel = object()
    observed: dict[str, object] = {}
    analyze = partial(_record_analysis, observed, sentinel)
    monkeypatch.setattr(api, "analyze_search_result", analyze)

    result = api.run_analysis(
        restored,
        cancelled=_never_cancelled,
        progress=_ignore_progress,
    )

    _assert_analysis_handler_call(observed, restored, result, sentinel)


def test_analysis_can_skip_bootstrap_and_profile_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _api()
    problem = _problem()
    search = _search_result(
        problem,
        tuple(_candidate(problem, f"E-{index}") for index in range(4)),
    )

    monkeypatch.setattr(
        module,
        "bootstrap_problem_local",
        lambda *_args, **_kwargs: pytest.fail("automatic analysis ran bootstrap"),
    )
    monkeypatch.setattr(
        module,
        "build_problem_profiles",
        lambda *_args, **_kwargs: pytest.fail("automatic analysis built profiles"),
    )

    request = module.AnalysisRequest(
        "curve",
        problem,
        search,
        profile_names=(),
        bootstrap_enabled=False,
    )
    result = module.run_analysis(request)

    assert result.uncertainty is not None
    assert result.uncertainty.bootstrap_performed is False
    assert result.uncertainty.profiles == ()


# Twelve variables used to sit on the exhaustive-profile boundary. Building a
# zero-profile preliminary report first is what preserves evidence selection
# while avoiding a full periodic profile scan.
def test_twelve_parameter_default_profiles_use_preliminary_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _api()
    problem = SimpleNamespace(variables=tuple(SimpleNamespace(name=f"parameter.{index}") for index in range(12)))
    preliminary = object()
    observed: dict[str, object] = {}

    def build(problem_value, candidates, **options):
        observed["build"] = (problem_value, candidates, options)
        return preliminary

    def select(problem_value, report, **options):
        observed["select"] = (problem_value, report, options)
        return ("parameter.0",)

    monkeypatch.setattr(module, "build_uncertainty_report", build)
    monkeypatch.setattr(module, "select_profile_names", select)
    candidates = (object(),)
    bootstrap = object()

    selected = module._selected_profile_names(
        problem,
        candidates,
        None,
        bootstrap,
        ("warning",),
        None,
    )

    assert selected == ("parameter.0",)
    assert observed["build"] == (
        problem,
        candidates,
        {
            "profile_names": (),
            "bootstrap": bootstrap,
            "cancelled": None,
        },
    )
    assert observed["select"] == (
        problem,
        preliminary,
        {"degeneracy_warnings": ("warning",)},
    )


@pytest.mark.parametrize("drift", ["structure", "data"], ids=["structure", "data"])
def test_analysis_request_rejects_search_result_from_another_context(drift: str) -> None:
    problem = _problem()
    if drift == "structure":
        stale_problem = _problem(thickness_a=140.0)
    else:
        data = replace(
            problem.data,
            source_sha256="b" * 64,
            intensity_raw=problem.data.intensity_raw * 0.75,
            intensity_normalized=problem.data.intensity_normalized * 0.75,
        )
        stale_problem = replace(problem, data=data)
    stale_search = _search_result(
        stale_problem,
        tuple(_candidate(stale_problem, f"E-{index}") for index in range(4)),
    )

    with pytest.raises(ValueError, match="search_result|candidate|context"):
        _api().AnalysisRequest("curve", problem, stale_search, profile_names=())


def test_analysis_request_rejects_bootstrap_parameter_ownership_drift() -> None:
    problem = _problem()
    search = _search_result(
        problem,
        tuple(_candidate(problem, f"E-{index}") for index in range(4)),
    )
    names = tuple(variable.name for variable in problem.variables)
    bootstrap = BootstrapResult(
        tuple(reversed(names)),
        np.ones((2, len(names))),
        (),
        0.0,
    )

    with pytest.raises(ValueError, match="bootstrap.*parameter|parameter.*bootstrap"):
        _api().AnalysisRequest(
            "curve",
            problem,
            search,
            profile_names=(),
            bootstrap=bootstrap,
        )


def test_analysis_request_construction_and_pickle_do_not_run_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    evaluation = import_module("xrr_fitter.evaluation")
    problem = _problem()
    search = _search_result(
        problem,
        tuple(_candidate(problem, f"E-{index}") for index in range(4)),
    )

    def unexpected_evaluation(*_args, **_kwargs):
        raise AssertionError("request value executed physics")

    monkeypatch.setattr(evaluation, "evaluate_model", unexpected_evaluation)

    request = api.AnalysisRequest("curve", problem, search, profile_names=())
    restored = pickle.loads(pickle.dumps(request))

    assert restored.dataset_id == "curve"


def test_analysis_request_rejects_tampered_published_candidate_payload() -> None:
    problem = _problem()
    search = _search_result(
        problem,
        tuple(_candidate(problem, f"E-{index}") for index in range(4)),
    )
    tampered = replace(
        search.candidates[0],
        sld_profile_a2=np.zeros_like(search.candidates[0].sld_profile_a2),
    )
    changed = replace(search, candidates=(tampered, *search.candidates[1:]))

    with pytest.raises(ValueError, match="search_result|provenance|context"):
        _api().AnalysisRequest("curve", problem, changed, profile_names=())


def test_analysis_request_rejects_tampered_nonfinal_candidate_payload() -> None:
    problem = _problem()
    final_search = _search_result(
        problem,
        tuple(_candidate(problem, f"E-{index}") for index in range(4)),
    )
    historical = _candidate(problem, "B-0")
    unsealed = replace(
        final_search,
        candidates=(historical, *final_search.candidates),
        best_index=final_search.best_index + 1,
        provenance_sha256=None,
    )
    search = replace(
        unsealed,
        provenance_sha256=fit_search_provenance_sha256(problem, unsealed),
    )
    tampered = replace(
        historical,
        sld_profile_a2=np.zeros_like(historical.sld_profile_a2),
    )
    changed = replace(search, candidates=(tampered, *search.candidates[1:]))

    with pytest.raises(ValueError, match="search_result|provenance|context"):
        _api().AnalysisRequest("curve", problem, changed, profile_names=())


def test_analysis_request_rejects_tampered_bootstrap_payload() -> None:
    problem = _problem()
    search = _search_result(
        problem,
        tuple(_candidate(problem, f"E-{index}") for index in range(4)),
    )
    candidate = search.best_candidate
    assert candidate is not None
    names = tuple(variable.name for variable in problem.variables)
    samples = np.ones((2, len(names)))
    intervals = tuple((name, 0.5, 1.5) for name in names)
    unsealed = BootstrapResult(
        names,
        samples,
        intervals,
        0.0,
    )
    bootstrap = replace(
        unsealed,
        candidate_id=candidate.candidate_id,
        provenance_sha256=bootstrap_provenance_sha256(problem, candidate, unsealed),
    )
    changed = replace(
        bootstrap,
        intervals=tuple((name, lower + 0.1, upper) for name, lower, upper in intervals),
    )

    with pytest.raises(ValueError, match="bootstrap|provenance|candidate"):
        _api().AnalysisRequest(
            "curve",
            problem,
            search,
            profile_names=(),
            bootstrap=changed,
        )


def test_analysis_accepts_expected_invalid_stage_e_evidence() -> None:
    api = _api()
    problem = _angle_offset_problem()
    candidates = (
        _candidate_at(problem, "E-0", np.asarray([0.5])),
        _candidate_at(problem, "E-1", np.asarray([0.0])),
    )
    assert candidates[0].valid is True
    assert candidates[1].valid is False
    search = _search_result(problem, candidates)

    request = api.AnalysisRequest("curve", problem, search, profile_names=())
    result = api.run_analysis(request)

    assert result.best_candidate.candidate_id == "E-0"
    assert result.candidates[1].stop_reason == "nonpositive_fitted_incident_angle"


def _empty_report(candidate_id: str | None) -> UncertaintyReport:
    return UncertaintyReport(
        correlation_names=(),
        correlation_matrix=np.empty((0, 0)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _analysis_candidates(problem):
    return tuple(_candidate(problem, f"E-{index}", offset) for index, offset in enumerate((0.0, 0.001, -0.001, 0.002)))


def _empty_owned_bootstrap(problem, search) -> BootstrapResult:
    candidate = search.best_candidate
    assert candidate is not None
    names = tuple(variable.name for variable in problem.variables)
    unsealed = BootstrapResult(
        names,
        np.empty((0, len(names))),
        (),
        0.0,
    )
    return replace(
        unsealed,
        candidate_id=candidate.candidate_id,
        provenance_sha256=bootstrap_provenance_sha256(
            problem,
            candidate,
            unsealed,
        ),
    )


def _record_profile_report(calls, report, report_calls, *_args, **kwargs):
    calls.append("report")
    report_calls.append(kwargs)
    kwargs["progress"](1, 1, "component.0.thickness_a")
    return report


def _record_bootstrap(calls, bootstrap_calls, problem, search, *_args, **kwargs):
    calls.append("bootstrap")
    bootstrap_calls.append(kwargs)
    report_progress = kwargs["progress"]
    report_progress(1, kwargs["sample_count"])
    report_progress(kwargs["sample_count"], kwargs["sample_count"])
    return _empty_owned_bootstrap(problem, search)


def _record_classification(
    calls,
    report,
    diagnostic,
    _problem,
    observed,
    _report,
    **_kwargs,
):
    calls.append("classify")
    _assert_classification_inputs(observed, report, diagnostic)
    return ConfidenceClass.CORRELATED, ("strong_correlation",)


def _assert_diagnostic_warning(warnings: tuple[str, ...]) -> None:
    assert any(
        warning.startswith("ideal_reflectivity_above_one:") and "full_data_indices=[1,2]" in warning
        for warning in warnings
    )


def test_build_report_rejects_missing_lineage_for_identified_candidate() -> None:
    problem = _problem()
    identified = replace(_candidate(problem, "E-0"), objective=1.0)
    legacy = _legacy(replace(_candidate(problem, "E-1"), objective=0.5))

    with pytest.raises(AttributeError, match="candidate_id"):
        _api().build_uncertainty_report(problem, (identified, legacy), profile_names=())


def test_build_report_rejects_none_lineage_for_identified_candidate() -> None:
    problem = _problem()
    identified = replace(_candidate(problem, "E-0"), objective=1.0)
    missing = _legacy(
        replace(_candidate(problem, "E-1"), objective=0.5),
        candidate_id=None,
    )

    with pytest.raises(AttributeError, match="candidate_id"):
        _api().build_uncertainty_report(problem, (identified, missing), profile_names=())


def test_build_report_allows_missing_lineage_for_legacy_candidate_double() -> None:
    problem = _problem()
    legacy = _legacy(_candidate(problem, "E-0"))

    report = _api().build_uncertainty_report(problem, (legacy,), profile_names=())

    assert report.candidate_id is None


def test_build_report_selects_the_persisted_global_ranking_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _api()
    problem = _problem()
    candidates = (
        replace(_candidate(problem, "E-0"), objective=0.10, ranking_objective=10.0),
        replace(_candidate(problem, "E-1"), objective=0.20, ranking_objective=1.0),
        replace(_candidate(problem, "E-2"), objective=0.30, ranking_objective=20.0),
        replace(_candidate(problem, "E-3"), objective=0.40, ranking_objective=30.0),
    )
    monkeypatch.setattr(
        module,
        "_correlation_evidence",
        lambda _problem, _unit, names: (np.eye(len(names)), (), (), np.ones(len(names))),
    )
    monkeypatch.setattr(module, "_profiles", lambda *_args: ())
    monkeypatch.setattr(
        module,
        "_residual_evidence",
        lambda _problem, _candidate: (False, (), False),
    )

    report = module.build_uncertainty_report(problem, candidates, profile_names=())

    assert report.candidate_id == "E-1"


def test_fit_dataset_real_uncertainty_report_is_attached() -> None:
    problem = _problem()
    candidates = _analysis_candidates(problem)
    search = _search_result(problem, candidates)

    result = _api().analyze_search_result(problem, search, profile_names=())

    assert result.uncertainty is not None
    assert result.uncertainty.candidate_id == search.best_candidate.candidate_id
    assert result.confidence is not None
    assert result.classification_evidence is not None


def _assert_classification_inputs(observed, report, diagnostic) -> None:
    assert tuple(candidate.candidate_id for candidate in observed) == (
        "E-0",
        "E-1",
        "E-2",
        "E-3",
    )
    winner = next(candidate for candidate in observed if candidate.candidate_id == report.candidate_id)
    assert diagnostic in winner.diagnostics


def _assert_bootstrap_invocation(
    module,
    problem,
    bootstrap_calls,
    task_runner,
) -> None:
    assert len(bootstrap_calls) == 1
    bootstrap_options = dict(bootstrap_calls[0])
    assert callable(bootstrap_options.pop("progress"))
    assert bootstrap_options.pop("task_runner") is task_runner
    assert bootstrap_options == {
        "sample_count": problem.config.budget.bootstrap_samples,
        "child_seed": module.uncertainty_seed(problem.config),
        "cancelled": None,
    }


def _assert_analyzed_result(result, report, diagnostic, search) -> None:
    assert result.uncertainty is report
    assert result.confidence is ConfidenceClass.CORRELATED
    assert result.classification_evidence == ("strong_correlation",)
    assert diagnostic in result.best_candidate.diagnostics
    assert result.stage_summaries[-1] == FitStageSummary(
        "uncertainty",
        ("E-0", "E-1", "E-2", "E-3"),
        search.best_candidate.objective,
        0,
        ("completed",),
    )


def test_fit_dataset_runs_uncertainty_before_classifying_result(monkeypatch) -> None:
    module = _api()
    problem = _problem()
    candidates = _analysis_candidates(problem)
    search = _search_result(problem, candidates)
    diagnostic = PhysicsDiagnostic(
        "ideal_reflectivity_above_one",
        "synthetic ideal-reflectivity diagnostic",
        (1, 2),
    )
    report = replace(
        _empty_report(search.best_candidate.candidate_id),
        diagnostics=(diagnostic,),
    )
    calls: list[str] = []
    bootstrap_calls: list[dict[str, object]] = []
    report_calls: list[dict[str, object]] = []

    def task_runner(tasks):
        return tuple(task() for task in tasks)

    monkeypatch.setattr(
        module,
        "bootstrap_problem_local",
        partial(_record_bootstrap, calls, bootstrap_calls, problem, search),
    )
    monkeypatch.setattr(
        module,
        "build_uncertainty_report",
        partial(_record_profile_report, calls, report, report_calls),
    )
    monkeypatch.setattr(
        module,
        "classify_result_with_evidence",
        partial(_record_classification, calls, report, diagnostic),
    )
    progress: list[FitProgress] = []

    result = module.analyze_search_result(
        problem,
        search,
        profile_names=("component.0.thickness_a",),
        dataset_id="curve",
        progress=progress.append,
        task_runner=task_runner,
    )

    assert calls == ["bootstrap", "report", "classify"]
    _assert_bootstrap_invocation(module, problem, bootstrap_calls, task_runner)
    assert len(report_calls) == 1
    assert report_calls[0]["task_runner"] is task_runner
    _assert_analyzed_result(result, report, diagnostic, search)
    assert progress == [
        FitProgress(
            "curve",
            "bootstrap",
            0,
            problem.config.budget.bootstrap_samples,
            search.best_candidate.objective,
            f"bootstrap 0/{problem.config.budget.bootstrap_samples}",
        ),
        FitProgress(
            "curve",
            "bootstrap",
            1,
            problem.config.budget.bootstrap_samples,
            search.best_candidate.objective,
            f"bootstrap 1/{problem.config.budget.bootstrap_samples}",
        ),
        FitProgress(
            "curve",
            "bootstrap",
            problem.config.budget.bootstrap_samples,
            problem.config.budget.bootstrap_samples,
            search.best_candidate.objective,
            f"bootstrap {problem.config.budget.bootstrap_samples}/{problem.config.budget.bootstrap_samples}",
        ),
        FitProgress(
            "curve",
            "profile",
            0,
            1,
            search.best_candidate.objective,
            "profile 0/1",
        ),
        FitProgress(
            "curve",
            "profile",
            1,
            1,
            search.best_candidate.objective,
            "profile 1/1: component.0.thickness_a",
        ),
        FitProgress(
            "curve",
            "finalizing",
            0,
            1,
            search.best_candidate.objective,
            "finalizing",
        ),
        FitProgress(
            "curve",
            "finalizing",
            1,
            1,
            search.best_candidate.objective,
            "completed",
        ),
    ]
    _assert_diagnostic_warning(result.warnings)


def test_profiles_collect_results_and_publish_progress_in_declaration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _api()
    problem = _problem()
    unit = encode_physical_vector(problem, {})
    requested = tuple(variable.name for variable in problem.variables)
    completed: list[str] = []
    progress: list[tuple[int, int, str]] = []

    def build(_problem, _unit, names, **options):
        assert options["task_runner"] is run_tasks
        for name in reversed(names):
            completed.append(name)
        return tuple(f"profile:{name}" for name in names)

    def run_tasks(tasks):
        values = tuple(tasks)
        assert progress == []
        results = [None] * len(values)
        for index in reversed(range(len(values))):
            results[index] = values[index]()
        return tuple(results)

    monkeypatch.setattr(module, "build_problem_profiles", build)

    profiles = module._profiles(
        problem,
        unit,
        requested,
        None,
        lambda index, total, name: progress.append((index, total, name)),
        task_runner=run_tasks,
    )

    assert completed == list(reversed(requested))
    assert profiles == tuple(f"profile:{name}" for name in requested)
    assert progress == [(index, len(requested), name) for index, name in enumerate(requested, start=1)]


def test_joint_result_uncertainty_uses_global_candidate_parameter_names() -> None:
    problem = _problem()
    candidate = _candidate(problem, "E-0")

    report = _api().build_uncertainty_report(
        problem,
        (candidate,),
        profile_names=(),
    )

    assert report.correlation_names == tuple(variable.name for variable in problem.variables)
    assert report.correlation_matrix.shape == (
        len(problem.variables),
        len(problem.variables),
    )


def test_problem_objective_information_uses_robust_weights_and_scale_prior() -> None:
    derivatives = import_module("xrr_fitter.analysis.derivatives")
    problem = _problem()
    unit = encode_physical_vector(problem, {})

    information = derivatives.objective_information(problem, unit)

    assert information.shape == (len(problem.variables), len(problem.variables))
    assert np.allclose(information, information.T)


def _with_priors(problem, priors):
    definitions = tuple(
        replace(definition, prior=priors[definition.name]) if definition.name in priors else definition
        for definition in problem.parameter_definitions
    )
    return replace(problem, parameter_definitions=definitions)


def test_uncertainty_report_point_estimate_conflict() -> None:
    problem = _with_priors(
        _problem(),
        {
            "component.0.density_scale": PriorSpec("normal", (0.6, 0.05)),
            "component.0.thickness_a": PriorSpec("normal", (20.0, 5.0)),
        },
    )
    candidates = _analysis_candidates(problem)

    report = _api().build_uncertainty_report(problem, candidates, profile_names=())

    assert "component.0.density_scale" in report.prior_conflicts
    assert "component.0.thickness_a" not in report.prior_conflicts


def test_prior_conflicts_do_not_enter_profile_selection() -> None:
    profiles = import_module("xrr_fitter.analysis.profiles")
    report = replace(_empty_report(None), prior_conflicts=("component.0.density_scale",))

    assert profiles._reported_profile_names(report) == set()
