"""Automatic fitting service orchestration contracts.

The suite covers readiness, seed allocation, adaptive search, recovery analysis,
absorption release, and project publication through the service boundary. Tests
keep search provenance and immutable project transitions visible while replacing
expensive numerical work with deterministic collaborators where appropriate.

The orchestration cases deliberately retain their call-sequence assertions:
they distinguish service composition defects from numerical fitting defects.
Prepared-dataset, automatic, joint, recovery, absorption, checkpoint, and
sidecar contracts share the same helpers so every phase is tested against one
consistent immutable problem shape.  Lightweight harnesses replace only the
expensive calculation boundary; source loading, request construction,
publication, invalidation, and error translation continue through production
service code.  This keeps failures attributable while preserving the complete
end-to-end ownership contract.

Seed and provenance assertions remain exact rather than approximate.  Error
cases likewise assert the owning service message so an invalid declaration
cannot be mistaken for a failed optimizer run.  Sidecar tests compile the
effective retained settings before judging priors, matching the runtime order.
The suite therefore records both the decisive input and published output.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.drift_cases import one_drift_block_structure
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    prepared_data,
    project,
    simple_structure,
)

from xrr_fitter.analysis.report import AnalysisRequest
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.automatic import candidate_from_physical_values
from xrr_fitter.fit.candidates import best_candidate_index, candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.pipeline import FitSearchRequest
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.fitting import FitConfig, FitSearchResult, FitStageSummary
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.operations import ProjectFitResult
from xrr_fitter.model.parameters import ParameterPrior, ParameterSetting, PriorSpec
from xrr_fitter.model.provenance import fit_search_provenance_sha256
from xrr_fitter.model.structure import MaterialSpec
from xrr_fitter.services import fitting
from xrr_fitter.services.datasets import add_dataset, service_seed_branches
from xrr_fitter.services.fitting_phases.automatic_absorption import (
    _automatic_absorption_problem,
    _automatic_absorption_search,
)
from xrr_fitter.services.fitting_phases.automatic_dataset import (
    fit_automatic_prepared_dataset as fit_automatic_prepared_dataset_phase,
)
from xrr_fitter.services.fitting_phases.base import (
    fit_prepared_dataset as fit_prepared_dataset_phase,
)
from xrr_fitter.services.fitting_phases.joint_execution import (
    fit_joint_datasets as fit_joint_datasets_phase,
)
from xrr_fitter.services.fitting_phases.operations import (
    automatic_worker_handler as automatic_worker_handler_phase,
)
from xrr_fitter.services.fitting_phases.operations import (
    fit_automatically as fit_automatically_phase,
)
from xrr_fitter.services.projects import new_project
from xrr_fitter.services.structures import set_structure


def _source(path: Path) -> Path:
    angles = np.linspace(0.1, 3.2, 48)
    path.write_bytes(xy_bytes(angles, np.geomspace(1.0, 1e-5, angles.size)))
    return path


def _project(tmp_path: Path):
    value = add_dataset(
        new_project(),
        _source(tmp_path / "curve.xy"),
        InstrumentSpec(instrument_id="fitting-service", footprint_mode="none"),
    )
    value = set_structure(value, "curve", simple_structure())
    return replace(value, fit_config=replace(value.fit_config, scale_prior_enabled=False))


def _automatic_problem():
    base = simple_structure()
    direct_sld = MaterialSpec("unknown-film", None, None, 20e-6 + 0.0j)
    structure = replace(
        base,
        components=(
            replace(
                base.components[0],
                material=direct_sld,
                density_scale=1.0,
            ),
        ),
    )
    return compile_fit_problem(
        prepared_data(size=48),
        structure,
        InstrumentSpec(
            footprint_mode="fit",
            resolution_domain="theta",
            background_kind="powerlaw",
        ),
        replace(FitConfig.fast(1201), scale_prior_enabled=False),
    )


def _stage_e_search(problem) -> FitSearchResult:
    unit = encode_physical_vector(problem, {})
    candidate = candidate_from_evaluation(
        problem,
        unit,
        evaluate_vector(problem, unit),
        candidate_id="E-0",
        seed_index=0,
        stop_reason="test candidate",
        nfev=1,
    )
    summary = FitStageSummary(
        "E",
        (candidate.candidate_id,),
        candidate.objective,
        candidate.nfev,
        (candidate.stop_reason,),
    )
    result = FitSearchResult(
        problem.parameter_definitions,
        (candidate,),
        0,
        (),
        (11,),
        (summary,),
        problem.region_labels,
        problem.weights,
    )
    return replace(
        result,
        provenance_sha256=fit_search_provenance_sha256(problem, result),
    )


def _automatic_prepared(problem, *, parameter_priors=()):
    dataset = replace(
        dataset_project("curve"),
        structure=problem.structure,
        instrument=problem.instrument,
        parameter_settings=(),
        parameter_priors=parameter_priors,
    )
    return fitting.PreparedDatasetFit("curve", 0, dataset, problem)


def _absorption_trial_candidate(search, *, gain: float, value: float):
    baseline = search.best_candidate
    assert baseline is not None
    parameters = tuple(
        replace(parameter, value=value) if parameter.name == "component.0.sld_imag_a2" else parameter
        for parameter in baseline.parameters
    )
    return replace(
        baseline,
        parameters=parameters,
        objective=baseline.objective - gain,
        stop_reason="automatic absorption trial",
        nfev=3,
    )


class _RecordingTaskRunner:
    events: list[object] = []
    instances: list[_RecordingTaskRunner] = []

    def __init__(self, max_workers):
        self.events.append(("created", max_workers))
        self.instances.append(self)

    def __enter__(self):
        self.events.append("entered")
        return self

    def __exit__(self, *_args):
        self.events.append("closed")

    def run(self, tasks):
        return tuple(task() for task in tasks)

    @classmethod
    def reset(cls) -> None:
        cls.events = []
        cls.instances = []


class _FittingHarness:
    def __init__(self, initial_search, decision, continued_search, analyzed):
        self.calls: list[object] = []
        self.initial_search = initial_search
        self.decision = decision
        self.continued_search = continued_search
        self.analyzed = analyzed

    def run_search(self, request, **kwargs):
        self.calls.append(
            (
                "search",
                request.dataset_id,
                kwargs["checkpoint"] is not None,
                kwargs["task_runner"].__self__,
            )
        )
        return self.initial_search

    def recover(self, problem, candidate, **_kwargs):
        self.calls.append(("recover", problem, candidate))
        return self.decision

    def continue_search(self, problem, search, center, **kwargs):
        self.calls.append(
            (
                "continue",
                problem,
                search,
                tuple(center),
                kwargs["parameter_name"],
                kwargs["task_runner"].__self__,
            )
        )
        return self.continued_search

    def analysis_request(
        self,
        dataset_id,
        problem,
        search,
        *,
        profile_names=None,
        parameter_priors=(),
    ):
        self.calls.append(
            (
                "analysis-request",
                dataset_id,
                problem,
                search,
                profile_names,
                parameter_priors,
            )
        )
        return "analysis-request"

    def run_analysis(self, request, **kwargs):
        self.calls.append(("analysis", request, kwargs["task_runner"].__self__))
        return self.analyzed


def _assert_fitting_calls(harness, decision, continued_search) -> None:
    assert [call[0] for call in harness.calls] == [
        "search",
        "recover",
        "continue",
        "analysis-request",
        "analysis",
    ]
    assert harness.calls[2][-2] == decision.parameter_name
    assert harness.calls[3][3] is continued_search
    assert harness.calls[3][4] is None


def _assert_shared_task_runner(harness, worker_count: int) -> None:
    assert _RecordingTaskRunner.events == [
        ("created", worker_count),
        "entered",
        "closed",
    ]
    assert len(_RecordingTaskRunner.instances) == 1
    assert harness.calls[0][-1] is _RecordingTaskRunner.instances[0]
    assert harness.calls[2][-1] is _RecordingTaskRunner.instances[0]
    assert harness.calls[4][-1] is _RecordingTaskRunner.instances[0]


def _assert_basin_progress(progress_events) -> None:
    assert [(event.stage, event.completed, event.total, event.message) for event in progress_events] == [
        ("basin-recovery", 0, 1, "checking profile basins"),
        ("basin-recovery", 1, 1, "basin recovery completed"),
    ]


def test_automatic_prepared_result_keeps_quality_reason_consistent() -> None:
    result = fitting.AutomaticPreparedResult(
        SimpleNamespace(),
        final_fit_result(),
        True,
        None,
    )

    assert result.passed is True

    with np.testing.assert_raises(ValueError):
        fitting.AutomaticPreparedResult(
            SimpleNamespace(),
            final_fit_result(),
            True,
            "needs review",
        )


def test_automatic_fit_returns_failed_result_when_stage_e_candidates_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _automatic_problem()
    prepared = _automatic_prepared(problem)
    valid_search = _stage_e_search(problem)
    candidate = valid_search.best_candidate
    assert candidate is not None
    invalid = replace(
        candidate,
        objective=float("inf"),
        valid=False,
        stop_reason="invalid physical candidate",
    )
    search = replace(
        valid_search,
        candidates=(invalid,),
        best_index=None,
        stage_summaries=(
            replace(
                valid_search.stage_summaries[0],
                best_objective=float("inf"),
                stop_reasons=(invalid.stop_reason,),
            ),
        ),
    )
    search = replace(
        search,
        provenance_sha256=fit_search_provenance_sha256(problem, search),
    )
    monkeypatch.setattr(
        fitting,
        "run_fit_search",
        lambda *_args, **_kwargs: search,
    )

    result = fitting.fit_automatic_prepared_dataset(
        prepared,
        local_workers=1,
    )

    assert result.passed is False
    assert result.reason == "no valid candidate"
    assert result.fit_result.best_candidate is None
    assert result.fit_result.candidates == (invalid,)


def test_automatic_absorption_problem_preserves_real_mode_constraints() -> None:
    problem = _automatic_problem()
    search = _stage_e_search(problem)
    baseline = search.best_candidate
    assert baseline is not None
    values = {parameter.name: parameter.value for parameter in baseline.parameters}

    trial = _automatic_absorption_problem(
        problem,
        ("component.0.sld_imag_a2",),
        values,
        compile_fit_problem=compile_fit_problem,
    )

    assert tuple(variable.name for variable in trial.variables) == ("component.0.sld_imag_a2",)
    absorption = next(
        definition for definition in trial.parameter_definitions if definition.name == "component.0.sld_imag_a2"
    )
    assert absorption.expert_only is True
    assert absorption.locked is False
    assert (
        next(
            definition for definition in trial.parameter_definitions if definition.name == "instrument.relative_sigma"
        ).initial
        == 0.0
    )
    unit = encode_physical_vector(trial, {absorption.name: 2e-6})
    assert evaluate_vector(trial, unit).valid is True


def test_automatic_absorption_rejects_insufficient_gain() -> None:
    problem = _automatic_problem()
    search = _stage_e_search(problem)
    baseline = search.best_candidate
    assert baseline is not None
    threshold = max(
        abs(baseline.objective) * problem.config.confidence.equivalent_cost_fraction,
        problem.config.confidence.equivalent_cost_floor,
    )
    trial = _absorption_trial_candidate(search, gain=0.5 * threshold, value=2e-6)
    updated_prepared, result = _automatic_absorption_search(
        _automatic_prepared(problem),
        search,
        ("component.0.sld_imag_a2",),
        cancelled=None,
        compile_fit_problem=compile_fit_problem,
        refit_from_physical_values=lambda *_args, **_kwargs: SimpleNamespace(best_candidate=trial),
        candidate_from_physical_values=candidate_from_physical_values,
        evaluate_vector=evaluate_vector,
        candidate_from_evaluation=candidate_from_evaluation,
        best_candidate_index=best_candidate_index,
        fit_search_provenance_sha256=fit_search_provenance_sha256,
    )

    assert updated_prepared.problem is problem
    assert result is search


def test_automatic_absorption_replaces_winner_and_preserves_stage_e_lineage() -> None:
    problem = _automatic_problem()
    search = _stage_e_search(problem)
    baseline = search.best_candidate
    assert baseline is not None
    threshold = max(
        abs(baseline.objective) * problem.config.confidence.equivalent_cost_fraction,
        problem.config.confidence.equivalent_cost_floor,
    )
    trial = _absorption_trial_candidate(search, gain=2.0 * threshold, value=2e-6)
    updated_prepared, result = _automatic_absorption_search(
        _automatic_prepared(problem),
        search,
        ("component.0.sld_imag_a2",),
        cancelled=None,
        compile_fit_problem=compile_fit_problem,
        refit_from_physical_values=lambda *_args, **_kwargs: SimpleNamespace(best_candidate=trial),
        candidate_from_physical_values=candidate_from_physical_values,
        evaluate_vector=evaluate_vector,
        candidate_from_evaluation=candidate_from_evaluation,
        best_candidate_index=best_candidate_index,
        fit_search_provenance_sha256=fit_search_provenance_sha256,
    )

    winner = result.best_candidate
    assert winner is not None
    imag_value = next(parameter.value for parameter in winner.parameters if parameter.name == "component.0.sld_imag_a2")
    fixed = next(
        definition
        for definition in updated_prepared.problem.parameter_definitions
        if definition.name == "component.0.sld_imag_a2"
    )
    setting = next(
        value
        for value in updated_prepared.updated_dataset.parameter_settings
        if value.name == "component.0.sld_imag_a2"
    )
    assert (
        winner is not baseline,
        winner.candidate_id,
        result.stage_summaries[-1].candidate_ids,
        imag_value,
        fixed.locked,
        fixed.initial,
        setting.initial,
        setting.locked,
        result.parameter_definitions,
        result.provenance_sha256,
    ) == (
        True,
        baseline.candidate_id,
        (baseline.candidate_id,),
        pytest.approx(2e-6),
        True,
        pytest.approx(2e-6),
        pytest.approx(2e-6),
        True,
        updated_prepared.problem.parameter_definitions,
        fit_search_provenance_sha256(updated_prepared.problem, result),
    )


def test_automatic_clean_evidence_skips_recovery_bootstrap_and_profiles() -> None:
    problem = _automatic_problem()
    from xrr_fitter.model.parameters import ParameterPrior, PriorSpec

    priors = (ParameterPrior("component.0.density_scale", PriorSpec("normal", (0.6, 0.05))),)
    prepared = _automatic_prepared(problem, parameter_priors=priors)
    search = _stage_e_search(problem)
    analyzed = final_fit_result()
    requests = []
    decision = SimpleNamespace(
        passed=True,
        search_upgrade=False,
        absorption_names=(),
        profile_names=(),
        reasons=(),
    )

    def analyze(request, **_kwargs):
        requests.append(request)
        return analyzed

    _RecordingTaskRunner.reset()
    result = fit_automatic_prepared_dataset_phase(
        prepared,
        local_workers=1,
        fit_search_request=FitSearchRequest,
        run_fit_search=lambda *_args, **_kwargs: search,
        analysis_request=AnalysisRequest,
        run_analysis=analyze,
        assess_automatic_quality=lambda *_args, **_kwargs: decision,
        automatic_profile_recovery=lambda *_args, **_kwargs: pytest.fail("clean evidence ran basin recovery"),
        automatic_absorption_search=lambda current, current_search, *_args, **_kwargs: (
            current,
            current_search,
        ),
        task_runner_factory=_RecordingTaskRunner,
    )

    assert result.passed is True
    assert len(requests) == 2
    assert all(request.bootstrap_enabled is False for request in requests)
    assert all(request.profile_names == () for request in requests)
    assert all(request.parameter_priors == priors for request in requests)


def test_automatic_search_upgrade_runs_profile_recovery_at_most_once() -> None:
    problem = _automatic_problem()
    prepared = _automatic_prepared(problem)
    search = _stage_e_search(problem)
    analyzed = final_fit_result()
    recovery_calls = []
    decisions = iter(
        (
            SimpleNamespace(
                passed=False,
                search_upgrade=True,
                absorption_names=(),
                profile_names=(),
                reasons=("distinct_equivalent_clusters",),
            ),
            SimpleNamespace(
                passed=False,
                search_upgrade=True,
                absorption_names=(),
                profile_names=("component.0.thickness_a",),
                reasons=("distinct_equivalent_clusters",),
            ),
            SimpleNamespace(
                passed=False,
                search_upgrade=True,
                absorption_names=(),
                profile_names=("component.0.thickness_a",),
                reasons=("distinct_equivalent_clusters",),
            ),
        )
    )
    requests = []

    def analyze(request, **_kwargs):
        requests.append(request)
        return analyzed

    def recover(*_args, **_kwargs):
        recovery_calls.append(True)
        return search

    _RecordingTaskRunner.reset()
    result = fit_automatic_prepared_dataset_phase(
        prepared,
        local_workers=1,
        fit_search_request=FitSearchRequest,
        run_fit_search=lambda *_args, **_kwargs: search,
        analysis_request=AnalysisRequest,
        run_analysis=analyze,
        assess_automatic_quality=lambda *_args, **_kwargs: next(decisions),
        automatic_profile_recovery=recover,
        automatic_absorption_search=lambda current, current_search, *_args, **_kwargs: (
            current,
            current_search,
        ),
        task_runner_factory=_RecordingTaskRunner,
    )

    assert recovery_calls == [True]
    assert len(requests) == 3
    assert requests[-1].profile_names == ("component.0.thickness_a",)
    assert result.passed is False
    assert result.reason == "distinct_equivalent_clusters"


def test_preflight_loads_current_sources_and_compiles_declared_structure(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)

    ready = fitting.preflight_fit(value)
    missing_structure = fitting.preflight_fit(replace(value, datasets=(replace(value.datasets[0], structure=None),)))

    assert ready.ready is True
    assert ready.message == "ready"
    assert missing_structure.ready is False
    assert "structure" in missing_structure.message


def test_preflight_rejects_stale_parameter_priors(tmp_path: Path) -> None:
    value = _project(tmp_path)
    stale = ParameterPrior("component.99.thickness_a", PriorSpec("uniform"))
    value = replace(
        value,
        datasets=(replace(value.datasets[0], parameter_priors=(stale,)),),
    )

    readiness = fitting.preflight_fit(value)

    assert readiness.ready is False
    assert "unknown parameter name" in readiness.message


def test_reconcile_parameter_sidecars_validates_priors_against_retained_settings(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    prepared = fitting.prepare_dataset_fit(value, "curve", value.master_seed)
    thickness = next(
        definition
        for definition in prepared.problem.parameter_definitions
        if definition.name == "component.0.thickness_a"
    )
    retained_setting = ParameterSetting(
        thickness.name,
        thickness.initial,
        10.0,
        30.0,
    )
    stale_prior = ParameterPrior(thickness.name, PriorSpec("normal", (50.0, 5.0)))
    value = replace(
        value,
        datasets=(
            replace(
                value.datasets[0],
                parameter_settings=(retained_setting,),
                parameter_priors=(stale_prior,),
            ),
        ),
    )

    reconciled = fitting._reconcile_parameter_sidecars(value, "curve")

    assert reconciled.datasets[0].parameter_settings == (retained_setting,)
    assert reconciled.datasets[0].parameter_priors == ()


def test_reconcile_parameter_sidecars_drops_generated_constraint_prior(
    tmp_path: Path,
) -> None:
    value = set_structure(_project(tmp_path), "curve", one_drift_block_structure())
    prior = ParameterPrior(
        "component.0.repeat.1.layer.0.thickness_a",
        PriorSpec("uniform"),
    )
    value = replace(
        value,
        datasets=(replace(value.datasets[0], parameter_priors=(prior,)),),
    )

    reconciled = fitting._reconcile_parameter_sidecars(value, "curve")

    assert reconciled.datasets[0].parameter_priors == ()


def test_fitting_composes_search_profile_recovery_and_analysis_in_order(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    prepared = fitting.prepare_dataset_fit(value, "curve", value.master_seed)
    initial_search = SimpleNamespace(best_candidate=SimpleNamespace(objective=0.25, ranking_objective=None))
    continued_search = object()
    decision = SimpleNamespace(
        parameter_name="component.0.thickness_a",
        unit_vector=np.array([0.25]),
    )
    analyzed = final_fit_result()
    progress_events = []
    harness = _FittingHarness(initial_search, decision, continued_search, analyzed)
    _RecordingTaskRunner.reset()

    result = fit_prepared_dataset_phase(
        prepared,
        progress=progress_events.append,
        checkpoint=lambda _checkpoint: None,
        fit_search_request=FitSearchRequest,
        run_fit_search=harness.run_search,
        recover_profile_basin=harness.recover,
        continue_profile_basin=harness.continue_search,
        analysis_request=harness.analysis_request,
        run_analysis=harness.run_analysis,
        task_runner_factory=_RecordingTaskRunner,
    )

    assert result is analyzed
    _assert_fitting_calls(harness, decision, continued_search)
    _assert_shared_task_runner(harness, prepared.problem.config.local_workers)
    _assert_basin_progress(progress_events)


# Statistical corpus fits reuse production analysis while requesting only
# metric-owned profiles. Pin this service boundary so they cannot silently
# regress to default full-profile selection.
def test_fitting_forwards_explicit_profile_names_to_analysis_request(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    prepared = fitting.prepare_dataset_fit(value, "curve", value.master_seed)
    search = SimpleNamespace(best_candidate=SimpleNamespace(objective=0.25, ranking_objective=None))
    analyzed = final_fit_result()
    harness = _FittingHarness(search, None, search, analyzed)
    requested = (
        "component.0.thickness_a",
        "component.0.density_scale",
    )

    result = fit_prepared_dataset_phase(
        prepared,
        profile_names=requested,
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
    assert request_call == (
        "analysis-request",
        prepared.dataset_id,
        prepared.problem,
        search,
        requested,
        (),
    )


def test_joint_fit_reports_finalizing_after_stage_e() -> None:
    prepared = (
        SimpleNamespace(
            dataset_id="first",
            problem=object(),
            updated_dataset=SimpleNamespace(checkpoint=None, parameter_priors=()),
        ),
        SimpleNamespace(
            dataset_id="second",
            problem=object(),
            updated_dataset=SimpleNamespace(checkpoint=None, parameter_priors=()),
        ),
    )
    searches = (SimpleNamespace(best_candidate=SimpleNamespace(objective=0.25, ranking_objective=None)),)
    analyzed = (object(), object())
    events = []

    result = fit_joint_datasets_phase(
        prepared,
        (),
        progress=events.append,
        compile_joint_problem=lambda *_args: object(),
        joint_fit_request=lambda problem, checkpoints: (problem, checkpoints),
        run_joint_fit=lambda *_args, **_kwargs: searches,
        analyze_joint_searches=lambda _problem, _searches, _priors: analyzed,
    )

    assert result is analyzed
    assert [
        (
            event.dataset_id,
            event.stage,
            event.completed,
            event.total,
            event.best_objective,
            event.message,
        )
        for event in events
    ] == [
        (None, "finalizing", 0, 1, 0.25, "finalizing joint fit"),
        (None, "finalizing", 1, 1, 0.25, "completed"),
    ]


def _automatic_dataset(
    dataset_id: str,
    import_batch_id: str,
    status: AutomaticStatus,
):
    reason = "quality review" if status is AutomaticStatus.REVIEW else None
    return replace(
        dataset_project(dataset_id),
        automation=DatasetAutomation(
            import_batch_id=import_batch_id,
            role=AutomaticRole.UNROUTED,
            status=status,
            reason=reason,
        ),
    )


def test_automatic_dataset_ids_select_only_runnable_statuses_and_batch() -> None:
    value = project(
        _automatic_dataset("pending", "batch-1", AutomaticStatus.PENDING),
        _automatic_dataset("refining", "batch-1", AutomaticStatus.REFINING),
        _automatic_dataset("review", "batch-2", AutomaticStatus.REVIEW),
        _automatic_dataset("passed", "batch-1", AutomaticStatus.PASSED),
        dataset_project("manual"),
    )

    assert fitting._automatic_dataset_ids(value, None) == (
        "pending",
        "refining",
        "review",
    )
    assert fitting._automatic_dataset_ids(value, "batch-1") == (
        "pending",
        "refining",
    )


def test_automatic_operations_phase_injects_spawn_safe_service_functions(
    tmp_path: Path,
) -> None:
    value = _project(tmp_path)
    current = replace(
        value,
        datasets=(
            replace(
                value.datasets[0],
                automation=DatasetAutomation(
                    import_batch_id="batch-1",
                    role=AutomaticRole.UNROUTED,
                    status=AutomaticStatus.PENDING,
                ),
            ),
        ),
        measurement_preset=MeasurementPreset(
            "lab",
            BeamSpec("monochromatic"),
            InstrumentSpec(instrument_id="lab"),
        ),
    )
    expected = ProjectFitResult("automatic", (), (), current)
    observed = []

    def progress(_value) -> None:
        return None

    def checkpoint(_value) -> None:
        return None

    def transaction(*args, **kwargs):
        observed.append((args, kwargs))
        return expected

    def prepare_dataset(*_args, **_kwargs):
        return SimpleNamespace(
            problem=SimpleNamespace(parameter_definitions=()),
            updated_dataset=SimpleNamespace(parameter_priors=()),
        )

    def evaluate_declared_initial(_problem):
        return SimpleNamespace(valid=True)

    def fit_dataset(*_args, **_kwargs):
        return None

    def fit_joint(*_args, **_kwargs):
        return ()

    result = fit_automatically_phase(
        current,
        "batch-1",
        progress_callback=progress,
        checkpoint_callback=checkpoint,
        fit_automatic_transaction=transaction,
        prepare_dataset_fit=prepare_dataset,
        validate_parameter_priors=lambda *_args, **_kwargs: None,
        evaluate_declared_initial=evaluate_declared_initial,
        fit_automatic_prepared_dataset=fit_dataset,
        fit_automatic_joint_group=fit_joint,
    )

    assert result is expected
    assert observed == [
        (
            (current, "batch-1", progress, checkpoint, None),
            {
                "seed_branches": service_seed_branches,
                "prepare_dataset": prepare_dataset,
                "fit_dataset": fit_dataset,
                "fit_joint": fit_joint,
            },
        )
    ]


def test_automatic_worker_phase_injects_spawn_safe_service_functions(
    tmp_path: Path,
) -> None:
    base = _project(tmp_path)
    current = replace(
        base,
        datasets=(
            replace(
                base.datasets[0],
                automation=DatasetAutomation(
                    import_batch_id="batch-1",
                    role=AutomaticRole.UNROUTED,
                    status=AutomaticStatus.PENDING,
                ),
            ),
        ),
        measurement_preset=MeasurementPreset(
            "lab",
            BeamSpec("monochromatic"),
            InstrumentSpec(instrument_id="lab"),
        ),
    )
    expected = ProjectFitResult("automatic", (), (), current)
    observed = []

    def transaction(*args, **kwargs):
        observed.append((args, kwargs))
        return expected

    def cancelled() -> bool:
        return False

    def prepare_dataset(*_args, **_kwargs):
        return SimpleNamespace(
            problem=SimpleNamespace(parameter_definitions=()),
            updated_dataset=SimpleNamespace(parameter_priors=()),
        )

    def evaluate_declared_initial(_problem):
        return SimpleNamespace(valid=True)

    def fit_dataset(*_args, **_kwargs):
        return None

    def fit_joint(*_args, **_kwargs):
        return ()

    result = automatic_worker_handler_phase(
        current,
        "batch-1",
        None,
        None,
        cancelled,
        fit_automatic_transaction=transaction,
        prepare_dataset_fit=prepare_dataset,
        validate_parameter_priors=lambda *_args, **_kwargs: None,
        evaluate_declared_initial=evaluate_declared_initial,
        fit_automatic_prepared_dataset=fit_dataset,
        fit_automatic_joint_group=fit_joint,
    )

    assert result is expected
    assert observed == [
        (
            (current, "batch-1", None, None, cancelled),
            {
                "seed_branches": service_seed_branches,
                "prepare_dataset": prepare_dataset,
                "fit_dataset": fit_dataset,
                "fit_joint": fit_joint,
            },
        )
    ]


# `_sld_bands` is the fitting boundary that replays retained MCMC draws into an
# SLD envelope while the report is assembled, so the three tests below pin its
# full contract against silent regressions. State one: without a report there is
# nothing to replay, so both the bands and the report come back None. State two:
# when a replay raises (an unknown parameter axis here), the reason is folded into
# the report's existing warnings channel and the fit still succeeds rather than
# aborting on a diagnostic-only failure. State three: a clean replay must leave
# the warnings tuple identical, proving the success path never touches it. The
# helper mints the smallest legal McmcReport carrying samples along one axis.
BANDS_WAVELENGTH_A = 1.5406


def _bands_mcmc_report(values, names=("component.0.thickness_a",)):
    from xrr_fitter.model.analysis import McmcConfig, McmcReport

    samples = np.asarray(values, dtype=float).reshape(-1, len(names))
    steps = samples.shape[0]
    walkers = 4
    return McmcReport(
        config=McmcConfig(walkers=walkers, burn_in=0, production_steps=steps),
        child_seed=7,
        parameter_names=names,
        samples_physical=samples,
        log_probability=np.zeros(steps),
        acceptance_fraction=np.full(walkers, 0.4),
        split_rhat=np.ones(len(names)),
        effective_sample_size=np.full(len(names), float(steps)),
        boundary_hits=(),
    )


def test_sld_bands_returns_none_when_no_mcmc_report_is_available() -> None:
    bands, report = fitting._sld_bands(simple_structure(), None, BANDS_WAVELENGTH_A)

    assert bands is None
    assert report is None


def test_sld_bands_appends_the_reason_to_the_report_warnings_when_replay_fails() -> None:
    report = _bands_mcmc_report(np.full((8, 1), 20.0), names=("component.9.thickness_a",))

    bands, updated = fitting._sld_bands(simple_structure(), report, BANDS_WAVELENGTH_A)

    assert bands is None
    assert updated is not report
    assert len(updated.warnings) == len(report.warnings) + 1
    assert any("component.9.thickness_a" in warning for warning in updated.warnings)


def test_sld_bands_replays_samples_into_bands_without_touching_warnings() -> None:
    from xrr_fitter.model.analysis import SldUncertaintyBands

    report = _bands_mcmc_report(np.linspace(18.0, 22.0, 12))

    bands, updated = fitting._sld_bands(simple_structure(), report, BANDS_WAVELENGTH_A)

    assert isinstance(bands, SldUncertaintyBands)
    assert updated.warnings == report.warnings
