from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    prepared_data,
    project,
    simple_structure,
)

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
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
from xrr_fitter.model.operations import FitReadiness, ProjectFitResult
from xrr_fitter.model.provenance import fit_search_provenance_sha256
from xrr_fitter.model.structure import MaterialSpec
from xrr_fitter.services import fitting
from xrr_fitter.services.datasets import add_dataset
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


def _automatic_prepared(problem):
    dataset = replace(
        dataset_project("curve"),
        structure=problem.structure,
        instrument=problem.instrument,
        parameter_settings=(),
    )
    return fitting.PreparedDatasetFit("curve", 0, dataset, problem)


def _absorption_trial_candidate(search, *, gain: float, value: float):
    baseline = search.best_candidate
    assert baseline is not None
    parameters = tuple(
        replace(parameter, value=value)
        if parameter.name == "component.0.sld_imag_a2"
        else parameter
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

    def analysis_request(self, dataset_id, problem, search):
        self.calls.append(("analysis-request", dataset_id, problem, search))
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
    assert harness.calls[3][-1] is continued_search


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
    assert [
        (event.stage, event.completed, event.total, event.message)
        for event in progress_events
    ] == [
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


def test_automatic_absorption_problem_preserves_real_mode_constraints() -> None:
    problem = _automatic_problem()
    search = _stage_e_search(problem)
    baseline = search.best_candidate
    assert baseline is not None
    values = {parameter.name: parameter.value for parameter in baseline.parameters}

    trial = fitting._automatic_absorption_problem(
        problem,
        ("component.0.sld_imag_a2",),
        values,
    )

    assert tuple(variable.name for variable in trial.variables) == (
        "component.0.sld_imag_a2",
    )
    absorption = next(
        definition
        for definition in trial.parameter_definitions
        if definition.name == "component.0.sld_imag_a2"
    )
    assert absorption.expert_only is True
    assert absorption.locked is False
    assert next(
        definition
        for definition in trial.parameter_definitions
        if definition.name == "instrument.relative_sigma"
    ).initial == 0.0
    unit = encode_physical_vector(trial, {absorption.name: 2e-6})
    assert evaluate_vector(trial, unit).valid is True


def test_automatic_absorption_rejects_insufficient_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _automatic_problem()
    search = _stage_e_search(problem)
    baseline = search.best_candidate
    assert baseline is not None
    threshold = max(
        abs(baseline.objective)
        * problem.config.confidence.equivalent_cost_fraction,
        problem.config.confidence.equivalent_cost_floor,
    )
    trial = _absorption_trial_candidate(search, gain=0.5 * threshold, value=2e-6)
    monkeypatch.setattr(
        fitting,
        "refit_from_physical_values",
        lambda *_args, **_kwargs: SimpleNamespace(best_candidate=trial),
    )

    updated_prepared, result = fitting._automatic_absorption_search(
        _automatic_prepared(problem),
        search,
        ("component.0.sld_imag_a2",),
        cancelled=None,
    )

    assert updated_prepared.problem is problem
    assert result is search


def test_automatic_absorption_replaces_winner_and_preserves_stage_e_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _automatic_problem()
    search = _stage_e_search(problem)
    baseline = search.best_candidate
    assert baseline is not None
    threshold = max(
        abs(baseline.objective)
        * problem.config.confidence.equivalent_cost_fraction,
        problem.config.confidence.equivalent_cost_floor,
    )
    trial = _absorption_trial_candidate(search, gain=2.0 * threshold, value=2e-6)
    monkeypatch.setattr(
        fitting,
        "refit_from_physical_values",
        lambda *_args, **_kwargs: SimpleNamespace(best_candidate=trial),
    )

    updated_prepared, result = fitting._automatic_absorption_search(
        _automatic_prepared(problem),
        search,
        ("component.0.sld_imag_a2",),
        cancelled=None,
    )

    winner = result.best_candidate
    assert winner is not None
    assert winner is not baseline
    assert winner.candidate_id == baseline.candidate_id
    assert result.stage_summaries[-1].candidate_ids == (
        baseline.candidate_id,
    )
    assert next(
        parameter.value
        for parameter in winner.parameters
        if parameter.name == "component.0.sld_imag_a2"
    ) == pytest.approx(2e-6)
    fixed = next(
        definition
        for definition in updated_prepared.problem.parameter_definitions
        if definition.name == "component.0.sld_imag_a2"
    )
    assert fixed.locked is True
    assert fixed.initial == pytest.approx(2e-6)
    setting = next(
        value
        for value in updated_prepared.updated_dataset.parameter_settings
        if value.name == "component.0.sld_imag_a2"
    )
    assert (setting.initial, setting.locked) == (pytest.approx(2e-6), True)
    assert result.parameter_definitions == updated_prepared.problem.parameter_definitions
    assert result.provenance_sha256 == fit_search_provenance_sha256(
        updated_prepared.problem,
        result,
    )


def test_automatic_clean_evidence_skips_recovery_bootstrap_and_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _automatic_problem()
    prepared = _automatic_prepared(problem)
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
    monkeypatch.setattr(fitting, "run_fit_search", lambda *_args, **_kwargs: search)
    monkeypatch.setattr(
        fitting,
        "run_analysis",
        lambda request, **_kwargs: requests.append(request) or analyzed,
    )
    monkeypatch.setattr(
        fitting,
        "assess_automatic_quality",
        lambda *_args, **_kwargs: decision,
    )
    monkeypatch.setattr(
        fitting,
        "_automatic_profile_recovery",
        lambda *_args, **_kwargs: pytest.fail("clean evidence ran basin recovery"),
    )

    result = fitting.fit_automatic_prepared_dataset(prepared, local_workers=1)

    assert result.passed is True
    assert len(requests) == 2
    assert all(request.bootstrap_enabled is False for request in requests)
    assert all(request.profile_names == () for request in requests)


def test_automatic_search_upgrade_runs_profile_recovery_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(fitting, "run_fit_search", lambda *_args, **_kwargs: search)
    monkeypatch.setattr(
        fitting,
        "run_analysis",
        lambda request, **_kwargs: requests.append(request) or analyzed,
    )
    monkeypatch.setattr(
        fitting,
        "assess_automatic_quality",
        lambda *_args, **_kwargs: next(decisions),
    )

    def recover(*_args, **_kwargs):
        recovery_calls.append(True)
        return search

    monkeypatch.setattr(fitting, "_automatic_profile_recovery", recover)

    result = fitting.fit_automatic_prepared_dataset(prepared, local_workers=1)

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
    missing_structure = fitting.preflight_fit(
        replace(value, datasets=(replace(value.datasets[0], structure=None),))
    )

    assert ready.ready is True
    assert ready.message == "ready"
    assert missing_structure.ready is False
    assert "structure" in missing_structure.message


def test_fitting_composes_search_profile_recovery_and_analysis_in_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    value = _project(tmp_path)
    initial_search = SimpleNamespace(
        best_candidate=SimpleNamespace(objective=0.25, ranking_objective=None)
    )
    continued_search = object()
    decision = SimpleNamespace(
        parameter_name="component.0.thickness_a",
        unit_vector=np.array([0.25]),
    )
    analyzed = final_fit_result()
    progress_events = []
    harness = _FittingHarness(initial_search, decision, continued_search, analyzed)
    _RecordingTaskRunner.reset()

    monkeypatch.setattr(
        fitting,
        "OrderedTaskRunner",
        _RecordingTaskRunner,
        raising=False,
    )
    monkeypatch.setattr(fitting, "run_fit_search", harness.run_search)
    monkeypatch.setattr(fitting, "recover_profile_basin", harness.recover)
    monkeypatch.setattr(fitting, "continue_profile_basin", harness.continue_search)
    monkeypatch.setattr(fitting, "AnalysisRequest", harness.analysis_request)
    monkeypatch.setattr(fitting, "run_analysis", harness.run_analysis)

    result = fitting.fit_project(
        value,
        progress_callback=progress_events.append,
        checkpoint_callback=lambda _project: None,
    )

    assert result.datasets[0].fit_result is analyzed
    _assert_fitting_calls(harness, decision, continued_search)
    _assert_shared_task_runner(harness, value.fit_config.local_workers)
    _assert_basin_progress(progress_events)


def test_joint_fit_reports_finalizing_after_stage_e(monkeypatch) -> None:
    prepared = (
        SimpleNamespace(
            dataset_id="first",
            problem=object(),
            updated_dataset=SimpleNamespace(checkpoint=None),
        ),
        SimpleNamespace(
            dataset_id="second",
            problem=object(),
            updated_dataset=SimpleNamespace(checkpoint=None),
        ),
    )
    searches = (
        SimpleNamespace(
            best_candidate=SimpleNamespace(objective=0.25, ranking_objective=None)
        ),
    )
    analyzed = (object(), object())
    monkeypatch.setattr(fitting, "compile_joint_problem", lambda *_args: object())
    monkeypatch.setattr(fitting, "run_joint_fit", lambda *_args, **_kwargs: searches)
    monkeypatch.setattr(
        fitting,
        "_analyze_joint_searches",
        lambda _problem, _searches: analyzed,
    )
    events = []

    result = fitting.fit_joint_datasets(prepared, (), progress=events.append)

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


def test_fit_automatically_injects_spawn_safe_service_functions(monkeypatch) -> None:
    from xrr_fitter.services import batch

    current = replace(
        project(_automatic_dataset("pending", "batch-1", AutomaticStatus.PENDING)),
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

    monkeypatch.setattr(
        fitting,
        "preflight_automatic_fit",
        lambda *_args, **_kwargs: FitReadiness(True, "ready"),
    )

    def transaction(*args, **kwargs):
        observed.append((args, kwargs))
        return expected

    monkeypatch.setattr(batch, "fit_automatic_transaction", transaction)

    result = fitting.fit_automatically(
        current,
        "batch-1",
        progress_callback=progress,
        checkpoint_callback=checkpoint,
    )

    assert result is expected
    assert observed == [
        (
            (current, "batch-1", progress, checkpoint, None),
            {
                "seed_branches": fitting.service_seed_branches,
                "prepare_dataset": fitting.prepare_dataset_fit,
                "fit_dataset": fitting.fit_automatic_prepared_dataset,
                "fit_joint": fitting.fit_automatic_joint_group,
            },
        )
    ]


def test_automatic_worker_handler_injects_spawn_safe_service_functions(
    monkeypatch,
) -> None:
    from xrr_fitter.services import batch

    current = project(
        _automatic_dataset("pending", "batch-1", AutomaticStatus.PENDING)
    )
    expected = ProjectFitResult("automatic", (), (), current)
    observed = []

    def transaction(*args, **kwargs):
        observed.append((args, kwargs))
        return expected

    def cancelled() -> bool:
        return False

    monkeypatch.setattr(batch, "fit_automatic_transaction", transaction)

    result = fitting.automatic_worker_handler(
        current,
        "batch-1",
        None,
        None,
        cancelled,
    )

    assert result is expected
    assert observed == [
        (
            (current, "batch-1", None, None, cancelled),
            {
                "seed_branches": fitting.service_seed_branches,
                "prepare_dataset": fitting.prepare_dataset_fit,
                "fit_dataset": fitting.fit_automatic_prepared_dataset,
                "fit_joint": fitting.fit_automatic_joint_group,
            },
        )
    ]
