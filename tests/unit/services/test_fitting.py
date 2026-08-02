from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.support.model_cases import final_fit_result, simple_structure
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.projects import new_project
from xrr_fitter.services.structures import set_structure
from xrr_fitter.services import fitting


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
