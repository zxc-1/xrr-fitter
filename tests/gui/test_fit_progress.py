from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QLabel, QProgressBar, QPushButton

import xrr_fitter.api as api


class _FakeJob:
    def __init__(self, events=()) -> None:
        self.events = tuple(events)
        self.is_running = True
        self.closed = False

    def poll(self):
        events, self.events = self.events, ()
        if any(event.kind == "stopped" for event in events):
            self.is_running = False
        return events

    def cancel(self) -> None:
        pass

    def force_stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path, count=1):
    project = api.new_project()
    air = api.MaterialSpec("Air", None, None, 0.0j)
    silicon = api.MaterialSpec("Si", "Si", 2.329)
    silica = api.MaterialSpec("SiO2", "SiO2", 2.20)
    structure = api.StructureSpec(
        air,
        (api.LayerSpec("film", silica, 40.0),),
        silicon,
    )
    for index in range(count):
        source = _write_curve(tmp_path / f"curve-{index}.xy")
        project = api.add_dataset(project, source, api.InstrumentSpec())
        project = api.set_structure(project, f"curve-{index}", structure)
    return project


def _panel(qtbot, tmp_path, count=1):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    panel = FitPanel(ProjectDocument(_project(tmp_path, count)))
    qtbot.addWidget(panel)
    return panel


def test_progress_view_renders_stage_counts_objective_and_message(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    progress = api.FitProgress("curve", "stage-c", 3, 10, 0.0125, "local search")

    view.set_progress(progress)

    bar = view.findChild(QProgressBar, "fitProgressBar")
    stage = view.findChild(QLabel, "fitProgressStage")
    detail = view.findChild(QLabel, "fitProgressDetail")
    assert (bar.value(), bar.maximum()) == (3, 10)
    assert stage.text() == "curve · stage-c"
    assert "local search" in detail.text()
    assert "0.0125" in detail.text()


def test_fit_panel_controls_running_state_and_publishes_result(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    updated = api.set_expert_mode(original, True)
    result = api.ProjectFitResult("independent", (), (), updated)
    job = _FakeJob(
        (
            api.OperationEvent(0, "fit_result", fit_result=result),
            api.OperationEvent(1, "stopped"),
        )
    )
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    start = panel.findChild(QPushButton, "startFitButton")
    cancel = panel.findChild(QPushButton, "cancelFitButton")

    assert panel.start_fit() is True
    assert start.isEnabled() is False
    assert cancel.isEnabled() is True

    panel.controller.poll_now()

    assert panel.document.project is updated
    assert panel.is_running is False
    assert start.isEnabled() is True
    assert cancel.isEnabled() is False


def test_fit_panel_checkpoint_adopts_project_before_terminal_result(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    checkpoint = api.set_expert_mode(panel.document.project, True)
    job = _FakeJob(
        (
            api.OperationEvent(0, "checkpoint", checkpoint=checkpoint),
            api.OperationEvent(1, "cancelled", cancellation="requested"),
            api.OperationEvent(2, "stopped"),
        )
    )
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    results: list[object] = []
    panel.result_published.connect(results.append)

    panel.start_fit()
    panel.controller.poll_now()

    assert panel.document.project is checkpoint
    assert results == []
    assert panel.status_text() == "已取消：requested"


def test_fit_panel_preflight_failure_does_not_start_worker(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    starts: list[object] = []
    monkeypatch.setattr(
        api,
        "preflight_fit",
        lambda _project: api.FitReadiness(False, "结构尚未准备"),
    )
    monkeypatch.setattr(api, "start_fit_job", starts.append)

    assert panel.start_fit() is False
    assert starts == []
    assert panel.status_text() == "结构尚未准备"


def test_batch_mode_selector_is_visible_persisted_and_rejects_one_dataset_joint(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    selector = panel.findChild(QComboBox, "batchModeSelector")

    assert [selector.itemData(index) for index in range(selector.count())] == [
        "independent",
        "joint",
    ]
    before = panel.document.project
    try:
        panel.set_batch_mode("joint")
    except ValueError as error:
        assert "requires at least two datasets" in str(error)
    else:
        raise AssertionError("one-dataset joint mode was accepted")
    assert panel.document.project is before

    joint = _panel(qtbot, tmp_path / "joint", count=2)
    assert joint.set_batch_mode("joint") is True
    assert joint.document.project.batch_mode == "joint"
