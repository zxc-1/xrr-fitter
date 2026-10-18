"""Keep live GUI evidence bound to the dataset that actually produced it."""

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPersistentModelIndex
from tests.gui.test_fit_progress import _FakeJob, _project

import xrr_fitter.api as api
from xrr_fitter.gui.data.panel import DataPanel
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.controller import FitController
from xrr_fitter.gui.fitting.panel import FitPanel
from xrr_fitter.gui.fitting.progress import ProgressView
from xrr_fitter.gui.main_window import MainWindow


def _frame(owner, stage="B", completed=1, total=10, objective=1.25, **kwargs):
    return api.FitProgress(owner, stage, completed, total, objective, "search", **kwargs)


def _event(sequence, owner, **kwargs):
    return api.OperationEvent(sequence, "progress", progress=_frame(owner, **kwargs))


def _stop(controller):
    controller._finish()


def test_coalescing_preserves_each_dataset_within_the_same_stage(qtbot):
    controller = FitController()
    seen = []
    controller.progress_changed.connect(lambda value: seen.append((value.dataset_id, value.completed)))

    controller._dispatch_batch((_event(0, "a"), _event(1, "a", completed=2), _event(2, "b")))

    assert seen == [("a", 2), ("b", 1)]


def test_interleaved_parallel_frames_draw_only_latest_per_member_each_poll(qtbot):
    controller = FitController()
    seen = []
    controller.progress_changed.connect(lambda value: seen.append((value.dataset_id, value.completed)))
    events = tuple(
        _event(2 * index + offset, owner, completed=index, total=100)
        for index in range(100)
        for offset, owner in enumerate(("a", "b"))
    )

    controller._dispatch_batch(events)

    assert seen == [("a", 99), ("b", 99)]


def test_first_finished_dataset_does_not_complete_the_whole_batch(qtbot, tmp_path, monkeypatch):
    panel = FitPanel(ProjectDocument(_project(tmp_path, count=2)))
    qtbot.addWidget(panel)
    job = _FakeJob()
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: job)
    try:
        assert panel.start_fit()
        panel.controller._dispatch_batch((_event(0, "curve-0", stage="finalizing", total=1),))
        assert panel.progress_view.overall_percent() == 50
        assert panel.progress_view.bar.value() == 500

        panel.controller._dispatch_batch((_event(1, "curve-1", stage="A", completed=0),))
        assert panel.progress_view.overall_percent() == 50
        assert "curve-1" in panel.progress_view.stage_label.text()

        panel.controller._dispatch_batch((_event(2, "curve-1", stage="finalizing", total=1),))
        assert panel.progress_view.overall_percent() == 99
    finally:
        _stop(panel.controller)


def test_stage_times_belong_to_current_dataset_not_previous_member(qtbot):
    readings = iter((0.0, 1.0, 10.0, 11.0, 13.0))
    view = ProgressView(clock=lambda: next(readings))
    qtbot.addWidget(view)
    for owner, stage in (("a", "A"), ("a", "B"), ("a", "finalizing"), ("b", "A"), ("b", "B")):
        view.set_progress(_frame(owner, stage))

    assert view._stage_rows["A"].metric.text() == "2s"
    view.freeze()


def test_independent_live_objectives_do_not_leak_into_other_dataset_rows(qtbot, tmp_path):
    panel = DataPanel(ProjectDocument(_project(tmp_path, count=2)))
    qtbot.addWidget(panel)
    panel.set_running(True)
    panel.set_run_progress(_frame("curve-0", objective=1.25))

    assert panel.run_subline("curve-0") == "拟合中 · J=1.25"
    assert panel.run_subline("curve-1") == ""

    panel.set_run_progress(_frame("curve-1", objective=4.5))
    assert panel.run_subline("curve-0") == "拟合中 · J=1.25"
    assert panel.run_subline("curve-1") == "拟合中 · J=4.50"


def test_joint_rows_use_reported_member_objectives_when_available(qtbot, tmp_path):
    project = api.set_batch_mode(_project(tmp_path, count=2), "joint")
    panel = DataPanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    panel.set_running(True)

    panel.set_run_progress(_frame(None, objective=5.0, dataset_objectives=(("curve-0", 1.0), ("curve-1", 4.0))))

    assert panel.run_subline("curve-0") == "拟合中 · J=1.00"
    assert panel.run_subline("curve-1") == "拟合中 · J=4.00"


def test_progress_updates_existing_rows_without_rebuilding_or_losing_selection(qtbot, tmp_path):
    panel = DataPanel(ProjectDocument(_project(tmp_path, count=2)))
    qtbot.addWidget(panel)
    panel.set_running(True)
    index = QPersistentModelIndex(panel.tree.model().index(0, 0))
    original = panel.tree.topLevelItem(0)
    panel.tree.setCurrentItem(original)
    resets = []
    panel.tree.model().modelReset.connect(lambda: resets.append(True))

    panel.set_run_progress(_frame("curve-0"))

    assert resets == []
    assert index.isValid()
    assert panel.tree.currentItem() is original


def test_preview_does_not_overlay_another_datasets_observations(qtbot, tmp_path):
    project = api.select_active_dataset(_project(tmp_path, count=2), "curve-0")
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.autosave.stop()
    frame = _frame(
        "curve-1",
        preview_qz_a_inv=np.array([0.01, 0.02, 0.03]),
        preview_model_normalized=np.array([0.9, 0.5, 0.1]),
    )
    try:
        window.fit_panel._project_preview(frame)
        assert window.plot_panel.view("log").preview_item is None
        window.fit_panel._project_preview(replace(frame, dataset_id="curve-0"))
        assert window.plot_panel.view("log").preview_item is not None
    finally:
        window.plot_panel.release_resources()


def test_ownerless_joint_preview_is_not_misattributed_to_active_curve(qtbot, tmp_path):
    project = api.set_batch_mode(_project(tmp_path, count=2), "joint")
    panel = FitPanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    emitted = []
    panel.preview_available.connect(lambda *values: emitted.append(values))

    panel._project_preview(
        _frame(
            None,
            preview_qz_a_inv=np.array([0.01, 0.02]),
            preview_model_normalized=np.array([0.9, 0.1]),
        )
    )

    assert emitted == []
