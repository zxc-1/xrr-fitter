"""Exercise request buttons while the worker is still winding down."""

from dataclasses import replace

from PySide6.QtCore import Qt
from tests.gui.test_fit_progress import _FakeJob, _project

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.panel import FitPanel


def _running_panel(qtbot, tmp_path, monkeypatch):
    panel = FitPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    panel.show()
    job = _FakeJob()
    calls = []
    job.pause = lambda: calls.append("pause")
    job.resume = lambda: calls.append("resume")
    job.cancel = lambda: calls.append("cancel")
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: job)
    assert panel.start_fit()
    panel.controller.progress_changed.emit(api.FitProgress("curve-0", "B", 1, 10, 1.0, "search"))
    return panel, job, calls


def test_cancel_disables_conflicting_and_duplicate_requests_but_keeps_force_stop(qtbot, tmp_path, monkeypatch):
    panel, job, calls = _running_panel(qtbot, tmp_path, monkeypatch)
    try:
        qtbot.mouseClick(panel.cancel_button, Qt.MouseButton.LeftButton)

        assert calls == ["cancel"]
        assert not panel.cancel_button.isEnabled()
        assert not panel.pause_button.isEnabled()
        assert not panel.skip_button.isEnabled()
        assert panel.force_button.isVisible() and panel.force_button.isEnabled()
        panel.cancel_shortcut.activated.emit()
        assert calls == ["cancel"]
    finally:
        panel.controller._finish()


def test_pause_button_acknowledges_a_request_without_claiming_worker_ack(qtbot, tmp_path, monkeypatch):
    panel, job, calls = _running_panel(qtbot, tmp_path, monkeypatch)
    try:
        qtbot.mouseClick(panel.pause_button, Qt.MouseButton.LeftButton)

        assert calls == ["pause"]
        assert "暂停已请求" in panel.progress_view.status_summary().position
        assert "已暂停" not in panel.status_text()
        assert "安全" in panel.pause_button.toolTip()
        assert not panel.skip_button.isEnabled()

        qtbot.mouseClick(panel.pause_button, Qt.MouseButton.LeftButton)

        assert calls == ["pause", "resume"]
        assert "暂停已请求" not in panel.progress_view.status_summary().position
        assert panel.skip_button.isEnabled()
    finally:
        panel.controller._finish()


def test_checkpoint_updates_do_not_recompile_preflight_while_running(qtbot, tmp_path, monkeypatch):
    panel, job, calls = _running_panel(qtbot, tmp_path, monkeypatch)
    checks = []

    def ready(kind):
        checks.append(kind)
        return api.FitReadiness(True, "ready")

    monkeypatch.setattr(api, "preflight_fit", lambda _p: ready("manual"))
    monkeypatch.setattr(api, "preflight_automatic_fit", lambda _p: ready("automatic"))
    try:
        panel._publish_checkpoint(api.set_expert_mode(panel.document.project, True))

        assert checks == []
        job.events = (api.OperationEvent(0, "cancelled", cancellation="requested"), api.OperationEvent(1, "stopped"))
        panel.controller.poll_now()
        assert checks == ["manual", "automatic"]
        assert "已保存检查点" in panel.status_text()
    finally:
        panel.controller._finish()


def test_automatic_run_clears_last_run_feedback_and_uses_only_selected_batch(qtbot, tmp_path, monkeypatch):
    project = _project(tmp_path, count=2)
    datasets = tuple(
        replace(
            dataset,
            automation=api.DatasetAutomation(
                import_batch_id=f"batch-{index}",
                role=api.AutomaticRole.UNROUTED,
                status=api.AutomaticStatus.PENDING,
            ),
        )
        for index, dataset in enumerate(project.datasets)
    )
    panel = FitPanel(ProjectDocument(replace(project, datasets=datasets)))
    qtbot.addWidget(panel)
    panel._checkpoint_saved = True
    panel._preview_last_emit = 123.0
    panel.progress_view.joint_label.setText("previous joint run")
    panel.progress_view.joint_label.show()
    monkeypatch.setattr(api, "preflight_automatic_fit", lambda *_a: api.FitReadiness(True, "ready"))
    job = _FakeJob()
    monkeypatch.setattr(api, "start_automatic_fit_job", lambda *_a, **_k: job)
    try:
        assert panel.start_automatic_fit("batch-1")

        assert not panel._checkpoint_saved
        assert panel._preview_last_emit is None
        assert panel.progress_view.joint_label.isHidden()
        panel.controller.progress_changed.emit(api.FitProgress("curve-1", "finalizing", 1, 1, 1.0, "done"))
        assert panel.progress_view.overall_percent() == 100
    finally:
        panel.controller._finish()
