"""Terminal feedback survives the real window's inspector/result-step routing."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication
from tests.gui.fit_feedback_cases import manual_case, terminal_case, with_checkpoint
from tests.gui.test_fit_progress import _FakeJob, _project
from tests.gui.test_fit_terminal_feedback import _publish

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.main_window import MainWindow


@contextmanager
def _running_window(qtbot, monkeypatch, project, *, guided=False, automatic=True):
    monkeypatch.setattr(api, "preflight_fit", lambda *_a: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "preflight_automatic_fit", lambda *_a: api.FitReadiness(True, "ready"))
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.autosave.stop()
    window.resize(1400, 900)
    window.set_guidance_visible(guided)
    window.show()
    qtbot.waitExposed(window)
    job = _FakeJob()
    monkeypatch.setattr(api, "start_automatic_fit_job", lambda *_a, **_k: job)
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: job)
    try:
        start = window.fit_panel.start_automatic_fit if automatic else window.fit_panel.start_fit
        assert start()
        yield window, job
    finally:
        window.fit_panel.controller._finish()


def _assert_visible_feedback(window, kind, reason):
    QApplication.processEvents()
    label = window.fit_panel.status_label
    visible_rect = QRect(label.mapTo(window, label.rect().topLeft()), label.size())
    assert label.property("statusKind") == kind
    assert reason in label.text()
    assert label.isVisibleTo(window)
    assert not label.visibleRegion().isEmpty()
    assert window.rect().contains(visible_rect)


@pytest.mark.parametrize("guided", (False, True))
@pytest.mark.parametrize(
    "states,kind",
    (
        ((api.AutomaticStatus.FAILED, api.AutomaticStatus.FAILED), "error"),
        ((api.AutomaticStatus.PASSED, api.AutomaticStatus.FAILED), "warn"),
        ((api.AutomaticStatus.REVIEW,), "warn"),
    ),
)
def test_batch_reason_stays_visible_after_result_and_stopped(qtbot, tmp_path, monkeypatch, guided, states, kind):
    project, result = terminal_case(tmp_path, states)
    project = api.select_active_dataset(project, project.datasets[0].dataset_id)
    updated = api.select_active_dataset(result.updated_project, project.datasets[0].dataset_id)
    result = replace(result, updated_project=updated)
    reason = next(row.automation.reason for row in updated.datasets if row.automation.reason)
    with _running_window(qtbot, monkeypatch, project, guided=guided) as (window, job):
        _publish(window.fit_panel, job, result)

        _assert_visible_feedback(window, kind, reason)
        assert not window.fit_panel.controller.is_running


def test_failed_manual_rerun_stays_visible_beside_the_retained_result(qtbot, tmp_path, monkeypatch):
    _initial, valid = manual_case(tmp_path, api.ConfidenceClass.TRUSTED)
    project = api.select_active_dataset(valid.updated_project, valid.datasets[0].dataset_id)
    reason = "no eligible candidate in the new run"
    failed = api.FitResult.failed(reason, valid.datasets[0].fit_result.region_labels.size)
    result = replace(valid, datasets=(replace(valid.datasets[0], fit_result=failed),), updated_project=project)
    with _running_window(qtbot, monkeypatch, project, automatic=False) as (window, job):
        _publish(window.fit_panel, job, result)

        _assert_visible_feedback(window, "error", reason)
        assert window.document.project.datasets[0].last_valid_result is valid.datasets[0].fit_result


@pytest.mark.parametrize("checkpoint", (False, True))
def test_cancel_checkpoint_feedback_is_visible_in_guided_mode(qtbot, tmp_path, monkeypatch, checkpoint):
    project = _project(tmp_path)
    snapshot = with_checkpoint(project) if checkpoint else project
    with _running_window(qtbot, monkeypatch, project, guided=True) as (window, job):
        job.events = (
            api.OperationEvent(0, "checkpoint", checkpoint=snapshot),
            api.OperationEvent(1, "cancelled", cancellation="requested"),
            api.OperationEvent(2, "stopped"),
        )
        window.fit_panel.controller.poll_now()

        reason = "已保存检查点，可从中恢复" if checkpoint else "未产生检查点"
        _assert_visible_feedback(window, "warn", reason)


def test_next_run_clears_the_previous_notice_without_a_blank_banner(qtbot, tmp_path, monkeypatch):
    project, result = terminal_case(tmp_path, (api.AutomaticStatus.REVIEW,))
    with _running_window(qtbot, monkeypatch, project) as (window, job):
        _publish(window.fit_panel, job, result)
        _assert_visible_feedback(window, "warn", "需复核")

        assert window.fit_panel.start_automatic_fit()
        QApplication.processEvents()

        assert not window.fit_panel.status_label.text()
        assert not window.fit_panel.status_label.isVisibleTo(window)


def test_ready_run_does_not_reserve_an_empty_notice_area(qtbot, tmp_path, monkeypatch):
    project, _result = terminal_case(tmp_path, (api.AutomaticStatus.PASSED,))
    with _running_window(qtbot, monkeypatch, project) as (window, _job):
        assert not window.fit_panel.status_label.text()
        assert not window.fit_panel.status_label.isVisibleTo(window)
