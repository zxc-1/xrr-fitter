"""MCMC actions must publish feedback in the dialog the user is operating.

Only process startup is replaced: buttons, validation, controller dispatch,
project publication and widget parenting run through production code. The fake
job supplies the same immutable progress/terminal values as the worker; these
are interaction regressions, not sampling convergence or timing benchmarks.

Elapsed-clock tests advance a monotonic input rather than sleeping through an
arbitrary hardware-dependent interval. Dialog visibility is checked relative to
the dialog itself because the main inspector intentionally hides secondary UI.
"""

from dataclasses import replace

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QProgressBar, QWidget
from tests.gui.test_uncertainty_dialog import _panel

import xrr_fitter.api as api


class _Job:
    def __init__(self):
        self.events = ()
        self.cancel_calls = 0
        self.force_calls = 0
        self.close_calls = 0

    def poll(self):
        events, self.events = self.events, ()
        return events

    def cancel(self):
        self.cancel_calls += 1

    def force_stop(self):
        self.force_calls += 1

    def close(self):
        self.close_calls += 1


@pytest.fixture
def sampling(qtbot, monkeypatch):
    panel = _panel(qtbot, expert=True)
    panel.set_secondary_visible(False)
    panel.burn_in.setValue(10)
    panel.production.setValue(90)
    dialog = panel.open_uncertainty_dialog()
    qtbot.addWidget(dialog)
    dialog.resize(860, 680)
    dialog.show()
    job = _Job()
    starts = []
    monkeypatch.setattr(api, "start_mcmc_job", lambda *args: (starts.append(args), job)[1])
    yield panel, dialog, job, starts
    if panel.controller.is_running:
        panel.controller._finish()


def _click(qtbot, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def _progress(completed, *, dataset_id="curve", stage="MCMC"):
    value = api.FitProgress(
        dataset_id,
        stage,
        completed,
        100,
        0.2,
        "MCMC sampling",
        iteration=completed,
        acceptance_rate=0.42,
        step_size=2.0,
    )
    return api.OperationEvent(completed, "progress", progress=value)


def _deliver(panel, job, *events):
    job.events = events
    panel.controller.poll_now()


def _visible_status(panel, dialog, fragment):
    assert panel.status_label.isVisibleTo(dialog)
    assert fragment in panel.status_text()


def _view(dialog):
    value = dialog.findChild(QWidget, "mcmcProgress")
    assert value is not None, "MCMC progress has no visible projection"
    assert value.isVisibleTo(dialog)
    return value


def _heartbeat(view):
    timer = view.findChild(QTimer, "mcmcProgressHeartbeat")
    assert timer is not None
    return timer


def _text(view):
    return "\n".join(label.text() for label in view.findChildren(QLabel))


def _idle(panel, dialog):
    assert not panel.controller.is_running
    assert not _heartbeat(_view(dialog)).isActive()
    assert not panel.cancel_button.isEnabled()
    assert not panel.force_button.isEnabled()


def test_invalid_walkers_error_is_visible_where_run_was_clicked(qtbot, sampling):
    panel, dialog, _job, starts = sampling
    panel.walkers.setValue(2)
    _click(qtbot, panel.mcmc_button)
    _visible_status(panel, dialog, "walkers")
    assert starts == []


def test_startup_failure_is_visible_and_leaves_no_running_clock(qtbot, sampling, monkeypatch):
    panel, dialog, _job, _starts = sampling

    def fail_start(*_args):
        raise OSError("worker spawn failed")

    monkeypatch.setattr(api, "start_mcmc_job", fail_start)
    source = panel.document.project
    _click(qtbot, panel.mcmc_button)
    _visible_status(panel, dialog, "worker spawn failed")
    assert panel.document.project is source
    _idle(panel, dialog)


def test_run_shows_source_and_elapsed_before_first_worker_frame(qtbot, sampling):
    panel, dialog, _job, starts = sampling
    _click(qtbot, panel.mcmc_button)
    _visible_status(panel, dialog, "MCMC 已启动")
    text = _text(_view(dialog))
    assert "curve" in text and "candidate-a" in text
    assert "已用" in text and "--:--" in text
    assert len(starts) == 1


def test_real_progress_signal_updates_visible_steps_and_acceptance(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _deliver(panel, job, _progress(37))
    view = _view(dialog)
    text = _text(view)
    assert "37 / 100" in text
    assert "42.0%" in text
    assert view.findChild(QProgressBar).value() == 37
    assert "--:--" in text


def test_eta_needs_two_advancing_frames_and_expires_without_false_zero(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    view = _view(dialog)
    now = [view._clock()]
    view._clock = lambda: now[0]
    _deliver(panel, job, _progress(20))
    now[0] += 10
    _deliver(panel, job, _progress(40))
    assert "预计剩余 00:30" in _text(view)
    now[0] += 60
    _heartbeat(view).timeout.emit()
    assert "预计剩余 00:00" not in _text(view)
    assert "估算中" in _text(view)


def test_dialog_reopening_retains_inflight_progress_and_clock(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _deliver(panel, job, _progress(37))
    view = _view(dialog)
    qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    assert not dialog.isVisible()
    _deliver(panel, job, _progress(52))
    reopened = panel.open_uncertainty_dialog()
    reopened.show()
    assert reopened is dialog and _view(reopened) is view
    assert "52 / 100" in _text(view)
    assert _heartbeat(view).isActive()


def test_cancel_acknowledges_request_and_disables_repeated_cancel(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _click(qtbot, panel.cancel_button)
    _visible_status(panel, dialog, "正在取消 MCMC")
    _click(qtbot, panel.cancel_button)
    assert job.cancel_calls == 1
    assert not panel.cancel_button.isEnabled()
    assert panel.force_button.isEnabled()
    assert not panel.mcmc_button.isEnabled() and not panel.walkers.isEnabled()


def test_cancel_state_survives_progress_and_readonly_reopening(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _click(qtbot, panel.cancel_button)
    _deliver(panel, job, _progress(37))
    panel.open_uncertainty_dialog(sampling=False).show()
    _visible_status(panel, dialog, "正在取消 MCMC")
    assert not panel.cancel_button.isEnabled()
    assert panel.force_button.isEnabled() and panel.force_button.isVisibleTo(dialog)


def test_force_stop_feedback_waits_for_worker_and_deduplicates_clicks(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _click(qtbot, panel.force_button)
    _visible_status(panel, dialog, "正在强制停止 MCMC")
    _click(qtbot, panel.force_button)
    assert job.force_calls == 1
    assert panel.controller.is_running
    assert not panel.force_button.isEnabled() and not panel.cancel_button.isEnabled()


@pytest.mark.parametrize("kind", ("cancelled", "error"))
def test_unsuccessful_terminal_keeps_project_and_freezes_visible_elapsed(qtbot, sampling, kind):
    panel, dialog, job, _starts = sampling
    source = panel.document.project
    _click(qtbot, panel.mcmc_button)
    value = (
        api.OperationEvent(2, "cancelled", cancellation="requested")
        if kind == "cancelled"
        else api.OperationEvent(2, "error", error=api.OperationError("ValueError", "bad chain", "traceback"))
    )
    _deliver(panel, job, _progress(37), value, api.OperationEvent(3, "stopped"))
    _visible_status(panel, dialog, "已取消" if kind == "cancelled" else "bad chain")
    _idle(panel, dialog)
    assert panel.document.project is source
    view = _view(dialog)
    before = _text(view)
    view._clock = lambda: 1e9
    _heartbeat(view).timeout.emit()
    assert _text(view) == before


def test_all_sampling_steps_do_not_claim_the_result_has_finished(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _deliver(panel, job, _progress(100))
    assert "等待结果" in _text(_view(dialog))
    assert "MCMC 完成" not in panel.status_text()
    assert panel.controller.is_running


def test_success_publishes_project_and_stops_clock_without_clearing_evidence(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    updated = replace(panel.document.project)
    _deliver(
        panel,
        job,
        _progress(100),
        api.OperationEvent(2, "mcmc_result", mcmc_result=updated),
        api.OperationEvent(3, "stopped"),
    )
    _visible_status(panel, dialog, "MCMC 完成")
    _idle(panel, dialog)
    assert panel.document.project is updated
    assert "100 / 100" in _text(_view(dialog))


@pytest.mark.parametrize("changes", ({"dataset_id": "another"}, {"stage": "B"}))
def test_sampling_progress_ignores_foreign_owner_or_operation(qtbot, sampling, changes):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _deliver(panel, job, _progress(37))
    view = _view(dialog)
    before = _text(view)
    _deliver(panel, job, _progress(52, **changes))
    assert _text(view) == before


def test_terminal_elapsed_does_not_leave_a_stale_remaining_estimate(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _deliver(
        panel,
        job,
        _progress(100),
        api.OperationEvent(2, "cancelled", cancellation="requested"),
        api.OperationEvent(3, "stopped"),
    )
    text = _text(_view(dialog))
    assert "已用" in text
    assert "预计剩余" not in text and "等待结果" not in text


def test_hidden_run_proxy_opens_dialog_before_showing_validation_error(qtbot, monkeypatch):
    """The main toolbar forwards to this hidden owned button's click method."""
    from PySide6.QtWidgets import QToolButton

    panel = _panel(qtbot, expert=True)
    panel.set_secondary_visible(False)
    panel.walkers.setValue(2)
    proxy = QToolButton()
    qtbot.addWidget(proxy)
    proxy.clicked.connect(panel.mcmc_button.click)
    proxy.show()
    starts = []
    monkeypatch.setattr(api, "start_mcmc_job", lambda *args: starts.append(args))
    _click(qtbot, proxy)
    dialog = panel._uncertainty_dialog
    assert dialog is not None and dialog.isVisible()
    _visible_status(panel, dialog, "walkers")
    assert starts == []


def test_new_document_does_not_display_the_previous_sampling_run(qtbot, sampling):
    panel, dialog, job, _starts = sampling
    _click(qtbot, panel.mcmc_button)
    _deliver(
        panel,
        job,
        _progress(37),
        api.OperationEvent(2, "cancelled", cancellation="requested"),
        api.OperationEvent(3, "stopped"),
    )
    view = _view(dialog)
    panel.document.new()
    assert view.isHidden()
    assert panel.status_text() == ""
