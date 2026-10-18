"""Keep the worker's input snapshot immutable while allowing plot inspection."""

import pytest
from tests.gui.test_fit_progress import _FakeJob, _project

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.main_window import MainWindow


def _running_window(qtbot, tmp_path, monkeypatch):
    project = api.set_expert_mode(_project(tmp_path), True)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.autosave.stop()
    window.show()
    job = _FakeJob()
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: job)
    assert window.fit_panel.start_fit()
    return window


@pytest.mark.parametrize("mode", ("range", "mask"))
def test_running_plot_cannot_arm_an_input_edit_but_keeps_navigation(qtbot, tmp_path, monkeypatch, mode):
    window = _running_window(qtbot, tmp_path, monkeypatch)
    try:
        action = window.plot_panel.mode_actions()[mode]
        action.trigger()
        assert window.plot_panel.toolbar.mode() == "view"
        assert not action.isEnabled()
        assert window.plot_panel.isEnabled()
        assert all(button.isEnabled() for button in window.plot_panel.toolbar.navigation_buttons().values())
    finally:
        window.fit_panel.controller._finish()


def test_running_input_editors_are_locked_including_reparented_inspector_controls(qtbot, tmp_path, monkeypatch):
    window = _running_window(qtbot, tmp_path, monkeypatch)
    controls = (
        window.structure_panel,
        window.parameters_panel,
        window.parameter_disposition,
        window.selected_layer_panel,
        window.batch_mode_group,
    )
    try:
        assert all(not control.isEnabled() for control in controls)
        window.fit_panel.controller._finish()
        assert all(control.isEnabled() for control in controls)
    finally:
        window.fit_panel.controller._finish()


def test_window_running_projection_does_not_repeat_scientific_preflight(qtbot, tmp_path, monkeypatch):
    window = _running_window(qtbot, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(api, "preflight_fit", lambda _p: calls.append(True) or api.FitReadiness(True, "ready"))
    try:
        window._refresh_operation_state()
        assert calls == []
    finally:
        window.fit_panel.controller._finish()
