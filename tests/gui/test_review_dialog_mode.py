"""An open shared evidence dialog follows current depth without losing live controls."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel, QToolButton
from tests.gui.plot_support import _candidate
from tests.gui.test_review_evidence_projection import _report
from tests.support.model_cases import final_fit_result, prepared_data, simple_structure

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.main_window import MainWindow
from xrr_fitter.model.parameters import ParameterValue


def _stop_fake_sampling(window):
    controller = window.result_panel.controller
    if controller.is_running:
        controller._job = None
        controller.running_changed.emit(False)


@pytest.fixture
def window(qtbot, tmp_path):
    data = prepared_data(size=32)
    path = tmp_path / "curve.xy"
    np.savetxt(path, np.column_stack((data.two_theta_deg, data.intensity_raw)))
    value = api.add_dataset(api.new_project(), path, api.InstrumentSpec())
    candidate = replace(
        _candidate(data),
        noise_model="gaussian",
        unit_vector=np.array([0.5, 0.5]),
        parameters=(ParameterValue("scale", 1.0, 0.5, 1.5), ParameterValue("thickness", 20.0, 10.0, 30.0)),
    )
    result = replace(
        final_fit_result(candidate),
        uncertainty=_report(),
        region_labels=np.zeros(32, dtype=int),
        region_weights=np.ones(32),
    )
    dataset = replace(value.datasets[0], last_valid_result=result, structure=simple_structure())
    value = replace(value, datasets=(dataset,))
    value = api.set_expert_mode(api.select_active_dataset(value, "curve"), True)
    widget = MainWindow(ProjectDocument(value))
    qtbot.addWidget(widget, before_close_func=_stop_fake_sampling)
    return widget


@pytest.fixture
def dispatch_calls(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window.result_panel.controller, "start_mcmc", lambda *args: (calls.append(args), True)[1])
    return calls


@pytest.fixture
def active_sampling(window):
    """Keep a real controller active without spawning a scientific worker or timer."""
    controller = window.result_panel.controller
    operations = []
    controller._job = SimpleNamespace(
        cancel=lambda: operations.append("cancel"),
        force_stop=lambda: operations.append("force"),
    )
    controller.running_changed.emit(True)
    return operations


def _open(qtbot, window, *, sampling=True):
    action = "openUncertaintyAction" if sampling else "viewInferenceEvidenceAction"
    window.chrome_actions[action].trigger()
    dialog = window.result_panel._uncertainty_dialog
    qtbot.addWidget(dialog)
    assert dialog.isVisible()
    return dialog


def _configuration_widgets(panel):
    controls = (panel.walkers, panel.burn_in, panel.production, panel.thin)
    labels = tuple(label for label in panel.mcmc_group.findChildren(QLabel) if label.buddy() in controls)
    assert len(labels) == 4
    return (*controls, *labels, panel.mcmc_button, panel.mcmc_group.recommend_button)


def test_open_expert_dialog_reprojects_immediately_when_depth_is_disabled(qtbot, window):
    panel = window.result_panel
    dialog = _open(qtbot, window)
    window.parameters_panel.expert_toggle.setChecked(False)

    assert not window.document.project.ui_state.expert_mode
    assert dialog.windowTitle() == dialog.accessibleName() == "推断证据"
    assert not panel.mcmc_group.isVisibleTo(dialog)
    assert not panel.mcmc_button.isEnabled()


def test_restoring_expert_depth_reuses_requested_sampling_controls_and_config(qtbot, window):
    panel = window.result_panel
    dialog = _open(qtbot, window)
    widgets = _configuration_widgets(panel)
    panel.production.setValue(1234)
    config = panel.mcmc_config()
    window.parameters_panel.expert_toggle.setChecked(False)
    window.parameters_panel.expert_toggle.setChecked(True)

    assert panel._uncertainty_dialog is dialog
    assert _configuration_widgets(panel) == widgets
    assert dialog.windowTitle() == dialog.accessibleName() == "不确定度分析"
    assert panel.mcmc_group.isVisibleTo(dialog)
    assert all(control.isVisibleTo(dialog) for control in widgets)
    assert panel.mcmc_config() == config


def test_disabling_depth_stops_the_open_dialog_button_from_dispatching(qtbot, window, dispatch_calls):
    panel = window.result_panel
    _open(qtbot, window)
    assert panel.mcmc_button.isEnabled()
    window.parameters_panel.expert_toggle.setChecked(False)

    panel.mcmc_button.click()

    assert dispatch_calls == []
    assert panel._mcmc_source_project is None


def test_actual_start_path_rechecks_depth_even_for_a_queued_activation(qtbot, window, dispatch_calls):
    panel = window.result_panel
    _open(qtbot, window)
    window.parameters_panel.expert_toggle.setChecked(False)

    with pytest.raises(ValueError, match="expert mode"):
        panel.start_mcmc()
    with qtbot.capture_exceptions() as exceptions:
        panel.mcmc_button.clicked.emit()

    assert not exceptions
    assert dispatch_calls == []
    assert "expert mode" in panel.status_text()
    assert panel._mcmc_source_project is None


def test_requested_read_only_mode_survives_depth_and_project_refresh(qtbot, window, dispatch_calls):
    panel = window.result_panel
    source = window.document.project
    dialog = _open(qtbot, window, sampling=False)
    assert window.document.project is source
    widgets = _configuration_widgets(panel)
    for expert in (False, True):
        window.parameters_panel.expert_toggle.setChecked(expert)
        current = window.document.project
        window.document.replace_project(replace(current))
        assert window.document.project.ui_state.expert_mode is expert
        assert panel._uncertainty_dialog is dialog
        assert _configuration_widgets(panel) == widgets
        assert dialog.windowTitle() == dialog.accessibleName() == "推断证据"
        assert not panel.mcmc_group.isVisibleTo(dialog)
    assert dispatch_calls == []


@pytest.fixture(params=("ordinary", "read-only"))
def restricted_live_dialog(qtbot, window, active_sampling, request):
    panel = window.result_panel
    dialog = _open(qtbot, window)
    widgets = _configuration_widgets(panel)
    cancel, force = panel.cancel_button, panel.force_button
    config = panel.mcmc_config()
    if request.param == "ordinary":
        window.parameters_panel.expert_toggle.setChecked(False)
    else:
        source = window.document.project
        window.chrome_actions["viewInferenceEvidenceAction"].trigger()
        assert window.document.project is source
    return dialog, widgets, (cancel, force), config


def test_restricted_live_dialog_hides_configuration_without_replacing_controls(window, restricted_live_dialog):
    panel = window.result_panel
    dialog, widgets, cancellation, config = restricted_live_dialog

    assert panel._uncertainty_dialog is dialog
    assert panel.mcmc_group.isVisibleTo(dialog)
    assert all(not control.isVisibleTo(dialog) for control in widgets)
    assert dialog.windowTitle() == dialog.accessibleName() == "推断证据"
    assert (panel.cancel_button, panel.force_button) == cancellation
    assert panel.mcmc_config() == config


def test_restricted_live_dialog_can_still_cancel_and_force_stop(restricted_live_dialog, active_sampling):
    dialog, _widgets, (cancel, force), _config = restricted_live_dialog

    assert cancel.isVisibleTo(dialog) and cancel.isEnabled()
    assert force.isVisibleTo(dialog) and force.isEnabled()
    assert active_sampling == []

    cancel.click()
    force.click()
    assert active_sampling == ["cancel", "force"]


def test_restricted_dialog_hides_live_controls_once_sampling_stops(window, restricted_live_dialog):
    panel = window.result_panel
    dialog, _widgets, (cancel, force), _config = restricted_live_dialog

    panel.controller._job = None
    panel.controller.running_changed.emit(False)
    assert not panel.mcmc_group.isVisibleTo(dialog)
    assert not cancel.isEnabled() and not force.isEnabled()


def test_live_restricted_controls_survive_an_empty_result_refresh(qtbot, window, active_sampling):
    panel = window.result_panel
    dialog = _open(qtbot, window)
    window.parameters_panel.expert_toggle.setChecked(False)
    source = window.document.project
    dataset = replace(source.datasets[0], last_valid_result=None)
    window.document.replace_project(replace(source, datasets=(dataset,)))

    assert panel.mcmc_group.isVisibleTo(dialog)
    assert all(not control.isVisibleTo(dialog) for control in _configuration_widgets(panel))
    assert panel.cancel_button.isVisibleTo(dialog) and panel.cancel_button.isEnabled()
    assert panel.force_button.isVisibleTo(dialog) and panel.force_button.isEnabled()
    assert active_sampling == []


def test_an_expert_toolbar_launch_remains_legal_after_opening_read_only_evidence(qtbot, window, dispatch_calls):
    panel = window.result_panel
    _open(qtbot, window, sampling=False)
    window.plot_panel.select_view("uncertainty")
    source = window.document.project
    command = window.findChild(QToolButton, "runMcmcToolButton")
    assert command.isEnabled()

    command.click()

    assert dispatch_calls == [(source, "curve", "candidate-a", panel.mcmc_config())]
    assert window.document.project is source
