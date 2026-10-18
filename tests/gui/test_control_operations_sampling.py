"""MCMC/evidence input routes; fake process boundary is explicitly not sampling."""

from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPlainTextEdit
from tests.gui.control_operation_driver import controls as controls
from tests.gui.test_control_operations_results import result_fixture
from tests.gui.test_fit_progress import _FakeJob
from tests.gui.test_uncertainty_dialog import _full_report

import xrr_fitter.api as api


def evidence_project(tmp_path):
    project = result_fixture(tmp_path)
    dataset = project.datasets[0]
    result = replace(dataset.last_valid_result, uncertainty=_full_report("selected"))
    return api.set_expert_mode(replace(project, datasets=(replace(dataset, last_valid_result=result),)), True)


def test_uncertainty_entry_fields_recommendation_and_reopen(controls, tmp_path):
    window = controls.window(evidence_project(tmp_path))
    controls.menu(window, "openUncertaintyAction")
    panel = window.result_panel
    dialog = panel._uncertainty_dialog
    assert dialog.isVisible()
    for name, value in (("mcmcWalkers", 48), ("mcmcBurnIn", 25), ("mcmcProduction", 200), ("mcmcThin", 2)):
        spin = controls.widget(dialog, name)
        controls.type(spin, str(value))
        assert spin.value() == value
        controls.record(
            name,
            f"type {value}; Tab to commit",
            "spin value is committed by keyboard",
            setup="display fixture; no sampling run",
        )
    chosen = panel.mcmc_config()
    controls.qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    controls.menu(window, "openUncertaintyAction")
    assert panel.mcmc_config() == chosen and panel._uncertainty_dialog is dialog
    controls.named_click(dialog, "mcmcRecommendButton")
    assert panel.mcmc_config() == api.McmcConfig.standard(1)
    controls.record(
        ("openUncertaintyAction", "mcmcRecommendButton"),
        "open; edit; Escape; reopen; recommend",
        "inputs survive reopen; recommended defaults return",
    )
    controls.qtbot.keyClick(dialog, Qt.Key.Key_Escape)


def start_sampling(controls, window, route):
    if route == "dialog":
        controls.menu(window, "openUncertaintyAction")
        controls.named_click(window.result_panel._uncertainty_dialog, "mcmcButton")
    else:
        controls.menu(window, "plotViewAction:uncertainty")
        controls.named_click(window, "runMcmcToolButton")


@pytest.mark.parametrize("route", ("dialog", "toolbar"))
def test_sampling_start_cancel_force_and_readonly_reopen(controls, tmp_path, monkeypatch, route):
    window = controls.window(evidence_project(tmp_path))
    panel = window.result_panel
    job = _FakeJob()
    calls = []
    starts = []
    job.cancel = lambda: calls.append("cancel")
    job.force_stop = lambda: calls.append("force_stop")
    monkeypatch.setattr(api, "start_mcmc_job", lambda *args: (starts.append(args), job)[1])
    try:
        start_sampling(controls, window, route)
        assert panel.controller.is_running and len(starts) == 1
        dialog = panel._uncertainty_dialog
        assert dialog.isVisible()
        controls.named_click(dialog, "cancelMcmcButton")
        assert calls == ["cancel"] and not panel.cancel_button.isEnabled()
        controls.qtbot.keyClick(dialog, Qt.Key.Key_Escape)
        controls.menu(window, "viewInferenceEvidenceAction")
        assert dialog.isVisible() and not panel.walkers.isVisibleTo(dialog)
        controls.named_click(dialog, "forceStopMcmcButton")
        assert calls == ["cancel", "force_stop"] and not panel.force_button.isEnabled()
        control = "mcmcButton" if route == "dialog" else "runMcmcToolButton"
        controls.record(
            (control, "cancelMcmcButton", "forceStopMcmcButton"),
            f"{route} run; cancel; read-only reopen; force stop",
            "one job starts, request gates survive dialog closure",
            setup="fake worker only; no sampling budget or convergence claim",
        )
    finally:
        job.events = (api.OperationEvent(2, "cancelled", cancellation="requested"), api.OperationEvent(3, "stopped"))
        panel.controller.poll_now()
        if panel._uncertainty_dialog is not None:
            panel._uncertainty_dialog.close()


def copy_readonly_evidence(controls, dialog):
    editors = [w for w in dialog.findChildren(QPlainTextEdit) if w.isVisible()]
    assert len(editors) == 1
    previous = QApplication.clipboard().text()
    try:
        controls.click(editors[0])
        QTest.keySequence(editors[0], QKeySequence(QKeySequence.StandardKey.SelectAll))
        QTest.keySequence(editors[0], QKeySequence(QKeySequence.StandardKey.Copy))
        assert QApplication.clipboard().text() == editors[0].toPlainText()
        assert editors[0].isReadOnly()
        before = editors[0].toPlainText()
        controls.qtbot.keyClicks(editors[0], "do not change evidence")
        assert editors[0].toPlainText() == before
    finally:
        QApplication.clipboard().setText(previous)


def test_evidence_copy_and_band_page(controls, tmp_path):
    window = controls.window(evidence_project(tmp_path))
    controls.menu(window, "viewInferenceEvidenceAction")
    dialog = window.result_panel._uncertainty_dialog
    pages = window.result_panel.uncertainty.pages
    controls.tab(pages, 2)
    assert pages.currentWidget() is pages.bands.canvas
    copy_readonly_evidence(controls, dialog)
    controls.record(
        ("uncertaintyEvidence:copy", "uncertaintyPages:sld_bands", "readOnly:uncertaintyEvidence"),
        "select band page; Copy; attempt text edit",
        "band page selected, exact evidence copied, evidence stays read-only",
    )
    dialog.close()
