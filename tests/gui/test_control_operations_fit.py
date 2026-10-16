"""Actual window input routes, with only worker startup replaced.

Controlled events prove UI dispatch and lifecycle, not optimizer performance.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from tests.gui.control_operation_driver import setup_project
from tests.gui.test_fit_progress import _FakeJob

import xrr_fitter.api as api


def running_window(driver, tmp_path, monkeypatch, *, route="toolbar"):
    project = api.set_expert_mode(setup_project(tmp_path), True)
    window = driver.window(project)
    job = _FakeJob()
    calls = []
    for name in ("cancel", "pause", "resume", "skip_stage", "force_stop"):
        setattr(job, name, lambda name=name: calls.append(name))
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: job)
    triggers = {
        "toolbar": lambda: driver.named_click(window, "startFitToolButton"),
        "menu": lambda: driver.menu(window, "startFitAction"),
        "shortcut": lambda: driver.shortcut(window, "startFitAction"),
    }
    triggers[route]()
    assert window.fit_panel.is_running
    window.fit_panel.controller.progress_changed.emit(api.FitProgress("curve", "B", 1, 10, 1.0, "search"))
    return window, job, calls


def stop_input(driver, window, route, focus):
    if route == "menu":
        driver.menu(window, "cancelFitAction")
        return
    focus.setFocus()
    driver.qtbot.wait(1)
    QTest.keySequence(focus, QKeySequence("Escape"))


def repeat_stop(driver, window, route, focus):
    # 停止请求已发出后，菜单里的「取消拟合」灰掉，只剩强停；再按 Esc 不再发第二次请求。
    if route == "menu":
        assert not window._workflow_actions["cancelFitAction"].isEnabled()
        return
    stop_input(driver, window, route, focus)


@pytest.mark.parametrize("route", ("menu", "shortcut"))
@pytest.mark.parametrize("focus_name", ("panel", "plot"))
def test_window_stop_routes_acknowledge_and_deduplicate(controls, tmp_path, monkeypatch, route, focus_name):
    window, job, calls = running_window(controls, tmp_path, monkeypatch)
    panel = window.fit_panel
    focus = panel.pause_button if focus_name == "panel" else window.plot_panel
    try:
        stop_input(controls, window, route, focus)
        assert calls == ["cancel"]
        assert panel._cancel_requested
        assert not panel.cancel_button.isEnabled()
        assert not panel.pause_button.isEnabled()
        assert not panel.skip_button.isEnabled()
        assert panel.force_button.isEnabled() and panel.force_button.isVisible()
        repeat_stop(controls, window, route, focus)
        assert calls == ["cancel"]
        identities = ("cancelFitAction",) if route == "menu" else ("cancelFitAction:shortcut",)
        controls.record(
            identities,
            f"{route} stop twice with {focus_name} focus",
            "one cancellation; stopping feedback and only force stop remain",
            setup="real window, preflight and controller; fake worker boundary",
        )
    finally:
        job.events = (api.OperationEvent(1, "stopped"),)
        panel.controller.poll_now()


@pytest.mark.parametrize("route", ("toolbar", "menu", "shortcut"))
def test_start_routes_reject_overlap_and_restore_after_stop(controls, tmp_path, monkeypatch, route):
    window, job, _calls = running_window(controls, tmp_path, monkeypatch, route=route)
    try:
        assert not window._workflow_actions["startFitAction"].isEnabled()
        assert not window.chrome_actions["reloadSourceAction"].isEnabled()
        assert not window.chrome_actions["relinkSourceAction"].isEnabled()
        identities = {"toolbar": "startFitToolButton", "menu": "startFitAction", "shortcut": "fitShortcut:Ctrl+Return"}
        controls.record(
            identities[route],
            f"{route} start",
            "real preflight starts once and mutating source/start routes are disabled",
            setup="fake worker; real window/controller",
        )
    finally:
        job.events = (api.OperationEvent(1, "stopped"),)
        window.fit_panel.controller.poll_now()
    assert window._workflow_actions["startFitAction"].isEnabled()


def progress_tabs(controls, window):
    from PySide6.QtWidgets import QTabBar

    bar = controls.widget(window, "fitCanvasTabs", QTabBar)
    for index, key in enumerate(("progress", "preview", "objective")):
        controls.qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(index).center())
        assert bar.currentIndex() == index
        controls.record(f"fitCanvasTabs:{key}", "click progress view tab", "requested progress pane becomes selected")


def pause_resume_and_skip(controls, window, calls, name):
    panel = window.fit_panel
    controls.named_click(window, name)
    assert calls == ["pause"] and "暂停已请求" in panel.progress_view.status_summary().position
    assert not panel.skip_button.isEnabled()
    controls.named_click(window, name)
    assert calls == ["pause", "resume"] and panel.skip_button.isEnabled()
    controls.named_click(window, "skipStageButton")
    assert calls[-1] == "skip_stage"
    panel.controller.progress_changed.emit(api.FitProgress("curve", "finalizing", 1, 1, 0.5, "publishing"))
    assert not panel.skip_button.isEnabled()


@pytest.mark.parametrize("surface", ("panel", "toolbar"))
def test_pause_skip_stop_force_and_progress_tabs(controls, tmp_path, monkeypatch, surface):
    window, job, calls = running_window(controls, tmp_path, monkeypatch)
    panel = window.fit_panel
    names = ("pauseFitButton", "cancelFitButton") if surface == "panel" else ("toolbarPauseButton", "toolbarStopButton")
    try:
        pause_resume_and_skip(controls, window, calls, names[0])
        progress_tabs(controls, window)
        controls.named_click(window, names[1])
        assert calls[-1] == "cancel" and not panel.pause_button.isEnabled()
        force = "forceStopFitButton" if surface == "panel" else "forceStopFitAction"
        if surface == "panel":
            controls.named_click(window, force)
        else:
            controls.menu(window, force)
        assert calls[-1] == "force_stop"
        controls.record(
            (*names, "skipStageButton", force),
            f"{surface} pause/resume; skip; stop; force",
            "requests reach worker in order and stage/stopping gates hold",
            setup="fake worker events, not numerical fitting",
        )
    finally:
        job.events = (api.OperationEvent(1, "stopped"),)
        panel.controller.poll_now()


def standalone_fit(driver, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    panel = FitPanel(ProjectDocument(project))
    driver.qtbot.addWidget(panel)
    driver.qtbot.addWidget(panel.live_metrics)
    panel.resize(650, 800)
    panel.show()
    panel.activateWindow()
    driver.qtbot.wait(1)
    return panel


def automatic_project(tmp_path):
    from tests.gui.control_operation_driver import source_curve

    preset = api.MeasurementPreset("controls", api.BeamSpec("monochromatic"), api.InstrumentSpec())
    source = source_curve(tmp_path / "curve Si+Zr.xy")
    preview = api.preview_import_batch((source,), preset)
    choices = {row.substrate_group_id: "Si" for row in preview.files if row.requires_substrate_choice}
    imported = api.import_dataset_batch(api.new_project(), preview, choices)
    assert not imported.failures and len(imported.imported_dataset_ids) == 1
    readiness = api.preflight_automatic_fit(imported.updated_project)
    assert readiness.ready, readiness.message
    return imported.updated_project


@pytest.mark.parametrize("automatic", (False, True))
def test_standalone_start_and_escape(controls, tmp_path, monkeypatch, automatic):
    project = automatic_project(tmp_path) if automatic else setup_project(tmp_path)
    panel = standalone_fit(controls, api.set_expert_mode(project, True))
    job = _FakeJob()
    calls = []
    job.cancel = lambda: calls.append("cancel")
    entry = "start_automatic_fit_job" if automatic else "start_fit_job"
    monkeypatch.setattr(api, entry, lambda *_a, **_k: job)
    name = "startAutomaticFitButton" if automatic else "startFitButton"
    try:
        controls.named_click(panel, name)
        assert panel.is_running and panel.cancel_shortcut.isEnabled()
        # The standalone metrics widget is a separate window; target the panel.
        panel.activateWindow()
        controls.qtbot.waitUntil(panel.isActiveWindow)
        panel.pause_button.setFocus()
        controls.qtbot.waitUntil(panel.pause_button.hasFocus)
        QTest.keySequence(panel.pause_button, QKeySequence("Escape"))
        assert calls == ["cancel"] and panel._cancel_requested
        controls.record(
            (name, "cancelFitShortcut"),
            "standalone start button; Escape",
            "real preflight and panel-only shortcut dispatch correctly",
            setup="fake worker; standalone production panel",
        )
    finally:
        job.events = (api.OperationEvent(1, "stopped"),)
        panel.controller.poll_now()


def test_noise_selector_commits_each_mode_without_starting(controls, tmp_path):
    panel = standalone_fit(controls, api.set_expert_mode(setup_project(tmp_path), True))
    for index in range(panel.noise_selector.count()):
        controls.select(panel.noise_selector, index)
        assert panel.document.project.fit_config.noise_model == panel.noise_selector.itemData(index)
        assert not panel.is_running
    controls.record(
        "noiseModelSelector",
        "select every noise mode with keyboard",
        "project noise mode follows selection and no worker starts",
    )
