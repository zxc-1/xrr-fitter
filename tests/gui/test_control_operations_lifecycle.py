"""Window lifecycle prompts and standalone aliases omitted by shell assembly."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QScrollArea
from tests.gui.control_operation_driver import setup_project
from tests.gui.test_control_operations_fit import running_window
from tests.gui.test_control_operations_plots import data_window
from tests.gui.test_control_operations_results import result_fixture

import xrr_fitter.api as api


def stop_job(driver, window, job, *, closing):
    def stop():
        job.events = (api.OperationEvent(1, "stopped"),)
        window.fit_panel.controller.poll_now()

    if closing and window.document.is_dirty:
        driver.modals(stop, [lambda dialog: driver.click(dialog.button(QMessageBox.StandardButton.Yes))])
    else:
        stop()
    driver.qtbot.wait(1)


@pytest.mark.parametrize("accept", (False, True))
def test_force_close_prompt_uses_real_deadline_and_buttons(controls, tmp_path, monkeypatch, accept):
    window, job, calls = running_window(controls, tmp_path, monkeypatch)
    window.close()
    controls.qtbot.waitUntil(lambda: window.force_close_prompt is not None, timeout=7000)
    prompt = window.force_close_prompt
    assert prompt.isVisible() and calls == ["cancel"]
    try:
        response = QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.No
        controls.click(prompt.button(response))
        assert calls == (["cancel", "force_stop"] if accept else ["cancel"])
        assert window.close_pending is accept
        assert window.isVisible()
        controls.record(
            f"forceClosePrompt:{'Yes' if accept else 'No'}",
            "wait through actual 5s deadline; click confirmation",
            "force-stop is dispatched only for Yes; window stays while worker has not stopped",
            setup="QWidget.close request and fake worker that ignores cancellation; not native titlebar or optimizer evidence",
        )
    finally:
        stop_job(controls, window, job, closing=accept)
    assert window.isVisible() is not accept


def test_standalone_result_entry_buttons_have_live_routes(controls, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    panel = ResultsPanel(ProjectDocument(api.set_expert_mode(result_fixture(tmp_path), True)))
    host = QScrollArea()
    host.setWidgetResizable(True)
    host.setWidget(panel)
    controls.qtbot.addWidget(host)
    host.resize(760, 760)
    host.show()
    controls.named_click(panel, "openUncertaintyDialogButton")
    dialog = panel._uncertainty_dialog
    assert dialog.isVisible() and not panel.controller.is_running
    controls.qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    # 清除先问一句（F27）：点「是」才清，和菜单那一条走同一个确认框。
    yes = [lambda confirm: controls.click(confirm.button(QMessageBox.StandardButton.Yes))]
    controls.modals(lambda: controls.named_click(panel, "clearResultsButton"), yes)
    assert panel.document.project.datasets[0].last_valid_result is None
    controls.record(
        ("openUncertaintyDialogButton", "clearResultsButton"),
        "standalone panel open uncertainty; Escape; Clear",
        "dialog is opened without sampling and result is removed through the real API",
        setup="production ResultsPanel in a scroll host; shell intentionally hides these aliases",
    )


@pytest.mark.parametrize("state", ("empty", "data", "structure", "result"))
def test_unreached_export_pipeline_step_does_not_navigate_or_export(controls, tmp_path, state):
    projects = {
        "empty": lambda: api.new_project(),
        "data": lambda: setup_project(tmp_path, layers=0),
        "structure": lambda: setup_project(tmp_path),
        "result": lambda: result_fixture(tmp_path),
    }
    window = controls.window(projects[state]())
    datasets = window.document.project.datasets
    step_before = window.pipeline_nav.current_step_index()
    row = controls.widget(window, "pipelineStep_导出")
    assert row.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert row.cursor().shape() == Qt.CursorShape.ArrowCursor
    controls.click(row)
    assert window.pipeline_nav.current_step_index() == step_before
    assert window.document.project.datasets is datasets
    assert window.export_workflow.manifest is None
    controls.record(
        "pipelineStep_导出",
        f"attempt click on unreached export milestone from {state}",
        "unreached milestone has no focus/hand affordance; navigation, export and datasets are unchanged",
    )


def test_plot_escape_is_scoped_to_the_plot_descendants(controls, tmp_path):
    window = data_window(controls, tmp_path)
    panel = window.plot_panel
    window.activateWindow()
    controls.qtbot.waitUntil(window.isActiveWindow)
    controls.named_click(window, "plotModeRange")
    focus = controls.widget(window, "plotModeRange")
    focus.setFocus()
    controls.qtbot.keyClick(focus, Qt.Key.Key_Escape)
    assert panel.interaction_mode() == "view"
    controls.named_click(window, "plotModeRange")
    window.data_panel.tree.setFocus()
    controls.qtbot.keyClick(window.data_panel.tree, Qt.Key.Key_Escape)
    assert panel.interaction_mode() == "range"
    controls.record(
        "plot:EscapeCancel",
        "Escape in plot toolbar; rearm; Escape in dataset tree",
        "selection cancels only within the owning plot subtree",
    )
