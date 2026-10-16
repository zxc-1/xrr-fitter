"""Keyboard and pointer diagnostic navigation on the complete window."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTabBar
from tests.gui.control_operation_driver import setup_project
from tests.gui.test_control_operations_results import result_fixture

import xrr_fitter.api as api

KEYS = ("log", "raw", "qz4", "residual", "candidates", "residual_map", "parameter_map", "uncertainty", "trend")


def bar_click(driver, bar, index):
    assert bar.isVisible() and bar.isTabEnabled(index) and bar.isTabVisible(index)
    position = bar.tabRect(index).center()
    driver.reach(bar, position)
    driver.qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=position)
    assert bar.currentIndex() == index


@pytest.mark.parametrize("route", ("menu", "shortcut", "tab"))
def test_all_diagnostic_routes_with_real_inputs(controls, tmp_path, route):
    window = controls.window(api.set_expert_mode(result_fixture(tmp_path), True))
    panel = window.plot_panel
    for index, key in enumerate(KEYS):
        if route == "menu":
            controls.menu(window, f"plotViewAction:{key}")
            identity = f"plotViewAction:{key}"
        elif route == "shortcut":
            panel.setFocus()
            QTest.keySequence(panel, QKeySequence(f"Alt+{index + 1}"))
            identity = f"plotShortcut:Alt+{index + 1}"
        else:
            if index in (0, 4):
                controls.menu(window, f"plotViewAction:{key}")
            name = "reflectivityCanvasTabs" if index < 4 else "analysisCanvasTabs"
            bar_click(controls, controls.widget(window, name, QTabBar), index if index < 4 else index - 4)
            identity = f"{name}:{key}"
        assert panel.current_view_key() == key
        assert window.chrome_actions[f"plotViewAction:{key}"].isChecked()
        controls.record(
            identity,
            f"{route} select {key}",
            "active view and menu check agree",
            setup="serializable result fixture with expert depth; no worker",
        )


def test_structure_canvas_tabs_are_operable(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    controls.named_click(window, "pipelineStep_结构")
    bar = controls.widget(window, "structureCanvasTabs", QTabBar)
    for index, key in enumerate(("structure", "sld", "parameters")):
        bar_click(controls, bar, index)
        controls.record(f"structureCanvasTabs:{key}", f"click {key} tab", "requested canvas page is selected")
