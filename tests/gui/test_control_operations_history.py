"""Undo/redo shortcut routing while editing and after the editor commits."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit
from tests.gui.control_operation_driver import controls as controls
from tests.gui.control_operation_driver import setup_project
from tests.gui.test_control_operations_parameters import _select_name, _table


def open_initial_editor(driver, window):
    _panel, table = _table(driver, window)
    row = _select_name(driver, table)
    cell = table.item(row, 1)
    position = table.visualItemRect(cell).center()
    driver.reach(table.viewport(), position)
    driver.qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    driver.qtbot.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    editors = [editor for editor in table.findChildren(QLineEdit) if editor.isVisible()]
    assert len(editors) == 1
    window.activateWindow()
    driver.qtbot.waitUntil(window.isActiveWindow)
    driver.click(editors[0])
    return table, editors[0]


def history_key(driver, window, focus, action):
    focus.setFocus()
    driver.qtbot.waitUntil(focus.hasFocus)
    QTest.keySequence(focus, window.chrome_actions[action].shortcut())


@pytest.mark.parametrize("focus_name", ("table", "window"))
def test_history_keys_preserve_local_edit_then_route_to_document(controls, tmp_path, focus_name):
    window = controls.window(setup_project(tmp_path))
    table, editor = open_initial_editor(controls, window)
    before = window.document.project.datasets[0].parameter_settings
    original_text = editor.text()
    QTest.keySequence(editor, QKeySequence(QKeySequence.StandardKey.SelectAll))
    controls.qtbot.keyClicks(editor, "4.5")
    history_key(controls, window, editor, "undoAction")
    assert editor.text() == original_text
    assert window.document.project.datasets[0].parameter_settings == before
    history_key(controls, window, editor, "redoAction")
    assert editor.text() == "4.5"
    assert window.document.project.datasets[0].parameter_settings == before
    controls.qtbot.keyClick(editor, Qt.Key.Key_Return)
    controls.qtbot.waitUntil(lambda: window.document.project.datasets[0].parameter_settings != before)
    edited = window.document.project.datasets[0].parameter_settings
    assert edited != before
    focus = table if focus_name == "table" else window
    history_key(controls, window, focus, "undoAction")
    assert window.document.project.datasets[0].parameter_settings == before
    history_key(controls, window, focus, "redoAction")
    assert window.document.project.datasets[0].parameter_settings == edited
    controls.record(
        ("historyShortcut:Ctrl+Z", "historyShortcut:Ctrl+Shift+Z"),
        f"Undo/Redo in active cell; commit; repeat with {focus_name} focus",
        "local text history is isolated; committed project change reverses and reapplies",
    )
