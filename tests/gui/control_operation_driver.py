"""Repeatable keyboard/pointer sweeps; fixture setup is not control coverage.

These tests use the actual MainWindow and real Qt dialogs.  File dialogs are
Qt's non-native variant so offscreen runs can replay them on every platform.
They are QTest evidence, not a claim of native OS or real optimizer coverage.
Set XRR_CONTROL_AUDIT_DIR to keep per-action evidence alongside pytest output.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QContextMenuEvent, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLineEdit,
    QMenu,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QWidget,
)

import xrr_fitter.api as api


def source_curve(path: Path, *, columns: int = 2, header: str = "") -> Path:
    """Small parser/UI fixture, deliberately not a scientific benchmark."""
    rows = []
    for index in range(64):
        values = (0.08 + index * 0.025, 1000.0 / (index + 1), 0.1, 0.002)
        rows.append(" ".join(f"{value:.10g}" for value in values[:columns]))
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def setup_project(tmp_path: Path, *, layers: int = 1, datasets: int = 1):
    """Only initial state uses api; every tested mutation uses the GUI."""
    value = api.new_project()
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0j),
        tuple(
            api.LayerSpec(f"film{index}", api.MaterialSpec("SiO2", "SiO2", 2.2), 40.0 + index, roughness_a=2.0)
            for index in range(layers)
        ),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    for index in range(datasets):
        name = "curve" if index == 0 else f"curve{index}"
        value = api.add_dataset(value, source_curve(tmp_path / f"{name}.xy"), api.InstrumentSpec())
        if layers:
            value = api.set_structure(value, name, structure)
    return api.select_active_dataset(value, "curve")


def _dialog_name(dialog) -> str:
    return dialog.objectName() or dialog.windowTitle() or type(dialog).__name__


def _close_modals() -> None:
    """Reject every open modal, innermost first, so a failed sweep fails instead of hanging.

    A dialog left open keeps its ``exec()`` loop, and with it the triggering call, blocked
    for good; the assertion recorded in ``modals`` is only raised once that call returns.
    """
    closed = []
    while (dialog := QApplication.activeModalWidget()) is not None and dialog not in closed:
        closed.append(dialog)
        if isinstance(dialog, QDialog):
            dialog.reject()
        else:
            dialog.close()


class ControlDriver:
    """Inputs stay at the Qt event boundary; assertions stay with each test."""

    def __init__(self, qtbot, nodeid: str):
        self.qtbot = qtbot
        self.nodeid = nodeid
        self.records = []
        self.windows = []
        self.context_event_actions = set()

    def window(self, project=None):
        from xrr_fitter.gui.document import ProjectDocument
        from xrr_fitter.gui.main_window import MainWindow

        window = MainWindow(ProjectDocument(project) if project is not None else None)
        self.qtbot.addWidget(window)
        self.windows.append(window)
        window.menuBar().setNativeMenuBar(False)
        window.resize(1440, 960)
        window.show()
        window.activateWindow()
        self.qtbot.wait(10)
        return window

    def widget(self, root, name: str, kind=QWidget):
        widget = root.findChild(kind, name)
        assert widget is not None, f"missing control {name}"
        return widget

    def click(self, widget):
        assert widget.isVisible(), f"hidden control {widget.objectName()}"
        assert widget.isEnabled(), f"disabled control {widget.objectName()}"
        position = (
            QPoint(8, widget.height() // 2) if isinstance(widget, (QCheckBox, QRadioButton)) else widget.rect().center()
        )
        self.reach(widget, position)
        self.qtbot.mouseClick(widget, Qt.MouseButton.LeftButton, pos=position)
        QApplication.processEvents()

    def reach(self, widget, position=None):
        """Scroll enclosing cards with keys instead of clicking clipped children."""
        position = widget.rect().center() if position is None else position
        parent = widget.parentWidget()
        while parent is not None and not parent.isWindow():
            if isinstance(parent, QScrollArea):
                bar = parent.verticalScrollBar()
                for _ in range(200):
                    point = widget.mapTo(parent.viewport(), position)
                    if 0 <= point.y() < parent.viewport().height():
                        break
                    before = bar.value()
                    key = Qt.Key.Key_Down if point.y() >= parent.viewport().height() else Qt.Key.Key_Up
                    self.qtbot.keyClick(bar, key)
                    QApplication.processEvents()
                    assert bar.value() != before, f"cannot reach {widget.objectName()} through {parent.objectName()}"
                else:
                    raise AssertionError(f"scroll did not reach {widget.objectName()}")
            parent = parent.parentWidget()

    def named_click(self, root, name: str):
        self.click(self.widget(root, name))

    def type(self, widget, value: str, *, commit: bool = True):
        editor = widget if isinstance(widget, QLineEdit) else widget.findChild(QLineEdit)
        assert editor is not None and editor.isVisible() and editor.isEnabled()
        self.click(editor)
        QTest.keySequence(editor, QKeySequence(QKeySequence.StandardKey.SelectAll))
        self.qtbot.keyClicks(editor, value)
        if commit:
            self.qtbot.keyClick(editor, Qt.Key.Key_Tab)
        QApplication.processEvents()

    def select(self, combo: QComboBox, index: int):
        assert 0 <= index < combo.count()
        self.click(combo)
        view = combo.view()
        assert view.isVisible()
        self.qtbot.keyClick(view, Qt.Key.Key_Home)
        for _ in range(index):
            self.qtbot.keyClick(view, Qt.Key.Key_Down)
        self.qtbot.keyClick(view, Qt.Key.Key_Return)
        assert combo.currentIndex() == index

    def tab(self, tabs: QTabWidget, index: int):
        bar = tabs.tabBar()
        assert tabs.isVisible() and tabs.isTabEnabled(index)
        self.reach(bar, bar.tabRect(index).center())
        self.qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(index).center())
        assert tabs.currentIndex() == index

    def menu(self, window, name: str):
        matches = []
        for menu in window.menuBar().findChildren(QMenu):
            matches.extend((menu, action) for action in menu.actions() if action.objectName() == name)
        assert len(matches) == 1, (name, len(matches))
        menu, action = matches[0]
        assert action.isEnabled() and action.isVisible(), name
        bar = window.menuBar()
        self.qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.actionGeometry(menu.menuAction()).center())
        assert menu.isVisible(), name
        self.qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
        QApplication.processEvents()

    def shortcut(self, window, name: str):
        action = window.findChild(QAction, name)
        assert action is not None and action.isEnabled() and not action.shortcut().isEmpty()
        window.activateWindow()
        window.setFocus()
        self.qtbot.wait(1)
        QTest.keySequence(window, action.shortcut())
        QApplication.processEvents()

    def buttons(self, dialog, standard):
        box = dialog.findChild(QDialogButtonBox)
        assert box is not None
        button = box.button(standard)
        assert button is not None
        self.click(button)

    def file(self, dialog, path: Path | None):
        """Choose ``path`` the way a reader types it: open its folder, type its name, press Open.

        Typing the whole absolute path is not equivalent. When the name field does not hold
        focus (the CI runner's dialog was not yet active), QFileDialog keeps every parent folder
        it passed through selected, and Open then navigates into "/" instead of accepting.
        """
        assert isinstance(dialog, QFileDialog)
        if path is None:
            self.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)
            return
        dialog.setDirectory(str(path.parent))
        self.type(self.widget(dialog, "fileNameEdit", QLineEdit), path.name, commit=False)
        box = dialog.findChild(QDialogButtonBox)
        accepted = [
            button for button in box.buttons() if box.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole
        ]
        assert len(accepted) == 1
        self.click(accepted[0])
        selected = dialog.selectedFiles()
        assert not dialog.isVisible(), f"file dialog stayed open choosing {path.name}: selected {selected[:3]}"

    def popup(self, trigger, handler):
        failures = []
        completed = []
        timer = QTimer()
        timer.setInterval(10)

        def visit():
            menu = QApplication.activePopupWidget()
            if not isinstance(menu, QMenu):
                return
            timer.stop()
            try:
                handler(menu)
            except BaseException as error:
                failures.append(error)
                menu.close()
            completed.append(True)

        timer.timeout.connect(visit)
        timer.start()
        trigger()
        try:
            self.qtbot.waitUntil(lambda: bool(completed), timeout=5000)
        finally:
            timer.stop()
        if failures:
            raise failures[0]

    def popup_action(self, menu, name: str):
        action = next(action for action in menu.actions() if action.objectName() == name)
        assert action.isEnabled() and action.isVisible(), name
        self.qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())

    def context_menu(self, widget, position, name: str):
        """Send the platform context request, then operate the real Qt menu.

        Offscreen mouse events do not consistently synthesize ContextMenu after
        modal dialogs. This explicit event is recorded separately from QTest
        pointer/keyboard coverage; native right-click needs its own evidence.
        """
        window = widget.window()
        window.activateWindow()
        self.reach(widget, position)
        hit = QApplication.widgetAt(widget.mapToGlobal(position))
        assert hit is widget or widget.isAncestorOf(hit), f"context target {widget.objectName()} is covered by {hit}"
        self.context_event_actions.add(name)
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, position, widget.mapToGlobal(position))
        self.popup(
            lambda: QApplication.sendEvent(widget, event),
            lambda menu: self.popup_action(menu, name),
        )

    def table_text(self, table, row: int, column: int, text: str):
        index = table.model().index(row, column)
        table.scrollTo(index)
        self.reach(table.viewport(), table.visualRect(index).center())
        self.qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualRect(index).center())
        self.qtbot.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualRect(index).center())
        editors = [editor for editor in table.findChildren(QLineEdit) if editor.isVisible()]
        assert len(editors) == 1
        editor = editors[0]
        self.type(editor, text, commit=False)
        self.qtbot.keyClick(editor, Qt.Key.Key_Return)
        QApplication.processEvents()

    def modals(self, trigger, handlers):
        """Drive nested exec loops without replacing dialog implementations."""
        remaining = list(handlers)
        failures = []
        seen = []
        attempts = [0]
        timer = QTimer()
        timer.setInterval(10)

        def visit():
            dialog = QApplication.activeModalWidget()
            if dialog is None or dialog in seen:
                attempts[0] += 1
                if attempts[0] > 300:
                    message = "expected modal dialog did not appear"
                    if dialog is not None:
                        message = f"modal dialog stayed open after its handler: {_dialog_name(dialog)}"
                    failures.append(AssertionError(message))
                    timer.stop()
                    _close_modals()
                return
            seen.append(dialog)
            timer.stop()
            try:
                assert isinstance(dialog, QDialog)
                assert remaining, f"unexpected modal dialog: {_dialog_name(dialog)}"
                remaining.pop(0)(dialog)
            except BaseException as error:
                failures.append(error)
                remaining.clear()
                _close_modals()
            if not failures:
                timer.start()

        timer.timeout.connect(visit)
        timer.start()
        trigger()
        self.qtbot.waitUntil(lambda: not remaining or bool(failures), timeout=5000)
        timer.stop()
        if failures:
            raise failures[0]
        assert not remaining

    def record(self, identities, action: str, assertion: str, *, setup: str = "api fixture only"):
        if isinstance(identities, str):
            identities = (identities,)
        self.records.extend(
            {
                "id": name,
                "modality": "qt_context_event" if name in self.context_event_actions else "qtest",
                "action": action,
                "assertion": assertion,
                "setup": setup,
            }
            for name in identities
        )

    def finish(self):
        for window in reversed(self.windows):
            assert not window._operation_is_running(), "operation sweep must not leak a worker"
            window.close()
        directory = os.environ.get("XRR_CONTROL_AUDIT_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256(self.nodeid.encode()).hexdigest()[:16] + ".json"
            (path / name).write_text(
                json.dumps({"test": self.nodeid, "operations": self.records}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def driven_controls(qtbot, request):
    """The body of the shared ``controls`` fixture (``tests/conftest.py``).

    Qt's non-native file dialogs stay on for the test so offscreen runs can drive them;
    the driver audits and closes its windows before the attribute is restored.
    """
    attribute = Qt.ApplicationAttribute.AA_DontUseNativeDialogs
    previous = QApplication.testAttribute(attribute)
    QApplication.setAttribute(attribute, True)
    driver = ControlDriver(qtbot, request.node.nodeid)
    yield driver
    driver.finish()
    QApplication.setAttribute(attribute, previous)
