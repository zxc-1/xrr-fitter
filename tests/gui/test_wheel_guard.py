"""A wheel notch over an editor the reader was only scrolling past must not edit it.

``QAbstractSpinBox`` steps on every notch whether or not it is focused, and Fusion
(Windows/Linux) lets a closed combo box change on the wheel too. One notch over the
noise selector used to switch the noise model and drop the fitted result; one notch
over a thickness box added a nanometre that was committed on the next focus change.
The wheels below go through ``QTest.wheelEvent``, the spontaneous route a real mouse
takes, so focus handling runs as it does for a user.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit, QScrollArea, QSpinBox, QVBoxLayout, QWidget
from tests.gui.test_control_operations_results import result_fixture


def _notch(widget: QWidget, steps: int = -1) -> None:
    top = widget.window()
    point = widget.mapTo(top, widget.rect().center())
    # The notch has to land on the editor itself, or the test proves nothing.
    landed = top.childAt(point)
    assert landed is widget or widget.isAncestorOf(landed)
    QTest.wheelEvent(top.windowHandle(), QPointF(point), QPoint(0, 120 * steps))


def _scrolling_host(qtbot) -> tuple[QScrollArea, QLineEdit, QSpinBox]:
    area = QScrollArea()
    qtbot.addWidget(area)
    inner = QWidget()
    layout = QVBoxLayout(inner)
    anchor = QLineEdit()
    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(50)
    layout.addWidget(anchor)
    layout.addSpacing(360)
    layout.addWidget(spin)
    layout.addSpacing(600)
    area.setWidget(inner)
    area.setWidgetResizable(True)
    area.resize(300, 300)
    area.show()
    assert QTest.qWaitForWindowExposed(area)
    area.activateWindow()
    anchor.setFocus()
    area.ensureWidgetVisible(spin)
    qtbot.waitUntil(anchor.hasFocus)
    return area, anchor, spin


def test_wheel_over_an_unfocused_spin_box_scrolls_the_panel_instead(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    area, _anchor, spin = _scrolling_host(qtbot)
    scrolled = area.verticalScrollBar().value()

    _notch(spin)

    assert spin.value() == 50
    assert not spin.hasFocus()
    assert area.verticalScrollBar().value() > scrolled


def test_wheel_still_steps_a_spin_box_the_reader_clicked_into(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    _area, _anchor, spin = _scrolling_host(qtbot)
    spin.setFocus()
    qtbot.waitUntil(spin.hasFocus)

    _notch(spin)

    assert spin.value() == 49


def test_wheel_over_the_unfocused_noise_selector_keeps_the_mode_and_the_result(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(result_fixture(tmp_path)))
    qtbot.addWidget(window)
    window.resize(1440, 900)
    window.show()
    assert QTest.qWaitForWindowExposed(window)
    window.pipeline_nav.select_step(2)
    selector = window.fit_panel.noise_selector
    qtbot.waitUntil(selector.isVisible)
    window.activateWindow()
    window.plot_panel.setFocus()
    mode = selector.currentData()
    result = window.document.project.datasets[0].last_valid_result
    assert result is not None

    _notch(selector)

    assert selector.currentData() == mode
    assert window.document.project.datasets[0].last_valid_result is result


def test_no_visible_workspace_editor_takes_focus_from_the_wheel(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(result_fixture(tmp_path)))
    qtbot.addWidget(window)
    window.resize(1440, 900)
    window.show()
    assert QTest.qWaitForWindowExposed(window)
    offenders: list[str] = []
    for step in range(len(window.pipeline_nav.step_labels)):
        window.pipeline_nav.select_step(step)
        offenders.extend(
            f"step {step}: {editor.accessibleName() or editor.objectName() or type(editor).__name__}"
            for editor in window.findChildren(QWidget)
            if isinstance(editor, (QAbstractSpinBox, QComboBox))
            and editor.isVisible()
            and editor.focusPolicy() == Qt.FocusPolicy.WheelFocus
        )

    assert offenders == []
