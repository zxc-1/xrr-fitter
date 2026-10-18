"""Audit helpers must fail boundedly rather than hang at an unexpected modal."""

import pytest
from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QLineEdit
from tests.gui.control_operation_driver import source_curve


class FocusThief(QObject):
    """Take focus straight back from a field, as the CI runner's inactive dialog did."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.FocusIn:
            QTimer.singleShot(0, watched.clearFocus)
        return False


def test_a_dialog_its_handler_left_open_fails_by_name_instead_of_hanging(controls):
    # The first CI run of the control sweeps sat inside a file dialog's exec loop for the
    # whole 60-minute job: the handler returned, the dialog stayed up, and nothing closed it.
    stuck = QDialog()
    stuck.setObjectName("stuckDialog")
    controls.qtbot.addWidget(stuck)
    with pytest.raises(AssertionError, match="stuckDialog"):
        controls.modals(stuck.exec, [lambda dialog: None])
    assert not stuck.isVisible()


def test_file_choice_does_not_depend_on_the_name_field_having_focus(controls, tmp_path):
    # Without focus, QFileDialog keeps every parent folder of a typed absolute path selected,
    # and Open then navigates into "/" instead of accepting: the CI hang in getOpenFileNames.
    source = source_curve(tmp_path / "P1 Si+Zr.xy")
    dialog = QFileDialog()
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    controls.qtbot.addWidget(dialog)
    editor = controls.widget(dialog, "fileNameEdit", QLineEdit)
    thief = FocusThief()
    editor.installEventFilter(thief)
    chosen = []

    def choose():
        if dialog.exec() == QDialog.DialogCode.Accepted:
            chosen.extend(dialog.selectedFiles())

    def answer(opened):
        editor.clearFocus()
        QApplication.processEvents()
        controls.file(opened, source)

    controls.modals(choose, [answer])
    assert chosen == [str(source)]


def test_unexpected_second_modal_is_rejected_with_its_identity(controls):
    first = QDialog()
    second = QDialog()
    second.setObjectName("unplannedDialog")
    controls.qtbot.addWidget(first)
    controls.qtbot.addWidget(second)
    with pytest.raises(AssertionError, match="unexpected modal dialog: unplannedDialog"):
        controls.modals(lambda: (first.exec(), second.exec()), [lambda dialog: dialog.reject()])
    assert not second.isVisible()
