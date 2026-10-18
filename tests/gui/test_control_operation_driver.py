"""Audit helpers must fail boundedly rather than hang at an unexpected modal."""

import pytest
from PySide6.QtWidgets import QDialog
from tests.gui.control_operation_driver import controls as controls


def test_unexpected_second_modal_is_rejected_with_its_identity(controls):
    first = QDialog()
    second = QDialog()
    second.setObjectName("unplannedDialog")
    controls.qtbot.addWidget(first)
    controls.qtbot.addWidget(second)
    with pytest.raises(AssertionError, match="unexpected modal dialog: unplannedDialog"):
        controls.modals(lambda: (first.exec(), second.exec()), [lambda dialog: dialog.reject()])
    assert not second.isVisible()
