"""Sharing controls are user-operable, not only programmatically writable."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
)
from tests.gui.test_parameter_sharing import _constraint_rule, _project, _rule

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.parameters.sharing import SharingEditor

THICKNESS = "component.0.thickness_a"
DENSITY = "component.0.density_scale"


def _editor(qtbot, tmp_path, *, rules=(), constraint=False):
    project = _project(tmp_path)
    if rules:
        project = api.set_sharing_rules(project, rules)
    if constraint:
        project = api.set_constraint_rules(project, (_constraint_rule(),))
    editor = SharingEditor(ProjectDocument(project))
    qtbot.addWidget(editor)
    editor.resize(560, 360)
    editor.show()
    qtbot.waitExposed(editor)
    return editor


def _button(editor, name):
    button = editor.findChild(QPushButton, name)
    assert button is not None, f"missing user-facing sharing control: {name}"
    return button


def _open_and_drive(qtbot, editor, operation):
    button = _button(editor, "sharingAddRuleButton")
    errors = []

    def drive():
        dialog = QApplication.activeModalWidget()
        try:
            assert isinstance(dialog, QDialog)
            assert dialog.objectName() == "sharingDialog"
            operation(dialog)
        except BaseException as error:
            errors.append(error)
        finally:
            if isinstance(dialog, QDialog) and dialog.isVisible():
                dialog.reject()

    QTimer.singleShot(0, drive)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    if errors:
        raise errors[0]


def _type(qtbot, editor, value):
    qtbot.mouseClick(editor, Qt.MouseButton.LeftButton)
    qtbot.keyClick(editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(editor, value)


def _member(qtbot, dialog, row):
    tree = dialog.findChild(QTreeWidget, "sharingDatasetList")
    item = tree.topLevelItem(row)
    qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(item).center())
    qtbot.keyClick(tree, Qt.Key.Key_Space)


def _parameter(qtbot, dialog, name):
    choice = dialog.findChild(QComboBox, "sharingParameterInput")
    index = choice.findData(name)
    assert index >= 0, f"missing eligible parameter: {name}"
    qtbot.mouseClick(choice, Qt.MouseButton.LeftButton)
    popup = choice.view()
    assert popup.isVisible()
    qtbot.keyClick(popup, Qt.Key.Key_Home)
    for _ in range(index):
        qtbot.keyClick(popup, Qt.Key.Key_Down)
    qtbot.keyClick(popup, Qt.Key.Key_Return)
    assert choice.currentData() == name
    assert dialog.isVisible()


def _configure(qtbot, dialog, *, key="shared-thickness", parameter=THICKNESS):
    _type(qtbot, dialog.findChild(QLineEdit, "sharingKeyInput"), key)
    _member(qtbot, dialog, 0)
    _member(qtbot, dialog, 1)
    _parameter(qtbot, dialog, parameter)


def _dialog_button(qtbot, dialog, standard):
    box = dialog.findChild(QDialogButtonBox, "sharingDialogButtons")
    qtbot.mouseClick(box.button(standard), Qt.MouseButton.LeftButton)


def _accept(qtbot, dialog):
    _dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Ok)


def _density_rule():
    return api.SharingRule(
        "shared-density", tuple(api.ParameterReference(name, DENSITY) for name in ("first", "second"))
    )


def test_add_rule_uses_real_clicks_and_keys_to_commit_project(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path)

    def commit(dialog):
        _configure(qtbot, dialog)
        _accept(qtbot, dialog)

    _open_and_drive(qtbot, editor, commit)

    assert editor.document.project.sharing_rules == (_rule(),)
    assert editor.document.is_dirty
    assert editor.tree.topLevelItem(0).text(0) == "shared-thickness"


def test_cancel_after_editing_keeps_exact_project_identity(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path)
    original = editor.document.project

    def cancel(dialog):
        _configure(qtbot, dialog)
        _dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Cancel)

    _open_and_drive(qtbot, editor, cancel)

    assert editor.document.project is original
    assert editor.tree.topLevelItemCount() == 0


def test_less_than_two_members_stays_open_with_visible_error(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path)
    original = editor.document.project

    def invalid(dialog):
        _type(qtbot, dialog.findChild(QLineEdit, "sharingKeyInput"), "one-member")
        _member(qtbot, dialog, 0)
        _accept(qtbot, dialog)
        error = dialog.findChild(QLabel, "sharingDialogError")
        assert dialog.isVisible()
        assert error.isVisibleTo(dialog)
        assert "至少" in error.text()
        assert editor.document.project is original

    _open_and_drive(qtbot, editor, invalid)


def test_duplicate_key_can_be_corrected_without_partial_commit(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path, rules=(_rule(),))
    original = editor.document.project

    def correct(dialog):
        _configure(qtbot, dialog, parameter=DENSITY)
        _accept(qtbot, dialog)
        error = dialog.findChild(QLabel, "sharingDialogError")
        assert dialog.isVisible() and error.isVisibleTo(dialog)
        assert "unique" in error.text()
        assert editor.document.project is original
        _type(qtbot, dialog.findChild(QLineEdit, "sharingKeyInput"), "shared-density")
        _accept(qtbot, dialog)

    _open_and_drive(qtbot, editor, correct)

    assert tuple(rule.sharing_key for rule in editor.rules) == ("shared-thickness", "shared-density")
    assert editor.rules[1].members == tuple(api.ParameterReference(name, DENSITY) for name in ("first", "second"))


@pytest.mark.parametrize("occupied", ["sharing", "constraint"])
def test_dialog_does_not_offer_already_owned_coordinates(qtbot, tmp_path, occupied):
    editor = _editor(
        qtbot, tmp_path, rules=(_rule(),) if occupied == "sharing" else (), constraint=occupied == "constraint"
    )

    def inspect(dialog):
        _member(qtbot, dialog, 0)
        _member(qtbot, dialog, 1)
        choices = dialog.findChild(QComboBox, "sharingParameterInput")
        assert choices.findData(THICKNESS) == -1
        assert choices.findData(DENSITY) >= 0
        _dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Cancel)

    _open_and_drive(qtbot, editor, inspect)


def test_delete_control_removes_only_selected_rule(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path, rules=(_rule(), _density_rule()))
    button = _button(editor, "sharingDeleteRuleButton")
    item = editor.tree.topLevelItem(1)
    qtbot.mouseClick(editor.tree.viewport(), Qt.MouseButton.LeftButton, pos=editor.tree.visualItemRect(item).center())
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert editor.document.project.sharing_rules == (_rule(),)
    assert editor.tree.topLevelItemCount() == 1


def test_delete_without_selection_reports_visible_error(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path)
    original = editor.document.project
    qtbot.mouseClick(_button(editor, "sharingDeleteRuleButton"), Qt.MouseButton.LeftButton)

    assert editor.document.project is original
    assert editor.error_label.isVisibleTo(editor)
    assert "选择" in editor.error_label.text()


def test_parent_disabled_gate_prevents_sharing_mutations(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path, rules=(_rule(),))
    button = _button(editor, "sharingDeleteRuleButton")
    original = editor.document.project
    item = editor.tree.topLevelItem(0)
    qtbot.mouseClick(editor.tree.viewport(), Qt.MouseButton.LeftButton, pos=editor.tree.visualItemRect(item).center())
    editor.setEnabled(False)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert not button.isEnabled()
    assert not _button(editor, "sharingAddRuleButton").isEnabled()
    assert editor.document.project is original


def test_blank_group_key_stays_editable_without_mutation(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path)
    original = editor.document.project

    def invalid(dialog):
        _configure(qtbot, dialog, key="   ")
        _accept(qtbot, dialog)
        error = dialog.findChild(QLabel, "sharingDialogError")
        assert dialog.isVisible() and error.isVisibleTo(dialog)
        assert "名称不能为空" in error.text()
        assert editor.document.project is original

    _open_and_drive(qtbot, editor, invalid)


def _lock_dataset(editor, dataset_id):
    project = editor.document.project
    settings = tuple(
        api.ParameterSetting(row.name, row.initial, row.lower, row.upper, api.ParameterFreedom.FIXED)
        for row in api.describe_parameters(project, dataset_id)
    )
    editor.document.replace_project(api.set_parameter_settings(project, dataset_id, settings))


def test_no_common_free_parameter_is_reported_inside_dialog(qtbot, tmp_path):
    editor = _editor(qtbot, tmp_path)
    _lock_dataset(editor, "second")
    original = editor.document.project

    def invalid(dialog):
        _type(qtbot, dialog.findChild(QLineEdit, "sharingKeyInput"), "no-free")
        _member(qtbot, dialog, 0)
        _member(qtbot, dialog, 1)
        assert dialog.findChild(QComboBox, "sharingParameterInput").count() == 0
        _accept(qtbot, dialog)
        error = dialog.findChild(QLabel, "sharingDialogError")
        assert error.isVisibleTo(dialog)
        assert "没有共同" in error.text()
        assert editor.document.project is original

    _open_and_drive(qtbot, editor, invalid)
