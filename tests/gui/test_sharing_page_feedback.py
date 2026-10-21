"""共享页说的话：失败只说在失败的地方，说中文，说读者认得的参数名。"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QLabel
from tests.gui.test_parameter_sharing import _project, _rule, _write_curve
from tests.gui.test_sharing_operations import (
    DENSITY,
    THICKNESS,
    _accept,
    _button,
    _configure,
    _dialog_button,
    _editor,
    _member,
    _open_and_drive,
)

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument


def test_a_duplicate_name_fails_inside_the_dialog_and_leaves_nothing_on_the_page(qtbot, tmp_path) -> None:
    editor = _editor(qtbot, tmp_path, rules=(_rule(),))
    original = editor.document.project

    def duplicate(dialog):
        _configure(qtbot, dialog, key="shared-thickness", parameter=DENSITY)
        _accept(qtbot, dialog)
        error = dialog.findChild(QLabel, "sharingDialogError")
        assert dialog.isVisible() and error.isVisibleTo(dialog)
        assert "shared-thickness" in error.text() and "已存在" in error.text()
        assert "unique" not in error.text()
        _dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Cancel)

    _open_and_drive(qtbot, editor, duplicate)

    assert editor.document.project is original
    assert not editor.error_label.isVisibleTo(editor)
    assert "unique" not in editor.error_label.text()


def test_adding_a_share_needs_a_second_dataset(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.parameters.sharing import SharingEditor

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "only.xy", 1000.0), api.InstrumentSpec())
    editor = SharingEditor(ProjectDocument(project))
    qtbot.addWidget(editor)
    button = _button(editor, "sharingAddRuleButton")

    assert not button.isEnabled()
    assert "两个数据集" in button.toolTip()

    editor.document.replace_project(
        api.add_dataset(project, _write_curve(tmp_path / "two.xy", 900.0), api.InstrumentSpec())
    )
    assert button.isEnabled()


def test_the_parameter_choice_reads_as_display_names(qtbot, tmp_path) -> None:
    editor = _editor(qtbot, tmp_path)
    definitions = {item.name: item.display_name for item in api.describe_parameters(editor.document.project, "first")}

    def inspect(dialog):
        _member(qtbot, dialog, 0)
        _member(qtbot, dialog, 1)
        choice = dialog.findChild(QComboBox, "sharingParameterInput")
        index = choice.findData(THICKNESS)
        assert index >= 0
        assert choice.itemText(index) == definitions[THICKNESS]
        assert all(not choice.itemText(row).startswith("component.") for row in range(choice.count()))
        _dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Cancel)

    _open_and_drive(qtbot, editor, inspect)


def test_the_cross_dataset_badge_follows_the_projects_sharing_rules(qtbot, tmp_path) -> None:
    """编译出来的声明从不带 ``sharing_key``，徽标按它判断就永远不亮。"""
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(api.set_sharing_rules(_project(tmp_path), (_rule(),))))
    qtbot.addWidget(panel)
    table = panel.parameter_table

    table.setCurrentCell(table.row_names.index(THICKNESS), 0)

    assert panel.disposition.sharing_badge.text() == "共享：跨 2 集共享厚度 d"

    table.setCurrentCell(table.row_names.index(DENSITY), 0)
    assert panel.disposition.sharing_badge.text() == ""
