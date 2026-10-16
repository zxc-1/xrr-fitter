"""约束构建器（F35）：运算符、参数、表达式用读者的话说，能撤销，拒绝原因说成中文。"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QPushButton
from tests.gui.test_parameter_sharing import _panel

import xrr_fitter.api as api

FIRST = api.ParameterReference("first", "component.0.thickness_a")
SECOND = api.ParameterReference("second", "component.0.thickness_a")
ROUGHNESS = api.ParameterReference("first", "component.0.roughness_a")
LABELS = {FIRST: "first · film 厚度", SECOND: "second · film 厚度"}
#: 原文里这些词一个都不该上屏。
ENGLISH = ("constraint", "unique", "cycle", "sharing", "roughness", "component.", "::")


def _dialog(qtbot, **kwargs):
    from xrr_fitter.gui.parameters.constraints import ConstraintDialog

    dialog = ConstraintDialog((FIRST,), (SECOND,), **kwargs)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def _select(combo, reference) -> None:
    combo.setCurrentIndex(next(index for index in range(combo.count()) if combo.itemData(index) == reference))


def _rule(target, expression) -> api.ConstraintRule:
    return api.ConstraintRule(target, expression)


def _ref(reference) -> api.ConstraintNode:
    return api.ConstraintNode("ref", reference=reference)


def test_operators_read_as_symbols_while_the_rule_keeps_op_codes(qtbot) -> None:
    dialog = _dialog(qtbot)
    combo = dialog.operator_input

    texts = [combo.itemText(index) for index in range(combo.count())]
    codes = [combo.itemData(index) for index in range(combo.count())]

    assert texts == ["+ 加", "− 减", "× 乘", "÷ 除", "^ 乘方"]
    assert codes == ["add", "sub", "mul", "div", "pow"]
    assert "左" in combo.toolTip() and "右" in combo.toolTip()


def test_builder_names_parameters_and_reads_the_formula(qtbot) -> None:
    dialog = _dialog(qtbot, labels=LABELS)

    assert dialog.target_input.itemText(0) == "first · film 厚度"
    assert dialog.source_input.itemText(0) == "second · film 厚度"
    # 原始标识留在悬停提示里：两个数据集里同名的层只能靠它分清。
    assert "second:component.0.thickness_a" in dialog.source_input.itemData(0, Qt.ItemDataRole.ToolTipRole)

    dialog.add_parameter_button.click()
    dialog.constant_input.setValue(1.1)
    dialog.add_constant_button.click()
    dialog.operator_input.setCurrentIndex(dialog.operator_input.findData("mul"))
    dialog.add_operator_button.click()

    root = dialog.expression_tree.topLevelItem(0)
    assert root.text(1) == "second · film 厚度 × 1.1"
    assert [root.child(index).text(1) for index in range(root.childCount())] == ["second · film 厚度", "1.1"]


def test_builder_undoes_one_step_at_a_time_and_clears(qtbot) -> None:
    dialog = _dialog(qtbot, labels=LABELS)
    undo = dialog.findChild(QPushButton, "constraintUndoButton")
    clear = dialog.findChild(QPushButton, "constraintClearButton")
    tree = dialog.expression_tree

    assert undo is not None and clear is not None
    assert not undo.isEnabled() and not clear.isEnabled()
    dialog.add_parameter_button.click()
    dialog.add_constant_button.click()
    dialog.add_operator_button.click()
    assert tree.topLevelItemCount() == 1

    undo.click()
    # 撤销一次运算要把它吃掉的两个操作数原样吐回来，而不是只弹掉栈顶。
    assert [tree.topLevelItem(index).text(1) for index in range(tree.topLevelItemCount())] == [
        "second · film 厚度",
        "0",
    ]
    clear.click()
    assert tree.topLevelItemCount() == 0 and not clear.isEnabled()
    undo.click()
    assert tree.topLevelItemCount() == 2
    undo.click()
    undo.click()
    assert tree.topLevelItemCount() == 0
    assert not undo.isEnabled() and not clear.isEnabled()


def test_pow_with_a_parameter_exponent_is_refused_in_chinese(qtbot) -> None:
    dialog = _dialog(qtbot, labels=LABELS)
    dialog.add_constant_button.click()
    dialog.add_parameter_button.click()
    dialog.operator_input.setCurrentIndex(dialog.operator_input.findData("pow"))

    dialog.add_operator_button.click()

    assert "指数" in dialog.error_label.text()
    assert not any(word in dialog.error_label.text() for word in ENGLISH)


def _rejection(panel, rules) -> str:
    with pytest.raises(ValueError):
        panel.apply_constraint_rules(rules)
    return panel.constraint_error_text()


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("duplicate", "约束目标重复"),
        ("cycle", "「first · film 厚度」 → 「second · film 厚度」 → 「first · film 厚度」"),
        ("shared", "已在共享组里"),
        ("roughness", "粗糙度"),
    ),
)
def test_editor_rejections_are_said_in_chinese_with_parameter_names(qtbot, tmp_path, case, expected) -> None:
    panel = _panel(qtbot, tmp_path)
    first_rule = _rule(FIRST, api.ConstraintNode("const", value=40.0))
    rules = {
        "duplicate": (first_rule, _rule(FIRST, api.ConstraintNode("const", value=41.0))),
        "cycle": (_rule(FIRST, _ref(SECOND)), _rule(SECOND, _ref(FIRST))),
        "shared": (_rule(FIRST, _ref(SECOND)),),
        "roughness": (_rule(FIRST, _ref(ROUGHNESS)),),
    }[case]
    if case == "shared":
        panel.apply_sharing_rules((api.SharingRule("shared-thickness", (FIRST, SECOND)),))

    text = _rejection(panel, rules)

    assert expected in text
    assert not any(word in text for word in ENGLISH), text


def test_editor_tree_names_targets_and_reads_the_formula(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    expression = api.ConstraintNode("mul", operands=(_ref(SECOND), api.ConstraintNode("const", value=1.1)))

    panel.apply_constraint_rules((_rule(FIRST, expression),))

    root = panel.constraint_editor.tree.topLevelItem(0)
    assert root.text(0) == "first · film 厚度"
    assert root.text(1) == "second · film 厚度 × 1.1"
    assert "first:component.0.thickness_a" in root.toolTip(0)


def test_add_command_hands_the_builder_the_same_names(qtbot, tmp_path, monkeypatch) -> None:
    from xrr_fitter.gui.parameters import constraints as module

    panel = _panel(qtbot, tmp_path)
    seen: list[object] = []

    class Capturing:
        def __init__(self, *_args, labels=None, **_kwargs):
            seen.append(labels)

        def exec(self):
            return 0

    monkeypatch.setattr(module, "ConstraintDialog", Capturing)
    panel.constraint_editor.add_button.click()

    assert seen and seen[0][FIRST] == "first · film 厚度"


def test_a_refusal_inside_the_builder_leaves_no_residue_on_the_page(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.apply_constraint_rules((_rule(FIRST, _ref(SECOND)),))
    editor = panel.constraint_editor
    seen: list[str] = []

    def drive() -> None:
        dialog = QApplication.activeModalWidget()
        try:
            _select(dialog.target_input, SECOND)
            _select(dialog.source_input, FIRST)
            dialog.add_parameter_button.click()
            dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
            seen.append(dialog.error_label.text())
        finally:
            dialog.reject()

    QTimer.singleShot(0, drive)
    editor.add_button.click()

    assert seen and "→" in seen[0]
    assert editor.error_label.isHidden()
    assert len(panel.constraint_rules) == 1
