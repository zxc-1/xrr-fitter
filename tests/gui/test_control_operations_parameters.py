"""Parameter and constraint edits reached from a real MainWindow."""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QTabBar
from tests.gui.control_operation_driver import setup_project

import xrr_fitter.api as api

NAME = "component.0.thickness_a"


def _table(controls, window):
    controls.named_click(window, "pipelineStep_参数")
    assert window.pipeline_nav.current_step_index() == 2
    panel = window.parameters_panel
    bar = controls.widget(window, "structureCanvasTabs", QTabBar)
    controls.reach(bar, bar.tabRect(2).center())
    controls.qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(2).center())
    controls.tab(controls.widget(window, "parameterTabs"), 0)
    return panel, panel.parameter_table


def _select_name(controls, table, name=NAME):
    row = table.row_names.index(name)
    item = table.item(row, 0)
    table.scrollToItem(item)
    controls.reach(table.viewport(), table.visualItemRect(item).center())
    controls.qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualItemRect(item).center())
    assert table.currentRow() == row
    return row


def _setting(window, name=NAME):
    return next(setting for setting in window.document.project.datasets[0].parameter_settings if setting.name == name)


def _context(controls, table, name, action):
    row = _select_name(controls, table, name)
    controls.context_menu(table.viewport(), table.visualItemRect(table.item(row, 0)).center(), action)


def _edit_numeric_parameters(controls, window, table, row):
    for column, value, stored_name, expected in (
        (1, "4.5", "initial", 45),
        (2, "1", "lower", 10),
        (3, "10", "upper", 100),
    ):
        controls.table_text(table, row, column, value)
        assert getattr(_setting(window), stored_name) == pytest.approx(expected)
        controls.record(
            f"parameterTable:{('initial', 'lower', 'upper')[column - 1]}",
            "double-click numeric cell, type, Return",
            "display nm commits to persisted angstrom parameter setting",
        )


def _reference_index(selector, name):
    return next(index for index in range(selector.count()) if selector.itemData(index).parameter_name == name)


def test_parameter_cells_and_three_freedom_gears_real_inputs(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    panel, table = _table(controls, window)
    row = _select_name(controls, table)
    _edit_numeric_parameters(controls, window, table, row)
    for freedom in (api.ParameterFreedom.RANGE_ONLY, api.ParameterFreedom.FIXED, api.ParameterFreedom.FREE):
        _select_name(controls, table)
        controls.qtbot.keyClick(table, Qt.Key.Key_Space)
        assert _setting(window).freedom == freedom
    controls.record(
        "parameterTable:freedomCheck",
        "select name cell and press Space three times",
        "tri-state cycles range_only/fixed/free and persists",
    )
    for name, freedom in (
        ("parameterFixedChoice", api.ParameterFreedom.FIXED),
        ("parameterRangeOnlyChoice", api.ParameterFreedom.RANGE_ONLY),
        ("parameterFreeChoice", api.ParameterFreedom.FREE),
    ):
        controls.named_click(window, name)
        assert _setting(window).freedom == freedom
        controls.record(
            name, "click visible disposition radio", "selected parameter's persisted freedom matches choice"
        )
    before = window.document.project
    controls.table_text(table, row, 2, "1000")
    assert window.document.project is before
    assert "下限" in panel.status_label.text() and panel.status_label.isVisible()
    controls.table_text(table, row, 2, "1")
    assert _setting(window).lower == 10
    controls.record(
        "parameterTable:lower",
        "enter inconsistent lower bound then correct",
        "bad bounds report inline error without committing; correction works",
    )
    _context(controls, table, NAME, "resetParameterAction")
    assert all(setting.name != NAME for setting in window.document.project.datasets[0].parameter_settings)
    controls.record(
        "resetParameterAction", "right-click name and click Reset", "override is removed and defaults restored"
    )


@pytest.mark.parametrize(
    "kind,values,expected",
    (
        ("uniform", (), ()),
        ("normal", ("4", "0.5"), (40.0, 5.0)),
        ("lognormal", ("1.3", "0.2"), (1.3 + math.log(10), 0.2)),
        ("soft_range", ("2", "8", "0.5"), (20.0, 80.0, 5.0)),
    ),
)
def test_prior_context_dialog_all_kinds_and_clear(controls, tmp_path, kind, values, expected):
    window = controls.window(setup_project(tmp_path))
    panel, table = _table(controls, window)

    def configure(dialog):
        controls.select(dialog.kind_select, dialog.kind_select.findText(kind))
        for index, value in enumerate(values):
            controls.type(dialog._params[index], value)
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(lambda: _context(controls, table, NAME, "editPriorAction"), [configure])
    prior = window.document.project.datasets[0].parameter_priors[0]
    assert prior.name == NAME
    assert prior.prior.kind == kind and prior.prior.parameters == pytest.approx(expected)
    identities = ["editPriorAction", "priorKindSelect", "priorDialogButtons:Ok"]
    identities += [f"priorParam{index}" for index in range(len(values))]
    controls.record(
        identities,
        f"real context menu; choose {kind}; keyboard scalars; OK",
        "prior kind and correctly transformed parameters persist",
    )
    original = window.document.project

    def cancel(dialog):
        controls.select(dialog.kind_select, dialog.kind_select.findText("normal"))
        controls.type(dialog._params[0], "3")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)

    controls.modals(lambda: _context(controls, table, NAME, "editPriorAction"), [cancel])
    assert window.document.project is original
    controls.record("priorDialogButtons:Cancel", "reopen existing prior; edit; Cancel", "prior/project is unchanged")
    _context(controls, table, NAME, "clearPriorAction")
    assert not window.document.project.datasets[0].parameter_priors
    controls.record("clearPriorAction", "right-click parameter then Clear prior", "prior override removed")


def test_advanced_checkbox_and_parameter_tabs(controls, tmp_path):
    window = controls.window(setup_project(tmp_path, datasets=2))
    panel, table = _table(controls, window)
    before = len(panel.visible_definitions)
    controls.click(panel.expert_toggle)
    assert panel.expert_mode and len(panel.visible_definitions) > before
    controls.click(panel.expert_toggle)
    assert not panel.expert_mode and len(panel.visible_definitions) == before
    controls.record(
        "expertModeToggle",
        "click checkbox twice",
        "advanced parameter definitions reveal/hide and expert depth persists",
    )
    tabs = controls.widget(window, "parameterTabs")
    for index, name in ((1, "共享"), (2, "约束"), (0, "参数")):
        controls.tab(tabs, index)
        assert tabs.currentWidget().isVisible()
        controls.record(
            f"parameterTabs:{name}",
            "click actual tab label",
            "corresponding parameter/sharing/constraint surface is visible",
        )


def test_constraint_builder_expression_add_delete_and_cancel(controls, tmp_path):
    window = controls.window(setup_project(tmp_path, layers=2))
    _table(controls, window)
    tabs = controls.widget(window, "parameterTabs")
    controls.tab(tabs, 2)
    editor = window.parameters_panel.constraint_editor
    original = window.document.project

    def cancel(dialog):
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)

    controls.modals(lambda: controls.click(editor.add_button), [cancel])
    assert window.document.project is original
    controls.record("constraintDialogButtons:Cancel", "Add constraint then Cancel", "project remains unchanged")

    def configure(dialog):
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)
        assert dialog.error_label.isVisible() and "不能为空" in dialog.error_label.text()
        target = _reference_index(dialog.target_input, "component.1.thickness_a")
        source = _reference_index(dialog.source_input, NAME)
        controls.select(dialog.target_input, target)
        controls.select(dialog.source_input, source)
        controls.select(dialog.operator_input, dialog.operator_input.findData("mul"))
        controls.click(dialog.add_operator_button)
        assert "两个操作数" in dialog.error_label.text()
        controls.type(dialog.constant_input, "1.1")
        controls.click(dialog.add_parameter_button)
        controls.click(dialog.add_constant_button)
        assert dialog.expression_tree.topLevelItemCount() == 2
        controls.click(dialog.add_operator_button)
        assert dialog.expression_tree.topLevelItemCount() == 1
        node = dialog.expression_tree.topLevelItem(0)
        assert node.childCount() == 2
        controls.qtbot.mouseClick(
            dialog.expression_tree.viewport(),
            Qt.MouseButton.LeftButton,
            pos=dialog.expression_tree.visualItemRect(node).center(),
        )
        controls.qtbot.keyClick(dialog.expression_tree, Qt.Key.Key_Left)
        assert not node.isExpanded()
        controls.qtbot.keyClick(dialog.expression_tree, Qt.Key.Key_Right)
        assert node.isExpanded()
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(lambda: controls.click(editor.add_button), [configure])
    rule = window.document.project.constraint_rules[0]
    assert rule.target.parameter_name == "component.1.thickness_a"
    assert rule.expression.op == "mul"
    assert rule.expression.operands[0].reference.parameter_name == NAME
    assert rule.expression.operands[1].value == pytest.approx(1.1)
    controls.record(
        (
            "constraintAddRuleButton",
            "constraintTargetInput",
            "constraintSourceInput",
            "constraintOperatorInput",
            "constraintConstantInput",
            "constraintAddConstantButton",
            "constraintAddParameterButton",
            "constraintAddOperatorButton",
            "constraintDialogButtons:Ok",
            "constraintExpressionTree",
        ),
        "Add; invalid empty expression; keyboard/popup operands; reduce multiply; expand/collapse; OK",
        "inline validation keeps dialog open; complete AST persists as source * 1.1",
    )
    tree = editor.tree
    node = tree.topLevelItem(0)
    controls.qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(node).center())
    controls.click(editor.delete_button)
    assert not window.document.project.constraint_rules and tree.topLevelItemCount() == 0
    controls.record(
        ("constraintTree", "constraintDeleteRuleButton"),
        "select committed rule and click Delete",
        "only selected constraint is removed",
    )
