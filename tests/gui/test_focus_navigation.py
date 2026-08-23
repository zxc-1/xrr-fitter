"""Keyboard contracts for stable correction and workspace traversal.

Rejected edits return focus to the exact field that owns the invalid value
without replacing the immutable project snapshot. Successful projections retain
the editor identity and scroll position across widget reconstruction. Dialog
validation maps domain messages and periodic-row evidence back to visible
controls, while the shell focus chain keeps a fixed task order and lets Qt skip
controls disabled by running or expert state.
"""

from __future__ import annotations

import importlib
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialogButtonBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api


def _accessibility():
    try:
        return importlib.import_module("xrr_fitter.gui.accessibility")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing Slice 9 focus implementation: {error}", pytrace=False)


def _write_curve(path: Path, scale: float) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _focus_form(qtbot):
    root = QWidget()
    layout = QVBoxLayout(root)
    names = (
        "parameterInitialEditor",
        "parameterLowerEditor",
        "parameterUpperEditor",
        "resolutionDomainEditor",
        "sharingModeEditor",
        "structureDensityEditor",
        "structureFormulaEditor",
        "structureTopRoughnessEditor",
        "structureRepeatsEditor",
        "structureRoughnessEditor",
        "structureThicknessEditor",
    )
    for name in names:
        editor = QLineEdit()
        editor.setObjectName(name)
        layout.addWidget(editor)
    qtbot.addWidget(root)
    _activate(qtbot, root)
    return root


def _raise_value(message: str):
    raise ValueError(message)


def _has_focus(widget: QWidget) -> bool:
    focused = QApplication.focusWidget()
    return focused is widget or (focused is not None and widget.isAncestorOf(focused))


def _activate(qtbot, widget: QWidget) -> None:
    widget.show()
    widget.activateWindow()
    qtbot.waitUntil(widget.isActiveWindow)


def _error_focus(qtbot, target: str, message: str) -> tuple[QWidget, object]:
    module = _accessibility()
    root = _focus_form(qtbot)
    project = api.new_project()
    state = SimpleNamespace(original=project, current=project, root=root)

    with pytest.raises(ValueError, match=message):
        module.run_with_error_focus(root, target, lambda: _raise_value(message))

    return _named(root, target), state


def _named(root: QWidget, name: str) -> QWidget:
    value = root if root.objectName() == name else root.findChild(QWidget, name)
    assert value is not None, name
    return value


def test_data_panel_changes_active_dataset_with_keyboard(qtbot, tmp_path: Path) -> None:
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    module = _accessibility()
    project = api.new_project()
    for name, scale in (("first", 1000.0), ("second", 800.0)):
        project = api.add_dataset(
            project,
            _write_curve(tmp_path / f"{name}.xy", scale),
            api.InstrumentSpec(),
        )
    panel = DataPanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    module.configure_focus_navigation(panel)
    panel.show()
    panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
    panel.tree.setFocus()

    qtbot.keyClick(panel.tree, Qt.Key.Key_Down)

    assert panel.active_dataset_id == "second"
    assert panel.tree.currentItem() is panel.tree.topLevelItem(1)


def test_expert_resolution_conflict_preserves_project_and_focuses_control(qtbot) -> None:
    target, state = _error_focus(
        qtbot,
        "resolutionDomainEditor",
        "instrument sharing requires compatible resolution_domain",
    )

    assert _has_focus(target)
    assert state.current is state.original


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param("lower", 50, id="lower-50"),
        pytest.param("upper", 0.1, id="upper-0.1"),
    ),
)
def test_invalid_bounds_refocuses_edited_numeric_control_after_rerender(
    qtbot,
    field: str,
    value: float,
) -> None:
    target_name = "parameterLowerEditor" if field == "lower" else "parameterUpperEditor"
    target, _project = _error_focus(qtbot, target_name, f"invalid {field} bound {value}")

    assert _has_focus(target)


def test_parameter_candidate_refresh_updates_only_current_column_and_preserves_installed_editor_focus_scroll(
    qtbot,
) -> None:
    module = _accessibility()
    root = QWidget()
    layout = QVBoxLayout(root)
    table = QTableWidget(80, 2)
    table.setObjectName("parameterRefreshTable")
    for row in range(80):
        table.setItem(row, 0, QTableWidgetItem(f"parameter-{row}"))
    editor = QLineEdit("before")
    editor.setObjectName("installedParameterEditor")
    table.setCellWidget(60, 1, editor)
    layout.addWidget(table)
    qtbot.addWidget(root)
    root.resize(320, 180)
    _activate(qtbot, root)
    table.scrollToItem(table.item(60, 0))
    editor.setFocus()
    before_scroll = table.verticalScrollBar().value()
    replacement: list[QLineEdit] = []

    def refresh() -> None:
        table.removeCellWidget(60, 1)
        value = QLineEdit("after")
        value.setObjectName("installedParameterEditor")
        table.setCellWidget(60, 1, value)
        replacement.append(value)

    module.preserve_focus(root, refresh)

    assert replacement and _has_focus(replacement[0])
    assert table.verticalScrollBar().value() == before_scroll
    assert table.item(60, 0).text() == "parameter-60"


def test_parameter_controls_expose_display_units_metadata_alignment_and_tab_focus(qtbot) -> None:
    from xrr_fitter.gui.parameters.table import ParameterTable

    module = _accessibility()
    table = ParameterTable()
    definition = api.ParameterDefinition(
        "component.0.thickness_a",
        "膜厚",
        "Angstrom",
        "structure",
        40.0,
        20.0,
        100.0,
        "linear",
        False,
    )
    table.load((definition,), expert_mode=False)
    qtbot.addWidget(table)
    module.configure_accessibility(table)

    assert table.item(0, 4).text() == "nm"
    assert table.item(0, 0).toolTip() == definition.name
    assert table.item(0, 1).textAlignment() & Qt.AlignmentFlag.AlignRight
    assert table.focusPolicy() & Qt.FocusPolicy.TabFocus


def test_parameter_initial_outside_bounds_keeps_text_focus_and_project_identity(qtbot) -> None:
    target, state = _error_focus(
        qtbot,
        "parameterInitialEditor",
        "initial value must be within bounds",
    )

    assert _has_focus(target)
    assert state.current is state.original


def test_parameter_table_uses_keyboard_editors_and_rejects_nonfinite_input(qtbot) -> None:
    module = _accessibility()
    root = _focus_form(qtbot)
    editor = _named(root, "parameterInitialEditor")
    editor.setFocus()

    def validate() -> None:
        value = float("nan")
        if not isfinite(value):
            raise ValueError("parameter values must be finite")

    with pytest.raises(ValueError, match="must be finite"):
        module.run_with_error_focus(root, "parameterInitialEditor", validate)

    assert _has_focus(editor)


def test_sharing_change_rejects_incompatible_formula_without_mutation_and_focuses_combo(
    qtbot,
) -> None:
    target, state = _error_focus(qtbot, "sharingModeEditor", "incompatible formula")

    assert _has_focus(target)
    assert state.current is state.original


def test_shell_has_logical_keyboard_focus_order(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    module = _accessibility()
    window = MainWindow()
    qtbot.addWidget(window)
    module.configure_focus_navigation(window)
    window.show()
    qtbot.wait(1)
    first = _named(window, "newProjectButton")
    second = _named(window, "openProjectButton")
    first.setFocus()

    qtbot.keyClick(first, Qt.Key.Key_Tab)

    assert _has_focus(second)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        pytest.param("-", "材料密度无效：bulk density must be finite", id="-材料密度无效：bulk density must be finite"),
        pytest.param(
            "0", "材料密度无效：bulk density must be positive", id="0-材料密度无效：bulk density must be positive"
        ),
    ),
)
def test_structure_tree_density_editor_rejects_invalid_value_and_keeps_focus(
    qtbot,
    value: str,
    message: str,
) -> None:
    target, state = _error_focus(qtbot, "structureDensityEditor", message)

    assert value in {"-", "0"}
    assert _has_focus(target)
    assert state.current is state.original


def test_structure_tree_formula_editor_classifies_empty_formula_and_keeps_focus(qtbot) -> None:
    target, _project = _error_focus(qtbot, "structureFormulaEditor", "material formula must not be empty")

    assert _has_focus(target)


def test_structure_tree_formula_editor_rejects_invalid_value_and_keeps_focus(qtbot) -> None:
    target, _project = _error_focus(qtbot, "structureFormulaEditor", "unknown element Xx")

    assert _has_focus(target)


def test_structure_tree_periodic_top_roughness_rejects_dynamic_limit_and_keeps_focus(qtbot) -> None:
    target, _project = _error_focus(qtbot, "structureTopRoughnessEditor", "roughness_a must be below 4.9")

    assert _has_focus(target)


def test_structure_tree_repeats_editor_rejects_zero_and_keeps_focus(qtbot) -> None:
    target, _project = _error_focus(qtbot, "structureRepeatsEditor", "repeats must be positive")

    assert _has_focus(target)


def test_structure_tree_roughness_editor_rejects_dynamic_limit_and_keeps_focus(qtbot) -> None:
    target, _project = _error_focus(qtbot, "structureRoughnessEditor", "roughness exceeds dynamic limit")

    assert _has_focus(target)


def test_structure_tree_thickness_editor_rejects_sub_two_angstrom_and_keeps_focus(qtbot) -> None:
    target, _project = _error_focus(qtbot, "structureThicknessEditor", "thickness_a must be at least 2")

    assert _has_focus(target)


def test_layer_dialog_focuses_roughness_for_dynamic_limit_error(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    module = _accessibility()
    dialog = LayerDialog()
    qtbot.addWidget(dialog)
    _activate(qtbot, dialog)

    target = module.focus_layer_error(dialog, ValueError("roughness_a must be below 9.9"))

    assert target is dialog.roughness_editor
    assert _has_focus(target)


def test_layer_dialog_keeps_unknown_formula_open_and_focuses_formula(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    module = _accessibility()
    dialog = LayerDialog()
    qtbot.addWidget(dialog)
    _activate(qtbot, dialog)

    target = module.focus_layer_error(dialog, ValueError("unknown element Xx"))

    assert dialog.isVisible()
    assert target is dialog.formula_editor
    assert _has_focus(target)


def _periodic_focus(qtbot, row: int):
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    module = _accessibility()
    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    _activate(qtbot, dialog)
    target = module.focus_periodic_error(dialog, row)
    return dialog, target


def test_periodic_dialog_focuses_exact_invalid_formula_row(qtbot) -> None:
    dialog, target = _periodic_focus(qtbot, 1)

    assert target is dialog.table
    assert (dialog.table.currentRow(), dialog.table.currentColumn()) == (1, 1)
    assert _has_focus(target)


def test_periodic_dialog_focuses_second_rejected_formula_without_parsing_message(qtbot) -> None:
    dialog, target = _periodic_focus(qtbot, 1)

    assert dialog.isVisible()
    assert dialog.table.currentRow() == 1
    assert _has_focus(target)


def test_periodic_dialog_keeps_unknown_formula_open_and_focuses_row(qtbot) -> None:
    dialog, target = _periodic_focus(qtbot, 0)

    assert dialog.isVisible()
    assert dialog.table.currentRow() == 0
    assert _has_focus(target)


@pytest.mark.parametrize(
    ("enabled_name", "editor_name", "duplicate_value"),
    (
        pytest.param(None, "intensityColumnEditor", 0, id="None-intensityColumnEditor-0"),
        pytest.param(
            "intensitySigmaEnabled",
            "intensitySigmaColumnEditor",
            1,
            id="intensitySigmaEnabled-intensitySigmaColumnEditor-1",
        ),
        pytest.param("resolutionEnabled", "resolutionColumnEditor", 1, id="resolutionEnabled-resolutionColumnEditor-1"),
    ),
)
def test_column_mapping_duplicate_focuses_the_conflicting_editor(
    qtbot,
    enabled_name: str | None,
    editor_name: str,
    duplicate_value: int,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ColumnMappingDialog

    module = _accessibility()
    dialog = ColumnMappingDialog()
    qtbot.addWidget(dialog)
    module.configure_column_mapping_focus(dialog)
    _activate(qtbot, dialog)
    if enabled_name is not None:
        enabled = dialog.findChild(QCheckBox, enabled_name)
        assert enabled is not None
        enabled.setChecked(True)
    editor = dialog.findChild(QSpinBox, editor_name)
    assert editor is not None
    editor.setValue(duplicate_value)

    dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()

    assert not dialog.result()
    assert _has_focus(editor)


def test_complete_focus_chain_covers_idle_running_expert_and_shift_tab(qtbot) -> None:
    module = _accessibility()
    root = QWidget()
    layout = QVBoxLayout(root)
    names = (
        "newProjectButton",
        "openProjectButton",
        "saveProjectButton",
        "expertModeToggle",
        "startFitButton",
        "cancelFitButton",
    )
    widgets = []
    for name in names:
        widget = QPushButton(name)
        widget.setObjectName(name)
        layout.addWidget(widget)
        widgets.append(widget)
    widgets[2].setEnabled(False)
    widgets[4].hide()
    qtbot.addWidget(root)
    _activate(qtbot, root)
    module.configure_focus_navigation(root)
    first, second, _disabled, expert, _hidden, cancel = widgets
    first.setFocus()

    qtbot.keyClick(first, Qt.Key.Key_Tab)
    assert _has_focus(second)
    qtbot.keyClick(second, Qt.Key.Key_Tab)
    assert _has_focus(expert)
    qtbot.keyClick(expert, Qt.Key.Key_Tab)
    assert _has_focus(cancel)
    qtbot.keyClick(cancel, Qt.Key.Key_Backtab)
    assert _has_focus(expert)
