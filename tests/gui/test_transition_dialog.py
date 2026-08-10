"""Interface-transition entry in the layer dialog and the structure tree hint.

A transition replaces Névot-Croce rather than stacking with it, so the dialog
must disable the roughness input instead of rejecting it at commit time. The
assertions land on the committed LayerSpec rather than on widget state, which is
what the fit actually consumes.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
)

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.structure.panel import StructurePanel

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(instrument_id="transition-gui"),
    )
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def _dialog(qtbot):
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    dialog = LayerDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "layerNameInput").setText("graded")
    dialog.findChild(QLineEdit, "layerFormulaInput").setText("SiO2")
    dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(2.2)
    dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(4.5)
    return dialog


def _fill_branch(table: QTableWidget, row: int, kind: str, weight: str, width: str) -> None:
    for column, text in enumerate((kind, weight, width)):
        table.setItem(row, column, QTableWidgetItem(text))


def _commit(qtbot, dialog) -> None:
    buttons = dialog.findChild(QDialogButtonBox, "layerDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)


def test_layer_dialog_without_transition_commits_none(qtbot) -> None:
    dialog = _dialog(qtbot)
    dialog.findChild(QDoubleSpinBox, "layerRoughnessInput").setValue(0.3)

    _commit(qtbot, dialog)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.layer() == api.LayerSpec("graded", SIO2, 45.0, roughness_a=3.0)
    assert dialog.layer().transition is None


def test_layer_dialog_disables_roughness_when_transition_is_enabled(qtbot) -> None:
    dialog = _dialog(qtbot)
    roughness = dialog.findChild(QDoubleSpinBox, "layerRoughnessInput")
    roughness.setValue(0.3)
    toggle = dialog.findChild(QCheckBox, "layerTransitionToggle")

    toggle.setChecked(True)

    assert roughness.isEnabled() is False
    assert roughness.value() == 0.0

    toggle.setChecked(False)

    assert roughness.isEnabled() is True


def test_layer_dialog_commits_normalized_branch_weights(qtbot) -> None:
    dialog = _dialog(qtbot)
    dialog.findChild(QCheckBox, "layerTransitionToggle").setChecked(True)
    dialog.findChild(QDoubleSpinBox, "layerMicroslabInput").setValue(0.1)
    table = dialog.findChild(QTableWidget, "layerTransitionTable")
    _fill_branch(table, 0, "erf", "2", "1.0")
    _fill_branch(table, 1, "linear", "2", "0.5")

    _commit(qtbot, dialog)

    assert dialog.result() == QDialog.DialogCode.Accepted
    transition = dialog.layer().transition
    assert transition is not None
    assert tuple(branch.kind for branch in transition.branches) == ("erf", "linear")
    assert tuple(branch.weight for branch in transition.branches) == (0.5, 0.5)
    assert tuple(branch.thickness_a for branch in transition.branches) == (10.0, 5.0)
    assert transition.microslab_max_a == 1.0
    assert dialog.layer().roughness_a == 0.0


def test_layer_dialog_reports_unknown_transition_kind(qtbot) -> None:
    dialog = _dialog(qtbot)
    dialog.findChild(QCheckBox, "layerTransitionToggle").setChecked(True)
    table = dialog.findChild(QTableWidget, "layerTransitionTable")
    _fill_branch(table, 0, "gaussian", "1", "1.0")
    _fill_branch(table, 1, "erf", "1", "1.0")

    _commit(qtbot, dialog)

    error = dialog.findChild(QLabel, "layerDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert error.isVisible() is True
    assert "gaussian" in error.text()


def test_layer_dialog_reports_incomplete_branch_row(qtbot) -> None:
    dialog = _dialog(qtbot)
    dialog.findChild(QCheckBox, "layerTransitionToggle").setChecked(True)
    table = dialog.findChild(QTableWidget, "layerTransitionTable")
    _fill_branch(table, 0, "erf", "1", "1.0")
    _fill_branch(table, 1, "linear", "", "1.0")

    _commit(qtbot, dialog)

    error = dialog.findChild(QLabel, "layerDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert error.isVisible() is True


def test_structure_tree_marks_transition_in_the_roughness_column(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.structure.editor import TREE_HEADERS

    graded = api.LayerSpec(
        "graded",
        SIO2,
        45.0,
        roughness_a=0.0,
        transition=api.InterfaceTransition(
            (api.TransitionBranch("erf", 1.0, 8.0),),
            microslab_max_a=2.0,
        ),
    )
    panel = _panel(qtbot, tmp_path)

    panel.set_structure(api.StructureSpec(AIR, (graded,), SI))

    tree = panel.findChild(QTreeWidget, "structureTree")
    assert len(TREE_HEADERS) == 6
    assert "erf" in tree.topLevelItem(1).toolTip(4)
