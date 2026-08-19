from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
)

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.structure.panel import StructurePanel

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path: Path, *, joint: bool = False) -> StructurePanel:
    project = api.new_project()
    count = 2 if joint else 1
    for index in range(count):
        project = api.add_dataset(
            project,
            _write_curve(tmp_path / f"sample-{index}.xy"),
            api.InstrumentSpec(instrument_id="backing-gui"),
        )
    if joint:
        project = api.set_batch_mode(project, "joint")
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    panel.set_structure(api.StructureSpec(AIR, (), SI))
    return panel


def _edit_backing(
    qtbot,
    panel: StructurePanel,
    *,
    formula: str,
    density: float,
    roughness_nm: float,
) -> None:
    button = panel.findChild(QPushButton, "editBackingButton")
    assert button is not None

    def configure() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        dialog.findChild(QLineEdit, "backingFormulaInput").setText(formula)
        dialog.findChild(QDoubleSpinBox, "backingDensityInput").setValue(density)
        dialog.findChild(QDoubleSpinBox, "backingRoughnessInput").setValue(roughness_nm)
        buttons = dialog.findChild(QDialogButtonBox, "backingDialogButtons")
        qtbot.mouseClick(
            buttons.button(QDialogButtonBox.StandardButton.Ok),
            Qt.MouseButton.LeftButton,
        )

    QTimer.singleShot(0, configure)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def test_structure_editor_edits_backing_material_and_roughness(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)

    _edit_backing(
        qtbot,
        panel,
        formula="Al2O3",
        density=3.95,
        roughness_nm=0.5,
    )

    assert panel.structure is not None
    assert panel.structure.backing == api.MaterialSpec("Al2O3", "Al2O3", 3.95)
    assert panel.structure.backing_roughness_a == 5.0


def test_structure_editor_editing_backing_propagates_in_joint_mode(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path, joint=True)

    _edit_backing(
        qtbot,
        panel,
        formula="SiO2",
        density=2.65,
        roughness_nm=0.2,
    )

    assert tuple(
        (dataset.structure.backing.formula, dataset.structure.backing_roughness_a)
        for dataset in panel.document.project.datasets
    ) == (("SiO2", 2.0), ("SiO2", 2.0))
