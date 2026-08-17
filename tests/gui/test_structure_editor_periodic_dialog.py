"""Periodic structure dialog and drift wiring contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
)

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _document(tmp_path, *, beam=None):
    from xrr_fitter.gui.document import ProjectDocument

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(instrument_id="structure-gui"),
        beam=beam,
    )
    return ProjectDocument(project)


def _panel(qtbot, tmp_path, *, beam=None):
    from xrr_fitter.gui.structure.panel import StructurePanel

    panel = StructurePanel(_document(tmp_path, beam=beam))
    qtbot.addWidget(panel)
    return panel


def _bare(backing=SI) -> api.StructureSpec:
    return api.StructureSpec(AIR, (), backing)


def test_add_periodic_dialog_commits_ordered_children_through_direct_method(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "periodicNameInput").setText("Mo/Si")
    table = dialog.findChild(QTableWidget, "periodicLayerTable")
    values = (("Mo", "Mo", "10.28", "2.5", "0.2"), ("Si", "Si", "2.329", "4", "0.3"))
    for row, fields in enumerate(values):
        for column, value in enumerate(fields):
            table.setItem(row, column, QTableWidgetItem(value))
    buttons = dialog.findChild(QDialogButtonBox, "periodicDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    block = dialog.block()
    assert block.name == "Mo/Si"
    assert [layer.name for layer in block.layers] == ["Mo", "Si"]
    assert [layer.thickness_a for layer in block.layers] == [25.0, 40.0]


def test_add_periodic_dialog_keeps_full_stack_error_open_then_commits_correction(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_bare())
    dialog = PeriodicDialog(panel, commit_block=panel.add_periodic_block)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.findChild(QLineEdit, "periodicNameInput").setText("thin repeat")
    table = dialog.findChild(QTableWidget, "periodicLayerTable")
    values = (
        ("first", "SiO2", "2.2", "1", "1"),
        ("second", "Si", "2.329", "1", "1"),
    )
    for row, fields in enumerate(values):
        for column, value in enumerate(fields):
            table.setItem(row, column, QTableWidgetItem(value))
    buttons = dialog.findChild(QDialogButtonBox, "periodicDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    error = dialog.findChild(QLabel, "periodicDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.isVisible()
    assert error is not None and error.isVisible()
    assert "roughness_a must be below 4.9 A" in error.text()
    assert panel.structure.components == ()

    table.item(0, 4).setText("0.1")
    table.item(1, 4).setText("0.1")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert tuple(block.name for block in panel.structure.components) == ("thin repeat",)


def _fill_periodic_layers(dialog) -> None:
    table = dialog.findChild(QTableWidget, "periodicLayerTable")
    values = (("Mo", "Mo", "10.28", "2.5", "0.2"), ("Si", "Si", "2.329", "4", "0.3"))
    for row, fields in enumerate(values):
        for column, value in enumerate(fields):
            table.setItem(row, column, QTableWidgetItem(value))


def test_periodic_dialog_without_drift_emits_no_constraint_rules(qtbot) -> None:
    from xrr_fitter.fit.drift import drift_constraint_rules
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    kind = dialog.findChild(QComboBox, "periodicDriftKind")
    assert kind is not None
    assert kind.currentData() is None
    dialog.findChild(QLineEdit, "periodicNameInput").setText("plain")
    _fill_periodic_layers(dialog)
    buttons = dialog.findChild(QDialogButtonBox, "periodicDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    block = dialog.block()
    assert block.drift is None
    assert drift_constraint_rules(api.StructureSpec(AIR, (block,), SI)) == ()


def test_periodic_dialog_drift_warning_tracks_repeats(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    warning = dialog.findChild(QLabel, "periodicDriftWarning")
    assert warning is not None
    assert not warning.isVisible()

    dialog.findChild(QComboBox, "periodicDriftKind").setCurrentText("线性")
    assert warning.isVisible()

    repeats = dialog.findChild(QSpinBox, "periodicRepeatsInput")
    repeats.setValue(5)
    text_five = warning.text()
    repeats.setValue(50)
    text_fifty = warning.text()

    assert "5" in text_five
    assert "50" in text_fifty
    assert text_five != text_fifty


def test_periodic_dialog_random_drift_shows_seed_source_without_randomize_button(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.findChild(QComboBox, "periodicDriftKind").setCurrentText("随机")

    seed_source = dialog.findChild(QLabel, "periodicDriftSeedSource")
    assert seed_source is not None
    assert seed_source.isVisible()
    assert "种子来源" in seed_source.text()
    assert "工程种子" in seed_source.text()

    labels = tuple(button.text() for button in dialog.findChildren(QPushButton))
    assert all("随机生成" not in text for text in labels)


def test_periodic_dialog_feeds_selected_drift_into_committed_block(qtbot) -> None:
    from xrr_fitter.fit.drift import drift_constraint_rules
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "periodicNameInput").setText("drifting")
    _fill_periodic_layers(dialog)
    dialog.findChild(QComboBox, "periodicDriftKind").setCurrentText("线性")
    dialog.findChild(QComboBox, "periodicDriftTarget").setCurrentText("厚度")
    dialog.findChild(QDoubleSpinBox, "periodicDriftAmount").setValue(0.05)
    buttons = dialog.findChild(QDialogButtonBox, "periodicDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    drift = dialog.block().drift
    assert drift is not None
    assert (drift.kind, drift.target) == ("linear", "thickness")
    assert drift.amount == pytest.approx(0.05)
    assert drift_constraint_rules(api.StructureSpec(AIR, (dialog.block(),), SI))


def test_periodic_block_row_tooltip_annotates_drift(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    block = api.PeriodicBlock(
        "drifted",
        (
            api.LayerSpec("Mo", api.MaterialSpec("Mo", "Mo", 10.28), 25.0, roughness_a=2.0),
            api.LayerSpec("Si", SI, 40.0, roughness_a=3.0),
        ),
        3,
        drift=api.DriftSpec("linear", "thickness", 0.05),
    )
    panel.editor.load(api.StructureSpec(AIR, (block,), SI))

    tree = panel.editor.findChild(QTreeWidget, "structureTree")
    item = tree.topLevelItem(1)
    assert item.text(0) == "drifted"
    tooltip = item.toolTip(0)
    assert "线性" in tooltip
    assert "厚度" in tooltip


def test_periodic_dialog_accepts_64bit_master_seed_without_overflow(qtbot) -> None:
    """真实工程 master_seed 为 64 位；对话框须折叠到 32 位而非溢出崩溃（终审 Finding #1）。"""
    from xrr_fitter.gui.structure.dialogs import DRIFT_SEED_MODULUS, PeriodicDialog

    master_seed = 2**63 + 12345
    block_offset = 3
    dialog = PeriodicDialog(master_seed=master_seed, block_offset=block_offset)
    qtbot.addWidget(dialog)

    seed = dialog.findChild(QSpinBox, "periodicDriftSeed")
    assert 0 <= seed.value() <= 2_147_483_647
    assert seed.value() == (master_seed + block_offset) % DRIFT_SEED_MODULUS

    twin = PeriodicDialog(master_seed=master_seed, block_offset=block_offset)
    qtbot.addWidget(twin)
    assert twin.findChild(QSpinBox, "periodicDriftSeed").value() == seed.value()

    source = dialog.findChild(QLabel, "periodicDriftSeedSource")
    assert "种子来源" in source.text()
    assert "工程种子" in source.text()
    assert str(seed.value()) in source.text()


def test_periodic_dialog_seed_source_tracks_manual_seed_edits(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import PeriodicDialog

    dialog = PeriodicDialog(master_seed=123, block_offset=2)
    qtbot.addWidget(dialog)
    seed = dialog.findChild(QSpinBox, "periodicDriftSeed")
    source = dialog.findChild(QLabel, "periodicDriftSeedSource")
    original = seed.value()

    seed.setValue(original + 1)

    assert "用户编辑" in source.text()
    assert str(seed.value()) in source.text()
    assert source.text() != f"种子来源：工程种子 123 + 块偏移 2，折叠至 32 位 = {original}"


def test_panel_set_structure_accepts_drifted_block_end_to_end(qtbot, tmp_path) -> None:
    """漂移块经 panel.set_structure → validate_structure → expand_structure 全链不崩，
    且声明逐位回存（终审 Finding #2：补齐 Task 12/13 边界的全链断言）。"""
    panel = _panel(qtbot, tmp_path)
    block = api.PeriodicBlock(
        "drifted",
        (
            api.LayerSpec("Mo", api.MaterialSpec("Mo", "Mo", 10.28), 25.0, roughness_a=2.0),
            api.LayerSpec("Si", SI, 40.0, roughness_a=3.0),
        ),
        3,
        drift=api.DriftSpec("linear", "thickness", 0.05),
    )
    structure = api.StructureSpec(AIR, (block,), SI)

    assert panel.set_structure(structure) is True
    assert panel.structure == structure
    assert panel.structure.components[0].drift == api.DriftSpec("linear", "thickness", 0.05)


def test_periodic_dialog_field_visibility_tracks_drift_kind(qtbot) -> None:
    """按漂移类型精确显隐字段并切换增量标签：无→周期/相位隐藏；正弦→显现（终审 Finding #3）。"""
    from xrr_fitter.gui.structure.dialogs import DRIFT_AMOUNT_LABELS, PeriodicDialog

    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    kind = dialog.findChild(QComboBox, "periodicDriftKind")
    period = dialog.findChild(QDoubleSpinBox, "periodicDriftPeriod")
    phase = dialog.findChild(QDoubleSpinBox, "periodicDriftPhase")

    assert kind.currentData() is None
    assert not period.isVisible() and not phase.isVisible()

    kind.setCurrentText("正弦")
    assert period.isVisible() and phase.isVisible()
    assert dialog.drift_amount_label.text() == DRIFT_AMOUNT_LABELS["sine"]

    kind.setCurrentText("线性")
    assert not period.isVisible() and not phase.isVisible()
    assert dialog.drift_amount_label.text() == DRIFT_AMOUNT_LABELS["linear"]

    kind.setCurrentText("随机")
    assert dialog.drift_amount_label.text() == "随机幅度"
