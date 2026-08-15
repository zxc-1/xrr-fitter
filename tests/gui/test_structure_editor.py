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
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


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


def _layer(name="film", material=SIO2, thickness=40.0) -> api.LayerSpec:
    return api.LayerSpec(name, material, thickness, roughness_a=3.0)


def _periodic() -> api.PeriodicBlock:
    return api.PeriodicBlock(
        "Mo/Si",
        (
            api.LayerSpec("Mo", api.MaterialSpec("Mo", "Mo", 10.28), 25.0),
            api.LayerSpec("Si", SI, 40.0),
        ),
        repeats=8,
    )


def test_structure_edit_replaces_active_dataset_project_structure(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    structure = api.StructureSpec(AIR, (_layer(),), SI)
    updated = api.set_structure(original, "sample", structure)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "set_structure",
        lambda project, dataset_id, value: (
            calls.append((project, dataset_id, value)),
            updated,
        )[1],
    )
    events: list[tuple[str, object]] = []
    panel.structure_changed.connect(lambda key, value: events.append((key, value)))

    panel.set_structure(structure)

    assert calls == [(original, "sample", structure)]
    assert panel.document.project is updated
    assert panel.structure == structure
    assert events == [("sample", structure)]


def test_imported_dataset_can_initialize_default_structure_from_visible_button(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    button = panel.findChild(QPushButton, "initializeStructureButton")
    add_layer = panel.findChild(QPushButton, "addLayerButton")

    assert button is not None and button.isVisibleTo(panel)
    assert add_layer is not None and add_layer.isEnabled() is False

    qtbot.mouseClick(button, Qt.LeftButton)

    assert panel.structure == _bare()
    assert button.isHidden()
    assert add_layer.isEnabled() is True


def test_structure_edit_failure_rolls_back_project_editor_and_signal(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    before = panel.document.project
    events: list[object] = []
    panel.structure_changed.connect(events.append)
    monkeypatch.setattr(
        api,
        "set_structure",
        lambda *_args: (_ for _ in ()).throw(ValueError("structure rejected")),
    )

    with pytest.raises(ValueError, match="structure rejected"):
        panel.set_structure(_bare())

    assert panel.document.project is before
    assert panel.structure is None
    assert events == []


def test_structure_editor_validates_with_active_mixed_kalpha_beam(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    beam = api.BeamSpec("mixed_kalpha")
    panel = _panel(qtbot, tmp_path, beam=beam)
    original = panel.document.project
    structure = _bare()
    updated = api.set_structure(original, "sample", structure)
    observed: list[api.BeamSpec] = []

    def set_structure(project, dataset_id, candidate):
        observed.append(project.datasets[0].beam)
        return updated

    monkeypatch.setattr(api, "set_structure", set_structure)

    panel.set_structure(structure)

    assert observed == [beam]


def test_structure_editor_adds_and_removes_ordinary_layer(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_bare())
    layer = _layer()

    panel.add_layer(layer)
    assert panel.structure.components == (layer,)

    panel.remove_component(0)
    assert panel.structure.components == ()


def test_structure_editor_adds_and_moves_periodic_block(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    layer = _layer()
    block = _periodic()
    panel.set_structure(api.StructureSpec(AIR, (layer,), SI))

    panel.add_periodic_block(block)
    changed = panel.move_component(1, 0)

    assert changed is True
    assert panel.structure.components == (block, layer)


@pytest.mark.parametrize(
    ("method", "arguments"),
    (
        ("remove_component", (-1,)),
        ("remove_component", (1,)),
        ("move_component", (-1, 0)),
        ("move_component", (0, 1)),
        ("move_component", (1, 0)),
    ),
)
def test_structure_editor_rejects_invalid_component_indices_transactionally(
    qtbot,
    tmp_path,
    method,
    arguments,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer(),), SI))
    before = panel.document.project
    events: list[object] = []
    panel.structure_changed.connect(events.append)

    with pytest.raises(IndexError, match="component index out of range"):
        getattr(panel, method)(*arguments)

    assert panel.document.project is before
    assert events == []


def test_structure_editor_noop_move_emits_nothing_and_preserves_identity(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.set_structure(api.StructureSpec(AIR, (_layer(),), SI))
    before = panel.document.project
    events: list[object] = []
    panel.structure_changed.connect(events.append)

    changed = panel.move_component(0, 0)

    assert changed is False
    assert panel.document.project is before
    assert events == []


def test_structure_editor_tree_keeps_fixed_roots_and_periodic_children_contained(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    block = _periodic()
    panel.set_structure(api.StructureSpec(AIR, (_layer(), block), SI))
    tree = panel.findChild(QTreeWidget, "structureTree")

    assert tree.topLevelItemCount() == 4
    assert tree.topLevelItem(0).text(0) == "Air"
    assert tree.topLevelItem(3).text(0) == "基底"
    periodic = tree.topLevelItem(2)
    assert periodic.text(0) == "Mo/Si"
    assert periodic.childCount() == 2
    assert [periodic.child(index).text(0) for index in range(2)] == ["Mo", "Si"]


def test_add_layer_dialog_commits_explicit_nm_fields_through_direct_method(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    dialog = LayerDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(QLineEdit, "layerNameInput").setText("cap")
    dialog.findChild(QLineEdit, "layerFormulaInput").setText("SiO2")
    dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(2.2)
    dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(4.5)
    dialog.findChild(QDoubleSpinBox, "layerRoughnessInput").setValue(0.3)
    buttons = dialog.findChild(QDialogButtonBox, "layerDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.layer() == api.LayerSpec("cap", SIO2, 45.0, roughness_a=3.0)


def test_add_layer_dialog_keeps_full_stack_error_open_then_commits_correction(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    panel = _panel(qtbot, tmp_path)
    panel.set_structure(_bare())
    dialog = LayerDialog(panel, commit_layer=panel.add_layer)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.findChild(QLineEdit, "layerNameInput").setText("too-rough")
    dialog.findChild(QLineEdit, "layerFormulaInput").setText("SiO2")
    dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(2.2)
    dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(1.0)
    roughness = dialog.findChild(QDoubleSpinBox, "layerRoughnessInput")
    roughness.setValue(1.0)
    buttons = dialog.findChild(QDialogButtonBox, "layerDialogButtons")

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    error = dialog.findChild(QLabel, "layerDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.isVisible()
    assert error is not None and error.isVisible()
    assert "roughness_a must be below 4.9 A" in error.text()
    assert panel.structure.components == ()

    roughness.setValue(0.1)
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert tuple(layer.name for layer in panel.structure.components) == ("too-rough",)


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
