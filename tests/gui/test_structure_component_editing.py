"""已有组件要能改、能复制，新组件落在读者指着的地方。

此前周期块和梯度层一旦加进去就只能删了重加（层序、设定一起丢）；没有新建梯度层的入口；
新层只能追加到末尾；周期表格固定两行，「每周期增量」不说是什么量。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
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


def _film(name: str = "film", thickness_a: float = 400.0) -> api.LayerSpec:
    return api.LayerSpec(name, SIO2, thickness_a, roughness_a=3.0)


def _block() -> api.PeriodicBlock:
    cell = (
        api.LayerSpec("Mo", api.MaterialSpec("Mo", "Mo", 10.28), 25.0, roughness_a=2.0),
        api.LayerSpec("Si", SI, 40.0, roughness_a=2.0),
    )
    return api.PeriodicBlock("ML", cell, 3, top_roughness_a=1.5, drift=api.DriftSpec("linear", "thickness", 0.01))


def _panel(qtbot, tmp_path, components):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.structure.panel import StructurePanel

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "sample.xy"), api.InstrumentSpec())
    project = api.set_structure(project, "sample", api.StructureSpec(AIR, tuple(components), SI))
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def _select_row(panel, row: int, child: int | None = None) -> None:
    item = panel.editor.tree.topLevelItem(row)
    panel.editor.tree.setCurrentItem(item if child is None else item.child(child))


def _drive(qtbot, trigger, operation) -> None:
    errors: list[BaseException] = []
    seen: list[QDialog] = []

    def visit() -> None:
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QDialog):
            return
        seen.append(dialog)
        try:
            operation(dialog)
        except BaseException as error:
            errors.append(error)
            dialog.reject()

    QTimer.singleShot(0, visit)
    trigger()
    assert seen, "expected a modal dialog"
    if errors:
        raise errors[0]


def _ok(qtbot, dialog) -> None:
    box = dialog.findChild(QDialogButtonBox)
    qtbot.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)


def test_an_existing_periodic_block_opens_prefilled_and_is_replaced_in_place(qtbot, tmp_path) -> None:
    block = _block()
    panel = _panel(qtbot, tmp_path, (_film(), block))
    _select_row(panel, 2)
    editor = panel.editor
    assert editor.edit_component_action.isEnabled()

    def edit(dialog):
        assert dialog.windowTitle() == "编辑周期块"
        assert dialog.findChild(QLineEdit, "periodicNameInput").text() == "ML"
        table = dialog.findChild(QTableWidget, "periodicLayerTable")
        assert table.rowCount() == 2 and table.item(1, 1).text() == "Si"
        assert dialog.drift_kind.currentData() == "linear"
        dialog.repeats_editor.setValue(5)
        _ok(qtbot, dialog)

    _drive(qtbot, editor.edit_component_action.trigger, edit)

    components = panel.structure.components
    assert len(components) == 2 and components[0] == _film()
    assert components[1] == api.PeriodicBlock("ML", block.layers, 5, top_roughness_a=1.5, drift=block.drift)


def test_a_child_row_of_a_block_points_at_the_block_and_edits_it(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.structure.selection import SelectedLayerCard

    panel = _panel(qtbot, tmp_path, (_film(), _block()))
    card = SelectedLayerCard(panel.editor.replace_component)
    qtbot.addWidget(card)
    panel.editor.component_selected.connect(card.show_component)

    _select_row(panel, 2, child=1)

    assert "周期块" in card.hint_label.text() and "ML" in card.hint_label.text()
    assert "在层堆叠里选一层" not in card.hint_label.text()
    assert panel.editor.edit_component_action.isEnabled()
    assert not panel.editor.remove_action.isEnabled()


def test_a_gradient_layer_can_be_added_below_the_selection_and_edited_later(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, (_film("top"), _film("bottom")))
    editor = panel.editor
    _select_row(panel, 1)

    def create(dialog):
        assert dialog.windowTitle() == "添加梯度层"
        dialog.findChild(QLineEdit, "gradientNameInput").setText("ramp")
        dialog.findChild(QDoubleSpinBox, "gradientUpperSldInput").setValue(18.9)
        dialog.findChild(QDoubleSpinBox, "gradientLowerSldInput").setValue(20.1)
        dialog.findChild(QDoubleSpinBox, "gradientThicknessInput").setValue(6.0)
        _ok(qtbot, dialog)

    _drive(qtbot, editor.add_gradient_action.trigger, create)

    names = [component.name for component in panel.structure.components]
    assert names == ["top", "ramp", "bottom"]
    ramp = panel.structure.components[1]
    assert isinstance(ramp, api.GradientLayerSpec)
    assert ramp.upper_sld_a2 == pytest.approx(18.9e-6) and ramp.thickness_a == pytest.approx(60.0)
    assert panel.editor.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == 1

    def edit(dialog):
        assert dialog.windowTitle() == "编辑梯度层"
        assert dialog.findChild(QLineEdit, "gradientNameInput").text() == "ramp"
        dialog.findChild(QDoubleSpinBox, "gradientThicknessInput").setValue(8.0)
        _ok(qtbot, dialog)

    _drive(qtbot, editor.edit_component_action.trigger, edit)
    edited = panel.structure.components[1]
    assert edited.thickness_a == pytest.approx(80.0)
    # 没碰过的数逐位照抄：10⁻⁶ 的显示换算来回一趟不能把 SLD 改掉末几位。
    assert edited.upper_sld_a2 == ramp.upper_sld_a2 and edited.lower_sld_a2 == ramp.lower_sld_a2


@pytest.mark.parametrize(
    ("row", "expected"), ((1, ["top", "new", "bottom"]), (0, ["new", "top", "bottom"]), (3, ["top", "bottom", "new"]))
)
def test_a_new_layer_lands_below_the_selected_row(qtbot, tmp_path, row, expected) -> None:
    panel = _panel(qtbot, tmp_path, (_film("top"), _film("bottom")))
    _select_row(panel, row)

    panel.editor.add_layer(_film("new"))

    assert [component.name for component in panel.structure.components] == expected


def test_the_periodic_table_grows_and_shrinks_and_says_what_its_numbers_are(qtbot) -> None:
    from xrr_fitter.gui.structure.periodic_dialog import PeriodicDialog

    dialog = PeriodicDialog()
    qtbot.addWidget(dialog)
    table = dialog.findChild(QTableWidget, "periodicLayerTable")
    add = dialog.findChild(QPushButton, "periodicAddRowButton")
    remove = dialog.findChild(QPushButton, "periodicRemoveRowButton")

    assert table.horizontalHeaderItem(2).text() == "密度 (g/cm³)"
    add.click()
    assert table.rowCount() == 3
    remove.click()
    remove.click()
    assert table.rowCount() == 1 and not remove.isEnabled()
    assert "无量纲" in dialog.drift_amount.toolTip() and "(1 + s·c_k)" in dialog.drift_amount.toolTip()

    table.setItem(0, 1, QTableWidgetItem("Si3N4"))
    assert table.item(0, 2).text() == "3.17"


def test_a_component_can_be_duplicated_directly_below_itself(qtbot, tmp_path) -> None:
    block = _block()
    panel = _panel(qtbot, tmp_path, (_film(), block))
    _select_row(panel, 2)

    panel.editor.duplicate_action.trigger()

    components = panel.structure.components
    assert [component.name for component in components] == ["film", "ML", "ML 副本"]
    assert components[2].layers == block.layers and components[2].repeats == block.repeats
    assert panel.editor.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == 2


def test_loading_a_structure_with_nothing_selected_lands_on_the_first_plain_layer(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.structure.selection import SelectedLayerCard

    panel = _panel(qtbot, tmp_path, (_block(), _film("first-film")))
    card = SelectedLayerCard(panel.editor.replace_component)
    qtbot.addWidget(card)
    panel.editor.component_selected.connect(card.show_component)

    # 默认选中发生在层堆叠上屏时：构造期间卡片还没接线，那时的选中没人听得见。
    assert panel.editor.tree.currentItem() is None
    panel.show()
    qtbot.waitExposed(panel)

    assert panel.editor.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == 1
    assert card.formula_editor.isEnabled() and card.formula_editor.text() == "SiO2"


def test_each_row_carries_its_numbers_for_screen_readers_and_tooltips(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, (_film(),))
    item = panel.editor.tree.topLevelItem(1)

    for role in (Qt.ItemDataRole.AccessibleTextRole, Qt.ItemDataRole.ToolTipRole):
        text = item.data(0, role)
        assert "film" in text and "厚度 40.00" in text and "粗糙 0.30" in text and "密度 2.20" in text


def test_deleting_a_layer_says_so_and_points_at_undo(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, (_film("top"), _film("bottom")))
    _select_row(panel, 1)

    panel.editor.remove_action.trigger()

    status = panel.editor.status_label
    assert [component.name for component in panel.structure.components] == ["bottom"]
    assert status.isVisibleTo(panel.editor)
    assert "已删除「top」" in status.text() and "撤销" in status.text()

    panel.editor.add_layer(_film("again"))
    assert not status.isVisibleTo(panel.editor)


def test_the_card_floor_is_the_models_two_angstroms_and_numbers_read_like_the_table(qtbot) -> None:
    from xrr_fitter.gui.structure.selection import SelectedLayerCard

    card = SelectedLayerCard(lambda _index, _layer: None)
    qtbot.addWidget(card)
    card.show_component(0, api.LayerSpec("film", SIO2, 487.031, roughness_a=4.4))

    assert card.thickness_editor.minimum() == pytest.approx(0.2)
    assert card.thickness_editor.text() == "48.7031"
    assert card.roughness_editor.text() == "0.4400"
