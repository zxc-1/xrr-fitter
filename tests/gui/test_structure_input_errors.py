"""结构输入被拒时：说清楚是哪一格、为什么，并且项目一个字都不动。

化学式写错（``sio2``、``SiO2;``）以前抛的是 pyparsing 的 ``ParseException``，不是
``ValueError``，卡片和三个对话框都只捕获后者——异常落进 excepthook，屏上什么也没有。
删除、移动、拖放产生非法粗糙度时 ``structureValidationError`` 从没被写过字，层堆叠一动
不动，读者以为点击没生效。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
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


def _panel(qtbot, tmp_path, layers):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.structure.panel import StructurePanel

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "sample.xy"), api.InstrumentSpec())
    project = api.set_structure(project, "sample", api.StructureSpec(AIR, tuple(layers), SI))
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def _card(qtbot, panel):
    from xrr_fitter.gui.structure.selection import SelectedLayerCard

    card = SelectedLayerCard(panel.editor.replace_component)
    qtbot.addWidget(card)
    panel.editor.component_selected.connect(card.show_component)
    return card


def _select(panel, index: int) -> None:
    tree = panel.editor.tree
    tree.setCurrentItem(tree.topLevelItem(index + 1))
    assert tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == index


def _retype(qtbot, editor, text: str) -> None:
    editor.setFocus()
    QTest.keySequence(editor, QKeySequence(QKeySequence.StandardKey.SelectAll))
    qtbot.keyClicks(editor, text)
    qtbot.keyClick(editor, Qt.Key.Key_Return)


@pytest.mark.parametrize("typed", ("sio2", "SiO2;"))
def test_card_rejects_an_unparseable_formula_in_chinese_and_rolls_the_field_back(qtbot, tmp_path, typed) -> None:
    panel = _panel(qtbot, tmp_path, (api.LayerSpec("film", SIO2, 40.0, roughness_a=2.0),))
    card = _card(qtbot, panel)
    _select(panel, 0)
    before = panel.document.project

    _retype(qtbot, card.formula_editor, typed)

    assert panel.document.project is before
    assert card.error_label.isVisibleTo(card)
    assert "化学式" in card.error_label.text() and typed in card.error_label.text()
    assert "Expected" not in card.error_label.text()
    assert card.formula_editor.text() == "SiO2"

    # 回滚之后框里写的就是项目里的那一层；再改别的数，不能拖着那串坏化学式一起提交。
    _retype(qtbot, card.thickness_editor.lineEdit(), "5")
    layer = panel.structure.components[0]
    assert layer.thickness_a == pytest.approx(50.0) and layer.material == SIO2
    assert not card.error_label.isVisibleTo(card)


def _layer_dialog(panel):
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    dialog = LayerDialog(panel, commit_layer=panel.add_layer)
    dialog.findChild(QLineEdit, "layerNameInput").setText("cap")
    dialog.findChild(QDoubleSpinBox, "layerDensityInput").setValue(2.2)
    dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(4.0)
    dialog.findChild(QDoubleSpinBox, "layerRoughnessInput").setValue(0.2)
    return dialog


def _press_ok(qtbot, dialog, name: str) -> None:
    buttons = dialog.findChild(QDialogButtonBox, name)
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)


@pytest.mark.parametrize("typed", ("sio2", "SiO2;", "Xx2"))
def test_layer_dialog_names_the_bad_formula_in_chinese_and_stays_open(qtbot, tmp_path, typed) -> None:
    panel = _panel(qtbot, tmp_path, ())
    dialog = _layer_dialog(panel)
    qtbot.addWidget(dialog)
    dialog.show()
    formula = dialog.findChild(QLineEdit, "layerFormulaInput")
    formula.setText(typed)

    _press_ok(qtbot, dialog, "layerDialogButtons")

    error = dialog.findChild(QLabel, "layerDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted and dialog.isVisible()
    assert panel.structure.components == ()
    assert error.isVisible() and "化学式" in error.text() and typed in error.text()
    # 对话框里没有「上一个有效值」可回滚：留住读者打的字并整段选中，重打一遍就是替换。
    assert formula.text() == typed and formula.selectedText() == typed


def test_backing_dialog_rejects_a_lowercase_formula_in_chinese(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.structure.dialogs import BackingDialog

    panel = _panel(qtbot, tmp_path, ())
    dialog = BackingDialog(SI, 2.0, panel, commit_backing=panel.editor.set_backing)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.findChild(QLineEdit, "backingFormulaInput").setText("si")
    # 表外的化学式要先填密度（见 test_material_density_defaults）；填上之后才轮到化学式本身被拒。
    dialog.findChild(QDoubleSpinBox, "backingDensityInput").setValue(2.329)

    _press_ok(qtbot, dialog, "backingDialogButtons")

    error = dialog.findChild(QLabel, "backingDialogError")
    assert dialog.isVisible() and panel.structure.backing == SI
    assert error.isVisible() and "化学式" in error.text() and "si" in error.text()


def test_periodic_dialog_rejects_a_bad_formula_cell_in_chinese(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.structure.periodic_dialog import PeriodicDialog

    panel = _panel(qtbot, tmp_path, ())
    dialog = PeriodicDialog(panel, commit_block=panel.add_periodic_block)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.findChild(QLineEdit, "periodicNameInput").setText("ML")
    table = dialog.findChild(QTableWidget, "periodicLayerTable")
    for row, fields in enumerate((("Mo", "Mo", "10.28", "2.5", "0.1"), ("Si", "SiO2;", "2.2", "4", "0.1"))):
        for column, value in enumerate(fields):
            table.setItem(row, column, QTableWidgetItem(value))

    _press_ok(qtbot, dialog, "periodicDialogButtons")

    error = dialog.findChild(QLabel, "periodicDialogError")
    assert dialog.isVisible() and panel.structure.components == ()
    assert error.isVisible() and "化学式" in error.text() and "SiO2;" in error.text()


def _film(name: str, thickness_a: float, roughness_a: float) -> api.LayerSpec:
    return api.LayerSpec(name, SIO2, thickness_a, roughness_a=roughness_a)


def test_delete_that_exposes_an_illegal_roughness_is_refused_out_loud(qtbot, tmp_path) -> None:
    """删掉 L2 后 L3 的 5 nm 粗糙度落到 2 nm 厚的 L1 上，上限只剩 0.98 nm。"""
    layers = (_film("L1", 20.0, 2.0), _film("L2", 400.0, 3.0), _film("L3", 400.0, 50.0))
    panel = _panel(qtbot, tmp_path, layers)
    _select(panel, 1)
    before = panel.document.project

    panel.editor.remove_action.trigger()

    error = panel.editor.error_label
    assert panel.document.project is before
    assert error.isVisibleTo(panel.editor)
    text = error.text()
    assert "删除" in text and "L1" in text and "L3" in text and "0.98 nm" in text
    assert "项目未改动" in text and "roughness_a" not in text


def test_move_or_drop_that_exposes_an_illegal_roughness_is_refused_out_loud(qtbot, tmp_path) -> None:
    """L1 下移到 2 nm 厚的 L2 下面，它自己 5 nm 的入射侧粗糙度就超了 0.98 nm 的上限。"""
    layers = (_film("L1", 400.0, 50.0), _film("L2", 20.0, 3.0))
    panel = _panel(qtbot, tmp_path, layers)
    _select(panel, 0)
    before = panel.document.project

    # 拖放与上移/下移走同一个信号；直接发它就是拖放落点算出来之后的那一步。
    panel.editor.tree.component_moved.emit(0, 1)

    error = panel.editor.error_label
    assert panel.document.project is before
    assert error.isVisibleTo(panel.editor)
    assert "移动" in error.text() and "L2" in error.text() and "L1" in error.text() and "0.98 nm" in error.text()

    panel.editor.down_action.trigger()
    assert panel.document.project is before and "移动" in error.text()

    # 下一次成功的编辑把那句拒绝收回去：它说的是上一次没生效的操作。
    panel.set_structure(api.StructureSpec(AIR, (_film("L1", 400.0, 3.0), _film("L2", 20.0, 3.0)), SI))
    assert not error.isVisibleTo(panel.editor)
