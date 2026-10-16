"""换材料就换它的密度：表里有的化学式带入名义密度，表外的要读者自己填。

此前从 ▾ 菜单选 Si3N4，层名仍叫 SiO2、密度仍是 SiO2 的 2.2；新建层对任何化学式都先填
1.0 g/cm³——一个没人写过的数，提交出去就是这层的密度。
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QLabel, QLineEdit

import xrr_fitter.api as api

SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.2)
SI = api.MaterialSpec("Si", "Si", 2.329)


def test_the_initial_density_table_is_public_and_read_only() -> None:
    table = api.INITIAL_DENSITIES_G_CM3
    assert table["Si3N4"] == pytest.approx(3.17) and table["SiO2"] == pytest.approx(2.20)
    with pytest.raises(TypeError):
        table["Al2O3"] = 3.95  # type: ignore[index]


def test_the_material_menu_lists_exactly_the_public_table() -> None:
    from xrr_fitter.gui.structure.selection import KNOWN_FORMULAS

    assert KNOWN_FORMULAS == tuple(api.INITIAL_DENSITIES_G_CM3)


def _card(qtbot, layer: api.LayerSpec):
    from xrr_fitter.gui.structure.selection import SelectedLayerCard

    commits: list[api.LayerSpec] = []
    card = SelectedLayerCard(lambda _index, value: commits.append(value))
    qtbot.addWidget(card)
    card.show_component(0, layer)
    return card, commits


def _choose(card, formula: str) -> None:
    menu = card.material_menu_button.menu()
    next(action for action in menu.actions() if action.text() == formula).trigger()


def test_choosing_a_material_renames_a_formula_named_layer_and_brings_its_density(qtbot) -> None:
    card, commits = _card(qtbot, api.LayerSpec("SiO2", SIO2, 40.0, roughness_a=2.0))

    _choose(card, "Si3N4")

    assert commits[-1] == api.LayerSpec("Si3N4", api.MaterialSpec("Si3N4", "Si3N4", 3.17), 40.0, roughness_a=2.0)


def test_choosing_a_material_keeps_a_name_the_reader_gave(qtbot) -> None:
    custom = api.LayerSpec("cap", api.MaterialSpec("a-Si", "Si", 2.28), 40.0, roughness_a=2.0)
    card, commits = _card(qtbot, custom)

    _choose(card, "SiO2")

    layer = commits[-1]
    assert (layer.name, layer.material.name) == ("cap", "a-Si")
    assert layer.material.formula == "SiO2" and layer.material.bulk_density_g_cm3 == pytest.approx(2.20)


def test_typing_an_unlisted_formula_keeps_the_density_already_on_screen(qtbot) -> None:
    card, commits = _card(qtbot, api.LayerSpec("SiO2", SIO2, 40.0, roughness_a=2.0))

    card.formula_editor.setText("Al2O3")
    card.formula_editor.editingFinished.emit()

    layer = commits[-1]
    assert layer.name == "Al2O3" and layer.material == api.MaterialSpec("Al2O3", "Al2O3", 2.2)


def _ok(qtbot, dialog, name: str) -> None:
    buttons = dialog.findChild(QDialogButtonBox, name)
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)


def test_new_layer_density_follows_the_formula_and_is_never_a_made_up_one(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    dialog = LayerDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    density = dialog.findChild(QDoubleSpinBox, "layerDensityInput")
    formula = dialog.findChild(QLineEdit, "layerFormulaInput")
    assert density.value() != pytest.approx(1.0) and density.text() == density.specialValueText()

    formula.setText("Si3N4")
    assert density.value() == pytest.approx(3.17)
    formula.setText("Al2O3")
    assert density.text() == density.specialValueText()

    dialog.findChild(QLineEdit, "layerNameInput").setText("alumina")
    dialog.findChild(QDoubleSpinBox, "layerThicknessInput").setValue(4.0)
    _ok(qtbot, dialog, "layerDialogButtons")
    error = dialog.findChild(QLabel, "layerDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert error.isVisible() and "密度" in error.text() and "Al2O3" in error.text()

    density.setValue(3.95)
    _ok(qtbot, dialog, "layerDialogButtons")
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.layer().material == api.MaterialSpec("Al2O3", "Al2O3", 3.95)


def test_a_density_the_reader_typed_survives_a_later_formula_change(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import LayerDialog

    dialog = LayerDialog()
    qtbot.addWidget(dialog)
    density = dialog.findChild(QDoubleSpinBox, "layerDensityInput")
    density.setValue(2.05)

    dialog.findChild(QLineEdit, "layerFormulaInput").setText("SiO2")

    assert density.value() == pytest.approx(2.05)


def test_backing_density_follows_a_changed_formula(qtbot) -> None:
    from xrr_fitter.gui.structure.dialogs import BackingDialog

    dialog = BackingDialog(SI, 2.0)
    qtbot.addWidget(dialog)
    density = dialog.findChild(QDoubleSpinBox, "backingDensityInput")
    formula = dialog.findChild(QLineEdit, "backingFormulaInput")
    assert density.value() == pytest.approx(2.329)

    formula.setText("SiO2")
    assert density.value() == pytest.approx(2.20)
    formula.setText("GaAs")
    assert density.text() == density.specialValueText()
    _ok(qtbot, dialog, "backingDialogButtons")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "密度" in dialog.findChild(QLabel, "backingDialogError").text()
