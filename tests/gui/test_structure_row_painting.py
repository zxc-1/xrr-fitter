"""Regression for duplicate selected labels under structure row widgets."""

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.structure.editor import StructureEditor


def _editor(qtbot, palette_name):
    editor = StructureEditor(lambda _value: None, lambda: None, lambda: None)
    qtbot.addWidget(editor)
    palette = theme.dark_palette() if palette_name == "dark" else theme.light_palette()
    editor.setPalette(palette)
    editor.setStyleSheet(theme.build_stylesheet(palette))
    editor.load(
        api.StructureSpec(
            api.MaterialSpec("Air", None, None, 0.0j),
            (api.LayerSpec("SiO2 film", api.MaterialSpec("SiO2", "SiO2", 2.2), 173.0),),
            api.MaterialSpec("Si", "Si", 2.329),
        )
    )
    editor.ensurePolished()
    return editor


def _paint(tree, option, index, *, textless):
    image = QImage(260, 48, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        if textless:
            clean = QStyleOptionViewItem(option)
            tree.itemDelegate().initStyleOption(clean, index)
            clean.text = ""
            tree.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, clean, painter, tree)
        else:
            tree.itemDelegate().paint(painter, option, index)
    finally:
        painter.end()
    return image


@pytest.mark.parametrize("palette_name", ["light", "dark"])
@pytest.mark.parametrize("selected", [False, True])
def test_native_delegate_does_not_overprint_the_row_name(qtbot, tmp_path, palette_name, selected):
    editor = _editor(qtbot, palette_name)
    tree = editor.tree
    item = tree.topLevelItem(1)
    index = tree.indexFromItem(item)
    option = QStyleOptionViewItem()
    option.initFrom(tree)
    option.widget = tree
    option.rect = QRect(0, 0, 260, 48)
    option.state |= QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Active
    if selected:
        option.state |= QStyle.StateFlag.State_Selected
    actual = _paint(tree, option, index, textless=False)
    expected = _paint(tree, option, index, textless=True)
    actual.save(str(tmp_path / "actual.png"))
    expected.save(str(tmp_path / "expected.png"))
    assert item.text(0) == "SiO2 film"
    assert actual == expected, f"raw model label overprinted: {palette_name=} {selected=}"
