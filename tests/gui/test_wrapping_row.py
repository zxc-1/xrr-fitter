"""A wrapping command bar must not also claim the height its column has spare."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _bar(qtbot, count: int = 6):
    from xrr_fitter.gui.wrapping import command_bar

    bar, row = command_bar(name="probeBar")
    for index in range(count):
        row.addWidget(QPushButton(f"命令 {index}"))
    qtbot.addWidget(bar)
    return bar, row


def test_the_row_asks_for_width_and_never_for_height(qtbot) -> None:
    """横向多给它就少换行，纵向多给它只剩空白。

    ``QLayout`` 默认横竖都声明要扩张，而 ``QWidgetItem`` 会把「内层布局要纵向扩张」
    的宿主 widget 一并提升成纵向可扩张——命令栏于是和真正要长高的那个控件分走剩余
    高度，自己撑出一片空白。
    """
    _, row = _bar(qtbot)

    directions = row.expandingDirections()

    assert bool(directions & Qt.Orientation.Horizontal) is True
    assert bool(directions & Qt.Orientation.Vertical) is False


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
        api.InstrumentSpec(instrument_id="wrap-gui"),
    )
    panel = StructurePanel(ProjectDocument(project))
    panel.set_structure(api.StructureSpec(AIR, (api.LayerSpec("cap", SIO2, 34.2, roughness_a=5.1),), SI))
    panel.resize(700, 620)
    qtbot.addWidget(panel)
    panel.show()
    return panel


def test_a_command_bar_in_a_tall_column_stays_as_tall_as_its_rows(qtbot, tmp_path) -> None:
    """结构画布的行头不该空出一大片。

    行头的高度由它排的那一行决定，多出来的高度归下面那张列表——它才是内容会长的那个。
    核的是排好版后的实际几何，因为策略正确而几何仍然错的话，屏幕上依旧是空白。
    """
    panel = _panel(qtbot, tmp_path)
    row = panel.editor.canvas_row

    wanted = row.sizeHint().height()

    assert row.height() == wanted, f"{row.objectName()} is {row.height()}px for {wanted}px of content"


def test_the_layer_list_takes_the_column_height_the_bars_do_not(qtbot, tmp_path) -> None:
    """空白归谁：列表是内容会变长的那个，剩余高度应当落在它身上。"""
    panel = _panel(qtbot, tmp_path)
    editor = panel.editor

    assert editor.tree.height() > editor.canvas_row.height()
