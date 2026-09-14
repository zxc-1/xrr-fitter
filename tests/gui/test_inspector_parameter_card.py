"""设计稿帧③ 右栏第二段的抬头写的是「参数化 · 厚度 d」，不是「参数化」。

那一段下面摆的是自由 / 固定 / 仅范围三档和两枚徽标（先验、共享），说的都是某一个量一个
人的事。抬头不点名，读者就得自己回到表里去看哪一行是选中的——而那张表在同一张卡里往下
滚，选中行常常已经不在眼前。

抬头此前是「参数 · 边界」：这个名字既没有出现在设计稿里，也把这张卡说成了「一堆边界」，
而它实际上一次只作用在一行上。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QWidget

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _window(qtbot, tmp_path, *, layers: int = 1):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        AIR,
        tuple(api.LayerSpec(f"film{index}", SIO2, 40.0, roughness_a=3.0) for index in range(layers)),
        SI,
    )
    project = api.select_active_dataset(api.set_structure(project, "curve", structure), "curve")
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def _heading(window) -> QLabel:
    heading = window.findChild(QLabel, "inspectorParametersTitle")
    assert heading is not None
    return heading


def _row_of(table, quantity: str) -> int:
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and item.text().startswith(quantity):
            return row
    raise AssertionError(f"参数表里没有 {quantity} 这一行")


def _row_of_name(table, name: str) -> int:
    """按参数名找行——两层结构里两行的行名都写作「厚度 d」，靠文本分不出是哪一层的。"""
    from PySide6.QtCore import Qt

    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
            return row
    raise AssertionError(f"参数表里没有 {name} 这一行")


def test_the_parameters_card_is_titled_the_way_the_design_titles_it(qtbot, tmp_path) -> None:
    """设计稿这一段叫「参数化」。

    「参数 · 边界」是自造的名字，而且照它去找边界会发现这张卡还管着先验和约束。
    """
    window = _window(qtbot, tmp_path)

    assert _heading(window).text().startswith("参数化")


def test_the_card_names_the_row_the_reader_is_standing_on(qtbot, tmp_path) -> None:
    """点表里的「厚度 d」，抬头就变成「参数化 · 厚度 d」。

    抬头里的量名和行名同源（行名格自己的文本），所以两处不可能各说各的；单位留给行，
    抬头只要那个量。
    """
    window = _window(qtbot, tmp_path)
    table = window.parameters_panel.parameter_table

    table.setCurrentCell(_row_of(table, "厚度 d"), 0)

    heading = _heading(window)
    assert heading.text() == "参数化 · 厚度 d"
    assert window.findChild(QWidget, "inspectorParameters").accessibleName() == "参数化 · 厚度 d"


def test_reloading_the_table_keeps_the_quantity_that_is_still_there(qtbot, tmp_path) -> None:
    """结构变了但读者站的那个量还在，抬头就还挂着它。

    改一格数、锁一个量都会重填整张表。若每次重填都把抬头清回「参数化」，读者改完一个数
    想接着改它的档位，得先回表里把同一行再点一次——而那三档说的正好就是这一行的事。

    认的是量本身（参数名），不是行号：重填的原因往往正是行的构成变了，第 n 行装的会是
    另一份声明。表重填时不发 ``currentCellChanged``（``load`` 整段套在 ``QSignalBlocker``
    里），所以这件事得由 ``load`` 自己交代。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(AIR, (api.LayerSpec("film0", SIO2, 40.0, roughness_a=3.0),), SI)
    project = api.select_active_dataset(api.set_structure(project, "curve", structure), "curve")
    document = ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    table = window.parameters_panel.parameter_table
    table.setCurrentCell(_row_of(table, "厚度 d"), 0)
    assert _heading(window).text() == "参数化 · 厚度 d"

    document.replace_project(
        api.set_structure(
            document.project,
            "curve",
            api.StructureSpec(AIR, (api.LayerSpec("film0", SIO2, 40.0),), SI),
        )
    )
    qtbot.wait(1)

    assert _heading(window).text() == "参数化 · 厚度 d"


def test_reloading_the_table_drops_the_quantity_that_is_gone(qtbot, tmp_path) -> None:
    """读者站的那个量随结构一起没了，抬头就退回「参数化」。

    抬头若还挂着旧名字，下面那三档就成了对着一个不存在的量在设置；而把光标顺手落到顶上
    那一行更糟——它读起来和「读者自己点了这一行」一模一样。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        AIR,
        tuple(api.LayerSpec(f"film{index}", SIO2, 40.0, roughness_a=3.0) for index in range(2)),
        SI,
    )
    project = api.select_active_dataset(api.set_structure(project, "curve", structure), "curve")
    document = ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    table = window.parameters_panel.parameter_table
    table.setCurrentCell(_row_of_name(table, "component.1.thickness_a"), 0)
    assert _heading(window).text() == "参数化 · 厚度 d"

    document.replace_project(
        api.set_structure(
            document.project,
            "curve",
            api.StructureSpec(AIR, (api.LayerSpec("film0", SIO2, 40.0, roughness_a=3.0),), SI),
        )
    )
    qtbot.wait(1)

    assert _heading(window).text() == "参数化"


def test_a_group_caption_row_is_not_a_quantity_and_does_not_rename_the_card(qtbot, tmp_path) -> None:
    """标题行（「film0」这种）说的是归属，不是可参数化的量，抬头就退回「参数化」。

    分组标题行在表里和参数行长得一样高，方向键一路往上就会停在它上面；那一刻抬头若还
    挂着上一行的量名，下面那三档就成了对着一个已经不在选中的量在设置。
    """
    window = _window(qtbot, tmp_path, layers=2)
    table = window.parameters_panel.parameter_table
    table.setCurrentCell(_row_of(table, "厚度 d"), 0)
    assert _heading(window).text() == "参数化 · 厚度 d"

    table.setCurrentCell(_row_of(table, "film0"), 0)

    assert _heading(window).text() == "参数化"
