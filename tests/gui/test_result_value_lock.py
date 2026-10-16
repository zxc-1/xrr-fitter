"""结果值表锁定行的那枚方框只画在参数名那一格里。

审计：``ResultValueDelegate._paint_lock`` 把方框放在「文字起点 + 全文宽度」处，而参数名长到被
截断时屏上只画了截断后的那一段——方框于是越过本格，压到「结果值」那一列的数字上，或者干脆
画到视口外面去，锁定这件事在屏上就没有了。
"""

from __future__ import annotations

from PySide6.QtWidgets import QTableWidgetItem

from xrr_fitter.gui import theme
from xrr_fitter.gui.results.values import LOCKED_ROLE, ResultValueTable

LONG_NAME = "第三层 · 钼硅混合过渡层的界面粗糙度（按导入预设锁定）"


def _table(qtbot) -> ResultValueTable:
    table = ResultValueTable()
    qtbot.addWidget(table)
    table.setRowCount(1)
    name = QTableWidgetItem(LONG_NAME)
    name.setData(LOCKED_ROLE, True)
    table.setItem(0, 0, name)
    for column, text in ((1, "3.420"), (2, "锁定"), (3, "nm")):
        table.setItem(0, column, QTableWidgetItem(text))
    table.resize(260, 80)
    table.show()
    qtbot.waitExposed(table)
    return table


def _accent_xs(table: ResultValueTable) -> list[float]:
    """Viewport x positions (logical px) of the pixels painted in the lock box fill."""
    accent = theme.token_colour(theme.palette_tokens(table.palette()).accent).rgb()
    image = table.viewport().grab().toImage()
    scale = image.width() / max(1, table.viewport().width())
    row = table.visualRect(table.model().index(0, 0))
    rows = range(int(row.top() * scale), int(row.bottom() * scale))
    return [x / scale for y in rows for x in range(image.width()) if image.pixel(x, y) == accent]


def test_the_lock_box_stays_inside_an_elided_name_cell(qtbot) -> None:
    table = _table(qtbot)
    cell = table.visualRect(table.model().index(0, 0))

    xs = _accent_xs(table)

    assert xs, "the lock box is not drawn anywhere in the visible table"
    assert max(xs) <= cell.right() + 1, "the lock box spills over the 结果值 column"
