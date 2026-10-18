"""Input attempts for read-only tables and real scrollbars."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit


def table_snapshot(table):
    return tuple(
        tuple(table.item(row, col).text() if table.item(row, col) else None for col in range(table.columnCount()))
        for row in range(table.rowCount())
    )


def attempt_table_edit(driver, table):
    assert table.rowCount() > 0
    before = table_snapshot(table)
    item = table.item(0, 0)
    table.scrollToItem(item)
    position = table.visualItemRect(item).center()
    driver.reach(table.viewport(), position)
    driver.qtbot.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    driver.qtbot.keyClicks(table, "not an editable value")
    assert table_snapshot(table) == before
    assert not any(editor.isVisible() for editor in table.findChildren(QLineEdit))
    driver.record(
        f"readOnly:{table.objectName()}",
        "double-click a rendered cell; attempt keyboard replacement",
        "all rendered cell values are unchanged and no editor opens",
    )


def scroll_roundtrip(driver, area, identity):
    bar = area.verticalScrollBar()
    assert bar.isVisible() and bar.maximum() > 0
    driver.qtbot.keyClick(bar, Qt.Key.Key_End)
    assert bar.value() == bar.maximum()
    driver.qtbot.keyClick(bar, Qt.Key.Key_Home)
    assert bar.value() == bar.minimum()
    driver.record(
        identity,
        "End then Home on visible scrollbar",
        "last content and first content can both be reached without model changes",
    )
