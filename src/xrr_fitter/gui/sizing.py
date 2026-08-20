"""Content-driven height for the scrolling trees that share a dock column.

A default ``QTreeWidget`` reports Qt's fixed 192 px height whatever it holds, so
two trees stacked in one dock column hand the window identical requests and are
given identical shares: a single dataset claims as much room as a twelve-layer
stack.  The tree here asks for the rows it actually shows, floored so a short
tree still reads as a tree and capped so a long one yields to its own scrollbar.
The same defect was fixed locally in ``results/candidates.py`` and
``results/uncertainty.py``; the two left-column trees share this class instead
because their sizing rule is identical.
"""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

VISIBLE_ROW_FLOOR = 3
VISIBLE_ROW_CEILING = 12


def _expanded_descendants(item: QTreeWidgetItem) -> int:
    """Count the rows a collapsed item hides and an expanded one shows."""
    if not item.isExpanded():
        return 0
    return sum(1 + _expanded_descendants(item.child(index)) for index in range(item.childCount()))


class ContentSizedTree(QTreeWidget):
    """A tree whose size hint follows the rows currently on screen."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        row_floor: int = VISIBLE_ROW_FLOOR,
        row_ceiling: int = VISIBLE_ROW_CEILING,
    ) -> None:
        super().__init__(parent)
        self._row_floor = row_floor
        self._row_ceiling = row_ceiling

    def sizeHint(self) -> QSize:
        """Ask for the visible rows, between the floor and the ceiling."""
        visible = sum(1 + _expanded_descendants(self.topLevelItem(index)) for index in range(self.topLevelItemCount()))
        rows = min(max(visible, self._row_floor), self._row_ceiling)
        row_height = self.sizeHintForRow(0)
        if row_height <= 0:
            row_height = self.fontMetrics().lineSpacing()
        header = 0 if self.isHeaderHidden() else self.header().sizeHint().height()
        return QSize(
            super().sizeHint().width(),
            rows * row_height + header + 2 * self.frameWidth(),
        )
