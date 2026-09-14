"""A command bar that spends rows instead of width when its column is narrow.

``QHBoxLayout`` has exactly one row and no way to be given a second, so a bar of
six commands reports the sum of their widths as a minimum however narrow the
column it sits in.  The left dock column is 264px wide and its scroll area keeps
the horizontal bar switched off, so the commands past that width were not
scrolled to -- they were clipped with nothing on screen saying a command had been
lost.  The mockup's ``.cmdbar`` is ``display:flex; flex-wrap:wrap``: the same
commands, laid left to right until the row fills and then continued below.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget

from xrr_fitter.gui import theme


class WrappingRow(QLayout):
    """Lay items left to right, continuing on a new row when the width runs out."""

    def __init__(self, parent: QWidget | None = None, *, spacing: int = theme.SPACE_XS) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    # -- QLayout's item protocol --

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt override
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt override
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt override
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    # -- the wrap itself --

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt override
        """Say yes, so the column asks how tall this is before assigning width."""
        return True

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802 - Qt override
        """Ask for width, never for height.

        ``QLayout`` claims both by default, and `QWidgetItem` promotes a host
        widget to expanding in any direction its own layout expands in provided
        the widget's policy can grow -- which ``Minimum`` can.  A bar left with
        the default therefore competes for the column's spare height with the
        list below it and holds the surplus as blank space, which is the opposite
        of what a wrap is for: extra width is what lets a row un-wrap, while
        extra height buys these rows nothing.
        """
        return Qt.Orientation.Horizontal

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override
        return self._reflow(QRect(0, 0, width, 0), place=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt override
        super().setGeometry(rect)
        self._reflow(rect, place=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Ask for a single row; a narrower column then wraps it into several."""
        margins = self.contentsMargins()
        width = sum(item.sizeHint().width() for item in self._items)
        width += self.spacing() * max(len(self._items) - 1, 0)
        height = max((item.sizeHint().height() for item in self._items), default=0)
        return QSize(
            width + margins.left() + margins.right(),
            height + margins.top() + margins.bottom(),
        )

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt override
        """The floor is the widest single command, not the sum of all of them.

        This is the whole point of the class: a row that may wrap only has to
        fit its largest item, so the column sets the width and the bar answers
        with however many rows that takes.
        """
        margins = self.contentsMargins()
        widest = max((item.minimumSize().width() for item in self._items), default=0)
        tallest = max((item.minimumSize().height() for item in self._items), default=0)
        return QSize(
            widest + margins.left() + margins.right(),
            tallest + margins.top() + margins.bottom(),
        )

    def _reflow(self, rect: QRect, *, place: bool) -> int:
        """Walk the items into rows, returning the height they take up.

        One pass serves both callers: ``heightForWidth`` measures without moving
        anything, ``setGeometry`` moves as it measures.  Two passes would be two
        chances for the answer and the arrangement to disagree.
        """
        margins = self.contentsMargins()
        left = rect.x() + margins.left()
        right = rect.right() - margins.right()
        x = left
        y = rect.y() + margins.top()
        row_height = 0
        for item in self._items:
            hint = item.sizeHint()
            if row_height and x + hint.width() - 1 > right:
                x = left
                y += row_height + self.spacing()
                row_height = 0
            if place:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self.spacing()
            row_height = max(row_height, hint.height())
        return y + row_height + margins.bottom() - rect.y()


def wrapping_row(
    parent: QWidget | None = None,
    *,
    name: str = "",
    spacing: int = theme.SPACE_XS,
) -> tuple[QWidget, WrappingRow]:
    """Host a wrapping row on a widget of its own.

    The items need a widget rather than a bare sub-layout, because the height a
    wrap costs is only negotiable between a widget and its column:
    ``heightForWidth`` has to be declared on the size policy for the enclosing
    layout to consult it at all.  Nested inside a ``QVBoxLayout`` the same row
    wraps on screen but never gets asked for the extra height, so the last row
    lands outside the widget it belongs to.
    """
    host = QWidget(parent)
    if name:
        host.setObjectName(name)
    layout = WrappingRow(host, spacing=spacing)
    policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    policy.setHeightForWidth(True)
    host.setSizePolicy(policy)
    return host, layout


def command_bar(parent: QWidget | None = None, *, name: str = "") -> tuple[QWidget, WrappingRow]:
    """Build the mockup's ``.cmdbar``: a named widget over a wrapping row."""
    bar, layout = wrapping_row(parent, name=name)
    bar.setProperty("commandBar", True)
    return bar, layout
