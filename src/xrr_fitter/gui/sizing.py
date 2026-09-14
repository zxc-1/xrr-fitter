"""Content-driven height for the scrolling views that share a column.

A default ``QTreeWidget`` reports Qt's fixed 192 px height whatever it holds, so
two trees stacked in one dock column hand the window identical requests and are
given identical shares: a single dataset claims as much room as a twelve-layer
stack.  The tree here asks for the rows it actually shows, floored so a short
tree still reads as a tree and capped so a long one yields to its own scrollbar.
The same defect was fixed locally in ``results/candidates.py`` and
``results/uncertainty.py``; the two left-column trees share this class instead
because their sizing rule is identical.  ``QTableWidget`` reports the same fixed
default, so the batch import preview takes the table variant below rather than a
second copy of the rule.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

VISIBLE_ROW_FLOOR = 3
VISIBLE_ROW_CEILING = 12

# A hint that shrinks to nothing has stopped being a hint, so elision stops here
# and the tooltip carries the rest.
MIN_ELIDED_WIDTH = 48

# Six digits still read as a rounded number while the stepper is dragged; a
# narrower spin box has stopped being a number field.
MIN_SPIN_CHARS = 6


def announce_hint_change(widget: QWidget) -> None:
    """Carry a size-hint change all the way up, not one level per event-loop pass.

    ``QWidget.updateGeometry`` invalidates the layout directly above the widget and
    posts a layout request; the widget owning that layout re-asks *its* own parent
    only when the request is delivered.  So an ancestor's ``sizeHint`` lags one pass
    per level: measured on the layer stack, the tree answered 210 px the instant its
    rows arrived while ``structureEditor`` still answered 194 and ``structurePanel``
    194 -- the editor caught up on the next pass, the panel on the one after.

    Anything that reads a height in the same call stack as the change therefore reads
    the height of the empty widget.  ``_refit_on_structure_change`` does exactly that:
    it re-deals the canvas splitter from ``structure_panel.sizeHint()`` the moment a
    structure loads, and was handing the layer stack the 194 px it needed while empty.
    Walking up closes the gap in one pass.
    """
    node: QWidget | None = widget
    while node is not None:
        node.updateGeometry()
        node = node.parentWidget()


class ElidingLabel(QLabel):
    """A label that shortens its text to the width it is given.

    ``QLabel`` without word wrap reports its whole string as a minimum width, so
    a caption in a fixed-width column sets the column's floor rather than living
    inside it.  Wrapping is the other way out, but it buys the width back with a
    second text row in every card that has a caption.  This elides at paint time
    instead: ``text()`` keeps the full string for tooltips and tests while the
    widget asks the column for almost nothing.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        # Deliberately not QSizePolicy.Policy.Ignored: that policy discards the
        # minimum size hint along with the preferred one, and a caption sharing a
        # row with a stretch then collapses to nothing -- the whole hint went
        # missing from the stack card rather than merely shortening.  Preferred
        # keeps sizeHint as the ask, so the text still shows in full wherever the
        # row has room, and leaves minimumSizeHint below as the floor it shrinks to.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Ask for a readable stub, not the string."""
        return QSize(min(MIN_ELIDED_WIDTH, super().minimumSizeHint().width()), super().minimumSizeHint().height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Draw the text elided to the current width, in the styled colour.

        The pen comes from the palette so the stylesheet's ``mutedText`` colour
        still applies; painting with the default pen would give every elided
        caption the body-text colour back.
        """
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        elided = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width())
        painter.drawText(self.contentsRect(), int(self.alignment()), elided)


class CompactDoubleSpinBox(QDoubleSpinBox):
    """A spin box whose widest conceivable value is not the layout's floor.

    ``QAbstractSpinBox`` sizes both of its hints to the widest string the range
    could ever produce.  A 0-10⁶ field at eight decimals is
    ``1000000.00000000`` -- 141 px -- while what it shows is ``1.5406``, and the
    import dialog's eight such boxes then refuse to be narrower than 1035 px.
    ``sizeHint`` is left alone, so a roomy layout still lays them out at their
    natural width; only the minimum is lowered, and the size policy gains the
    shrink flag a plain spin box lacks, so a cramped layout squeezes the boxes
    instead of pushing the dialog past the edge of the screen.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # QDoubleSpinBox ships QSizePolicy.Minimum: grow-only.  Without the
        # shrink flag a layout takes max(sizeHint, minimumSizeHint) as the
        # minimum and the override below never gets read.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Ask for a readable number, not the widest one in range."""
        hint = super().minimumSizeHint()
        metrics = self.fontMetrics()
        widest = metrics.horizontalAdvance(self.textFromValue(self.maximum()))
        narrow = metrics.horizontalAdvance("0" * MIN_SPIN_CHARS)
        return QSize(max(hint.width() - max(widest - narrow, 0), narrow), hint.height())


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
        # Qt caches a widget's hint in the layout above it and only drops that copy
        # when the widget says its hint moved.  Rows arriving change what sizeHint
        # answers here but nothing in Qt knows that, so an ancestor keeps handing out
        # the height of the empty tree: the layer stack still asked for 194px right
        # after a two-layer structure loaded, and the splitter pane it lives in was
        # dealt that.  ``announce_hint_change`` is the announcement.
        self.itemExpanded.connect(self._hint_moved)
        self.itemCollapsed.connect(self._hint_moved)
        model = self.model()
        model.rowsInserted.connect(self._hint_moved)
        model.rowsRemoved.connect(self._hint_moved)
        model.modelReset.connect(self._hint_moved)

    def _hint_moved(self, *_args: object) -> None:
        """Tell the layouts above that the row count they sized us by has changed."""
        announce_hint_change(self)

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


class ContentSizedTable(QTableWidget):
    """A table whose size hint follows the rows it currently holds.

    The batch import preview is as long as the reader's selection, so a fixed
    height either hangs a blank strip under a short batch or hides the tail of a
    long one.  Both ends are bounded: the floor keeps a single-file preview
    reading as a table, the ceiling hands a folder import to its own scrollbar.
    """

    def __init__(
        self,
        rows: int = 0,
        columns: int = 0,
        parent: QWidget | None = None,
        *,
        row_floor: int = VISIBLE_ROW_FLOOR,
        row_ceiling: int = VISIBLE_ROW_CEILING,
    ) -> None:
        super().__init__(rows, columns, parent)
        self._row_floor = row_floor
        self._row_ceiling = row_ceiling

    def sizeHint(self) -> QSize:
        """Ask for the rows on screen, between the floor and the ceiling."""
        rows = min(max(self.rowCount(), self._row_floor), self._row_ceiling)
        return QSize(super().sizeHint().width(), self._height_for(rows))

    def minimumSizeHint(self) -> QSize:
        """Defend the floor, because a hint is what a crowded layout gives away first.

        ``sizeHint`` alone left the table at QTableWidget's own minimum -- a header
        and about two rows -- whenever the widgets under it were rigid enough to
        want the difference.  A preview cut to two rows no longer previews the
        batch, so the floor moves where a layout has to honour it.
        """
        return QSize(super().minimumSizeHint().width(), self._height_for(self._row_floor))

    def _height_for(self, rows: int) -> int:
        """The pixel height of ``rows`` data rows plus the chrome around them."""
        row_height = self.rowHeight(0) if self.rowCount() else 0
        if row_height <= 0:
            row_height = self.fontMetrics().lineSpacing()
        head = self.horizontalHeader()
        header = 0 if head.isHidden() else head.sizeHint().height()
        return rows * row_height + header + 2 * self.frameWidth()


class ContentSizedScroll(QScrollArea):
    """A scroll area that asks for the height of what it holds, and no more.

    ``QScrollArea`` is Expanding and its own hint is bounded at 24 lines, so a
    splitter hands it whatever share the stretch factors compute and stretches
    the scrolled panel with it.  设计稿的 ``.nav`` 是一列 flex：数据集列表和管线各取
    自然高度，机架多出来的高度靠 ``.ds-summary{margin-top:auto}`` 一次顶到页脚之上，
    不垫在列表和「分析管线」之间——那两块设计稿画的是紧挨着的。

    The vertical policy keeps ``ShrinkFlag``, which makes the content height a
    cap rather than a fixed size: a list longer than the rail still yields to its
    own scrollbar instead of pushing the pipeline out of the window.  Asking the
    scrolled widget on every pass also sidesteps the cached hint behind
    ``QScrollArea``'s own, which lagged a row-height change by one layout.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def sizeHint(self) -> QSize:
        """Report the scrolled widget's own height request, unbounded."""
        inner = self.widget()
        hint = super().sizeHint()
        return hint if inner is None else QSize(hint.width(), inner.sizeHint().height())


class AnchorSizedScroll(QScrollArea):
    """A scroll area that asks for one designated child's height, not all of it.

    ``QScrollArea`` reports the whole scrolled content (bounded at 24 lines), and a
    ``QSplitter`` hands out its first round of height by ``sizeHint``.  画布列上半段
    装的是层堆叠那张卡加「参数总览」——后者是滚出来看的，把它算进这一段的高度请求，
    splitter 就从下半段扣三百来像素给一张读者此刻没在看的表，而下半段画的是同一张设计稿
    里的 SLD 深度剖面。

    Anchoring the request to the layer stack keeps the two apart: 这一段的高度预算归
    ``anchor``，别的内容靠滚动去够。``QScrollArea``'s own hint stays the ceiling, so a
    twelve-layer stack yields to its own scrollbar exactly as before.
    """

    def __init__(self, anchor: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._anchor = anchor

    def sizeHint(self) -> QSize:
        """The anchor's height request, capped by the one Qt would have given.

        ``sizeHint`` rather than ``viewportSizeHint``: ``QScrollArea`` overrides the
        former and reads the scrolled widget's size directly, so the viewport hook
        (and ``sizeAdjustPolicy`` with it) is never consulted on this class.
        """
        hint = super().sizeHint()
        return QSize(hint.width(), min(hint.height(), self._anchor.sizeHint().height()))


class ContentSizedSplitter(QSplitter):
    """A splitter that lays its children out at the heights they ask for.

    设计稿的左栏不是一个可拖的比例：``.nav`` 是一列 flex，数据集列表和管线各取自然高度。
    分栏尺寸照旧存进项目文件，可一个旧文件里的 ``(360, 360)`` 跟按内容算出来的上限是矛盾
    的，而 ``QSplitter`` 把这种超出上限的请求当 hint 直接分下去（qGeomCalc 只保证不破下
    限），于是管线拿到 360——比它需要的多 14px——列表反过来少 14px，第四个数据集被裁在视
    口外。这里把请求换成内容尺寸：谁来设都一样，两块永远是它们自己的高度。
    """

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self._imposed: list[int] = []

    def setSizes(self, sizes: list[int]) -> None:
        """Lay out by content, whatever ratio the caller asked for."""
        content = self._content_sizes()
        self._imposed = content
        super().setSizes(content or list(sizes))

    def event(self, event: QEvent) -> bool:
        """Re-impose the content heights once a child's request changes.

        A hint changes when rows arrive or a stylesheet lands, and the child posts
        a layout request for it.  Without this the splitter keeps distributing the
        sizes it was last given, which are the ones the content just outgrew.
        """
        handled = super().event(event)
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Show):
            content = self._content_sizes()
            if content and content != self._imposed:
                self._imposed = content
                super().setSizes(content)
        return handled

    def _content_sizes(self) -> list[int]:
        """Each child's own height request, floored by whatever minimum it pins."""
        children = [self.widget(index) for index in range(self.count())]
        return [max(child.minimumHeight(), child.sizeHint().height()) for child in children if child is not None]
