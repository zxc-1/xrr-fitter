"""设计稿 ``.stack .lyr`` 的一行：拖拽柄、色块、名字与副行、右端读数。

层堆叠此前是一张六列表格（名称 / 材料 / 密度 / 厚度 / 粗糙度 / 重复）。六列在 264px 的
画布列里每列摊到四十来个像素，数字全被截断成 ``2.19…``；而设计稿这张卡从来就不是表格，
它是四行「一层一行」的列表：颜色、名字、这一层是什么、右端一个读数。这个模块只负责画那
一行，选中与拖排仍归 ``ReorderableTree``。
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from xrr_fitter.gui import theme

# 设计稿 ``.lyr{padding:9px 12px;gap:10px}``。行高由内容加这两个边距决定，不写死：
# 副行是可选的，只有一行字的行本来就该矮一截。
ROW_MARGIN_H = 12
ROW_MARGIN_V = 9
ROW_GAP = 10

# 设计稿 ``.lyr .drag`` 的 ⋮⋮。它不是按钮：拖排由树的 drag-and-drop 负责，这里只是告诉
# 读者「这一行可以拖」——所以它和整行一起对鼠标透明，点它等于点行。
DRAG_GLYPH = "⋮⋮"

SWATCH_OBJECT = "structureLayerSwatch"
NAME_OBJECT = "structureLayerName"
DETAIL_OBJECT = "structureLayerDetail"
READING_OBJECT = "structureLayerReading"
DRAG_OBJECT = "structureLayerDrag"
ROW_OBJECT = "structureLayerRow"


def _transparent(widget: QWidget) -> None:
    """让整行连同子控件都不参与命中测试。

    行是 ``setItemWidget`` 放进树里的，落在它上面的点击如果被它自己吃掉，树就收不到
    ——选中、拖动排序、右键菜单会一起失灵。命中测试关掉之后事件直接落到 viewport，
    行只剩「画」这一个职责。
    """
    widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    for child in widget.findChildren(QWidget):
        child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


class _ElidedLabel(QLabel):
    """设计稿 ``.nm{flex:1;min-width:0}``：名字这一栏可以被压窄。

    ``QLabel`` 的最小宽度就是整句话的宽度，压不下去——面板窄到 300px 时，行里放不下的
    那一截会把右端的读数（「3 参数」）直接顶出可视区，而读数是这一行唯一的定量信息。
    把最小宽度放开、画的时候按当前宽度截断，让让步发生在最长也最可省的那一栏上。
    """

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 的钩子名
        metrics = self.fontMetrics()
        text = metrics.elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())
        painter = QPainter(self)
        # 走 ``drawItemText`` 而不是自己 ``setPen``：QSS 的 ``color:`` 是通过样式落到调色板
        # 上的，绕开样式画就等于把副行那一档灰漏掉，两级文字会变成同一个颜色。
        self.style().drawItemText(
            painter,
            self.rect(),
            int(self.alignment()),
            self.palette(),
            self.isEnabled(),
            text,
            self.foregroundRole(),
        )
        painter.end()


class LayerRow(QWidget):
    """One ``.lyr``: drag handle, swatch, name over detail, and a reading."""

    def __init__(
        self,
        colour: str | None,
        name: str,
        detail: str,
        reading: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(ROW_OBJECT)
        # QSS 要给 QWidget 子类画背景，得先声明它有样式化背景，否则 ``semi`` 那一档
        # 的底色规则会被静默丢掉。
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(ROW_MARGIN_H, ROW_MARGIN_V, ROW_MARGIN_H, ROW_MARGIN_V)
        layout.setSpacing(ROW_GAP)
        layout.addWidget(self._drag_handle())
        layout.addWidget(self._swatch(colour), 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(self._caption(name, detail), 1)
        self._reading_label = self._reading(reading)
        layout.addWidget(self._reading_label)
        _transparent(self)

    # ------------------------------------------------------------------ 构件

    def _drag_handle(self) -> QLabel:
        label = QLabel(DRAG_GLYPH, self)
        label.setObjectName(DRAG_OBJECT)
        # 读屏器念「⋮⋮」只会得到两个不成词的标点；这一行能拖这件事由树的
        # accessibleDescription 说，柄本身是纯装饰。
        label.setAccessibleName("")
        return label

    def _swatch(self, colour: str | None) -> QLabel:
        """色块。周期块的子层不占一个 hue，所以 `colour` 允许为空——那时留一块等宽的
        空位，行与行的文字才仍然对齐在同一条竖线上。"""
        label = QLabel(self)
        label.setObjectName(SWATCH_OBJECT)
        label.setAccessibleName("")
        if colour is not None:
            label.setPixmap(theme.stack_swatch(colour))
        label.setFixedSize(theme.STACK_SWATCH_PX, theme.STACK_SWATCH_PX)
        return label

    def _caption(self, name: str, detail: str) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self._name_label = _ElidedLabel(name, self)
        self._name_label.setObjectName(NAME_OBJECT)
        column.addWidget(self._name_label)
        self._detail_label = _ElidedLabel(detail, self) if detail else None
        if self._detail_label is not None:
            self._detail_label.setObjectName(DETAIL_OBJECT)
            column.addWidget(self._detail_label)
        return column

    def _reading(self, reading: str) -> QLabel:
        label = QLabel(reading, self)
        label.setObjectName(READING_OBJECT)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    # ------------------------------------------------------------------ 读数

    def name_text(self) -> str:
        return self._name_label.text()

    def detail_text(self) -> str:
        return "" if self._detail_label is None else self._detail_label.text()

    def reading_text(self) -> str:
        return self._reading_label.text()

    # ------------------------------------------------------------------ 状态

    def set_semi_infinite(self, semi: bool) -> None:
        """设计稿 ``.lyr.semi``：半无限介质那两行压一档底色，和可编辑的层分开。"""
        self.setProperty("semiInfinite", bool(semi))

    def set_last(self, last: bool) -> None:
        """设计稿 ``.lyr:last-child{border-bottom:none}``：末行不画分隔线。

        画了的话它会和 ``.stack`` 自己的下边框叠成两道，盒子底部凭空粗一档。
        """
        self.setProperty("lastRow", bool(last))

    def set_selected(self, selected: bool) -> None:
        """设计稿 ``.lyr.sel``：左缘 3px 强调色。

        树自己的选中高亮画在行控件底下，被半无限那两行的底色盖住，所以选中态得由行
        自己再戴一次。
        """
        if self.property("rowSelected") == bool(selected):
            return
        self.setProperty("rowSelected", bool(selected))
        theme.repolish(self)
