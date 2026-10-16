"""The transient series key row on the reflectivity plot cards.

卡片上那行常驻色标（``PLOT_CARD_LEGENDS``）只认每一次投影都会画的标记；叠加对比的曲线、搜索中的模型
是读者开了才有、换了数据集就换的，所以另起一行，有东西时才露。曲线的颜色与线型也从这里取：图上那一笔
和色标里那一格出自同一处，两边才不会各自轮转。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from xrr_fitter.gui import theme

# 有这行动态色标的两张卡：叠加对比只画在对数页，搜索中的模型两页都画。
SERIES_KEY_CARDS = ("log", "raw")

PEN_STYLES = {
    "solid": Qt.PenStyle.SolidLine,
    "dash": Qt.PenStyle.DashLine,
    "dot": Qt.PenStyle.DotLine,
    "dashdot": Qt.PenStyle.DashDotLine,
}

# 叠加曲线不用这几种颜色：观测点、拟合模型、搜索中的模型和拟合窗口已经占了它们，再用一次，读者就分不清
# 那抹橙是窗口还是第二条曲线。剩下的颜色不够分，所以颜色轮完一圈换线型——灰度打印时线型照样分得开。
RESERVED_COLOURS = (theme.DATA_OBSERVED, theme.DATA_CANDIDATE, theme.DATA_PREVIEW, theme.DATA_RANGE)
OVERLAY_DASHES = ("dash", "dot", "dashdot")


def overlay_style(index: int) -> tuple[str, str]:
    """第 ``index`` 条叠加曲线的 ``(颜色, 线型)``。"""
    colours = [colour for colour in theme.DATA_SEQUENCE if colour not in RESERVED_COLOURS]
    cycle, position = divmod(index, len(colours))
    return colours[position], OVERLAY_DASHES[cycle % len(OVERLAY_DASHES)]


def _line_swatch(colour: str, dash: str) -> QPixmap:
    """一格线段色标：同色的两条曲线靠线型分开，所以色标得把线型也画出来。"""
    pixmap = QPixmap(theme.LEGEND_SWATCH_W, theme.LEGEND_SWATCH_H)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(colour))
    pen.setWidthF(2.0)
    pen.setStyle(PEN_STYLES[dash])
    painter.setPen(pen)
    middle = theme.LEGEND_SWATCH_H // 2
    painter.drawLine(0, middle, theme.LEGEND_SWATCH_W, middle)
    painter.end()
    return pixmap


def build_series_key(parent: QWidget, name: str) -> QWidget:
    """一行空着、藏着的动态色标，由 ``show_series_key`` 来填。"""
    row = QWidget(parent)
    row.setObjectName(name)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_MD)
    layout.addStretch(1)
    row.hide()
    return row


def show_series_key(row: QWidget, entries: tuple[tuple[str, str, str], ...]) -> None:
    """把 ``(caption, colour, dash)`` 填进这一行；没有条目就整行藏起来，不占卡片的高度。"""
    layout = row.layout()
    # 旧的几格先摘下来再删：``deleteLater`` 要等事件循环，这之前它们还挂在行里、还读得到。
    while layout.count() > 1:
        widget = layout.takeAt(0).widget()
        widget.setParent(None)
        widget.deleteLater()
    for position, (caption, colour, dash) in enumerate(entries):
        swatch = QLabel(row)
        swatch.setPixmap(_line_swatch(colour, dash))
        text = QLabel(caption, row)
        text.setProperty("mutedText", True)
        layout.insertWidget(2 * position, swatch)
        layout.insertWidget(2 * position + 1, text)
    row.setVisible(bool(entries))
