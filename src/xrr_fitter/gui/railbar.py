"""设计稿的 ``.railbar``：把一个数字被允许落在哪一段里，画在它旁边。

输入框只报「现在是 48.7」。拟合器动的正是这个数，而它能走多远写在声明里——读者要判断
「快贴边了吗」，就得离开当前这张卡去参数表里翻那一行的上下界。这根条把那段话搬到数字
旁边：``40 ≤ 48.7 ≤ 60``，填充长度画出它落在哪儿。

条自己不知道界限从哪儿来，也不换算单位；喂进来的三个数已经是要显示的那三个数。这样
同一根条可以贴在任何一个量旁边，而「界限是谁说的」这件事只由喂它的人负责。

颜色全部从 palette 现取：填充沿用参数表值条的 ``Highlight`` 加低透明度，两者画的本来
就是同一件事（值在区间里的位置），同色才不会被读成两种含义。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import QSizePolicy, QWidget

from xrr_fitter.gui import theme

# 设计稿 ``.railbar{height:16px;min-width:70px;border-radius:3px}``。高度写死：条和 96px
# 的输入框并排在同一行，高度一旦跟着字体走，这一行就比相邻两行高出一截。
HEIGHT_PX = 16
MINIMUM_WIDTH_PX = 70
RADIUS_PX = 3

# ``.railbar i{border-right:2px solid var(--accent)}``：填充的右缘是一条实线，它才是
# 「当前值在这里」的那个刻度——只靠一片半透明填充的边界，在浅色主题下几乎看不见。
EDGE_WIDTH_PX = 2

# ``.railbar span{padding:0 5px}``
TEXT_PADDING_PX = 5

# 参数表的 ``ValuePositionDelegate.BAR_ALPHA`` 是同一个数：两处画的是同一件事，色差会
# 被读成含义差。
FILL_ALPHA = 48

# ``--panel-2`` 与 ``--border`` 在两套主题里都是「正文色压到很低的不透明度」——浅色下
# 是淡黑，深色下是淡白。照角色取色而不是写死十六进制，条才会跟着系统外观翻转。
TRACK_ALPHA = 13
BORDER_ALPHA = 38
MUTED_ALPHA = 145

# 两端的界限和中间的当前值之间的分隔。写成 ``范围 40–60`` 会漏掉最要紧的那个数，所以
# 三个数排成一列不等式，符号本身就说明了谁大谁小。
SEPARATOR = " ≤ "

# 小数位上限。设计稿的 ``40 ≤ 48.7 ≤ 60`` 一个尾随零都没有：70px 的条里固定两位会让
# 一半字符是没有信息的零，而先被挤掉的恰好是右端的上界。
DECIMALS = 2


def format_number(value: float) -> str:
    """两位小数封顶，去掉尾随零和光秃秃的小数点。"""
    text = f"{value:.{DECIMALS}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


class RailBar(QWidget):
    """Draw where a value sits between the bounds declared for it."""

    HEIGHT_PX = HEIGHT_PX
    MINIMUM_WIDTH_PX = MINIMUM_WIDTH_PX

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self.setFixedHeight(HEIGHT_PX)
        self.setMinimumWidth(MINIMUM_WIDTH_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._span: tuple[float, float, float] | None = None

    def set_span(self, low: float, value: float, high: float) -> None:
        """Report ``value`` as sitting between ``low`` and ``high``."""
        self._span = (float(low), float(value), float(high))
        self.setAccessibleName(self.text())
        self.setToolTip(self.text())
        self.update()

    def clear(self) -> None:
        """Say nothing.

        留着上一层的 ``40 ≤ 48.7 ≤ 60`` 会读成「当前这一层的厚度界限」，而这恰好是它
        此刻唯一不能表示的意思。
        """
        self._span = None
        self.setAccessibleName("")
        self.setToolTip("")
        self.update()

    def text(self) -> str:
        if self._span is None:
            return ""
        low, value, high = self._span
        return SEPARATOR.join(format_number(number) for number in (low, value, high))

    def fraction(self) -> float | None:
        """Where the value sits in its span, clamped to ``[0, 1]``; ``None`` when empty.

        初值可以落在界限外——导入的项目、手敲进输入框的数字都可能越界，而越界正是这
        根条最该说清楚的时刻。钳的是填充长度，不是读数：文本照实写那个越界的数。

        上下界重合的量没有「落在哪儿」可言，给满：这里不判断该不该有这种声明，只保证
        画它的时候不拿 0 去除。
        """
        if self._span is None:
            return None
        low, value, high = self._span
        width = high - low
        if width <= 0.0:
            return 1.0
        return min(1.0, max(0.0, (value - low) / width))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        ink = palette.color(QPalette.ColorRole.Text)
        track = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_tinted(ink, TRACK_ALPHA))
        painter.drawRoundedRect(track, RADIUS_PX, RADIUS_PX)
        fraction = self.fraction()
        if fraction is not None:
            self._paint_fill(painter, track, fraction, palette)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(_tinted(ink, BORDER_ALPHA))
        painter.drawRoundedRect(track, RADIUS_PX, RADIUS_PX)
        if self._span is not None:
            self._paint_reading(painter, ink)
        painter.end()

    def _paint_fill(self, painter: QPainter, track: QRectF, fraction: float, palette: QPalette) -> None:
        fill = QRectF(track)
        fill.setWidth(track.width() * fraction)
        # 填充左端跟着轨道圆角，右端是齐平的截断面——那条截断面就是刻度。
        painter.setClipRect(fill)
        painter.setBrush(_tinted(palette.color(QPalette.ColorRole.Highlight), FILL_ALPHA))
        painter.drawRoundedRect(track, RADIUS_PX, RADIUS_PX)
        painter.setClipping(False)
        if fill.width() >= EDGE_WIDTH_PX:
            edge = QRectF(fill.right() - EDGE_WIDTH_PX, fill.top(), EDGE_WIDTH_PX, fill.height())
            painter.setBrush(QColor(theme.palette_tokens(palette).accent))
            painter.drawRect(edge)

    def _paint_reading(self, painter: QPainter, ink: QColor) -> None:
        """两端弱、中间强：界限是背景信息，当前值才是读者来看的那个数。"""
        assert self._span is not None
        low, value, high = self._span
        font = painter.font()
        font.setPointSize(theme.FONT_PT_SM)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        segments = (
            (f"{format_number(low)}{SEPARATOR}", False),
            (format_number(value), True),
            (f"{SEPARATOR}{format_number(high)}", False),
        )
        muted = _tinted(ink, MUTED_ALPHA)
        # 三段分开画，所以基线得自己算：逐段居中会让加粗那段因为字高不同而错位。
        baseline = (self.height() + metrics.ascent() - metrics.descent()) / 2.0
        cursor = float(TEXT_PADDING_PX)
        for text, strong in segments:
            font.setBold(strong)
            painter.setFont(font)
            painter.setPen(ink if strong else muted)
            painter.drawText(QPointF(cursor, baseline), text)
            cursor += painter.fontMetrics().horizontalAdvance(text)


def _tinted(colour: QColor, alpha: int) -> QColor:
    tinted = QColor(colour)
    tinted.setAlpha(alpha)
    return tinted
