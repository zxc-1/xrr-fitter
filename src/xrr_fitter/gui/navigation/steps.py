"""左栏那一列 ``.pstep`` 行的构件：一枚状态圆点、名字、小字，以及往下一行的连接线。

这些构件本来长在 ``panel.py`` 里，只服务「分析管线」那六步。设计稿帧⑤ 的左栏在同一栏里
用同一种行画了第二段——「不确定度方法」的四步（见 ``methods.py``），于是行的画法得有一个
两段共用的出处：两处各画一遍，同一个圆点就会在同一屏上出现两种直径、两种间距。

这里只管一行长什么样，不管这一行处在什么状态、更不管状态从哪读来——状态是 ``theme``
那套 ``stepDot``/``stepState`` 记号的事，读判据是各自那段面板的事。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from xrr_fitter.gui import theme
from xrr_fitter.gui.sizing import ElidingLabel

# ``.pstep .body`` 的两行字：标题 ``font-size:13px;font-weight:600``，说明
# ``font-size:11.5px``。按像素给而不是折算成 pt：9.75pt / 8.625pt 取整后会并成同一号字，而
# 两行的大小差是「哪句是步骤名」的唯一线索。11.5 就近取 12（Qt 的像素字号是整数），这一档
# 的行盒与 11.5px 同为 15px。
STEP_TITLE_FONT_PX = 13
STEP_DESC_FONT_PX = 12

# A dot the size the design draws it, and a connector thin enough to read as a
# rail rather than as a column of its own.
MARKER_CELL_PX = 20
RAIL_W = 2
# ``.pstep .line{width:2px;flex:1;min-height:10px}``: the connector fills whatever
# the row's two lines of text leave beside the dot, so it has a floor rather than a
# height.  Giving it a fixed height instead is what turned it into a row of its own.
RAIL_MIN_H = 10
# ``.pstep{padding:7px var(--md)}``.  Seven is not a spacing token and should not
# become one -- it is this row's vertical padding in the design and nothing else
# reads it.
STEP_PAD_V_PX = 7
# 量说明那行行盒高度用的样字。行盒不该跟着文字所属的字体走：一行中文要向后备字体借更高
# 的盒（12px 字量到 17），一枚 ``—`` 用的是基准字体的（15），于是「未开始」那行比「—」
# 那行高 2px，六行的间距忽宽忽窄，而设计稿的六个圆点是等距落下来的一列。取两种字里高的
# 那种，也就是这个样字量出来的高度，钉给每一行的说明。
CAPTION_LINE_SAMPLE = "—字"


class StepRow(QWidget):
    """One ``.pstep`` row, clickable once the project has walked past it.

    左栏此前只是个进度条：它算得出项目最远走到哪，但读者没法说「我现在要看哪一步」。这两
    件事不是一回事——设计稿帧③ 里结构早就建好了（算出来是「参数」），画的却是停在「结构」
    上的那一屏。更要紧的是右栏和画布跟着这一步换内容，所以拟合出结果之后，选中层那张卡就
    再也够不着了：想改一层得先把结果删掉。

    往前的步不接受点击——回头看得见东西，是因为那些东西项目里已经有了；往前点没有这个前提，
    点过去只会换来一屏空卡。够不着的行不给手型光标也不进 Tab 序，免得读者去试。
    """

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._reachable = False
        self._pressed = False

    def set_reachable(self, reachable: bool) -> None:
        if reachable == self._reachable:
            return
        self._reachable = reachable
        self.setCursor(Qt.CursorShape.PointingHandCursor if reachable else Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus if reachable else Qt.FocusPolicy.NoFocus)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # 得在按下时就把事件收下，否则它会一路冒泡到父级，松开也就不会送到这里来。
        if self._reachable and event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pressed, self._pressed = self._pressed, False
        # 按下之后拖出行外再松手是「反悔」，按钮就是这么约定的。
        if pressed and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            if self.rect().contains(event.position().toPoint()):
                self.clicked.emit()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._reachable and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            event.accept()
            self.clicked.emit()
            return
        super().keyPressEvent(event)


@dataclass(frozen=True, slots=True)
class StepParts:
    """The widgets one step is made of, handed back for wiring.

    ``build_step`` assembles four widgets that later have to be reached
    individually -- the row goes into the layout, the marker and label take a
    state, and the description is only ever read by tests -- so returning them
    named beats returning a tuple the caller has to remember the order of.
    """

    row: StepRow
    marker: QLabel
    label: QLabel
    description: QLabel


def _build_rail(parent: QWidget, title: str) -> QFrame:
    """The 2px connector that chains one step's marker down to the next.

    It is a painted frame rather than a spacer because its whole job is to be
    seen: without the line the markers read as an unordered list of links, not
    as one ordered run.  The colour comes from the palette so the rail tracks a
    theme switch along with the border it matches.

    It stretches instead of measuring: ``flex:1`` in the design means the line
    takes the height the row's text leaves under the dot, so the connector is
    free -- which is the whole reason six steps fit a 264px column beside a
    dataset list and a summary.
    """
    rail = QFrame(parent)
    rail.setObjectName(f"pipelineRail_{title}")
    rail.setFrameShape(QFrame.Shape.VLine)
    rail.setFixedWidth(RAIL_W)
    rail.setMinimumHeight(RAIL_MIN_H)
    rail.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    rail.setProperty("pipelineRail", True)
    return rail


def build_step(parent: QWidget, title: str, description: str, *, connected: bool) -> StepParts:
    """One step: a marker column on the left, name and purpose stacked right.

    The description is a visible label rather than the tooltip it used to be --
    a newcomer scanning the column never hovers, and a screen reader reaches a
    tooltip only on focus, so the one line that says what the step is for was
    invisible to both.

    ``connected`` hangs the rail inside this row's own marker column, which is
    where ``.pstep .line`` lives; ``.pstep:last-child .line{display:none}`` is the
    last step passing ``False``.
    """
    row = StepRow(parent)
    row.setObjectName(f"pipelineStep_{title}")
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(theme.SPACE_MD, STEP_PAD_V_PX, theme.SPACE_MD, STEP_PAD_V_PX)
    row_layout.setSpacing(theme.SPACE_SM)

    marker = QLabel("", row)
    marker.setObjectName(f"pipelineDot_{title}")
    marker.setFixedSize(MARKER_CELL_PX, MARKER_CELL_PX)
    marker.setAlignment(Qt.AlignmentFlag.AlignCenter)
    # ``.pstep .dot`` is a filled 20px circle, and the guided header already draws
    # its ordinals through this property, so the two step maps of one project share
    # one marker vocabulary instead of each inventing a shape.
    marker.setProperty("stepDot", True)

    label = QLabel(title, row)
    label.setObjectName(f"pipelineLabel_{title}")
    label.setToolTip(description)
    # ``.pstep .body .t{font-size:13px;font-weight:600}``：粗细落在字体对象上，因为当前那步
    # 的 QSS 规则只管颜色——设计稿里六步的标题一样粗，只有颜色跟着状态走。
    title_font = label.font()
    title_font.setPixelSize(STEP_TITLE_FONT_PX)
    title_font.setWeight(QFont.Weight.DemiBold)
    label.setFont(title_font)

    caption = ElidingLabel(description, row)
    caption.setObjectName(f"pipelineDescription_{title}")
    # ``.pstep .body .d{color:var(--ink-faint)}``：比 ``mutedText`` 再浅一档，跟同一栏里
    # 「数据集」「分析管线」两句抬头同色，正文才是这栏里最深的字。
    caption.setProperty("faintText", True)
    # 不折行：``.pstep .body .d`` 在设计稿里是一行，264px 的栏宽也放得下每一句。折行标签的
    # sizeHint 不是「这段文字实际要多高」，而是 QLabel 猜出来的一个方块——中英混排的
    # 「导出 ORSO、Excel 与图」被猜成两行（hint 22，而 heightForWidth(220) 是 11），那一行
    # 于是按 52px 要地方，屏幕上却只画 44。六行里有一行这样，整条管线就凭空多要 8px，而这
    # 8px 是从同一栏里的数据列表和摘要那儿拿的。改成 elide：一行到底，窄了收字不收行。
    caption.setToolTip(description)
    caption_font = caption.font()
    caption_font.setPixelSize(STEP_DESC_FONT_PX)
    caption.setFont(caption_font)
    # 行盒钉死（见 ``CAPTION_LINE_SAMPLE``）：这一行的高度是字号的事，不是这一步此刻写着
    # 什么的事。不钉，同一条管线里中文那几行就比 ``—`` 那几行高 2px。
    caption.setFixedHeight(QFontMetrics(caption_font).boundingRect(CAPTION_LINE_SAMPLE).height())

    body = QVBoxLayout()
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(0)
    body.addWidget(label)
    body.addWidget(caption)

    dots = QVBoxLayout()
    dots.setContentsMargins(0, 0, 0, 0)
    dots.setSpacing(0)
    dots.addWidget(marker)
    if connected:
        dots.addWidget(_build_rail(row, title), 1, Qt.AlignmentFlag.AlignHCenter)
    else:
        # 最后一行不画连接线（``.pstep:last-child .line{display:none}``），而线是这一列里唯一
        # 会伸展的东西。线一撤，QVBoxLayout 里只剩一枚定尺圆点，剩余高度均分到它上下两侧，
        # 圆点被居中——设计稿的 ``align-items:flex-start`` 要的是顶边与标题齐平。补一段空白
        # 替线占住下方，六个点于是等距一列。
        dots.addStretch(1)

    row_layout.addLayout(dots)
    row_layout.addLayout(body, 1)
    return StepParts(row=row, marker=marker, label=label, description=caption)
