"""帧① 左栏的数据集卡片：一枚字形，一行名字，一行小字。

设计稿把每个数据集画成一张两行卡（``.ds``）——左边置信度字形，右边上行文件名、下行
一句小字，选中的那张多一条 accent 左边框。此前同样的数据摊成七列表格行里的三列：
名字、状态、拟合各占一列。264px 的窄栏里三列互相挤宽度，而且同一个数据集的点数和
拟合判定被拆到相隔一列的两处，读者要横着扫过去才凑得出「这条能不能用」。

自绘用 delegate 而不是给每行塞控件，理由与候选解行相同：列表因此仍是一个焦点目标，
方向键就能在数据集之间走；每行塞控件会按数据集个数往 Tab 序列里加停靠点。

七列仍留在模型里，只是显示收起——详情标签和各列 tooltip 都从它们读。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from xrr_fitter.gui import theme

# 卡片的四段挂在这个角色上，DisplayRole 仍留着数据集名给读屏。
DATASET_CARD_ROLE = Qt.ItemDataRole.UserRole + 9

# ``.ds{padding:8px var(--md)}``：上下 8px 决定两行卡的高度，左右 12px 是内容起点。
CARD_PAD_V_PX = 8
CARD_PAD_H_PX = theme.SPACE_MD

# ``.ds{border-left:3px solid transparent}``，选中时 ``.ds.on`` 把它染成 accent。
# 每张卡都留这条宽度，所以亮起来时名字不会横向挪一下。
ACCENT_BAR_W_PX = 3

# 小字与名字之间不留行距：设计稿的两行是紧挨着的，中间空一行会让卡片读成两条目。
LINE_GAP_PX = 0

# 「可拟合」不写进小字。它回答的是「这条数据本身够不够拟合」，而在拟合判定旁边这句
# 是多余的一跳——读者要先绕过一句「能拟合」才看到「还没拟合」。所以只有它被替换掉，
# 「数据点不足」「源文件缺失」这些仍照原样写出来。
STATUS_FITTABLE = "可拟合"

# 设计稿帧① 那四行小字的尾巴（HTML 368-371）：● 与 ◆ 都写「已拟合」，▲ 写「需复核」。
# 判定名（可信 / 可用但相关 / 多解 / 不可信）由左边那枚字形负责，尾巴给的是「所以该怎么
# 办」；两处都写判定名，卡片就有一半的字在重复。键取的是 ``ConfidenceClass`` 的取值，与
# ``theme.CONFIDENCE_GLYPHS`` 同一套；查不到的（也就是「未拟合」）没有下一步可说，那一段
# 整段不写——设计稿帧③ 三行未拟合的卡写的正是「θ/2θ · 512 点」，到点数就收住。
VERDICT_ACTIONS = {
    "可信": "已拟合",
    "可用但相关": "已拟合",
    "多解": "需复核",
    "不可信": "需复核",
}

# ``.ds .nm .t{font-size:13px;font-weight:600}`` 对着 ``.s{font-size:11.5px}``：两行差
# 一号字加一档字重。这里按像素而不是点数钉，因为设计稿写的就是像素。11.5px 表达不出来
# （``QFont.setPixelSize`` 只收整数），小字取 12px——与管线步骤的说明字
# （``navigation/steps`` 的 ``STEP_DESC_FONT_PX``）同一号，左栏两处小字于是同高。
TITLE_FONT_PX = 13
SUBLINE_FONT_PX = 12

# 扫描约定按项目字段读，不是猜的：``input_angle_kind`` 只接受 ``two_theta_deg``（模型
# 校验拒绝其余取值），θ/2θ 是它的常用写法。同一条曲线按 θ 还是按 2θ 读，算出来的层厚
# 差一倍，所以点数前面先说清这 512 个数沿什么轴排。这一格说的是模型内部的轴，跟
# ``MeasurementPreset.angle_convention``（源文件那一列量的是什么角）是两件事：入射角源
# 文件在导入时就已 ×2 归一，落到这里的一律是散射角。
ANGLE_KIND_LABELS = {"two_theta_deg": "θ/2θ"}

# 设计稿帧④ 的成员行字形：``<span class="cf" style="color:var(--accent)">◐</span>``。
# 半实心读作「在跑」，而 accent 是这一屏正在进行的那件事的颜色，不是四档判定里的任何
# 一种状态色。
RUNNING_GLYPH = "◐"
RUNNING_KIND = "accent"


@dataclass(frozen=True)
class DatasetCard:
    """一张数据集卡的四段，外加画字形要用的状态色名。"""

    glyph: str
    kind: str
    title: str
    subline: str


def _verdict_shape(verdict: str) -> tuple[str, str]:
    """把拟合判定读成形状和颜色。

    未拟合取空心环而不是空字形：卡片里字形格空了，这一行的名字就比上下行左移一格，
    几张卡的名字对不齐一列。``fit_status_marker`` 对未拟合返回空串是对的——那是给
    表格单元用的，旁边一格已经写着「未拟合」，再补一枚实心字形会读成「有结果」；
    这里的代价不同，所以在卡片这一侧补上空心环。
    """
    glyph = theme.CONFIDENCE_GLYPHS.get(verdict)
    if glyph is None:
        return theme.CONFIDENCE_FALLBACK_GLYPH, theme.CONFIDENCE_FALLBACK_KIND
    return glyph, theme.CONFIDENCE_COMPARISON_KINDS[verdict]


def _subline(panel, dataset_id: str) -> str:
    """扫描约定与点数，加此刻最要紧的那一句。

    设计稿写「θ/2θ · 512 点 · 已拟合」——约定、点数、一句状态。应用手上有两句状态：
    源校验加可拟合性（``status_text``），以及拟合判定（``fit_status_text``）。两句都写
    就成了「32 点 · 可拟合 · 未拟合」，所以第三段按紧急度取一句：源文件出问题说源文件，
    点数不够说点数不够，其余把判定读成下一步（``VERDICT_ACTIONS``）。源文件那句还带上
    它的警示字形，因为左边那枚字形报的是拟合判定，此刻回答不了「这条还能不能用」。

    判定那截尾巴只在结构这一步收起（设计稿帧③ 的三行小字都停在点数，只有帧① 带尾巴）：
    改结构问的是「我编的这叠层落在哪几条曲线上」，点数回答得了，而「已拟合」说的是上一轮
    的事，正要被这次编辑作废。字形不收——判定仍旧报，只是不再用词重复一遍。
    """
    kind = ANGLE_KIND_LABELS.get(panel.document.project.input_angle_kind)
    points = panel.point_count_text(dataset_id)
    if kind is not None:
        points = f"{kind} · {points}"
    status = panel.status_text(dataset_id)
    if status != STATUS_FITTABLE:
        marker = panel.status_marker(dataset_id)
        head = f"{marker} {points}" if marker else points
        return f"{head} · {status}"
    if panel.structure_step_is_current():
        return points
    action = VERDICT_ACTIONS.get(panel.fit_status_text(dataset_id))
    return f"{points} · {action}" if action else points


def dataset_card(panel, dataset_id: str) -> DatasetCard:
    """把一个数据集读成可分段对齐的一张卡。

    读的全是 ``DataPanel`` 已有的那几个口径（点数、源校验、拟合判定），所以卡片和
    详情标签、tooltip 说的是同一件事。
    """
    running = panel.run_subline(dataset_id)
    if running:
        # 运行中这一枚报的不是判定而是「正在跑」：上一轮的判定此刻回答不了「跑到哪儿」，
        # 而半实心的 ◐ 在四档形状之外，不会被读成第五档可信度。
        return DatasetCard(
            glyph=RUNNING_GLYPH,
            kind=RUNNING_KIND,
            title=panel.display_name_text(dataset_id),
            subline=running,
        )
    glyph, kind = _verdict_shape(panel.fit_status_text(dataset_id))
    return DatasetCard(
        glyph=glyph,
        kind=kind,
        title=panel.display_name_text(dataset_id),
        subline=_subline(panel, dataset_id),
    )


def _title_font(base: QFont) -> QFont:
    """名字那一行的字体：13px，字重 600。

    600 在 Qt 里是 ``DemiBold``；此前用的 ``setBold(True)`` 是 700，比设计稿粗一档，
    名字于是压过左边那枚字形，而设计稿里这两者是并列的。
    """
    font = QFont(base)
    font.setPixelSize(TITLE_FONT_PX)
    font.setWeight(QFont.Weight.DemiBold)
    return font


def _small_font(base: QFont) -> QFont:
    """小字那一行的字体：12px，比名字小一号。

    此前这里设的是 ``FONT_PT_SM``，而在当前基准下它就等于基准（离屏实测名字与小字的
    ``QFontMetrics.height()`` 都是 15），等于没设——两行同号，只靠颜色分层次，设计稿
    「名字大、小字小」这层结构就没了。
    """
    font = QFont(base)
    font.setPixelSize(SUBLINE_FONT_PX)
    return font


class DatasetCardDelegate(QStyledItemDelegate):
    """自绘字形、名字、小字和选中时那条 accent 左边框。

    背景与选中底色仍交给样式画，所以悬停和高亮跟着主题走；这里只接管内容。
    """

    def paint(self, painter, option, index) -> None:  # noqa: N802 - Qt override
        card = index.data(DATASET_CARD_ROLE)
        style_option = QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        if card is None:
            super().paint(painter, option, index)
            return
        # 文本清空后再让样式画背景，否则 DisplayRole 那个名字会画在自绘内容底下。
        style_option.text = ""
        widget = style_option.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, style_option, painter, widget)
        painter.save()
        tokens = theme.palette_tokens(style_option.palette)
        rect = self._paint_accent_bar(painter, style_option, tokens)
        rect = self._paint_glyph(painter, rect, card, tokens)
        self._paint_name(painter, rect, card, style_option, tokens)
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802 - Qt override
        size = super().sizeHint(option, index)
        if index.data(DATASET_CARD_ROLE) is None:
            return size
        lines = QFontMetrics(_title_font(option.font)).height() + QFontMetrics(_small_font(option.font)).height()
        size.setHeight(max(size.height(), lines + LINE_GAP_PX + 2 * CARD_PAD_V_PX))
        return size

    def _paint_accent_bar(self, painter, style_option, tokens) -> QRect:
        """选中的那张卡在左缘画一条 accent 竖条，返回内容可用的矩形。

        条宽在两种状态下都从矩形里扣掉，所以选中与否名字都在同一个横坐标上。
        """
        rect = style_option.rect
        if style_option.state & QStyle.StateFlag.State_Selected:
            bar = QRect(rect.left(), rect.top(), ACCENT_BAR_W_PX, rect.height())
            painter.fillRect(bar, QColor(tokens.accent))
        return rect.adjusted(ACCENT_BAR_W_PX + CARD_PAD_H_PX, CARD_PAD_V_PX, -CARD_PAD_H_PX, -CARD_PAD_V_PX)

    def _paint_glyph(self, painter, rect: QRect, card: DatasetCard, tokens) -> QRect:
        """字形跨两行居中——设计稿的 ``.ds`` 是 ``align-items:center``。"""
        cell = QRect(rect)
        cell.setWidth(theme.GLYPH_CELL_PX)
        painter.setPen(QColor(getattr(tokens, card.kind)))
        painter.drawText(cell, int(Qt.AlignmentFlag.AlignCenter), card.glyph)
        return rect.adjusted(theme.GLYPH_CELL_PX + theme.SPACE_SM, 0, 0, 0)

    def _paint_name(self, painter, rect: QRect, card: DatasetCard, style_option, tokens) -> None:
        title_font = _title_font(style_option.font)
        title_height = QFontMetrics(title_font).height()
        title_rect = QRect(rect.left(), rect.top(), rect.width(), title_height)
        self._paint_line(painter, title_rect, card.title, title_font, style_option.palette.text().color())
        small_font = _small_font(style_option.font)
        subline_rect = QRect(
            rect.left(),
            title_rect.bottom() + 1 + LINE_GAP_PX,
            rect.width(),
            QFontMetrics(small_font).height(),
        )
        self._paint_line(painter, subline_rect, card.subline, small_font, theme.token_colour(tokens.faint_text))

    def _paint_line(self, painter, rect: QRect, text: str, font: QFont, colour: QColor) -> None:
        """一行文字，放不下就省略中段。

        设计稿的 ``text-overflow:ellipsis`` 只能截尾，而这些名字的区别常在尾部
        （``aSi_ML_25C`` 对 ``aSi_ML_400C``）。截尾会把三张卡截成同一个名字，所以
        省略中段，两端都留住。
        """
        painter.setFont(font)
        painter.setPen(colour)
        metrics = painter.fontMetrics()
        alignment = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        painter.drawText(rect, int(alignment), metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, rect.width()))
