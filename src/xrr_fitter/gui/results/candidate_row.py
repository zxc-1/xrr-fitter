"""帧① 候选解一节的一行。

设计稿的每个候选是一只带边框的圆角盒子，里面四段：左边一枚可信度形状，中间加粗的
名字，右边贴着右缘的 ``J=…``，末尾一枚标记——打开着的那行写「当前」，其余每行写
「切换」。实测是一行散文，六个字段用点号串成一句，要读者自己在里面找目标值。

形状画的是可信度（``●◆▲■○``）。可信度是整份结果的判定，``FitCandidate`` 上没有这个
字段，也不能加：这个类要进 checkpoint，加字段会让指纹漂移、旧存档 resume 不上。但判定
本来就属于被采信的那一行——它就是被判定的那个解，所以判定随结果一起传进来，画在那一行
上。其余候选没有被判定过，它们的存在本身就是「目标值相近的候选不止一个」，也就是
``多解 ▲``；解不出来的那些是 ``不可信 ■``，这一条压过采信标记，因为一个不该采信的解不
因为被打开而变得可信。三态里的「早期淘汰」在散文与悬停上照旧写全（见
``candidates._audit_state``），形状这一列让给判定。

「切换」是画出来的，不是塞进去的控件：列表因此仍是一个焦点目标，方向键就能在候选之间
走。每行塞一枚真按钮会多出八个 Tab 停靠点，而它做的事和点这一行是同一件事——设计稿把
它画成无边框的主色文字，正是「点这行」的提示，不是第二个动作。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from xrr_fitter.gui import theme

# 判定这三档不来自结果本身，所以在这里落成常量：``多解`` 是「还有别的解跟它一样好」，
# ``不可信`` 是「这个解算不出来」，``不可用`` 是旧存档没写判定字段。前两个必须是
# ``theme.CONFIDENCE_GLYPHS`` 的键，形状与判定卡才是同一套。
STATE_MULTIPLE = "多解"
STATE_UNTRUSTED = "不可信"
STATE_UNAVAILABLE = "不可用"

MARK_CURRENT = "当前"
MARK_SWITCH = "切换"

TITLE_PREFIX = "候选"
ORDINAL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OBJECTIVE_SIGNIFICANT = 6

# 设计稿的一行：``padding:6px 8px``、``border:1px``、``border-radius:6px``，行与行之间
# ``gap:6px``。缝算进行高而不是交给 ``QListView::setSpacing``——那个开关四边都插，首行
# 上方和末行下方于是各多出三像素，而这一段紧贴着上一节的横线开始。
ROW_GAP_PX = 6
ROW_RADIUS_PX = 6
ROW_PAD_H_PX = 8
ROW_PAD_V_PX = 6

# 未选中的行只有一道 ``--border`` 细框；选中的那行换主色框加一层 ``--selection`` 底。
# 悬停这一层设计稿没有画，但列表原本靠样式表的 hover 提示「这行可点」，自绘接管背景之后
# 要把它接回来：画的就是选中那层底色压淡一档——悬停预演的正是点下去会发生的事。这个数是
# 那层底色自身透明度的倍数，不是绝对透明度；``--selection`` 本身才 46/255，把 0.55 当绝对
# 值写会让悬停比选中还重。
ROW_HOVER_ALPHA = 0.55

# 「当前」那枚胶囊的尺寸与两档透明度取自 ``theme.BADGE_*``：设计稿的 ``.badge`` 是同一只
# 胶囊，这里再抄一份局部常量会在版面调整时分叉——原先抄的那份横向内边距还写成了 6px，
# 而设计稿是 ``padding:2px 8px``。字号字重同理走 ``theme.BADGE_FONT_PX`` 与
# ``theme.BADGE_FONT_WEIGHT``。

# 一行里四段各有字号：形状 14px、名字 12.5px、读数 12px、胶囊 11px（设计稿 453-468 行）。
# 四个数写在同一行上，它们的全部意义就是互相比大小——形状最大（它是这一行的判定），名字次之
# （它是这一行是谁），读数再小一档（它是拿来扫着比的），胶囊最小（它只是一枚标记）。按 px 给
# 而不是折算成点数：这四档要直接比大小，混用两套单位就得先猜当前 DPI 才知道谁大，实测
# ``FONT_PT_MD`` 的 11pt 落到 15px 而不是 14px，正是这么差出来的。名字取整到 13：
# ``setPixelSize`` 只吃整数，而 12 会让名字与读数同号，「谁是主字段」就只剩字重在说。
GLYPH_FONT_PX = 14
TITLE_FONT_PX = 13
VALUE_FONT_PX = 12
# 「切换」是 ``.btn.sm`` 叠 ``.btn.ghost``：12px / 600，比胶囊大一档而比名字轻不了。
SWITCH_FONT_PX = 12

CANDIDATE_ROW_ROLE = Qt.ItemDataRole.UserRole + 8


def candidate_is_selectable(candidate: object) -> bool:
    """Whether a candidate can be adopted as the answer.

    An early-eliminated candidate is kept for inspection but never adopted: it was
    stopped before converging, so its objective is not comparable with the others.
    """
    if not getattr(candidate, "valid", False):
        return False
    if getattr(candidate, "stop_reason", "") == "early_eliminated":
        return False
    objective = getattr(candidate, "objective", float("nan"))
    ranking = getattr(candidate, "ranking_objective", None)
    if not isfinite(objective):
        return False
    return ranking is None or isfinite(ranking)


def candidate_is_archived(candidate: object) -> bool:
    """Whether a candidate was stopped early rather than having failed."""
    if not getattr(candidate, "valid", False):
        return False
    if getattr(candidate, "stop_reason", "") != "early_eliminated":
        return False
    return isfinite(getattr(candidate, "objective", float("nan")))


@dataclass(frozen=True)
class CandidateRow:
    """一行候选的四段，外加画它们要用的状态色名。"""

    glyph: str
    state: str
    kind: str
    title: str
    objective: str
    ranking: str
    mark: str


def _number(value: float) -> str:
    return f"{value:.{OBJECTIVE_SIGNIFICANT}g}"


def _grade(state: str) -> tuple[str, str, str]:
    """把一档判定读成「形状 · 判定名 · 色名」，与判定卡取自同一张表。"""
    return theme.CONFIDENCE_GLYPHS[state], state, theme.CONFIDENCE_COMPARISON_KINDS[state]


def _shape(candidate: object, *, adopted: bool, confidence: str | None) -> tuple[str, str, str]:
    """这一行画哪一档判定。

    次序是要紧的：算不出来的解先被拦下，采信标记压不过它——否则一份「可信」的结果会
    把它自己那个发散的候选也画成 ●。
    """
    if not candidate_is_selectable(candidate):
        return _grade(STATE_UNTRUSTED)
    if not adopted:
        return _grade(STATE_MULTIPLE)
    if confidence in theme.CONFIDENCE_GLYPHS:
        return _grade(confidence)
    return theme.CONFIDENCE_FALLBACK_GLYPH, STATE_UNAVAILABLE, theme.CONFIDENCE_FALLBACK_KIND


def candidate_title(ordinal: int) -> str:
    """这一行在这一节里的位置，就是它的名字。

    求解器给的 ID（``E-0``、``automatic-refit-2``）写的是这个候选从哪条种子来，回答不了
    读者在这一节要问的「有几个解、我在看第几个」。字母用完了退回编号：自动拟合一次能留下
    几十个候选，循环回 A 会让两行同名，那个问题就彻底没有答案。ID 仍在同一行的悬停里。
    """
    if 0 <= ordinal < len(ORDINAL_LETTERS):
        return f"{TITLE_PREFIX} {ORDINAL_LETTERS[ordinal]}"
    return f"{TITLE_PREFIX} {ordinal + 1}"


def candidate_row(
    candidate: object,
    *,
    selected: bool,
    adopted: bool = False,
    confidence: str | None = None,
    ordinal: int = 0,
) -> CandidateRow:
    """把一个候选读成设计稿那一行的四段。"""
    glyph, state, kind = _shape(candidate, adopted=adopted, confidence=confidence)
    objective = getattr(candidate, "objective", float("nan"))
    ranking = getattr(candidate, "ranking_objective", None)
    return CandidateRow(
        glyph=glyph,
        state=state,
        kind=kind,
        title=candidate_title(ordinal),
        objective=f"J={_number(objective)}",
        # 一次单阶段求解里两者相等，写两遍等于让读者去比较两个一样的数。
        ranking="" if ranking is None or ranking == objective else f"排序 J={_number(ranking)}",
        mark=MARK_CURRENT if selected else MARK_SWITCH,
    )


def glyph_colour(row: CandidateRow, palette: QPalette) -> QColor:
    """形状那一列的墨色。

    走 ``theme.token_colour`` 而不是 ``QColor(token)``：空心圈那一档取的是 ``muted_text``，
    它在两套外观里都写成 ``rgba(...)``，而 ``QColor`` 解不出这种写法——直接交给它，这一枚圈
    会画成不透明的墨黑，比四档判定里最重的 ■ 还重。
    """
    return theme.token_colour(getattr(theme.palette_tokens(palette), row.kind))


def glyph_font(base: QFont) -> QFont:
    """判定形状那一段：14px，这一行最大的一档字。

    它最大是因为它是这一行的判定——读者先看形状决定这一解可不可信，再看它是谁。
    """
    font = QFont(base)
    font.setPixelSize(GLYPH_FONT_PX)
    return font


def title_font(base: QFont, *, selected: bool) -> QFont:
    """候选名那一段：13px，选中那行 700，其余 600。

    设计稿选中那行的名字是 ``<b>``，其余每行显式写 ``font-weight:600``。三行并排时读者要一眼
    看出「我在看哪一个」；反过来，每一行都写 700 就等于没写——加粗成了这一节的常态，它就不再
    指向任何一行。
    """
    font = QFont(base)
    font.setPixelSize(TITLE_FONT_PX)
    font.setWeight(QFont.Weight.Bold if selected else QFont.Weight.DemiBold)
    return font


def value_font(base: QFont) -> QFont:
    """``J=…`` 那一段：12px 的等宽数字。

    设计稿这一段带 ``.tnum``。比例数字里 1 比 8 窄，``J=1.83`` 与 ``J=2.41`` 于是等宽不同，
    右对齐只让末位齐、小数点仍错开一两像素——而这一列的用处正是上下扫着比大小。Qt 的样式表
    没有 ``font-variant-numeric``，所以这一档只能落字体对象。
    """
    font = QFont(base)
    font.setPixelSize(VALUE_FONT_PX)
    font.setFeature(QFont.Tag("tnum"), 1)
    return font


def value_colour(palette: QPalette) -> QColor:
    """``J=…`` 那一段的墨色：两档灰里更淡的那一档。

    设计稿写 ``.faint``（``--ink-faint``）而不是 ``.muted``。两档灰的分工是固定的：静音那档给
    还要读的次要文字，最淡那档给读者扫过去、只在需要时停下来看的数。这一行的主角是名字与判定
    形状，给 ``J=…`` 静音档会让它跟名字抢同一级注意力。
    """
    return theme.token_colour(theme.palette_tokens(palette).faint_text)


def badge_font(base: QFont) -> QFont:
    """「当前」那枚胶囊的字：11px / 600，与样式表画的 ``.badge`` 同一档。

    字号字重都读 ``theme.BADGE_*``，与内边距和两档透明度走同一条路——同一只胶囊在两条通路上
    被画，两边各写一份就会在版面调整时分叉。
    """
    font = QFont(base)
    font.setPixelSize(theme.BADGE_FONT_PX)
    font.setWeight(theme.BADGE_FONT_WEIGHT)
    return font


def switch_font(base: QFont) -> QFont:
    """「切换」那段主色文字：12px / 600，设计稿的 ``.btn.sm`` 叠 ``.btn.ghost``。

    600 而不是 700：它比这一行的名字（未选中也是 600）重不了——一枚提示不该压过它所提示的
    那一行是谁。
    """
    font = QFont(base)
    font.setPixelSize(SWITCH_FONT_PX)
    font.setWeight(QFont.Weight.DemiBold)
    return font


def _mark_font(base: QFont, mark: str) -> QFont:
    """右缘那枚标记的字：胶囊与幽灵按钮各一档。"""
    return badge_font(base) if mark == MARK_CURRENT else switch_font(base)


def _row_text_height(base: QFont) -> int:
    """一行装得下的最矮高度：五档字里最高的那一档。

    按 ``option.fontMetrics`` 算会拿到应用默认的 12px，而形状那一段是 14px——行高照默认字号
    给，最大的那一档就被裁掉半个字。胶囊那一档连它的上下内边距一起算，跟 ``_paint_mark`` 里
    那个算式是同一个，否则胶囊会被压回行高、圆角变成扁圆。
    """
    heights = [
        QFontMetrics(glyph_font(base)).height(),
        QFontMetrics(title_font(base, selected=True)).height(),
        QFontMetrics(value_font(base)).height(),
        QFontMetrics(switch_font(base)).height(),
        QFontMetrics(badge_font(base)).height() + 2 * theme.BADGE_PAD_V_PX,
    ]
    return max(heights)


def _tint(colour: QColor, alpha: float) -> QColor:
    """同一个色相压到某个透明度，设计稿的 ``rgba(46,125,50,.08)`` 就是这么来的。"""
    tinted = QColor(colour)
    tinted.setAlphaF(alpha)
    return tinted


class CandidateRowDelegate(QStyledItemDelegate):
    """设计稿的候选行：一只圆角盒子，里面四段各自对齐。

    整行由这里画完，``super().paint`` 与 ``QStyle.CE_ItemViewItem`` 都不调用——默认那条通路
    只把 DisplayRole 那句散文按一列铺出来，而设计稿要的是「形状 · 名字 · 贴右缘的 ``J=…`` ·
    一枚标记」四段分别对齐。散文没有丢：它仍是 item 的文本与悬停，读屏和 ``candidate_text``
    照旧拿得到。
    """

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """行高只按字算，行与行之间留出设计稿的 ``gap:6px``。

        不与 ``super().sizeHint()`` 取大：DisplayRole 上挂的是那句散文，列表一旦开着自动换行
        它会把一行撑成三四行高——而这一行根本不画那句话。宽度同理只数这四段，散文那句话有
        三四百像素长，拿它当宽度提示会给这一栏挤出一条横向滚动条。

        缝算成每行的上缘留白而不是交给 ``QListView::setSpacing``：那个开关四边都插，首行上方
        和末行下方于是各多出三像素，而这一段紧贴着上一节的横线开始。
        """
        height = _row_text_height(option.font) + 2 * ROW_PAD_V_PX + (0 if index.row() == 0 else ROW_GAP_PX)
        row = index.data(CANDIDATE_ROW_ROLE)
        if row is None:
            return QSize(super().sizeHint(option, index).width(), height)
        # 每段按自己那一档字量：五段统一用 ``option.font`` 量，会把 14px 的形状与 13px 的名字
        # 都算成 12px 的宽，右边那两栏于是被挤出框、名字提早吃到省略号。选中态在这里读不到
        # （``sizeHint`` 拿到的 ``option.state`` 不带它），名字取更宽的 700 那一档。
        segments = (
            (glyph_font(option.font), row.glyph),
            (title_font(option.font, selected=True), row.title),
            (value_font(option.font), row.ranking),
            (value_font(option.font), row.objective),
            (_mark_font(option.font, row.mark), row.mark),
        )
        width = 2 * ROW_PAD_H_PX + sum(
            QFontMetrics(font).horizontalAdvance(text) + theme.SPACE_SM for font, text in segments if text
        )
        return QSize(width, height)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        row = index.data(CANDIDATE_ROW_ROLE)
        if row is None:
            super().paint(painter, option, index)
            return
        view_option = QStyleOptionViewItem(option)
        self.initStyleOption(view_option, index)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        box = self._box(view_option, index)
        self._paint_box(painter, view_option, box)
        self._paint_segments(painter, view_option, box, row)
        painter.restore()

    def _box(self, option: QStyleOptionViewItem, index) -> QRect:
        """盒子占的是这一行减去上缘那道缝。"""
        box = QRect(option.rect)
        if index.row():
            box.setTop(box.top() + ROW_GAP_PX)
        return box

    def _paint_box(self, painter: QPainter, option: QStyleOptionViewItem, box: QRect) -> None:
        """三种状态三种盒子：选中、悬停、其余。"""
        tokens = theme.palette_tokens(option.palette)
        wash = theme.token_colour(tokens.selection_bg)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(theme.token_colour(tokens.accent))
            painter.setBrush(wash)
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.setPen(theme.token_colour(tokens.surface_border))
            painter.setBrush(_tint(wash, wash.alphaF() * ROW_HOVER_ALPHA))
        else:
            painter.setPen(theme.token_colour(tokens.surface_border))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        # 半像素内缩：一像素的笔画压在整数边界上会被两侧各吃掉一半，圆角于是发灰。
        painter.drawRoundedRect(QRectF(box).adjusted(0.5, 0.5, -0.5, -0.5), ROW_RADIUS_PX, ROW_RADIUS_PX)

    def _paint_segments(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        box: QRect,
        row: CandidateRow,
    ) -> None:
        """四段从两头往中间排：形状靠左，标记与目标值贴右缘，名字吃掉剩下的宽。"""
        content = box.adjusted(ROW_PAD_H_PX, ROW_PAD_V_PX, -ROW_PAD_H_PX, -ROW_PAD_V_PX)
        shape_font = glyph_font(option.font)
        painter.setFont(shape_font)
        painter.setPen(glyph_colour(row, option.palette))
        painter.drawText(content, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), row.glyph)
        left = content.left() + QFontMetrics(shape_font).horizontalAdvance(row.glyph) + theme.SPACE_SM
        right = self._paint_numbers(painter, option, content, self._paint_mark(painter, option, content, row), row)
        title = QRect(content)
        title.setLeft(left)
        title.setRight(max(left, right - theme.SPACE_SM))
        name_font = title_font(option.font, selected=bool(option.state & QStyle.StateFlag.State_Selected))
        painter.setFont(name_font)
        # 选中那行的名字照旧是墨色：``initStyleOption`` 不换 ``Text``（只有 ``CE_ItemViewItem``
        # 会换成 ``HighlightedText``），而设计稿画的正是主色底上一行墨色的粗体名字。
        painter.setPen(option.palette.color(QPalette.ColorRole.Text))
        painter.drawText(
            title,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            QFontMetrics(name_font).elidedText(row.title, Qt.TextElideMode.ElideRight, title.width()),
        )

    def _paint_numbers(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        content: QRect,
        right: int,
        row: CandidateRow,
    ) -> int:
        """目标值贴着标记左侧，排序目标值再贴在它左侧；返回最左那一段的左缘。

        目标值占住靠右那一格，所以有没有排序目标值都不影响它上下对齐——排序值往左长，
        每行的 ``J=…`` 仍在同一列上，读者扫这一列就能比大小。
        """
        font = value_font(option.font)
        metrics = QFontMetrics(font)
        painter.setFont(font)
        painter.setPen(value_colour(option.palette))
        for text in (row.objective, row.ranking):
            if not text:
                continue
            cell = QRect(content)
            cell.setRight(right - theme.SPACE_SM)
            cell.setLeft(cell.right() - metrics.horizontalAdvance(text))
            painter.drawText(cell, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), text)
            right = cell.left()
        return right

    def _paint_mark(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        content: QRect,
        row: CandidateRow,
    ) -> int:
        """右缘那枚标记，返回它的左缘——左边几段排到这里为止。

        「当前」是设计稿的 ``.badge.ok``，一枚描边填色的圆角胶囊；「切换」是 ``.btn.ghost``，
        无边框的主色文字。后者不画框正是「点这一行」的提示，不是第二个可点的东西。
        """
        tokens = theme.palette_tokens(option.palette)
        font = _mark_font(option.font, row.mark)
        metrics = QFontMetrics(font)
        painter.setFont(font)
        width = metrics.horizontalAdvance(row.mark)
        if row.mark != MARK_CURRENT:
            cell = QRect(content)
            cell.setLeft(content.right() - width)
            painter.setPen(theme.token_colour(tokens.accent))
            painter.drawText(cell, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), row.mark)
            return cell.left()
        height = min(content.height(), metrics.height() + 2 * theme.BADGE_PAD_V_PX)
        pill = QRect(
            content.right() - width - 2 * theme.BADGE_PAD_H_PX,
            content.center().y() - height // 2,
            width + 2 * theme.BADGE_PAD_H_PX,
            height,
        )
        ok = theme.token_colour(tokens.ok)
        painter.setPen(_tint(ok, theme.BADGE_BORDER_ALPHA))
        painter.setBrush(_tint(ok, theme.BADGE_FILL_ALPHA))
        # ``border-radius:999px`` 落到 Qt 就是半个高：再大不会更圆，只会画坏。
        painter.drawRoundedRect(QRectF(pill).adjusted(0.5, 0.5, -0.5, -0.5), height / 2, height / 2)
        painter.setPen(ok)
        painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter), row.mark)
        return pill.left()
