"""帧① 检视器的「参数 · 结果值」表。

设置区的边界表写的是用户交给求解器的初值与上下限；这张表写的是求解器交回来的
值。两者用同一套分组与单位规则（``parameters.grouping``），所以同一个厚度在两张
表里读出的数量级一致。分组行的两半在这里互换过顺序——层列表读「材料 · 角色」，
这张表读「角色 · 材料」，理由见 ``grouping.group_caption``。

误差只认 ``UncertaintyReport.parameter_sigma``——它与 ``correlation_names`` 同序，
是逐参数的 1σ。bootstrap 区间是分位区间，半宽不是 1σ，拿它顶替会把一个百分位
当成标准差报出去；旧存档没有 sigma 字段时该列写「不可用」。锁定的参数根本没有
参与求解，它的误差列写「锁定」而不是 0，因为 0 会读成「测得极准」。

排版跟着设计稿的 ``table.grid``：没有竖线也没有外框，每行一道下边框，分组行压一层
底色，整张表按行数占高——检视器那一列自己会滚，表里再套一层滚动条就会把最后几层
藏在折叠线下。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHeaderView,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.parameters.grouping import (
    DENSITY_UNIT,
    caption_text,
    display_scale,
    display_unit,
    group_caption,
    group_key,
    owner_prefix,
    row_layout,
    row_name,
)

HEADERS = ("参数", "结果值", "±1σ", "单位")

# 设计稿这一组只留两行：每条曲线都要跟着拟合的强度标度与本底。声明有十条，余下八条是分辨率、
# 足迹、幂律本底这些仪器机械量——它们描述的是仪器怎么装的，读者在结果里要核对的是样品，八行
# 机械量插在三层之间会把「这层多厚」和「那层多密」隔开好几屏。
#
# 少画不等于丢掉：设置区那张边界表没有结果值这一列，这里藏掉一行就等于整个界面读不到它的
# 拟合值。所以被省下的每一行都挂在「仪器」这条分组行的悬停提示里，一个值都不少。
INSTRUMENT_GROUP_KEY = "instrument"
INSTRUMENT_RESULT_NAMES = ("instrument.scale", "instrument.background")

# 基底密度不是求解量，它是那块衬底的体密度（``MaterialSpec.bulk_density_g_cm3``），没有对应的
# 可拟合声明——补一条真声明会让 ``fit.checkpoint`` 的参数指纹漂移，所有旧存档都续跑不上。设计稿
# 仍把它画成一行锁定行，因为读者要拿它跟上面两层的密度比，所以这一行由显示层合成。
BACKING_DENSITY_NAME = "backing.bulk_density_g_cm3"
BACKING_DENSITY_LABEL = "密度"

# 设计稿这张表的每个数都写三位小数（``3.420``/``0.080``/``2.190``）。设置区那张边界表
# 写六位有效数字，因为那里的数是算出来的界限，位数由算法定；这张表的数要被人读出来、
# 抄进论文，位数由版面定，多出来的几位只是把小数点右边排成一片噪声。
DECIMALS = 3
# 小到定点写不出来的量（设计稿的本底 ``3.1e-7``）改写科学记数，尾数留一位。
MANTISSA_DECIMALS = 1
# 定点写法的两侧界限：低于 ``SCI_LOW`` 三位小数只剩 ``0.000``，高于 ``SCI_HIGH``
# 整数部分会把列撑开。
SCI_LOW = 5e-4
SCI_HIGH = 1e5

# 设计稿 ``.grouprow`` 的底色 ``--panel-2``：比参数行深一档。抬头靠加粗和字号已经分出
# 层级，但十几行里插三条分组行时，仅靠字重读者仍要逐行辨认；一层底色让「哪几行属于
# 同一层」在扫一眼时就成块。
GROUP_SHADE_ALPHA = 0.05
# 设计稿 ``.grouprow td`` 的 ``letter-spacing:.4px``。
GROUP_LETTER_SPACING_PX = 0.4
# 设计稿 ``table.grid td`` 的 ``border-bottom``，每行都有，末行不例外。
HAIRLINE_ALPHA = 0.16

# 设计稿 ``.lock.on``：一枚填成主色的圆角方框，里面一枚白勾。
LOCK_BOX_PX = 12
LOCK_RADIUS_PX = 3
LOCK_GLYPH = "✓"
# 这枚方框由委托画，所以「这一行是锁的」必须随 item 一起存下来。
LOCKED_ROLE = Qt.ItemDataRole.UserRole + 9

# 无量纲的声明在单位列写破折号而不是留空：空白读起来像没加载出来。
DIMENSIONLESS = "—"
LOCKED_TEXT = "锁定"
UNAVAILABLE_TEXT = "不可用"
ELIDED_TOOLTIP_HEAD = "未列出的仪器参数拟合值："


def _exponent(value: float) -> int | None:
    """这个数要不要改写科学记数——要的话用哪个指数，不要就是 ``None``。"""
    magnitude = abs(value)
    if magnitude == 0.0 or SCI_LOW <= magnitude < SCI_HIGH:
        return None
    return int(math.floor(math.log10(magnitude)))


def _number(value: float, exponent: int | None) -> str:
    if exponent is None:
        return f"{value:.{DECIMALS}f}"
    return f"{value / 10.0**exponent:.{MANTISSA_DECIMALS}f}e{exponent}"


def _sigma_by_name(report: object | None) -> dict[str, float]:
    """Pair the per-parameter sigma with the names it was published against."""
    if report is None or report.parameter_sigma is None:
        return {}
    return {name: float(value) for name, value in zip(report.correlation_names, report.parameter_sigma, strict=True)}


def _values_by_name(candidate: object | None) -> dict[str, float]:
    if candidate is None:
        return {}
    return {value.name: float(value.value) for value in candidate.parameters}


def _candidate(result: object, candidate_id: str | None) -> object | None:
    if candidate_id is None:
        return None
    return next(
        (value for value in result.candidates if value.candidate_id == candidate_id),
        None,
    )


def _sigma_text(
    definition: api.ParameterDefinition,
    sigma: Mapping[str, float],
    exponent: int | None,
    bulk_density: float | None = None,
) -> str:
    if definition.locked or definition.constrained:
        return LOCKED_TEXT
    value = sigma.get(definition.name)
    if value is None:
        return UNAVAILABLE_TEXT
    scaled = value * display_scale(definition, bulk_density)
    # 误差跟着值的指数走。两列各自取自己的指数时，屏幕上是 ``3.1e-7`` 配 ``4.0e-8``——
    # 同一行里两个指数，读者得先在心里对齐一次幂次才知道误差是值的百分之几。值本身没走
    # 科学记数时误差才自己判断，否则一个 4e-8 会被定点压成 ``0.000``。
    return _number(scaled, exponent if exponent is not None else _exponent(scaled))


def _backing_density_row(density: float | None) -> api.ParameterDefinition | None:
    """帧① 基底那组的第二行：一条只为显示存在的锁定声明。

    上下界与初值取同一个数，因为这个数没有区间可言——它不是解出来的，是那块衬底本身。
    """
    if density is None:
        return None
    return api.ParameterDefinition(
        name=BACKING_DENSITY_NAME,
        display_name=BACKING_DENSITY_LABEL,
        unit=DENSITY_UNIT,
        category="material",
        initial=density,
        lower=density,
        upper=density,
        transform="linear",
        locked=True,
    )


def _shown_definitions(
    definitions: tuple[api.ParameterDefinition, ...],
    densities: Mapping[str, float],
) -> tuple[tuple[api.ParameterDefinition, ...], tuple[api.ParameterDefinition, ...]]:
    """Split the declarations into the rows the design draws and the ones it elides."""
    shown: list[api.ParameterDefinition] = []
    elided: list[api.ParameterDefinition] = []
    for definition in definitions:
        if group_key(definition) == INSTRUMENT_GROUP_KEY and definition.name not in INSTRUMENT_RESULT_NAMES:
            elided.append(definition)
        else:
            shown.append(definition)
    synthetic = _backing_density_row(densities.get("backing"))
    if synthetic is not None:
        shown.append(synthetic)
    return tuple(shown), tuple(elided)


def _read_only(text: str, *, numeric: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if numeric:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _shade(palette: QPalette) -> QColor:
    """分组行的底色：正文色压到几个百分点，两种外观各自成立。"""
    colour = QColor(palette.color(QPalette.ColorRole.Text))
    colour.setAlphaF(GROUP_SHADE_ALPHA)
    return colour


class ResultValueDelegate(QStyledItemDelegate):
    """设计稿 ``table.grid`` 的两笔自绘：每行一道下边框，锁定行一枚打勾的方框。

    横线不走 ``setShowGrid``：那个开关连竖线一起画，把四列切成四只格子，读者的眼睛于是
    横着走一格停一下；靠右对齐的数字列自己就成列了，不需要线。

    方框也不往行里塞控件。它只是在陈述「这个数是我按住的，不是求解器解出来的」，不接受
    点击；而误差列的「锁定」二字要读到第三列才见到，行名旁边这一笔在扫参数名时就答了。
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        super().paint(painter, option, index)
        view_option = QStyleOptionViewItem(option)
        self.initStyleOption(view_option, index)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_hairline(painter, view_option)
        if index.data(LOCKED_ROLE):
            self._paint_lock(painter, view_option)
        painter.restore()

    def _paint_hairline(self, painter: QPainter, option: QStyleOptionViewItem) -> None:
        colour = QColor(option.palette.text().color())
        colour.setAlphaF(HAIRLINE_ALPHA)
        painter.setPen(colour)
        bottom = option.rect.bottom()
        painter.drawLine(option.rect.left(), bottom, option.rect.right(), bottom)

    def _paint_lock(self, painter: QPainter, option: QStyleOptionViewItem) -> None:
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        text_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, option, widget)
        advance = QFontMetrics(option.font).horizontalAdvance(option.text)
        box = QRect(
            text_rect.left() + advance + theme.SPACE_XS,
            option.rect.center().y() - LOCK_BOX_PX // 2,
            LOCK_BOX_PX,
            LOCK_BOX_PX,
        )
        tokens = theme.palette_tokens(option.palette)
        accent = theme.token_colour(tokens.accent)
        painter.setPen(accent)
        painter.setBrush(accent)
        painter.drawRoundedRect(box, LOCK_RADIUS_PX, LOCK_RADIUS_PX)
        glyph = QFont(option.font)
        glyph.setPointSize(theme.FONT_PT_SM)
        glyph.setBold(True)
        painter.setFont(glyph)
        painter.setPen(theme.token_colour(tokens.accent_text))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), LOCK_GLYPH)


class ResultValueTable(QTableWidget):
    """Project one candidate's fitted values, their 1σ, and their units."""

    HEADERS = HEADERS

    def __init__(self) -> None:
        super().__init__(0, len(HEADERS))
        self.setObjectName("resultValueTable")
        self.setAccessibleName("参数结果值")
        self.setHorizontalHeaderLabels(HEADERS)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.setShowGrid(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWordWrap(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setItemDelegate(ResultValueDelegate(self))
        vertical = self.verticalHeader()
        vertical.hide()
        vertical.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(HEADERS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(self._content_height())
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setHeight(self._content_height())
        return hint

    def _content_height(self) -> int:
        """表高 = 表头 + 每行行高，一像素不多。

        不滚动就得把高度让给内容。留着 ``QTableWidget`` 默认的那份高度，检视器里这张表
        会先撑出一大片空白，再把候选解那一段顶到视口以下。
        """
        rows = sum(self.rowHeight(row) for row in range(self.rowCount()))
        return self.horizontalHeader().sizeHint().height() + rows

    def project_result(
        self,
        result: object,
        candidate_id: str | None,
        captions: Mapping[str, str],
        densities: Mapping[str, float] | None = None,
    ) -> None:
        """Render the named candidate, or clear when the result no longer holds it.

        ``densities`` 是归属前缀到材料体密度的表（``grouping.bulk_densities``）：给了它，密度
        行就按帧① 写绝对密度，基底那组也补上它自己的密度行；不给就仍按声明的倍率显示。
        """
        candidate = _candidate(result, candidate_id)
        if candidate is None:
            self.clear_projection()
            return
        bulk = dict(densities or {})
        shown, elided = _shown_definitions(tuple(result.parameter_definitions), bulk)
        rows = row_layout(shown, captions)
        values = _values_by_name(candidate)
        sigma = _sigma_by_name(result.uncertainty)
        backing_density = bulk.get("backing")
        if backing_density is not None:
            values[BACKING_DENSITY_NAME] = backing_density
        self.setRowCount(len(rows))
        self.clearSpans()
        # ``row_layout`` emits no captions for a single group, so the owner is
        # only dropped from a name once a caption above the row has named it.
        caption: str | None = None
        for row, (key, definition) in enumerate(rows):
            if definition is None:
                # 剥离行名用的是原名，屏幕上写的是互换过的那份：两半一换，
                # ``strip_caption`` 就对不上前缀，每行都会重复一遍层名。
                caption = caption_text(key, captions)
                self._render_caption(row, group_caption(key, captions), self._elided_tooltip(key, elided, values, bulk))
            else:
                self._render_row(row, definition, caption, values, sigma, bulk.get(owner_prefix(definition.name)))
        self.setVisible(bool(rows))
        self.updateGeometry()

    def _elided_tooltip(
        self,
        key: str,
        elided: tuple[api.ParameterDefinition, ...],
        values: Mapping[str, float],
        densities: Mapping[str, float],
    ) -> str:
        """把这一组少画的行写成悬停提示，一行一个「量名 值 单位」。"""
        if key != INSTRUMENT_GROUP_KEY or not elided:
            return ""
        lines = [ELIDED_TOOLTIP_HEAD]
        for definition in elided:
            bulk = densities.get(owner_prefix(definition.name))
            value = values.get(definition.name)
            scaled = None if value is None else value * display_scale(definition, bulk)
            text = UNAVAILABLE_TEXT if scaled is None else _number(scaled, _exponent(scaled))
            unit = display_unit(definition, bulk)
            lines.append(f"{row_name(definition.display_name, None, bulk)} {text}{f' {unit}' if unit else ''}")
        return "\n".join(lines)

    def clear_projection(self) -> None:
        self.clearSpans()
        self.setRowCount(0)
        self.setVisible(False)
        self.updateGeometry()

    def _render_caption(self, row: int, text: str, tooltip: str = "") -> None:
        item = _read_only(text)
        font = item.font()
        font.setBold(True)
        font.setPointSize(theme.FONT_PT_SM)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, GROUP_LETTER_SPACING_PX)
        item.setFont(font)
        item.setBackground(_shade(self.palette()))
        if tooltip:
            item.setToolTip(tooltip)
        self.setItem(row, 0, item)
        self.setSpan(row, 0, 1, len(HEADERS))

    def _render_row(
        self,
        row: int,
        definition: api.ParameterDefinition,
        caption: str | None,
        values: Mapping[str, float],
        sigma: Mapping[str, float],
        bulk_density: float | None = None,
    ) -> None:
        name = _read_only(row_name(definition.display_name, caption, bulk_density))
        name.setToolTip(f"{definition.display_name}\n({definition.name})")
        if definition.locked or definition.constrained:
            name.setData(LOCKED_ROLE, True)
        self.setItem(row, 0, name)
        value = values.get(definition.name)
        scaled = None if value is None else value * display_scale(definition, bulk_density)
        exponent = None if scaled is None else _exponent(scaled)
        text = UNAVAILABLE_TEXT if scaled is None else _number(scaled, exponent)
        self.setItem(row, 1, _read_only(text, numeric=True))
        self.setItem(row, 2, _read_only(_sigma_text(definition, sigma, exponent, bulk_density), numeric=True))
        self.setItem(row, 3, _read_only(display_unit(definition, bulk_density) or DIMENSIONLESS))
