"""Visible parameter-definition projection with explicit display-unit rules."""

from __future__ import annotations

from collections.abc import Mapping
from math import exp, isfinite

from PySide6.QtCore import QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QHeaderView,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
)

import xrr_fitter.api as api
from xrr_fitter.gui.parameters.grouping import (
    caption_text,
    display_scale,
    row_layout,
    uses_nm,
)

# Aliased because the table publishes a ``display_unit(name)`` method of its own,
# which resolves a name through the visible declarations rather than taking one.
from xrr_fitter.gui.parameters.grouping import display_unit as parameter_display_unit
from xrr_fitter.gui.parameters.grouping import row_name as parameter_row_name

HEADERS = ("参数", "初值", "下限", "上限", "先验")
# Editable numeric columns: initial, lower, upper.
VALUE_COLUMNS = (1, 2, 3)
# Significant digits shown for a numeric cell.  A computed bound such as
# 0.37167741227 carries far more digits than a user reads, and the three numeric
# columns share this format, so a wide bound used to stretch the whole column.
DISPLAY_SIGNIFICANT = 6
# Holds the unrounded value behind a numeric cell.  The commit path reads the
# whole row back off screen, so without this the rounded text of the two cells
# the user did not touch would be persisted in place of their exact values.
EXACT_VALUE_ROLE = Qt.ItemDataRole.UserRole + 1
# Fraction in [0, 1] locating the initial value between its lower and upper bound,
# painted as a faint fill behind the 初值 cell by ValuePositionDelegate.  A value
# railed against a bound then reads as an empty or a full bar at a glance, the way
# GenX marks parameters that have hit their limits.  Left unset (None) when the
# bounds have no width, so a pinned parameter draws no misleading bar.
VALUE_POSITION_ROLE = Qt.ItemDataRole.UserRole + 3
# Per-column width policy.  The three numeric columns render DISPLAY_SIGNIFICANT
# digits of a computed limit and are sized to that, so one column has to absorb the
# surplus dock width.  That column is the name: it carries the longest text, and now
# also the lock and the unit, and is the one a user reads to tell rows apart.
# Letting 先验 stretch as well split the surplus evenly between them, and since most
# projects configure no priors at all, a blank column then held half the width while
# the names elided to stubs -- 幂律背景幅值 B₂ and 幂律背景指数 p both rendered as
# "幂律背..." and became indistinguishable.
#
# The name column is Interactive rather than Stretch because Stretch has no floor to
# give it; ``_apply_name_width`` reproduces the surplus-absorbing behaviour and adds
# the floor NAME_MIN_WIDTH_FRACTION documents.  The three bounds are Interactive for
# the mirror-image reason: ``ResizeToContents`` sized them to the widest limit any row
# happened to hold and would not give a pixel back, so ``_apply_bounds_widths`` sizes
# them to the same hint and caps it.
COLUMN_RESIZE_MODES = (
    QHeaderView.ResizeMode.Interactive,
    QHeaderView.ResizeMode.Interactive,
    QHeaderView.ResizeMode.Interactive,
    QHeaderView.ResizeMode.Interactive,
    QHeaderView.ResizeMode.Interactive,
)
PRIOR_COLUMN = 4
# Floor under the name column's share of the viewport.  Stretch hands that column
# whatever the others leave, which is the right rule only while something is left:
# at the right column's own 340px budget the six sized columns took 269px of a 213px
# viewport, leaving the names 74px and three columns off-screen behind a horizontal
# scrollbar.  Two of those columns existed to show a glyph -- 单位 held at most four
# characters and 锁定 one checkbox, and each was floored at 44px by its own header --
# so they now ride inside the name cell where the design draws them, which is what
# lets the four remaining columns fit without scrolling.  The floor stays because the
# bounds are not padding and cannot be squeezed below their digits.
NAME_MIN_WIDTH_FRACTION = 0.35
# Ceiling on the prior column's share of the viewport.  A configured prior such as
# soft_range([0.1, 0.9], σ=0.05) is wider than any name, so sizing that column to
# its contents let it take the width the names need; the summary that no longer
# fits stays reachable through the cell's tooltip.
PRIOR_MAX_WIDTH_FRACTION = 0.3
# Shown on the name cell of a constraint-driven row, where the lock indicator is
# read-only rather than a user toggle.
CONSTRAINT_DRIVEN_TOOLTIP = "该参数由表达式约束驱动，数值不可手动编辑"
# 名字格那个勾的三档。Qt 自带的 partial 档正好装「仅范围」，所以三态不需要额外一列或一个
# 对话框；``ItemIsUserTristate`` 让点击在三档之间轮转。两张表按枚举建，读回时用反查，避免
# 「勾的状态」和「档位」在两处各写一份对应关系。
FREEDOM_CHECK_STATES: dict[api.ParameterFreedom, Qt.CheckState] = {
    api.ParameterFreedom.FREE: Qt.CheckState.Unchecked,
    api.ParameterFreedom.RANGE_ONLY: Qt.CheckState.PartiallyChecked,
    api.ParameterFreedom.FIXED: Qt.CheckState.Checked,
}
FREEDOM_BY_CHECK_STATE: dict[Qt.CheckState, api.ParameterFreedom] = dict(
    zip(FREEDOM_CHECK_STATES.values(), FREEDOM_CHECK_STATES.keys(), strict=True)
)
#: 三档各自说的是拟合器拿这个量怎么办。名字格的勾和检视区那三个圆点悬停出同一句话：同一个
#: 档位在两处说法不同，读者会以为自己在调两件事。「仅范围」这句必须点明声明的上下限仍是硬
#: 边界——省掉这半句，它就和「固定」读不出区别，也解释不了为什么值得单独占一档。
FREEDOM_TOOLTIPS: dict[api.ParameterFreedom, str] = {
    api.ParameterFreedom.FREE: "自由：在声明的上下限内自由拟合",
    api.ParameterFreedom.RANGE_ONLY: "仅范围：优先待在这一行填的区间内，越界按 soft_range 先验渐进受罚，声明的上下限仍是硬边界",
    api.ParameterFreedom.FIXED: "固定：不参与拟合，值停在初值上",
}

# A caption is identified by carrying no parameter name in UserRole, so nothing
# else may be stored there; the group key lives in its own role.
GROUP_KEY_ROLE = Qt.ItemDataRole.UserRole + 2
# Placeholder held by ``row_names`` where a caption sits, so a caller locating a
# row by name still gets the physical row index back.
CAPTION_ROW_NAME = ""


def _prior_display_scale(definition: api.ParameterDefinition) -> float:
    # Roughness-fraction priors live on [0, 1], even though the corresponding
    # physical-value columns display the decoded roughness in nm.
    return 1.0 if definition.transform == "roughness_fraction" else display_scale(definition)


def _number(value: float) -> str:
    return f"{value:.{DISPLAY_SIGNIFICANT}g}"


def _prior_body(prior: api.PriorSpec, scale: float) -> str:
    values = prior.parameters
    if prior.kind == "normal":
        return f"μ={_number(values[0] * scale)}, σ={_number(values[1] * scale)}"
    if prior.kind == "lognormal":
        # loc/scale live in log space: the physical center is exp(loc) and takes
        # the display scale, but the log-space spread is dimensionless and is
        # shown as stored so the summary never implies a length it lacks.
        return f"μ={_number(exp(values[0]) * scale)}, σ={_number(values[1])}"
    if prior.kind == "soft_range":
        low, high, std = values
        return f"[{_number(low * scale)}, {_number(high * scale)}], σ={_number(std * scale)}"
    return ""  # uniform carries no scalar parameters


def prior_summary(definition: api.ParameterDefinition) -> str:
    """这个量的先验写成一行，没有先验时是空串。

    表的最后一列和帧③「参数化」那枚徽章说的是同一件事，所以共用这一份：各写各的话，
    同一个先验会在两处呈现出不同的形状，而两处都没标注自己用的是哪种写法。
    """
    prior = definition.prior
    if prior is None:
        return ""
    return f"{prior.kind}({_prior_body(prior, _prior_display_scale(definition))})"


class ValuePositionDelegate(QStyledItemDelegate):
    """Paint a faint fill bar behind a cell marking where its value sits in range.

    The bar is drawn first and the cell's text over it, so the reading stays
    legible while the fill gives an at-a-glance sense of how close a parameter is
    to a bound.  The tint follows the palette's highlight colour at low opacity so
    it reads the same under the light and the dark appearance, and a cell carrying
    no VALUE_POSITION_ROLE (a caption, or a pinned parameter) renders plainly.
    """

    # Opacity of the fill over the cell background: strong enough to see, faint
    # enough to leave the digits legible.
    BAR_ALPHA = 48

    def paint(self, painter, option, index) -> None:  # noqa: N802 - Qt override
        fraction = index.data(VALUE_POSITION_ROLE)
        if fraction is not None:
            colour = option.palette.color(QPalette.ColorRole.Highlight)
            colour.setAlpha(self.BAR_ALPHA)
            bar = QRectF(option.rect)
            bar.setWidth(bar.width() * float(fraction))
            painter.save()
            painter.fillRect(bar, colour)
            painter.restore()
        super().paint(painter, option, index)


class ParameterTable(QTableWidget):
    """Render immutable declarations without owning persisted settings."""

    # 整张表换了一批声明。跟着当前行走的东西（右栏那张卡的抬头）靠这个退回去：重填顺带清掉
    # 的当前行是在 ``QSignalBlocker`` 里清的，``currentCellChanged`` 一声不响。发信方是表
    # 自己，连接因此和表同生共死——挂到 ``model()`` 上的连接会在表的 Python 包装失效之后再
    # 响一次，那一响什么都碰不得。
    rows_reloaded = Signal()

    def __init__(self) -> None:
        super().__init__(0, len(HEADERS))
        self.setObjectName("parameterTable")
        self.setAccessibleName("拟合参数")
        self.setHorizontalHeaderLabels(HEADERS)
        # The design's grid has no row-number gutter, and at a 340px column those
        # 35px are a third of what the names get to work with.
        self.verticalHeader().setVisible(False)
        # ``QTableView`` wraps by default, and the rows stay one line tall: a name
        # too wide for its column folded onto a second line that the row then cut
        # in half, while its neighbours in the same column elided cleanly.  One
        # behaviour for the whole column -- elide, with the full text in the
        # tooltip, which is what _prior_item already does.
        self.setWordWrap(False)
        header = self.horizontalHeader()
        # Qt floors every section at 16px by default, which would leave the prior
        # column occupying space while showing nothing; _apply_prior_width collapses
        # it to zero when no declaration carries a prior, and this lets zero mean it.
        header.setMinimumSectionSize(0)
        for column, mode in enumerate(COLUMN_RESIZE_MODES):
            header.setSectionResizeMode(column, mode)
        # The 初值 column carries a value-position bar behind its text; the bounds
        # columns stay plain.  VALUE_COLUMNS[0] is that initial column.
        self.setItemDelegateForColumn(VALUE_COLUMNS[0], ValuePositionDelegate(self))
        self._definitions: tuple[api.ParameterDefinition, ...] = ()
        self._rows: tuple[tuple[str, api.ParameterDefinition | None], ...] = ()
        # 档位落在 setting 上，声明里没有；表只画不存，所以按名字接一份只读映射。缺名字读作
        # 「自由/固定」由声明的 ``locked`` 决定，这样没传映射的调用点行为不变。
        self._freedom: dict[str, api.ParameterFreedom] = {}

    @property
    def definitions(self) -> tuple[api.ParameterDefinition, ...]:
        return self._definitions

    def freedom_of(self, name: str) -> api.ParameterFreedom:
        """这一行当前的档位。

        没有持久化 setting 的参数只有声明上的 ``locked``，两态；有 setting 的按 setting 走。
        表自己不存档位，读的是 :meth:`load` 收到的那份映射。
        """
        stored = self._freedom.get(name)
        if stored is not None:
            return stored
        matches = tuple(item for item in self._definitions if item.name == name)
        locked = bool(matches and (matches[0].locked or matches[0].constrained))
        return api.ParameterFreedom.from_locked(locked)

    @property
    def row_names(self) -> tuple[str, ...]:
        """Name per physical row, so a caption keeps the indices behind it honest.

        Callers locate a row with ``row_names.index(name)`` and then address cells
        by that index, so the sequence has to stay aligned with the table rather
        than with the declarations; a caption contributes ``CAPTION_ROW_NAME``.
        """
        return tuple(CAPTION_ROW_NAME if value is None else value.name for _key, value in self._rows)

    def load(
        self,
        definitions: tuple[api.ParameterDefinition, ...],
        *,
        expert_mode: bool,
        captions: Mapping[str, str] | None = None,
        freedom: Mapping[str, api.ParameterFreedom] | None = None,
    ) -> None:
        visible = tuple(definition for definition in definitions if expert_mode or not definition.expert_only)
        names = {} if captions is None else captions
        self._freedom = {} if freedom is None else dict(freedom)
        rows = row_layout(visible, names)
        # 读者站在哪个量上，按名字记下来。行号记不得：重填的原因往往正是行的构成变了
        # （换数据集、切高级选项），第 n 行装的会是另一份声明。
        standing_on = self.current_name()
        blocker = QSignalBlocker(self)
        self.clearContents()
        self.setRowCount(len(rows))
        self._definitions = visible
        self._rows = rows
        # ``row_layout`` emits no captions for a single group, and a row whose
        # owner is named nowhere above it has to name it itself, so the prefix is
        # only dropped under a caption that was actually drawn.
        caption: str | None = None
        for row, (key, definition) in enumerate(rows):
            if definition is None:
                caption = self._render_caption(row, key, names)
            else:
                self._render_row(row, definition, caption=caption)
        if standing_on is not None:
            self._restore_current(standing_on)
        del blocker
        # 重填顺带把当前行清成了 -1，而那是在 ``QSignalBlocker`` 里清的（否则每写一格都要走
        # 一遍 ``itemChanged``），没人听得见；上面把它落回了同一个量，解除屏蔽之后在这里补
        # 一声，让跟着当前行走的东西（帧③ 的抬头与那三档）知道该重读一次。
        self.rows_reloaded.emit()
        # A reload changes which declarations are present, so both widths are stale
        # until they are recomputed -- and no resize follows a mere expert-mode
        # toggle to recompute them for us.
        self._apply_prior_width()
        self._apply_bounds_widths()
        self._apply_name_width()

    def _restore_current(self, name: str) -> None:
        """把当前行落回 ``name`` 那一行；那个量不在了就不落。

        不落回顶上那一行：抬头会改口说另一个量，而它读起来和「读者自己点了这一行」
        一模一样。
        """
        for row, (_key, definition) in enumerate(self._rows):
            if definition is not None and definition.name == name:
                self.setCurrentCell(row, 0)
                return

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        # Order matters: the prior column's width is one of the widths the name
        # column measures itself against, and the bounds cap themselves against
        # what is left once the prior has taken its share.
        self._apply_prior_width()
        self._apply_bounds_widths()
        self._apply_name_width()

    def _apply_prior_width(self) -> None:
        """Give the prior column only what its summaries need, capped by the share.

        An empty column still reports a non-zero size hint, so the decision is made
        from the declarations rather than from pixels: with no prior configured the
        column collapses and the names take the whole surplus.
        """
        available = self.viewport().width()
        if available <= 0:
            return
        if not any(definition.prior is not None for definition in self._definitions):
            self.setColumnWidth(PRIOR_COLUMN, 0)
            return
        cap = int(available * PRIOR_MAX_WIDTH_FRACTION)
        self.setColumnWidth(PRIOR_COLUMN, min(self.sizeHintForColumn(PRIOR_COLUMN), cap))

    def _apply_bounds_widths(self) -> None:
        """Size the three bounds to their digits, capped so none outgrows the name.

        ``ResizeToContents`` gave each of them the widest limit any row happened to
        hold and never handed a pixel back: at the inspector's own budget that was
        47/63/69px of a 237px viewport, and the names -- which take whatever is left
        -- ran on 58.  The column a reader scans to tell 幂律背景幅值 B₂ from
        幂律背景指数 p was the narrowest thing on the screen.  A bound that overruns
        its cell still reads in full in the editor it opens into; a name has no
        editor to fall back on, so the bounds are the ones that give.
        """
        available = self.viewport().width() - self.columnWidth(PRIOR_COLUMN)
        if available <= 0:
            return
        hints = [self.sizeHintForColumn(column) for column in VALUE_COLUMNS]
        # Water-fill: the widest bound gives back a pixel at a time until what is
        # left for the name is at least as wide as the widest bound still standing.
        cap = max(hints, default=0)
        while cap > 0 and available - sum(min(hint, cap) for hint in hints) < cap:
            cap -= 1
        for column, hint in zip(VALUE_COLUMNS, hints, strict=True):
            self.setColumnWidth(column, min(hint, cap))

    def _apply_name_width(self) -> None:
        """Hand the name column the surplus, reclaiming it from 先验 before starving.

        The name identifies the row and cannot be traded for a bound summary, so a
        surplus under the floor is topped up out of the prior column, whose overflow
        stays reachable on its own tooltip.  What the prior column cannot give is
        taken as read: the other three columns hold editable bounds, and clipping one
        of those to pad a name would put a value out of reach behind a horizontal
        scrollbar -- which is the one thing a 340px column has no room to absorb.
        """
        available = self.viewport().width()
        if available <= 0:
            return
        bounds = sum(self.columnWidth(column) for column in VALUE_COLUMNS)
        floor = int(available * NAME_MIN_WIDTH_FRACTION)
        surplus = available - bounds - self.columnWidth(PRIOR_COLUMN)
        if surplus < floor:
            self.setColumnWidth(PRIOR_COLUMN, max(available - bounds - floor, 0))
            surplus = available - bounds - self.columnWidth(PRIOR_COLUMN)
        width = max(surplus, 0)
        if width != self.columnWidth(0):
            self.setColumnWidth(0, width)

    def clear_parameters(self) -> None:
        self.load((), expert_mode=False)

    def definition(self, name: str) -> api.ParameterDefinition:
        matches = tuple(value for value in self._definitions if value.name == name)
        if len(matches) != 1:
            raise KeyError(f"unknown visible parameter: {name}")
        return matches[0]

    def display_values(self, name: str) -> tuple[float, float, float]:
        definition = self.definition(name)
        scale = display_scale(definition)
        return tuple(value * scale for value in (definition.initial, definition.lower, definition.upper))

    def current_name(self) -> str | None:
        """当前行是哪个参数，站在分组标题行或者没有当前行时是 ``None``。

        分组标题行（「表面氧化层 · SiO₂」）的名字格里不存参数名，见 ``_parameter_item``：
        它说的是归属，不是一个可以设自由/固定的量。
        """
        row = self.currentRow()
        if row < 0:
            return None
        cell = self.item(row, 0)
        if cell is None:
            return None
        name = cell.data(Qt.ItemDataRole.UserRole)
        return None if name is None else str(name)

    def current_quantity(self) -> str | None:
        """当前行说的是哪个量，比如 ``厚度 d``。

        量名取自行名格自己的文本，而不是另去声明里重算一遍：抬头、档位和行名从此不可能
        各说各的。单位（``厚度 d（nm）`` 里那截全角括号）留给行——设计稿的抬头只写量名。
        """
        if self.current_name() is None:
            return None
        cell = self.item(self.currentRow(), 0)
        quantity, _, _ = cell.text().partition("（")
        return quantity or None

    def display_unit(self, name: str) -> str:
        definition = self.definition(name)
        return parameter_display_unit(definition)

    def entered_value(self, item: QTableWidgetItem) -> float:
        """Read a numeric cell back without losing digits the display rounded off.

        A cell shows six significant digits while carrying its unrounded value in
        EXACT_VALUE_ROLE.  Text still matching what this table rendered means the
        user left the cell alone, so the exact value is returned; any other text
        is what they just typed and is taken as entered.
        """
        exact = item.data(EXACT_VALUE_ROLE)
        if exact is not None and item.text() == _number(float(exact)):
            return float(exact)
        return float(item.text())

    def to_persisted_values(
        self,
        name: str,
        initial: float,
        lower: float,
        upper: float,
    ) -> tuple[float, float, float]:
        definition = self.definition(name)
        scale = 10.0 if uses_nm(definition) else 1.0
        return initial * scale, lower * scale, upper * scale

    def _render_caption(self, row: int, key: str, captions: Mapping[str, str]) -> str:
        """Open a group with a bold, inert row naming its owner.

        Only the name column is populated: leaving the numeric columns empty keeps
        the commit path from ever reading a caption as a parameter row, since
        ``_read_row`` rejects an incomplete row.  The row is unselectable so
        keyboard navigation lands on parameters only, and it holds no name in
        UserRole, which is what distinguishes it from a declaration.

        The text is returned so the rows underneath can drop the owner it already
        names.
        """
        text = caption_text(key, captions)
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setData(GROUP_KEY_ROLE, key)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.setItem(row, 0, item)
        return text

    def _render_row(
        self,
        row: int,
        definition: api.ParameterDefinition,
        caption: str | None = None,
    ) -> None:
        numbers = self._display_values(definition)
        values = (
            _name_text(definition, caption=caption),
            _number(numbers[0]),
            _number(numbers[1]),
            _number(numbers[2]),
        )
        # A constraint-driven value is computed from other parameters, so its
        # numeric columns join the always-read-only name column; an unconstrained
        # row keeps 1/2/3 editable exactly as before.
        readonly_columns = (0, 1, 2, 3) if definition.constrained else (0,)
        for column, value in enumerate(values):
            self.setItem(
                row,
                column,
                _parameter_item(
                    definition,
                    column,
                    value,
                    numbers,
                    readonly_columns,
                    self.freedom_of(definition.name),
                ),
            )
        self.setItem(row, PRIOR_COLUMN, _prior_item(definition))

    def _display_values(
        self,
        definition: api.ParameterDefinition,
    ) -> tuple[float, float, float]:
        scale = display_scale(definition)
        return (
            definition.initial * scale,
            definition.lower * scale,
            definition.upper * scale,
        )


def _parameter_item(
    definition: api.ParameterDefinition,
    column: int,
    value: str,
    numbers: tuple[float, float, float],
    readonly_columns: tuple[int, ...],
    freedom: api.ParameterFreedom,
) -> QTableWidgetItem:
    item = QTableWidgetItem(value)
    if column in readonly_columns:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if column == 0:
        item.setData(Qt.ItemDataRole.UserRole, definition.name)
        _apply_lock(item, definition, freedom)
    if column in VALUE_COLUMNS:
        exact = numbers[column - VALUE_COLUMNS[0]]
        item.setData(EXACT_VALUE_ROLE, exact)
        if item.text() != repr(exact):
            item.setToolTip(repr(exact))
    if column == VALUE_COLUMNS[0]:
        fraction = _value_position(numbers)
        if fraction is not None:
            item.setData(VALUE_POSITION_ROLE, fraction)
    return item


def _value_position(numbers: tuple[float, float, float]) -> float | None:
    """Where the initial value sits between its bounds, or None if that is ill-defined.

    The fraction is a ratio of like-scaled quantities, so it is identical whether
    the row shows Å or nm.  A zero-width or non-finite interval has no position to
    report, and the result is clamped so a value at or past a bound never paints a
    bar wider than the cell.
    """
    initial, lower, upper = numbers
    width = upper - lower
    if not isfinite(width) or width <= 0.0:
        return None
    return min(1.0, max(0.0, (initial - lower) / width))


def _apply_lock(
    item: QTableWidgetItem,
    definition: api.ParameterDefinition,
    freedom: api.ParameterFreedom,
) -> None:
    """Give the name cell the design's in-place lock glyph and its full tooltip.

    Frame ① draws the lock inside the name cell -- ``<td>密度 ρ <span class="lock
    on"></span></td>`` -- and frame ③ puts it in the field's own label.  A table
    item's check indicator is exactly that glyph: it renders at the head of the
    cell, before the text, and is toggled by clicking it, so the lock costs no
    column of its own.

    Three states need three indicator states, and Qt already has one: the partial
    check is 「仅范围」, sitting between 自由 (unchecked) and 固定 (checked).  Granting
    ``ItemIsUserTristate`` makes a click cycle 自由 → 仅范围 → 固定 in place, so the
    middle gear is reachable without a second column or a dialog.

    A constraint-driven row keeps the indicator as a read-only annotation: checked,
    because the value is not free, but not user-checkable, because unlocking it
    would have to delete the constraint.  The reason is appended to the tooltip the
    cell already carries rather than replacing it -- the identifying name is what
    tells two 厚度 rows apart and must not be traded for the explanation.
    """
    tooltip = f"{definition.display_name}\n({definition.name})"
    if definition.constrained:
        item.setToolTip(f"{tooltip}\n{CONSTRAINT_DRIVEN_TOOLTIP}")
        # A fresh item is user-checkable by default, so a driven row has to have the
        # flag taken away rather than merely not granted.
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        return
    item.setToolTip(f"{tooltip}\n{FREEDOM_TOOLTIPS[freedom]}")
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsUserTristate)
    item.setCheckState(FREEDOM_CHECK_STATES[freedom])


def _name_text(definition: api.ParameterDefinition, *, caption: str | None) -> str:
    """What the name cell reads: the quantity, its unit, and nothing else.

    The unit rides with the quantity the way the design writes it -- ``<label>厚度
    d（nm）</label>`` -- in fullwidth parens, and is omitted entirely when the
    declaration is dimensionless: frame ① renders that unit cell as ``—``, and
    ``（）`` around nothing would read as a value that failed to load.

    The owner is dropped when a caption above the row already names it.  Frame ①'s
    ``<tr class="grouprow">`` carries 表面氧化层 · SiO₂ and its rows carry only
    ``厚度 d``, while the declarations arrive prefixed: a layer named film
    contributes "film 厚度".  The match is exact and includes the separating space,
    so 基底's group -- whose single row reads 基底连接界面粗糙度, one word rather
    than a prefixed quantity -- keeps every character through the strip; it shortens
    to ``粗糙度 σ`` one step later, by whole-name rewrite (``grouping.ROW_ALIASES``).
    """
    text = parameter_row_name(definition.display_name, caption)
    unit = parameter_display_unit(definition)
    return f"{text}（{unit}）" if unit else text


def _prior_item(definition: api.ParameterDefinition) -> QTableWidgetItem:
    summary = prior_summary(definition)
    item = QTableWidgetItem(summary)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if summary:
        # The column is capped so the names stay readable, which elides the longer
        # summaries; the tooltip keeps the full text reachable.
        item.setToolTip(summary)
    return item
