"""Visible parameter-definition projection with explicit display-unit rules."""

from __future__ import annotations

from collections.abc import Mapping
from math import exp

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

import xrr_fitter.api as api

HEADERS = ("参数", "初值", "下限", "上限", "单位", "锁定", "先验")
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
# Per-column width policy.  Sizing every column to its contents clipped 单位 and
# 锁定 outright, so one column has to absorb the surplus dock width.  That column
# is the name: it carries the longest text and is the one a user reads to tell
# rows apart.  Letting 先验 stretch as well split the surplus evenly between them,
# and since most projects configure no priors at all, a blank column then held
# half the width while the names elided to stubs -- 幂律背景幅值 B₂ and
# 幂律背景指数 p both rendered as "幂律背..." and became indistinguishable.
COLUMN_RESIZE_MODES = (
    QHeaderView.ResizeMode.Stretch,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.Interactive,
)
PRIOR_COLUMN = 6
# Ceiling on the prior column's share of the viewport.  A configured prior such as
# soft_range([0.1, 0.9], σ=0.05) is wider than any name, so sizing that column to
# its contents let it take the width the names need; the summary that no longer
# fits stays reachable through the cell's tooltip.
PRIOR_MAX_WIDTH_FRACTION = 0.3
# Shown on the lock cell of a constraint-driven row, where the checkbox is a
# read-only indicator rather than a user toggle.
CONSTRAINT_DRIVEN_TOOLTIP = "该参数由表达式约束驱动，数值不可手动编辑"

# Declarations already arrive clustered by owner -- every parameter of one layer is
# adjacent, then the next layer, then the backing, then the instrument -- but
# seventeen identically styled adjacent rows hid that structure completely, and ten
# of those seventeen belong to the instrument rather than to the sample.  A caption
# row opens each cluster so a user can tell which layer a 厚度 row belongs to
# without hovering it.  Grouping by ``category`` instead would scatter one layer's
# thickness, density and roughness across three distant blocks, which reads worse
# than the flat table it replaced.
BACKING_CAPTION = "基底"
INSTRUMENT_CAPTION = "仪器"
# A caption is identified by carrying no parameter name in UserRole, so nothing
# else may be stored there; the group key lives in its own role.
GROUP_KEY_ROLE = Qt.ItemDataRole.UserRole + 2
# Placeholder held by ``row_names`` where a caption sits, so a caller locating a
# row by name still gets the physical row index back.
CAPTION_ROW_NAME = ""


def _uses_nm(definition: api.ParameterDefinition) -> bool:
    return definition.name.endswith((".thickness_a", ".roughness_a"))


def _display_scale(definition: api.ParameterDefinition) -> float:
    return 0.1 if _uses_nm(definition) else 1.0


def _prior_display_scale(definition: api.ParameterDefinition) -> float:
    # Roughness-fraction priors live on [0, 1], even though the corresponding
    # physical-value columns display the decoded roughness in nm.
    return 1.0 if definition.transform == "roughness_fraction" else _display_scale(definition)


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


def _prior_summary(definition: api.ParameterDefinition) -> str:
    prior = definition.prior
    if prior is None:
        return ""
    return f"{prior.kind}({_prior_body(prior, _prior_display_scale(definition))})"


def _group_key(definition: api.ParameterDefinition) -> str:
    """Identify the layer, the backing or the instrument owning a declaration.

    Component parameters are keyed on ``component.{index}`` rather than on the
    leading segment alone, so a periodic block's per-layer and per-repeat rows stay
    with the block that produced them.
    """
    head, _, _ = definition.name.partition(".")
    if head != "component":
        return head
    return ".".join(definition.name.split(".")[:2])


def _caption_text(key: str, captions: Mapping[str, str]) -> str:
    """Name a group the way the structure editor names it.

    A component's caption cannot be recovered from its rows: a periodic block named
    ML contributes rows reading "W 厚度" and "Si 厚度", which share no token with
    the block.  The owner therefore supplies the component names and the key is
    only a fallback for callers that hold no structure.
    """
    if key == "instrument":
        return INSTRUMENT_CAPTION
    if key == "backing":
        return BACKING_CAPTION
    return captions.get(key, key)


def _row_layout(
    definitions: tuple[api.ParameterDefinition, ...],
    captions: Mapping[str, str],
) -> tuple[tuple[str, api.ParameterDefinition | None], ...]:
    """Interleave group captions with the declarations they introduce.

    A caption naming the only group present separates nothing, so a table holding
    one group is left exactly as it was before grouping existed.
    """
    keys = tuple(dict.fromkeys(_group_key(definition) for definition in definitions))
    if len(keys) < 2:
        return tuple((_group_key(value), value) for value in definitions)
    rows: list[tuple[str, api.ParameterDefinition | None]] = []
    current: str | None = None
    for definition in definitions:
        key = _group_key(definition)
        if key != current:
            rows.append((key, None))
            current = key
        rows.append((key, definition))
    return tuple(rows)


class ParameterTable(QTableWidget):
    """Render immutable declarations without owning persisted settings."""

    def __init__(self) -> None:
        super().__init__(0, len(HEADERS))
        self.setObjectName("parameterTable")
        self.setAccessibleName("拟合参数")
        self.setHorizontalHeaderLabels(HEADERS)
        header = self.horizontalHeader()
        for column, mode in enumerate(COLUMN_RESIZE_MODES):
            header.setSectionResizeMode(column, mode)
        self._definitions: tuple[api.ParameterDefinition, ...] = ()
        self._rows: tuple[tuple[str, api.ParameterDefinition | None], ...] = ()

    @property
    def definitions(self) -> tuple[api.ParameterDefinition, ...]:
        return self._definitions

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
    ) -> None:
        visible = tuple(definition for definition in definitions if expert_mode or not definition.expert_only)
        names = {} if captions is None else captions
        rows = _row_layout(visible, names)
        blocker = QSignalBlocker(self)
        self.clearContents()
        self.setRowCount(len(rows))
        self._definitions = visible
        self._rows = rows
        for row, (key, definition) in enumerate(rows):
            if definition is None:
                self._render_caption(row, key, names)
            else:
                self._render_row(row, definition)
        del blocker
        self._apply_prior_width()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._apply_prior_width()

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

    def clear_parameters(self) -> None:
        self.load((), expert_mode=False)

    def definition(self, name: str) -> api.ParameterDefinition:
        matches = tuple(value for value in self._definitions if value.name == name)
        if len(matches) != 1:
            raise KeyError(f"unknown visible parameter: {name}")
        return matches[0]

    def display_values(self, name: str) -> tuple[float, float, float]:
        definition = self.definition(name)
        scale = _display_scale(definition)
        return tuple(value * scale for value in (definition.initial, definition.lower, definition.upper))

    def display_unit(self, name: str) -> str:
        definition = self.definition(name)
        return "nm" if _uses_nm(definition) else definition.unit

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
        scale = 10.0 if _uses_nm(definition) else 1.0
        return initial * scale, lower * scale, upper * scale

    def _render_caption(self, row: int, key: str, captions: Mapping[str, str]) -> None:
        """Open a group with a bold, inert row naming its owner.

        Only the name column is populated: leaving the numeric columns empty keeps
        the commit path from ever reading a caption as a parameter row, since
        ``_read_row`` rejects an incomplete row.  The row is unselectable so
        keyboard navigation lands on parameters only, and it holds no name in
        UserRole, which is what distinguishes it from a declaration.
        """
        item = QTableWidgetItem(_caption_text(key, captions))
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setData(GROUP_KEY_ROLE, key)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.setItem(row, 0, item)

    def _render_row(self, row: int, definition: api.ParameterDefinition) -> None:
        numbers = self._display_values(definition)
        values = (
            definition.display_name,
            _number(numbers[0]),
            _number(numbers[1]),
            _number(numbers[2]),
            "nm" if _uses_nm(definition) else definition.unit,
        )
        # A constraint-driven value is computed from other parameters, so its
        # numeric columns join the always-read-only display-name/unit columns;
        # an unconstrained row keeps 1/2/3 editable exactly as before.
        readonly_columns = (0, 1, 2, 3, 4) if definition.constrained else (0, 4)
        for column, value in enumerate(values):
            self.setItem(
                row,
                column,
                _parameter_item(definition, column, value, numbers, readonly_columns),
            )
        self.setItem(row, 5, _lock_item(definition))
        self.setItem(row, 6, _prior_item(definition))

    def _display_values(
        self,
        definition: api.ParameterDefinition,
    ) -> tuple[float, float, float]:
        scale = _display_scale(definition)
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
) -> QTableWidgetItem:
    item = QTableWidgetItem(value)
    if column in readonly_columns:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if column == 0:
        item.setData(Qt.ItemDataRole.UserRole, definition.name)
        item.setToolTip(definition.name)
    if column in VALUE_COLUMNS:
        exact = numbers[column - VALUE_COLUMNS[0]]
        item.setData(EXACT_VALUE_ROLE, exact)
        if item.text() != repr(exact):
            item.setToolTip(repr(exact))
    return item


def _lock_item(definition: api.ParameterDefinition) -> QTableWidgetItem:
    item = QTableWidgetItem()
    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    if not definition.constrained:
        flags |= Qt.ItemFlag.ItemIsUserCheckable
    item.setFlags(flags)
    item.setCheckState(
        Qt.CheckState.Checked if definition.locked or definition.constrained else Qt.CheckState.Unchecked
    )
    if definition.constrained:
        item.setToolTip(CONSTRAINT_DRIVEN_TOOLTIP)
    return item


def _prior_item(definition: api.ParameterDefinition) -> QTableWidgetItem:
    summary = _prior_summary(definition)
    item = QTableWidgetItem(summary)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if summary:
        # The column is capped so the names stay readable, which elides the longer
        # summaries; the tooltip keeps the full text reachable.
        item.setToolTip(summary)
    return item
