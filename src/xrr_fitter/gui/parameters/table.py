"""Visible parameter-definition projection with explicit display-unit rules."""

from __future__ import annotations

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
# Per-column width policy.  Sizing every column to its contents left the two
# widest columns fighting for the dock width and clipped 单位 and 锁定 outright;
# the name and prior columns absorb the surplus instead, and the short columns
# keep exactly the width their text needs.
COLUMN_RESIZE_MODES = (
    QHeaderView.ResizeMode.Stretch,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.ResizeToContents,
    QHeaderView.ResizeMode.Stretch,
)
# Shown on the lock cell of a constraint-driven row, where the checkbox is a
# read-only indicator rather than a user toggle.
CONSTRAINT_DRIVEN_TOOLTIP = "该参数由表达式约束驱动，数值不可手动编辑"


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

    @property
    def definitions(self) -> tuple[api.ParameterDefinition, ...]:
        return self._definitions

    @property
    def row_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self._definitions)

    def load(
        self,
        definitions: tuple[api.ParameterDefinition, ...],
        *,
        expert_mode: bool,
    ) -> None:
        visible = tuple(definition for definition in definitions if expert_mode or not definition.expert_only)
        blocker = QSignalBlocker(self)
        self.clearContents()
        self.setRowCount(len(visible))
        self._definitions = visible
        for row, definition in enumerate(visible):
            self._render_row(row, definition)
        del blocker

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
    item = QTableWidgetItem(_prior_summary(definition))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item
