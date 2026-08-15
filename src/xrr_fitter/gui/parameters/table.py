"""Visible parameter-definition projection with explicit display-unit rules."""

from __future__ import annotations

from math import exp

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

import xrr_fitter.api as api

HEADERS = ("参数", "初值", "下限", "上限", "单位", "锁定", "先验")

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
    return f"{value:.12g}"


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
        self.resizeColumnsToContents()
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
        initial, lower, upper = self._display_values(definition)
        values = (
            definition.display_name,
            _number(initial),
            _number(lower),
            _number(upper),
            "nm" if _uses_nm(definition) else definition.unit,
        )
        # A constraint-driven value is computed from other parameters, so its
        # numeric columns join the always-read-only display-name/unit columns;
        # an unconstrained row keeps 1/2/3 editable exactly as before.
        readonly_columns = (0, 1, 2, 3, 4) if definition.constrained else (0, 4)
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column in readonly_columns:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, definition.name)
                item.setToolTip(definition.name)
            self.setItem(row, column, item)
        locked = QTableWidgetItem()
        lock_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not definition.constrained:
            # Only a free row exposes an interactive lock toggle; a driven row
            # keeps the checkbox as a read-only indicator of its locked state.
            lock_flags |= Qt.ItemFlag.ItemIsUserCheckable
        locked.setFlags(lock_flags)
        locked.setCheckState(
            Qt.CheckState.Checked if definition.locked or definition.constrained else Qt.CheckState.Unchecked
        )
        if definition.constrained:
            locked.setToolTip(CONSTRAINT_DRIVEN_TOOLTIP)
        self.setItem(row, 5, locked)
        prior = QTableWidgetItem(_prior_summary(definition))
        prior.setFlags(prior.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 6, prior)

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
