"""Visible parameter-definition projection with explicit display-unit rules."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

import xrr_fitter.api as api


HEADERS = ("参数", "初值", "下限", "上限", "单位", "锁定")


def _uses_nm(definition: api.ParameterDefinition) -> bool:
    return definition.name.endswith((".thickness_a", ".roughness_a"))


def _display_scale(definition: api.ParameterDefinition) -> float:
    return 0.1 if _uses_nm(definition) else 1.0


def _number(value: float) -> str:
    return f"{value:.12g}"


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
        visible = tuple(
            definition
            for definition in definitions
            if expert_mode or not definition.expert_only
        )
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
        return tuple(
            value * scale
            for value in (definition.initial, definition.lower, definition.upper)
        )

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
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, definition.name)
                item.setToolTip(definition.name)
            self.setItem(row, column, item)
        locked = QTableWidgetItem()
        locked.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        locked.setCheckState(
            Qt.CheckState.Checked if definition.locked else Qt.CheckState.Unchecked
        )
        self.setItem(row, 5, locked)

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
