"""Active-dataset parameter, sharing, and expert-mode coordinator."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QMenu,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.parameters.constraints import ConstraintEditor
from xrr_fitter.gui.parameters.sharing import SharingEditor
from xrr_fitter.gui.parameters.table import VALUE_COLUMNS, ParameterTable

# Wash of the error color behind a cell whose entered bound is self-inconsistent.
INVALID_CELL_BRUSH = QBrush(QColor(179, 38, 30, 48))
CONSTRAINT_CONTEXT_TOOLTIP = "该参数由表达式约束驱动，请先删除约束再修改"


def bounds_problem(initial: float, lower: float, upper: float) -> str | None:
    """Name the first self-inconsistency in an entered bound triple.

    This is a display-space sanity check that runs the instant a cell is
    edited, so the user sees why a value is rejected without waiting for the
    fit preflight. The public API stays the sole authority on whether a
    consistent triple is scientifically admissible; this only catches the
    ordering mistakes that are wrong on their face.
    """
    if lower > upper:
        return "下限不能大于上限"
    if initial < lower:
        return "初值不能小于下限"
    if initial > upper:
        return "初值不能大于上限"
    return None


class ParametersPanel(QWidget):
    """Project parameters whose only mutation boundary is the public API."""

    settings_changed = Signal(str, tuple)
    sharing_changed = Signal(tuple)
    constraints_changed = Signal(tuple)
    expert_mode_changed = Signal(bool)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("parametersPanel")
        self._definitions: tuple[api.ParameterDefinition, ...] = ()
        self.expert_toggle = QCheckBox("专家模式")
        self.expert_toggle.setObjectName("expertModeToggle")
        self.expert_toggle.setAccessibleName("切换专家参数")
        self.parameter_table = ParameterTable()
        self.sharing_editor = SharingEditor(document)
        self.sharing_editor.rules_changed.connect(self.sharing_changed.emit)
        self.constraint_editor = ConstraintEditor(document)
        self.constraint_editor.constraints_changed.connect(self.constraints_changed.emit)
        self.status_label = QLabel()
        self.status_label.setObjectName("parameterStatus")
        self.status_label.setProperty("mutedText", True)
        self.status_label.setWordWrap(True)
        tabs = QTabWidget()
        tabs.setObjectName("parameterTabs")
        tabs.addTab(self.parameter_table, "参数")
        tabs.addTab(self.sharing_editor, "共享")
        tabs.addTab(self.constraint_editor, "约束")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(self.expert_toggle)
        layout.addWidget(tabs)
        layout.addWidget(self.status_label)
        self.expert_toggle.toggled.connect(self._toggle_expert_mode)
        self.parameter_table.itemChanged.connect(self._table_setting_changed)
        self.parameter_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.parameter_table.customContextMenuRequested.connect(self._show_row_context_menu)
        document.project_changed.connect(self._refresh)
        self._refresh()

    @property
    def definitions(self) -> tuple[api.ParameterDefinition, ...]:
        return self._definitions

    @property
    def visible_definitions(self) -> tuple[api.ParameterDefinition, ...]:
        return self.parameter_table.definitions

    @property
    def row_names(self) -> tuple[str, ...]:
        return self.parameter_table.row_names

    @property
    def expert_mode(self) -> bool:
        return self.document.project.ui_state.expert_mode

    @property
    def sharing_rules(self) -> tuple[api.SharingRule, ...]:
        return self.sharing_editor.rules

    @property
    def constraint_rules(self) -> tuple[api.ConstraintRule, ...]:
        return self.constraint_editor.rules

    def display_values(self, name: str) -> tuple[float, float, float]:
        return self.parameter_table.display_values(name)

    def display_unit(self, name: str) -> str:
        return self.parameter_table.display_unit(name)

    def set_parameter(
        self,
        name: str,
        *,
        initial: float,
        lower: float,
        upper: float,
        locked: bool,
    ) -> bool:
        self._definition(name)
        setting = api.ParameterSetting(name, initial, lower, upper, locked)
        dataset_id = self._require_active_dataset_id()
        dataset = self._dataset(dataset_id)
        settings = self._with_setting(dataset.parameter_settings, setting)
        current = self.document.project
        updated = api.set_parameter_settings(current, dataset_id, settings)
        if updated is current:
            return False
        self.document.replace_project(updated)
        persisted = self._dataset(dataset_id).parameter_settings
        self.settings_changed.emit(dataset_id, persisted)
        return True

    def set_display_parameter(
        self,
        name: str,
        *,
        initial: float,
        lower: float,
        upper: float,
        locked: bool,
    ) -> bool:
        values = self.parameter_table.to_persisted_values(
            name,
            initial,
            lower,
            upper,
        )
        return self.set_parameter(
            name,
            initial=values[0],
            lower=values[1],
            upper=values[2],
            locked=locked,
        )

    def set_expert_mode(self, enabled: bool) -> bool:
        current = self.document.project
        updated = api.set_expert_mode(current, enabled)
        if updated is current:
            return False
        self.document.replace_project(updated)
        self.expert_mode_changed.emit(enabled)
        return True

    def reset_parameter(self, name: str) -> bool:
        """Drop a parameter's persisted setting so it falls back to its default.

        The declared default lives in ``describe_parameters`` and is derived
        from the structure; removing the user's override lets that default
        reassert itself without the user having to remember the original number.
        """
        self._definition(name)
        dataset_id = self._require_active_dataset_id()
        dataset = self._dataset(dataset_id)
        retained = tuple(value for value in dataset.parameter_settings if value.name != name)
        if len(retained) == len(dataset.parameter_settings):
            return False
        current = self.document.project
        updated = api.set_parameter_settings(current, dataset_id, retained)
        if updated is current:
            return False
        self.document.replace_project(updated)
        self.settings_changed.emit(
            dataset_id,
            self._dataset(dataset_id).parameter_settings,
        )
        return True

    def apply_sharing_rules(self, rules) -> bool:
        return self.sharing_editor.apply_rules(rules)

    def remove_sharing_rule(self, sharing_key: str) -> bool:
        return self.sharing_editor.remove_rule(sharing_key)

    def eligible_sharing_names(self, dataset_ids) -> tuple[str, ...]:
        return self.sharing_editor.eligible_names(dataset_ids)

    def sharing_error_text(self) -> str:
        return self.sharing_editor.error_text()

    def apply_constraint_rules(self, rules) -> bool:
        return self.constraint_editor.apply_rules(rules)

    def remove_constraint_rule(self, target: api.ParameterReference) -> bool:
        return self.constraint_editor.remove_rule(target)

    def eligible_constraint_targets(self, dataset_id: str) -> tuple[api.ParameterReference, ...]:
        return self.constraint_editor.eligible_targets(dataset_id)

    def eligible_constraint_sources(self, dataset_id: str) -> tuple[api.ParameterReference, ...]:
        return self.constraint_editor.eligible_sources(dataset_id)

    def constraint_error_text(self) -> str:
        return self.constraint_editor.error_text()

    def _toggle_expert_mode(self, enabled: bool) -> None:
        self.set_expert_mode(enabled)

    def _show_row_context_menu(self, position: object) -> None:
        item = self.parameter_table.itemAt(position)
        if item is None:
            return
        name_item = self.parameter_table.item(item.row(), 0)
        if name_item is None:
            return
        name = str(name_item.data(Qt.ItemDataRole.UserRole))
        menu = self._row_context_menu(name)
        actions = {action.objectName(): action for action in menu.actions()}
        reset = actions["resetParameterAction"]
        edit_prior = actions["editPriorAction"]
        clear_prior = actions["clearPriorAction"]
        chosen = menu.exec(self.parameter_table.viewport().mapToGlobal(position))
        if chosen is reset:
            self._reset_parameter_row(name)
        elif chosen is edit_prior:
            self._edit_prior_row(name)
        elif chosen is clear_prior:
            self._clear_prior_row(name)

    def _row_context_menu(self, name: str) -> QMenu:
        definition = self._definition(name)
        menu = QMenu(self.parameter_table)
        reset = menu.addAction("恢复默认值")
        reset.setObjectName("resetParameterAction")
        edit_prior = menu.addAction("编辑先验")
        edit_prior.setObjectName("editPriorAction")
        clear_prior = menu.addAction("清除先验")
        clear_prior.setObjectName("clearPriorAction")
        if definition.constrained:
            for action in (reset, edit_prior, clear_prior):
                action.setEnabled(False)
                action.setToolTip(CONSTRAINT_CONTEXT_TOOLTIP)
        return menu

    def _reset_parameter_row(self, name: str) -> None:
        try:
            if self.reset_parameter(name):
                self.status_label.setText(f"{name} 已恢复默认值")
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))

    def _edit_prior_row(self, name: str) -> None:
        from xrr_fitter.gui.parameters.dialogs import PriorDialog

        definition = self._definition(name)
        dialog = PriorDialog(
            definition,
            self,
            existing_prior=definition.prior,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        spec = dialog.spec()
        if spec is None:
            return
        self._commit_prior(name, self._with_prior(name, spec))

    def _clear_prior_row(self, name: str) -> None:
        self._commit_prior(name, self._without_prior(name))

    def _commit_prior(self, name: str, priors: tuple[api.ParameterPrior, ...]) -> None:
        try:
            dataset_id = self._require_active_dataset_id()
            current = self.document.project
            updated = api.set_parameter_priors(current, dataset_id, priors)
            if updated is current:
                return
            self.document.replace_project(updated)
            self.status_label.setText(f"{name} 先验已更新")
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))

    def _with_prior(self, name: str, spec: api.PriorSpec) -> tuple[api.ParameterPrior, ...]:
        retained = self._without_prior(name)
        return (*retained, api.ParameterPrior(name, spec))

    def _without_prior(self, name: str) -> tuple[api.ParameterPrior, ...]:
        dataset_id = self._require_active_dataset_id()
        priors = self._dataset(dataset_id).parameter_priors
        return tuple(value for value in priors if value.name != name)

    def _table_setting_changed(self, item: object) -> None:
        if item.column() not in (1, 2, 3, 5):
            return
        row = item.row()
        try:
            name, values, locked = self._read_row(row)
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))
            return
        problem = bounds_problem(*values)
        if problem is not None:
            # Keep the user's entry on screen and point at the offending bound
            # instead of silently reverting the whole row on a fixable typo.
            self._mark_row_invalid(row, problem)
            return
        # Boundary is consistent; submit normally. The resulting project_changed
        # signal triggers _refresh() which rebuilds the table, clearing any stale
        # red highlights from earlier edits automatically.
        try:
            self.set_display_parameter(
                name,
                initial=values[0],
                lower=values[1],
                upper=values[2],
                locked=locked,
            )
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))

    def _read_row(self, row: int) -> tuple[str, tuple[float, float, float], bool]:
        name_item = self.parameter_table.item(row, 0)
        value_items = tuple(self.parameter_table.item(row, column) for column in VALUE_COLUMNS)
        lock_item = self.parameter_table.item(row, 5)
        if name_item is None or lock_item is None or any(value is None for value in value_items):
            raise ValueError("parameter row is incomplete")
        name = str(name_item.data(Qt.ItemDataRole.UserRole))
        initial, lower, upper = (self.parameter_table.entered_value(value) for value in value_items)
        return name, (initial, lower, upper), lock_item.checkState() == Qt.CheckState.Checked

    def _mark_row_invalid(self, row: int, problem: str) -> None:
        for column in VALUE_COLUMNS:
            cell = self.parameter_table.item(row, column)
            if cell is not None:
                cell.setBackground(INVALID_CELL_BRUSH)
                cell.setToolTip(problem)
        self.status_label.setText(problem)

    def _refresh(self, *_args) -> None:
        blocker = QSignalBlocker(self.expert_toggle)
        self.expert_toggle.setChecked(self.expert_mode)
        del blocker
        dataset_id = self.document.active_dataset_id
        if dataset_id is None or self._dataset(dataset_id).structure is None:
            self._clear_projection("当前数据集尚未定义结构")
            return
        try:
            self._definitions = api.describe_parameters(
                self.document.project,
                dataset_id,
            )
        except (OSError, ValueError) as error:
            self._clear_projection(str(error))
            return
        self.parameter_table.load(self._definitions, expert_mode=self.expert_mode)
        # The row count is legible from the table itself, so the status line
        # stays empty here and is reserved for validation problems and the
        # outcome of a reset.
        self.status_label.clear()

    def _clear_projection(self, message: str) -> None:
        self._definitions = ()
        self.parameter_table.clear_parameters()
        self.status_label.setText(message)

    def _definition(self, name: str) -> api.ParameterDefinition:
        matches = tuple(value for value in self._definitions if value.name == name)
        if len(matches) != 1:
            raise KeyError(f"unknown parameter: {name}")
        return matches[0]

    def _require_active_dataset_id(self) -> str:
        dataset_id = self.document.active_dataset_id
        if dataset_id is None:
            raise ValueError("an active dataset is required")
        return dataset_id

    def _dataset(self, dataset_id: str) -> api.DatasetProject:
        matches = tuple(dataset for dataset in self.document.project.datasets if dataset.dataset_id == dataset_id)
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return matches[0]

    def _with_setting(
        self,
        existing: tuple[api.ParameterSetting, ...],
        setting: api.ParameterSetting,
    ) -> tuple[api.ParameterSetting, ...]:
        retained = tuple(value for value in existing if value.name != setting.name)
        position = next(
            (index for index, value in enumerate(existing) if value.name == setting.name),
            len(retained),
        )
        values = list(retained)
        values.insert(position, setting)
        return tuple(values)
