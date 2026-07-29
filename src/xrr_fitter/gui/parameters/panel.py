"""Active-dataset parameter, sharing, and expert-mode coordinator."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QTabWidget, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.parameters.sharing import SharingEditor
from xrr_fitter.gui.parameters.table import ParameterTable


class ParametersPanel(QWidget):
    """Project parameters whose only mutation boundary is the public API."""

    settings_changed = Signal(str, tuple)
    sharing_changed = Signal(tuple)
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
        self.status_label = QLabel()
        self.status_label.setObjectName("parameterStatus")
        self.status_label.setWordWrap(True)
        tabs = QTabWidget()
        tabs.setObjectName("parameterTabs")
        tabs.addTab(self.parameter_table, "参数")
        tabs.addTab(self.sharing_editor, "共享")
        layout = QVBoxLayout(self)
        layout.addWidget(self.expert_toggle)
        layout.addWidget(tabs)
        layout.addWidget(self.status_label)
        self.expert_toggle.toggled.connect(self._toggle_expert_mode)
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

    def apply_sharing_rules(self, rules) -> bool:
        return self.sharing_editor.apply_rules(rules)

    def remove_sharing_rule(self, sharing_key: str) -> bool:
        return self.sharing_editor.remove_rule(sharing_key)

    def eligible_sharing_names(self, dataset_ids) -> tuple[str, ...]:
        return self.sharing_editor.eligible_names(dataset_ids)

    def sharing_error_text(self) -> str:
        return self.sharing_editor.error_text()

    def _toggle_expert_mode(self, enabled: bool) -> None:
        self.set_expert_mode(enabled)

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
        self.status_label.setText(f"{len(self.parameter_table.definitions)} 个可见参数")

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
        matches = tuple(
            dataset
            for dataset in self.document.project.datasets
            if dataset.dataset_id == dataset_id
        )
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
