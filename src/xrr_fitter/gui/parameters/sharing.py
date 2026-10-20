"""API-backed joint-parameter sharing rule editor."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.parameters.sharing_dialog import SharingDialog


class SharingEditor(QWidget):
    """Validate and commit complete sharing-rule sequences atomically."""

    rules_changed = Signal(tuple)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("sharingEditor")
        self.tree = QTreeWidget()
        self.tree.setObjectName("sharingTree")
        self.tree.setAccessibleName("参数共享组列表")
        self.tree.setHeaderLabels(("共享组", "成员"))
        self.error_label = QLabel()
        self.error_label.setObjectName("sharingValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._commands())
        hint = QLabel("选择至少两个数据集中的同名自由参数建立共享组；共享规则在联合拟合中生效。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.tree)
        layout.addWidget(self.error_label)
        document.project_changed.connect(self._render)
        self._render()

    def _commands(self) -> QHBoxLayout:
        self.add_button = QPushButton("添加共享")
        self.add_button.setObjectName("sharingAddRuleButton")
        self.add_button.setAccessibleName("添加参数共享组")
        self.add_button.setToolTip("选择数据集及共同自由参数，创建共享组")
        self.add_button.clicked.connect(self._add_rule)
        self.delete_button = QPushButton("删除共享")
        self.delete_button.setObjectName("sharingDeleteRuleButton")
        self.delete_button.setAccessibleName("删除参数共享组")
        self.delete_button.setToolTip("删除当前选中的共享组，不删除数据集")
        self.delete_button.clicked.connect(self._delete_selected_rule)
        commands = QHBoxLayout()
        commands.addWidget(self.add_button)
        commands.addWidget(self.delete_button)
        commands.addStretch(1)
        return commands

    @property
    def rules(self) -> tuple[api.SharingRule, ...]:
        return self.document.project.sharing_rules

    def apply_rules(self, rules) -> bool:
        values = tuple(rules)
        current = self.document.project
        try:
            validated = api.validate_sharing_rules(current, values)
            updated = api.set_sharing_rules(current, validated)
        except (TypeError, ValueError) as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            raise
        self.error_label.hide()
        if updated is current:
            return False
        self.document.replace_project(updated)
        self.rules_changed.emit(updated.sharing_rules)
        return True

    def remove_rule(self, sharing_key: str) -> bool:
        retained = tuple(rule for rule in self.rules if rule.sharing_key != sharing_key)
        if len(retained) == len(self.rules):
            return False
        return self.apply_rules(retained)

    def eligible_names(self, dataset_ids) -> tuple[str, ...]:
        identifiers = tuple(dataset_ids)
        if len(identifiers) < 2:
            return ()
        definition_sets = tuple(self._available_names(dataset_id) for dataset_id in identifiers)
        common = set.intersection(*definition_sets)
        return tuple(sorted(common))

    def _available_names(self, dataset_id: str) -> set[str]:
        owned = {
            member.parameter_name for rule in self.rules for member in rule.members if member.dataset_id == dataset_id
        }
        return {
            definition.name
            for definition in api.describe_parameters(self.document.project, dataset_id)
            if not definition.locked and not definition.constrained and definition.name not in owned
        }

    def _add_rule(self) -> None:
        dialog = SharingDialog(
            tuple(dataset.dataset_id for dataset in self.document.project.datasets),
            self.eligible_names,
            self._commit_rule,
            self,
        )
        dialog.exec()
        dialog.deleteLater()

    def _commit_rule(self, rule: api.SharingRule) -> None:
        self.apply_rules((*self.rules, rule))

    def _delete_selected_rule(self) -> None:
        selected = self.tree.currentItem()
        if selected is None:
            self.error_label.setText("请先选择要删除的共享组")
            self.error_label.show()
            return
        try:
            self.remove_rule(selected.data(0, Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return

    def error_text(self) -> str:
        return self.error_label.text()

    def _render(self, *_args) -> None:
        self.tree.clear()
        for rule in self.rules:
            members = ", ".join(f"{member.dataset_id}:{member.parameter_name}" for member in rule.members)
            item = QTreeWidgetItem((rule.sharing_key, members))
            item.setData(0, Qt.ItemDataRole.UserRole, rule.sharing_key)
            self.tree.addTopLevelItem(item)
