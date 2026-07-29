"""API-backed joint-parameter sharing rule editor."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument


class SharingEditor(QWidget):
    """Validate and commit complete sharing-rule sequences atomically."""

    rules_changed = Signal(tuple)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("sharingEditor")
        self.tree = QTreeWidget()
        self.tree.setObjectName("sharingTree")
        self.tree.setHeaderLabels(("共享组", "成员"))
        self.error_label = QLabel()
        self.error_label.setObjectName("sharingValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        layout.addWidget(self.error_label)
        document.project_changed.connect(self._render)
        self._render()

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
        definition_sets = tuple(
            {
                definition.name
                for definition in api.describe_parameters(self.document.project, dataset_id)
                if not definition.locked
            }
            for dataset_id in identifiers
        )
        common = set.intersection(*definition_sets)
        return tuple(sorted(common))

    def error_text(self) -> str:
        return self.error_label.text()

    def _render(self, *_args) -> None:
        self.tree.clear()
        for rule in self.rules:
            members = ", ".join(
                f"{member.dataset_id}:{member.parameter_name}"
                for member in rule.members
            )
            self.tree.addTopLevelItem(QTreeWidgetItem((rule.sharing_key, members)))
