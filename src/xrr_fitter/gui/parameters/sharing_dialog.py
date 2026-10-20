"""Atomic user input for one explicit cross-dataset sharing declaration."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.accessibility import localize_standard_buttons


class SharingDialog(QDialog):
    """Validate and commit before closing, so rejected input remains editable."""

    def __init__(
        self,
        dataset_ids: tuple[str, ...],
        eligible_names: Callable[[tuple[str, ...]], tuple[str, ...]],
        commit_rule: Callable[[api.SharingRule], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sharingDialog")
        self.setWindowTitle("添加参数共享组")
        self.setAccessibleName("添加参数共享组")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._eligible_names = eligible_names
        self._commit_rule = commit_rule
        self.key_input = QLineEdit()
        self.key_input.setObjectName("sharingKeyInput")
        self.key_input.setAccessibleName("共享组名称")
        self.key_input.setPlaceholderText("例如 shared-thickness")
        self.parameter_input = QComboBox()
        self.parameter_input.setObjectName("sharingParameterInput")
        self.parameter_input.setAccessibleName("共享参数")
        self.dataset_list = self._dataset_list(dataset_ids)
        self.error_label = QLabel()
        self.error_label.setObjectName("sharingDialogError")
        self.error_label.setAccessibleName("共享组校验错误")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self._arrange()
        self.dataset_list.itemChanged.connect(self._refresh_parameters)
        self._refresh_parameters()

    def _dataset_list(self, dataset_ids: tuple[str, ...]) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setObjectName("sharingDatasetList")
        tree.setAccessibleName("选择共享数据集，至少两个")
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(False)
        for dataset_id in dataset_ids:
            item = QTreeWidgetItem((dataset_id,))
            item.setData(0, Qt.ItemDataRole.UserRole, dataset_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            tree.addTopLevelItem(item)
        return tree

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("共享组名称", self.key_input)
        form.addRow("共同自由参数", self.parameter_input)
        hint = QLabel("勾选至少两个数据集。已固定、被约束或已属于共享组的参数不会列出。")
        hint.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.setObjectName("sharingDialogButtons")
        localize_standard_buttons(self.buttons, confirm="添加共享")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self.dataset_list)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _selected_ids(self) -> tuple[str, ...]:
        items = (self.dataset_list.topLevelItem(row) for row in range(self.dataset_list.topLevelItemCount()))
        return tuple(
            item.data(0, Qt.ItemDataRole.UserRole) for item in items if item.checkState(0) == Qt.CheckState.Checked
        )

    def _refresh_parameters(self, *_args) -> None:
        previous = self.parameter_input.currentData()
        self.parameter_input.clear()
        try:
            names = self._eligible_names(self._selected_ids())
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._show_error(str(error))
            return
        for name in names:
            self.parameter_input.addItem(name, name)
        index = self.parameter_input.findData(previous)
        if index >= 0:
            self.parameter_input.setCurrentIndex(index)
        self.error_label.hide()

    def _rule(self) -> api.SharingRule:
        key = self.key_input.text().strip()
        if not key:
            raise ValueError("共享组名称不能为空")
        identifiers = self._selected_ids()
        if len(identifiers) < 2:
            raise ValueError("请至少选择两个数据集")
        name = self.parameter_input.currentData()
        if name is None:
            raise ValueError("选中的数据集没有共同的可共享自由参数")
        return api.SharingRule(key, tuple(api.ParameterReference(identifier, name) for identifier in identifiers))

    def _accept_fields(self) -> None:
        try:
            self._commit_rule(self._rule())
        except (TypeError, ValueError) as error:
            self._show_error(str(error))
            return
        self.error_label.hide()
        self.accept()

    def _show_error(self, text: str) -> None:
        self.error_label.setText(text)
        self.error_label.show()
