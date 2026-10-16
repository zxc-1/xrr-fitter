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

#: 服务层与模型层对共享规则的拒绝原文 → 屏上那句话。认不出的原样上屏：编一个原因比露出原文更糟。
SHARING_ERROR_TEXTS = {
    "sharing_key values must be unique": "共享组名称重复，每个共享组需要唯一的名称",
    "sharing coordinate has multiple ownership": "同一个参数不能同时属于两个共享组",
    "sharing rule references a missing dataset": "共享组引用了不存在的数据集",
}
CONSTRAINT_CONFLICT_PREFIX = "parameters are driven by both a sharing and a constraint rule: "

#: (参数名, 显示名) 按显示名列出；参数名留作数据，显示名给读者。
Choices = tuple[tuple[str, str], ...]


def sharing_error_text(error: BaseException, key: str | None = None) -> str:
    """把一次共享规则拒绝说成中文；``key`` 是对话框里正要添加的组名。"""
    message = str(error)
    if message == "sharing_key values must be unique" and key:
        return f"共享组名称「{key}」已存在，请换一个名称"
    if message.startswith(CONSTRAINT_CONFLICT_PREFIX):
        return f"这些参数已由约束驱动，不能再共享：{message[len(CONSTRAINT_CONFLICT_PREFIX) :]}"
    return SHARING_ERROR_TEXTS.get(message, message)


class SharingDialog(QDialog):
    """Validate and commit before closing, so rejected input remains editable."""

    def __init__(
        self,
        dataset_ids: tuple[str, ...],
        eligible_choices: Callable[[tuple[str, ...]], Choices],
        commit_rule: Callable[[api.SharingRule], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sharingDialog")
        self.setWindowTitle("添加参数共享组")
        self.setAccessibleName("添加参数共享组")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._eligible_choices = eligible_choices
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
            choices = self._eligible_choices(self._selected_ids())
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._show_error(str(error))
            return
        for name, label in choices:
            self.parameter_input.addItem(label, name)
            # 两层同名时显示名会重复；原名放在悬停里，读者仍分得清是哪一个。
            self.parameter_input.setItemData(self.parameter_input.count() - 1, name, Qt.ItemDataRole.ToolTipRole)
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
            self._show_error(sharing_error_text(error, self.key_input.text().strip()))
            return
        self.error_label.hide()
        self.accept()

    def _show_error(self, text: str) -> None:
        self.error_label.setText(text)
        self.error_label.show()
