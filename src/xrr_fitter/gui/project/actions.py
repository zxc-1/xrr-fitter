"""Concrete project command panel and workflow coordinator."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.project import dialogs


@dataclass(frozen=True, slots=True)
class _CommandSpec:
    text: str
    object_name: str
    callback_name: str
    accessible_name: str
    tooltip: str


BUTTON_SPECS = (
    _CommandSpec("新建", "newProjectButton", "new_project_dialog", "新建项目", "创建新的空项目"),
    _CommandSpec("打开", "openProjectButton", "open_project_dialog", "打开项目", "打开已有 XRR 项目"),
    _CommandSpec("保存", "saveProjectButton", "save_project_dialog", "保存项目", "保存当前 XRR 项目"),
    _CommandSpec(
        "另存为",
        "saveAsProjectButton",
        "save_project_as_dialog",
        "项目另存为",
        "将当前项目保存到新位置并重定位相对数据源",
    ),
    _CommandSpec(
        "重载源",
        "reloadSourceButton",
        "reload_source_dialog",
        "重新加载活动数据源",
        "从当前路径重新读取活动数据集并核对哈希",
    ),
    _CommandSpec(
        "重链接源",
        "relinkSourceButton",
        "relink_source_dialog",
        "重新链接活动数据源",
        "为活动数据集选择新源文件并核对哈希",
    ),
)

ACTION_SPECS = (
    ("newProjectAction", "新建项目", "new_project_dialog", QKeySequence.StandardKey.New),
    ("openProjectAction", "打开项目", "open_project_dialog", QKeySequence.StandardKey.Open),
    ("saveProjectAction", "保存项目", "save_project_dialog", QKeySequence.StandardKey.Save),
    ("saveAsProjectAction", "另存为", "save_project_as_dialog", QKeySequence.StandardKey.SaveAs),
)


class ProjectActions(QWidget):
    """Render project commands and coordinate only their API-backed workflows."""

    def __init__(self, owner: QWidget, document: ProjectDocument) -> None:
        super().__init__(owner)
        self.setObjectName("projectActions")
        self._owner = owner
        self._document = document
        self._buttons: dict[str, QPushButton] = {}
        self._build_buttons()
        self._build_actions()
        document.project_changed.connect(self.refresh)
        document.source_validation_changed.connect(self.refresh)
        self.refresh()

    def _build_buttons(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        for index, spec in enumerate(BUTTON_SPECS):
            button = QPushButton(spec.text)
            button.setObjectName(spec.object_name)
            button.setAccessibleName(spec.accessible_name)
            button.setToolTip(spec.tooltip)
            button.clicked.connect(
                lambda _checked=False, name=spec.callback_name: getattr(
                    self._owner,
                    name,
                )()
            )
            grid.addWidget(button, index // 2, index % 2)
            self._buttons[spec.object_name] = button
        layout.addLayout(grid)
        self.source_status_label = QLabel()
        self.source_status_label.setObjectName("sourceStatusLabel")
        self.source_status_label.setAccessibleName("活动数据源状态")
        self.source_status_label.setWordWrap(True)
        layout.addWidget(self.source_status_label)

    def _build_actions(self) -> None:
        for object_name, text, callback_name, shortcut in ACTION_SPECS:
            action = QAction(text, self._owner)
            action.setObjectName(object_name)
            action.setToolTip(text)
            action.setStatusTip(text)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(
                lambda _checked=False, name=callback_name: getattr(
                    self._owner,
                    name,
                )()
            )
            self._owner.addAction(action)

    def refresh(self, *_args) -> None:
        dataset_id = self._document.active_dataset_id
        enabled = dataset_id is not None
        self._buttons["reloadSourceButton"].setEnabled(enabled)
        self._buttons["relinkSourceButton"].setEnabled(enabled)
        if dataset_id is None:
            self.source_status_label.setText("当前没有活动数据集")
            return
        warning = self._document.source_warning(dataset_id)
        self.source_status_label.setText(warning or f"数据集 {dataset_id} 的源文件校验通过")

    def _may_replace_project(self) -> bool:
        return not self._document.is_dirty or self._owner._confirm_discard_changes()

    def new_project(self) -> None:
        if self._may_replace_project():
            self._owner.new_project(discard_unsaved=True)

    def open_project(self) -> None:
        if not self._may_replace_project():
            return
        target = dialogs.choose_project_to_open(self)
        if target is None:
            return
        try:
            self._owner.open_project(target, discard_unsaved=True)
        except dialogs.EXPECTED_DIALOG_ERRORS as error:
            dialogs.show_workflow_error(
                self,
                "打开项目失败",
                error,
                "请检查项目文件、读取权限和格式后重试。",
            )

    def save_project(self, *, save_as: bool) -> None:
        target = self._document.path
        if save_as or target is None:
            target = dialogs.choose_project_to_save(
                self,
                self._document.path,
                save_as=save_as,
            )
        if target is None:
            return
        title = "项目另存为失败" if save_as else "保存项目失败"
        try:
            self._owner.save_project(target)
        except dialogs.EXPECTED_DIALOG_ERRORS as error:
            dialogs.show_workflow_error(
                self,
                title,
                error,
                "请检查目标目录、写入权限和数据源状态后重试。",
            )

    def reload_source(self) -> None:
        self._source_update(None, relink=False)

    def relink_source(self) -> None:
        dataset_id = self._document.active_dataset_id
        if dataset_id is None:
            QMessageBox.critical(self, "重新链接数据源失败", "当前没有活动数据集")
            return
        replacement = dialogs.choose_source_to_relink(self)
        if replacement is not None:
            self._source_update(replacement, relink=True)

    def _source_update(self, replacement, *, relink: bool) -> None:
        dataset_id = self._document.active_dataset_id
        title = "重新链接数据源失败" if relink else "重新加载数据源失败"
        if dataset_id is None:
            QMessageBox.critical(self, title, "当前没有活动数据集")
            return
        try:
            preview = self._document.preview_source_update(dataset_id, replacement)
            record = self._document.source_record(dataset_id)
            if not dialogs.confirm_source_update(
                self,
                preview,
                record,
                relink=relink,
            ):
                return
            removed = self._document.accept_source_update(preview)
            dialogs.show_removed_settings(self, dataset_id, removed)
        except dialogs.EXPECTED_DIALOG_ERRORS as error:
            dialogs.show_workflow_error(
                self,
                title,
                error,
                "请检查数据源路径、读取权限和文件格式后重试。",
            )
