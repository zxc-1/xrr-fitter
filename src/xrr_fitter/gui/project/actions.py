"""Concrete project command panel and workflow coordinator."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from xrr_fitter.gui import status_bar, theme
from xrr_fitter.gui.command_icons import command_icon
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.project import dialogs


@dataclass(frozen=True, slots=True)
class _CommandSpec:
    text: str
    object_name: str
    callback_name: str
    accessible_name: str
    tooltip: str


# 命令栏只摆设计稿六张帧都画着的这三颗。另存为 / 重载源 / 重链接源 一次都没上过货架，
# 它们连同快捷键留在 文件 菜单（见 ``chrome._install_file_menu``）。
BUTTON_SPECS = (
    _CommandSpec("新建", "newProjectButton", "new_project_dialog", "新建项目", "创建新的空项目"),
    _CommandSpec("打开", "openProjectButton", "open_project_dialog", "打开项目", "打开已有 XRR 项目"),
    _CommandSpec("保存", "saveProjectButton", "save_project_dialog", "保存项目", "保存当前 XRR 项目"),
)

ACTION_SPECS = (
    ("newProjectAction", "新建项目", "new_project_dialog", QKeySequence.StandardKey.New),
    ("openProjectAction", "打开项目", "open_project_dialog", QKeySequence.StandardKey.Open),
    ("saveProjectAction", "保存项目", "save_project_dialog", QKeySequence.StandardKey.Save),
    ("saveAsProjectAction", "另存为", "save_project_as_dialog", QKeySequence.StandardKey.SaveAs),
)

# 设计稿帧① 底栏「源校验：<b>通过 ✓</b>」的取值。对勾跟着结论走而不是另画一个图标：
# 这一段在状态栏里只占一个词的宽度，图标反而要额外一格。
SOURCE_OK_TEXT = "通过 ✓"


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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        for spec in BUTTON_SPECS:
            button = QPushButton(spec.text)
            button.setObjectName(spec.object_name)
            button.setAccessibleName(spec.accessible_name)
            button.setToolTip(spec.tooltip)
            button.setProperty("commandBar", True)
            # 字形只挂在 ``_build_actions`` 建的那条 QAction 上：设计稿命令栏那三枚是
            # ``.btn sm`` 纯文字格，图标留给菜单——那里没有并排的三个字来互相区分。
            button.clicked.connect(
                lambda _checked=False, name=spec.callback_name: getattr(
                    self._owner,
                    name,
                )()
            )
            layout.addWidget(button)
            self._buttons[spec.object_name] = button
        self.source_status_label = QLabel()
        self.source_status_label.setObjectName("sourceStatusLabel")
        self.source_status_label.setAccessibleName("活动数据源状态")

    def button(self, object_name: str) -> QPushButton:
        return self._buttons[object_name]

    def _build_actions(self) -> None:
        for object_name, text, callback_name, shortcut in ACTION_SPECS:
            action = QAction(text, self._owner)
            action.setObjectName(object_name)
            action.setToolTip(text)
            action.setStatusTip(text)
            action.setShortcut(QKeySequence(shortcut))
            action.setIcon(command_icon(callback_name))
            action.triggered.connect(
                lambda _checked=False, name=callback_name: getattr(
                    self._owner,
                    name,
                )()
            )
            self._owner.addAction(action)

    def refresh(self, *_args) -> None:
        """源校验状态。

        重载源 / 重链接源 的可用性跟着活动数据集走，但那两条命令只在 文件 菜单里，
        由 ``chrome.refresh_status`` 一处开关，不在这里重复一遍。
        """
        dataset_id = self._document.active_dataset_id
        if dataset_id is None:
            self._show_source_status("", "", kind="")
            return
        warning = self._document.source_warning(dataset_id)
        if warning:
            self._show_source_status(warning.splitlines()[0], warning, kind="error")
            return
        # 说明文字「源校验：」住在状态栏那一段里（见 ``status_bar``），所以这里只报结论。
        self._show_source_status(SOURCE_OK_TEXT, "", kind="ok")

    def _show_source_status(self, text: str, tooltip: str, *, kind: str) -> None:
        self.source_status_label.setText(text)
        self.source_status_label.setToolTip(tooltip)
        theme.set_status_kind(self.source_status_label, kind)
        # 这一段没有结论可报时整段该藏起来，而藏不藏是段自己的事——文字一变就让状态栏重算。
        status_bar.sync(self._owner)

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
