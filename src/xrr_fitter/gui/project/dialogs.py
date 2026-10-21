"""Native file, confirmation, and error dialogs for project workflows."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

import xrr_fitter.api as api

EXPECTED_DIALOG_ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError)


class UnsavedChoice(Enum):
    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


PROJECT_FILTER = "XRR 项目 (*.xrrproj.json *.json);;所有文件 (*)"
SAVE_FILTER = "XRR 项目 (*.xrrproj.json);;JSON (*.json)"
SOURCE_FILTER = "XRR 数据 (*.xy *.dat *.txt);;所有文件 (*)"


def choose_project_to_open(parent: QWidget) -> Path | None:
    name, _selected = QFileDialog.getOpenFileName(
        parent,
        "打开 XRR 项目",
        "",
        PROJECT_FILTER,
    )
    return None if not name else Path(name)


def choose_project_to_save(
    parent: QWidget,
    current_path: Path | None,
    *,
    save_as: bool,
) -> Path | None:
    title = "XRR 项目另存为" if save_as else "保存 XRR 项目"
    suggested = "workspace.xrrproj.json" if current_path is None else current_path.name
    name, _selected = QFileDialog.getSaveFileName(
        parent,
        title,
        suggested,
        SAVE_FILTER,
    )
    return None if not name else Path(name)


def choose_source_to_relink(parent: QWidget) -> Path | None:
    name, _selected = QFileDialog.getOpenFileName(
        parent,
        "重新链接数据源",
        "",
        SOURCE_FILTER,
    )
    return None if not name else Path(name)


def source_confirmation_text(preview: api.SourceUpdatePreview, record: object) -> str:
    actual = getattr(record, "actual_sha256", None) or "不可用"
    return (
        f"dataset: {preview.dataset_id}\n"
        f"旧路径：{preview.current_source_path}\n"
        f"旧 expected SHA-256：{preview.expected_sha256}\n"
        f"旧 actual SHA-256：{actual}\n"
        f"新路径：{preview.proposed_source_path}\n"
        f"新 SHA-256：{preview.observed_sha256}\n\n"
        "只有在明确确认后才会更新源声明。"
    )


def confirm_source_update(
    parent: QWidget,
    preview: api.SourceUpdatePreview,
    record: object,
    *,
    relink: bool,
) -> bool:
    title = "确认重新链接数据源" if relink else "确认重新加载数据源"
    response = QMessageBox.question(
        parent,
        title,
        source_confirmation_text(preview, record),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return response == QMessageBox.StandardButton.Yes


def unsaved_changes_box(parent: QWidget) -> QMessageBox:
    """The three-way question every route that drops the open project asks.

    The buttons are named after what they do rather than 是/否: a yes/no question
    could only offer to throw the changes away. Saving is the answer that loses
    nothing, so it is the default and Esc cancels.
    """
    box = QMessageBox(parent)
    box.setObjectName("unsavedChangesPrompt")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("未保存的项目更改")
    box.setText("当前项目有未保存的更改。要先保存吗？")
    box.setInformativeText("不保存的话，这些更改会丢失。")
    save = box.addButton("保存", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("不保存", QMessageBox.ButtonRole.DestructiveRole)
    cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(save)
    box.setEscapeButton(cancel)
    return box


def _answered_role(box: QMessageBox) -> QMessageBox.ButtonRole:
    box.exec()
    clicked = box.clickedButton()
    role = QMessageBox.ButtonRole.RejectRole if clicked is None else box.buttonRole(clicked)
    box.deleteLater()
    return role


def ask_unsaved_changes(parent: QWidget) -> UnsavedChoice:
    role = _answered_role(unsaved_changes_box(parent))
    if role == QMessageBox.ButtonRole.AcceptRole:
        return UnsavedChoice.SAVE
    if role == QMessageBox.ButtonRole.DestructiveRole:
        return UnsavedChoice.DISCARD
    return UnsavedChoice.CANCEL


def dropping_results_box(parent: QWidget, change: str, count: int) -> QMessageBox:
    box = QMessageBox(parent)
    box.setObjectName("dropResultsPrompt")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(change)
    box.setText(f"{change}会清除 {count} 个数据集的拟合结果与检查点。")
    box.setInformativeText("切换之后要重新拟合；清除的内容可在「编辑 ▸ 撤销」恢复。")
    box.addButton("清除并切换", QMessageBox.ButtonRole.DestructiveRole)
    keep = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(keep)
    box.setEscapeButton(keep)
    return box


def confirm_dropping_results(parent: QWidget, change: str, count: int) -> bool:
    """Ask before a settings switch that clears every dataset's fit work.

    Only the two whole-project switches ask (噪声模式, 批量模式): one click there
    throws away every result and checkpoint at once. Edits to one layer or range
    stay a single step and say what they cleared afterwards instead.
    """
    return _answered_role(dropping_results_box(parent, change, count)) == QMessageBox.ButtonRole.DestructiveRole


def stop_for_exit_box(parent: QWidget) -> QMessageBox:
    box = QMessageBox(parent)
    box.setObjectName("stopForExitPrompt")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("后台计算仍在运行")
    box.setText("退出前要先停止正在运行的计算。")
    box.setInformativeText("停止的效果与按「停止」相同；停下后，有未保存的更改还会再问一次是否保存。")
    box.addButton("停止并退出", QMessageBox.ButtonRole.DestructiveRole)
    keep = box.addButton("继续运行", QMessageBox.ButtonRole.RejectRole)
    # 默认留在运行：一次手滑的回车不该把跑了半小时的拟合停掉。
    box.setDefaultButton(keep)
    box.setEscapeButton(keep)
    return box


def confirm_stop_for_exit(parent: QWidget) -> bool:
    return _answered_role(stop_for_exit_box(parent)) == QMessageBox.ButtonRole.DestructiveRole


def show_workflow_error(
    parent: QWidget,
    title: str,
    error: BaseException,
    recovery: str,
) -> None:
    QMessageBox.critical(
        parent,
        title,
        f"{type(error).__name__}: {error}\n\n{recovery}",
    )


def show_removed_settings(
    parent: QWidget,
    dataset_id: str,
    removed: tuple[str, ...],
) -> None:
    if not removed:
        return
    names = "\n".join(f"• {name}" for name in removed)
    QMessageBox.information(
        parent,
        "数据源已更新",
        f"数据集 {dataset_id} 的新数据源已接受。以下不兼容参数设置已移除：\n{names}",
    )
