"""Native file, confirmation, and error dialogs for project workflows."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

import xrr_fitter.api as api

EXPECTED_DIALOG_ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError)
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
