"""Result export workflow and bounded, scrollable completion summary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument


EXPECTED_EXPORT_ERRORS = (OSError, TypeError, ValueError, RuntimeError)


def export_summary(manifest: api.ExportManifest) -> str:
    """Render every manifest record as its actual published path."""
    dataset_ids = ", ".join(item.dataset_id for item in manifest.datasets)
    files = "\n".join(
        str(manifest.run_directory / record.path)
        for record in manifest.files
    )
    return f"导出完成：{manifest.run_directory}\n数据集：{dataset_ids}\n{files}"


class ExportSummaryDialog(QDialog):
    """Show a potentially long immutable export manifest without clipping."""

    def __init__(self, summary: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("exportSummaryDialog")
        self.setWindowTitle("导出完成")
        self.setAccessibleName("导出完成")
        self.setMinimumSize(640, 360)
        text = QPlainTextEdit(summary)
        text.setObjectName("exportSummaryText")
        text.setAccessibleName("导出文件清单")
        text.setReadOnly(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.setObjectName("exportSummaryButtons")
        close_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        close_button.setAccessibleName("关闭导出摘要")
        close_button.setToolTip("关闭导出文件清单")
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(text)
        layout.addWidget(buttons)


class ExportWorkflow:
    """Keep the latest successful summary while delegating export to the API."""

    def __init__(
        self,
        document: ProjectDocument,
        *,
        is_running: Callable[[], bool],
    ) -> None:
        self._document = document
        self._is_running = is_running
        self._summary_text = ""
        self._manifest: api.ExportManifest | None = None

    @property
    def summary_text(self) -> str:
        return self._summary_text

    @property
    def manifest(self) -> api.ExportManifest | None:
        return self._manifest

    def export_results(self, directory: str | Path) -> api.ExportManifest:
        if self._is_running():
            raise RuntimeError("cannot export a project while an operation is running")
        manifest = api.export_result(self._document.project, Path(directory))
        summary = export_summary(manifest)
        self._manifest = manifest
        self._summary_text = summary
        return manifest

    def export_results_dialog(
        self,
        parent: QWidget | None,
    ) -> api.ExportManifest | None:
        name = QFileDialog.getExistingDirectory(parent, "导出拟合结果")
        if not name:
            return None
        destination = Path(name)
        try:
            manifest = self.export_results(destination)
        except EXPECTED_EXPORT_ERRORS as error:
            QMessageBox.critical(
                parent,
                "导出失败",
                f"目标目录：{destination}\n{type(error).__name__}: {error}\n"
                "请检查目标目录的写入权限和可用空间后重试。",
            )
            return None
        ExportSummaryDialog(self._summary_text, parent).exec()
        return manifest
