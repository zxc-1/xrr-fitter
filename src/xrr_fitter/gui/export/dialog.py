"""Result export workflow and bounded, scrollable completion summary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
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

# 产出 .ort 时追加到摘要末尾：置信度/可复现种子/模型曲线不属于 ORSO 标准反射率列，
# 分装在三个冻结扩展命名空间里（与 ``io/orso.py`` 的 ``_extensions`` 逐字对应）。
ORT_EXTENSION_NOTE = (
    "已随每个数据集导出 ORSO .ort。拟合置信度、可复现种子与拟合配置、模型曲线与残差"
    "不属于 ORSO 标准反射率列，分别封装在扩展命名空间 xrr_fitter.confidence、"
    "xrr_fitter.reduction、xrr_fitter.model 中。"
)
# 缺逐参数 sigma 时无法派生协方差；受影响数据集的 .ort 会在 ``xrr_fitter.confidence``
# 段以 ``covariance_absent_reason`` 字段记录缺席原因，摘要在此如实披露。
COVARIANCE_ABSENT_NOTE = (
    "注意：本次拟合缺少逐参数标准差，无法派生协方差矩阵；受影响数据集的 .ort 已在 "
    "xrr_fitter.confidence 段以 covariance_absent_reason 字段记录该缺席原因。"
)


def _covariance_absent(project: api.XrrProject) -> bool:
    """镜像 ``services.exports`` 的判定：任一导出数据集缺协方差即为真。"""
    for dataset in project.datasets:
        result = dataset.last_valid_result
        if result is None:
            continue
        report = result.uncertainty
        covariance = None if report is None else report.covariance
        if covariance is None:
            return True
    return False


def export_summary(manifest: api.ExportManifest) -> str:
    """Render every manifest record as its actual published path."""
    dataset_ids = ", ".join(item.dataset_id for item in manifest.datasets)
    files = "\n".join(str(manifest.run_directory / record.path) for record in manifest.files)
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


class OrtOptionDialog(QDialog):
    """Ask, before choosing a directory, whether to also emit ORSO ``.ort`` files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ortOptionDialog")
        self.setWindowTitle("导出选项")
        self.setAccessibleName("导出选项")
        checkbox = QCheckBox("额外产出 ORSO .ort 文件")
        checkbox.setObjectName("ortOptionCheckbox")
        checkbox.setAccessibleName("额外产出 ORSO .ort 文件")
        checkbox.setToolTip("为每个数据集额外写出 ORSO 标准 .ort 文件（不勾选则导出内容与不支持 ORSO 时逐位一致）。")
        checkbox.setChecked(True)
        self._checkbox = checkbox
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.setObjectName("ortOptionButtons")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(checkbox)
        layout.addWidget(buttons)

    @property
    def include_ort(self) -> bool:
        return self._checkbox.isChecked()


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

    def export_results(
        self,
        directory: str | Path,
        *,
        include_ort: bool = False,
    ) -> api.ExportManifest:
        if self._is_running():
            raise RuntimeError("cannot export a project while an operation is running")
        manifest = api.export_result(self._document.project, Path(directory), include_ort=include_ort)
        summary = export_summary(manifest)
        if any(str(record.path).endswith(".ort") for record in manifest.files):
            summary = f"{summary}\n\n{ORT_EXTENSION_NOTE}"
            if _covariance_absent(self._document.project):
                summary = f"{summary}\n{COVARIANCE_ABSENT_NOTE}"
        self._manifest = manifest
        self._summary_text = summary
        return manifest

    def export_results_dialog(
        self,
        parent: QWidget | None,
    ) -> api.ExportManifest | None:
        option = OrtOptionDialog(parent)
        if option.exec() != QDialog.DialogCode.Accepted:
            return None
        include_ort = option.include_ort
        name = QFileDialog.getExistingDirectory(parent, "导出拟合结果")
        if not name:
            return None
        destination = Path(name)
        try:
            manifest = self.export_results(destination, include_ort=include_ort)
        except EXPECTED_EXPORT_ERRORS as error:
            QMessageBox.critical(
                parent,
                "导出失败",
                f"目标目录：{destination}\n{type(error).__name__}: {error}\n请检查目标目录的写入权限和可用空间后重试。",
            )
            return None
        ExportSummaryDialog(self._summary_text, parent).exec()
        return manifest
