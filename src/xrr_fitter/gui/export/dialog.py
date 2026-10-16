"""Result export workflow and bounded, scrollable completion summary."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QAction, QCursor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.accessibility import localize_standard_buttons
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.progress import stage_text
from xrr_fitter.gui.sizing import ContentSizedScroll

EXPECTED_EXPORT_ERRORS = (OSError, TypeError, ValueError, RuntimeError)

# 帧⑥b 的 ``.dh``：设计稿把导入与导出画成并排的两张卡，各自靠头一行说明自己是哪一张。
# 窗口标题栏在设计稿里画的是应用外壳，对话框卡上没有它，所以这一行是唯一的自我介绍。
EXPORT_HEADING = "📤 导出结果"

# 产出 .ort 时追加到摘要末尾：置信度/可复现种子/模型曲线不属于 ORSO 标准反射率列，
# 分装在三个冻结扩展命名空间里（与 ``io/orso.py`` 的 ``_extensions`` 逐字对应）。
ORT_EXTENSION_NOTE = (
    "已随每个数据集导出 ORSO .ort。拟合置信度、可复现种子与拟合配置、模型曲线与残差"
    "不属于 ORSO 标准反射率列，分别封装在扩展命名空间 xrr_fitter.confidence、"
    "xrr_fitter.reduction、xrr_fitter.model 中。"
)
# 缺逐参数 sigma 或 uncertainty 不属于实际选中候选时，受影响数据集的 .ort 会在
# ``xrr_fitter.confidence`` 段以 ``covariance_absent_reason`` 记录原因，摘要在此披露。
COVARIANCE_ABSENT_NOTE = (
    "注意：本次导出的部分选中候选缺少可用协方差矩阵；受影响数据集的 .ort 已在 "
    "xrr_fitter.confidence 段以 covariance_absent_reason 字段记录具体原因。"
)


def _selected_candidate(result, selected_id: str | None):
    if selected_id is None:
        return result.best_candidate
    return next(
        (candidate for candidate in result.candidates if candidate.candidate_id == selected_id),
        None,
    )


def _selected_report(project: api.XrrProject, dataset: api.DatasetProject):
    """The uncertainty report the export attaches to ``dataset``, or ``None``.

    镜像 ``io.export_tables.DatasetExportData.selected_uncertainty``：报告只在属于实际选中
    的候选时才随导出发布，否则参数表里只剩 ``sigma_unavailable_reason``。
    """
    result = dataset.last_valid_result
    selected = _selected_candidate(result, dict(project.ui_state.selected_candidate_ids).get(dataset.dataset_id))
    report = result.uncertainty
    if selected is None or report is None or report.candidate_id != selected.candidate_id:
        return None
    return report


def _covariance_absent(project: api.XrrProject) -> bool:
    """镜像 ``services.exports`` 的判定：任一导出数据集缺协方差即为真。"""
    for dataset in project.datasets:
        if dataset.last_valid_result is None:
            continue
        report = _selected_report(project, dataset)
        if report is None or report.covariance is None:
            return True
    return False


def _datasets_without_interval(project: api.XrrProject) -> tuple[str, ...]:
    """Name the fitted datasets whose parameter table will carry no 16–84% interval."""
    return tuple(
        dataset.display_name
        for dataset in project.datasets
        if dataset.last_valid_result is not None and _selected_report(project, dataset) is None
    )


def export_unavailable_reason(project: api.XrrProject, *, running: bool) -> str | None:
    """Say why export is greyed out, naming the datasets still without a result.

    与 ``operation_state.has_exportable_results`` 同一判据：每个数据集都要有拟合结果、且没有
    运行在跑。返回 ``None`` 表示可以导出。
    """
    if running:
        return "拟合或 MCMC 正在运行，结束后才能导出。"
    if not project.datasets:
        return "项目里还没有数据集，先导入数据。"
    missing = [dataset.display_name for dataset in project.datasets if dataset.last_valid_result is None]
    if missing:
        return f"还没有拟合结果：{'、'.join(missing)}。每个数据集都拟合过才能导出。"
    return None


def show_export_unavailable_reason(entries: Sequence[QWidget | QAction], reason: str | None) -> None:
    """Append the greyed-out reason to each export entry's own tooltip, or take it back off.

    灰掉的按钮不说为什么，读者只能猜是哪一步没做。原提示存在属性里，每次从它重新拼，
    可用之后旧原因不会残留。
    """
    for entry in entries:
        base = entry.property("exportBaseToolTip")
        if base is None:
            base = entry.toolTip()
            entry.setProperty("exportBaseToolTip", base)
        text = base if reason is None else f"{base}\n{reason}"
        entry.setToolTip(text)
        if isinstance(entry, QAction):
            entry.setStatusTip(text)


def export_summary(manifest: api.ExportManifest) -> str:
    """Render every manifest record as its actual published path."""
    dataset_ids = ", ".join(item.dataset_id for item in manifest.datasets)
    files = "\n".join(
        f"{manifest.run_directory / record.path} ({record.size} bytes, sha256 {record.sha256})"
        for record in manifest.files
    )
    return f"导出完成：{manifest.run_directory}\n数据集：{dataset_ids}\n{files}"


class ExportSummaryDialog(QDialog):
    """Show a potentially long immutable export manifest without clipping."""

    def __init__(self, summary: str, parent: QWidget | None = None, *, run_directory: Path) -> None:
        super().__init__(parent)
        self._run_directory = Path(run_directory)
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
        # Every other dialog routes its footer through this; this one was missed
        # and read "Close" beside the option dialog's 「取消」 one step earlier.
        localize_standard_buttons(buttons)
        close_button.setAccessibleName("关闭导出摘要")
        close_button.setToolTip("关闭导出文件清单")
        # 清单里是一长串绝对路径；读者真正要的是去那个目录，而不是复制路径再自己找。
        open_button = QPushButton("打开文件夹")
        open_button.setObjectName("exportOpenFolderButton")
        open_button.setAccessibleName("打开导出所在文件夹")
        open_button.setToolTip(f"在文件管理器中打开 {self._run_directory}")
        open_button.clicked.connect(self._open_run_directory)
        buttons.addButton(open_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(text)
        layout.addWidget(buttons)

    def _open_run_directory(self) -> None:
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._run_directory))):
            QMessageBox.warning(self, "无法打开文件夹", f"系统没有打开这个目录：\n{self._run_directory}")


def _option_checkbox(
    name: str,
    label: str,
    tooltip: str,
    *,
    checked: bool = False,
    locked: bool = False,
) -> QCheckBox:
    """Build one export artifact switch, named for both tests and screen readers."""
    box = QCheckBox(label)
    box.setObjectName(name)
    box.setAccessibleName(label)
    box.setToolTip(tooltip)
    box.setChecked(checked)
    box.setEnabled(not locked)
    return box


def _locked_checkbox(name: str, label: str, artifact: str) -> QCheckBox:
    """Show an artifact every run publishes, checked but not offered as a choice.

    The three switches this dialog really owns are the optional formats on top of
    :data:`api.DEFAULT_FORMATS`; unchecking anything else would have to ask
    ``export_result`` for a tree without a format the default set publishes, or delete
    a provenance file the export contract promises unconditionally. So these read as
    state rather than as switches.
    """
    return _option_checkbox(name, label, f"{artifact}始终随每次导出发布，无法取消。", checked=True, locked=True)


def _interval_row(missing: tuple[str, ...]) -> QCheckBox:
    """The locked parameter-table row, promising an interval only where one exists.

    参数表每次都发布；16–84% 区间只在选中候选自己拥有不确定度报告时才有。缺报告的数据集
    在行上点名，不让锁定的勾替它们承诺。
    """
    if not missing:
        return _locked_checkbox("parameterTableCheckbox", "参数表与 16–84% 置信区间", "参数表与置信区间")
    names = "、".join(missing)
    tooltip = (
        f"参数表始终随每次导出发布，无法取消。{names} 的选中候选没有不确定度报告，"
        "参数表里不含 16–84% 区间，原因记在 sigma_unavailable_reason 列。"
    )
    return _option_checkbox(
        "parameterTableCheckbox", f"参数表（{names} 无置信区间）", tooltip, checked=True, locked=True
    )


class _OptionsScroll(ContentSizedScroll):
    """Ask for the options' own width as a floor, measured whenever the layout asks.

    横向滚动关着，窄过选项就直接裁字。地板每次现量：样式表落定后字体会变，构造时量的数会过时。
    """

    def minimumSizeHint(self) -> QSize:
        inner = self.widget()
        hint = super().minimumSizeHint()
        if inner is None:
            return hint
        width = inner.minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width()
        return QSize(max(hint.width(), width), hint.height())


def _option_group(name: str, title: str, boxes: tuple[QCheckBox, ...]) -> QGroupBox:
    group = QGroupBox(title)
    group.setObjectName(name)
    layout = QVBoxLayout(group)
    for box in boxes:
        layout.addWidget(box)
    return group


class OrtOptionDialog(QDialog):
    """Choose the optional export formats, and state what the run will conclude.

    The redesign groups nine artifacts into 数据格式 / 图件 / 随附内容. Only ``.ort``,
    CSV and SVG are real switches in :mod:`xrr_fitter.services.exports`; the other six
    are written by every publication and stay visible but locked, so the dialog never
    offers a toggle the export contract cannot honour.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        master_seed: int | None = None,
        project: api.XrrProject | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ortOptionDialog")
        self.setWindowTitle("导出结果")
        self.setAccessibleName("导出结果")
        if master_seed is None and project is not None:
            master_seed = project.master_seed
        self._master_seed = master_seed
        self._plan: api.ExportPlan | None = None if project is None else api.describe_export_plan(project)
        self._ort = _option_checkbox(
            "ortOptionCheckbox",
            "ORSO .ort — 规范反射率交换格式（推荐）",
            "为每个数据集额外写出 ORSO 标准 .ort 文件（不勾选则导出内容与不支持 ORSO 时逐位一致）。",
            checked=True,
        )
        self._csv = _option_checkbox(
            "csvOptionCheckbox",
            "CSV — 逐数据集纯文本",
            "为每个数据集额外写出纯文本参数表 CSV（不勾选则导出内容与不支持 CSV 时逐位一致）。",
        )
        self._svg = _option_checkbox(
            "sldSvgCheckbox",
            "SLD 深度剖面 SVG（矢量）",
            "SLD 深度剖面 PNG 每次都会发布；勾选后额外写出矢量 SVG（不勾选则导出内容与不支持 SVG 时逐位一致）。",
            checked=True,
        )
        formats = _option_group(
            "exportFormatsGroup",
            "数据格式",
            (
                self._ort,
                _locked_checkbox("xlsxOptionCheckbox", "Excel .xlsx — 参数表 + 不确定度 + 数据", "参数表 Excel"),
                self._csv,
            ),
        )
        plots = _option_group(
            "exportPlotsGroup",
            "图件",
            (
                _locked_checkbox("overviewPngCheckbox", "反射率图 PNG（300 dpi）", "反射率总览 PNG"),
                self._svg,
                _locked_checkbox("residualsPngCheckbox", "加权残差图 PNG", "加权残差 PNG"),
            ),
        )
        extras = _option_group(
            "exportExtrasGroup",
            "随附内容",
            (
                _interval_row(() if project is None else _datasets_without_interval(project)),
                _locked_checkbox("correlationCheckbox", "相关矩阵与收敛诊断", "相关矩阵与收敛诊断"),
                _locked_checkbox("manifestCheckbox", "拟合清单 manifest（含 master_seed）", "导出清单"),
            ),
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.setObjectName("ortOptionButtons")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # 确认按钮说出去向：点下去弹的是目录选择器，不是导出本身。
        localize_standard_buttons(buttons, confirm="导出到文件夹…")
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_heading())
        layout.addWidget(self._build_options_scroll((formats, plots, extras)))
        # 窗口拉得比内容高时，多出的高度垫在页脚之上，而不是摊进各组之间。
        layout.addStretch(1)
        layout.addWidget(buttons)
        for box in (self._ort, self._csv, self._svg):
            box.toggled.connect(self._refresh_manifest_preview)

    def _build_options_scroll(self, groups: tuple[QGroupBox, ...]) -> ContentSizedScroll:
        """Scroll the three groups and the preview; keep the heading and footer fixed.

        平铺时布局最小高约 800 px，比 1366×768 笔记本的可用高度还高，「导出到文件夹…」会落到
        屏幕外。滚动区按内容要高度：大屏一次铺开，小屏让内容滚动、页脚留在原处。
        """
        body = QWidget()
        body.setObjectName("exportOptionsBody")
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        for group in groups:
            column.addWidget(group)
        column.addWidget(self._build_manifest_caption())
        column.addWidget(self._build_manifest_preview())
        scroll = _OptionsScroll()
        scroll.setObjectName("exportOptionsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        return scroll

    def _build_heading(self) -> QLabel:
        """帧⑥b 的头一行，与导入对话框同一套 ``sectionHeader`` 样式。"""
        heading = QLabel(EXPORT_HEADING)
        heading.setObjectName("exportDialogHeading")
        heading.setProperty("sectionHeader", True)
        return heading

    def _build_manifest_caption(self) -> QLabel:
        """Name the file the preview stands for, using its published name.

        ``services.exports`` publishes ``export_manifest.json``, which is the name
        written here; the mock-up's ``run_manifest.json`` names no file the export
        produces.
        """
        caption = QLabel("清单预览 · export_manifest.json")
        caption.setObjectName("exportManifestCaption")
        caption.setAccessibleName("导出清单预览标题")
        return caption

    def _build_manifest_preview(self) -> QLabel:
        """Show the conclusions this export would reach, before anything is written.

        ``services.exports`` derives ``mode``/``stages``/``confidence``/``reproducible``
        on the way out. Reaching the same derivation through
        :func:`xrr_fitter.api.describe_export_plan` keeps the promise on screen and the
        record in the manifest from drifting apart.

        The payload is quoted as a file rather than set as prose: it is indented
        JSON, and proportional glyph widths would pull that indentation out of
        column -- which is the only reason it is pretty-printed at all.
        """
        preview = QLabel()
        preview.setObjectName("exportManifestPreview")
        preview.setAccessibleName("导出清单预览")
        theme.mark_code_block(preview)
        # 不折行：折行会把 20 位的 master_seed 拆成两行、排歪缩进，还让高度随宽度变，
        # 滚动区按内容要的高度就差出一行。宽度由滚动区的地板兜住。
        preview.setWordWrap(False)
        self._preview = preview
        self._refresh_manifest_preview()
        return preview

    def _refresh_manifest_preview(self, *_toggle: object) -> None:
        payload: dict[str, object] = {"app": "xrr-fitter", "master_seed": self._master_seed}
        plan = self._plan
        if plan is not None:
            payload["datasets"] = len(plan.dataset_ids)
            payload["mode"] = plan.mode
            # 阶段按拟合卡上的名字写（名字末尾保留原码），裸码 A/B/E 只对读过源码的人有意义。
            payload["stages"] = [stage_text(stage) for stage in plan.stages]
            payload["confidence"] = plan.confidence.value
            payload["reproducible"] = plan.reproducible
        # 预览里写的是 ``export_result`` 真正会收到的那个格式集合，而不是三个开关各自的
        # 真假值：锁定的默认格式也会发布，只报三个可选开关会让预览少说三种产物。
        payload["formats"] = [value.value for value in self.selected_formats]
        self._preview.setText(json.dumps(payload, ensure_ascii=False, indent=2))

    @property
    def include_ort(self) -> bool:
        return self._ort.isChecked()

    @property
    def include_csv(self) -> bool:
        return self._csv.isChecked()

    @property
    def include_svg(self) -> bool:
        return self._svg.isChecked()

    @property
    def selected_formats(self) -> tuple[api.ExportFormat, ...]:
        """The three checkboxes read as the format set ``export_result`` takes.

        This is the one place the dialog's boxes become an export request. The locked
        boxes are exactly :data:`api.DEFAULT_FORMATS`, which is why they are shown as
        state rather than as switches -- unchecking one would have to ask for a tree the
        default set does not describe.
        """
        optional = (
            (api.ExportFormat.ORT, self.include_ort),
            (api.ExportFormat.CSV, self.include_csv),
            (api.ExportFormat.SVG, self.include_svg),
        )
        return (*api.DEFAULT_FORMATS, *(value for value, checked in optional if checked))


@contextmanager
def _busy_export(parent: QWidget | None, dataset_count: int) -> Iterator[None]:
    """Show that the export is writing while the main thread is blocked on it.

    导出在主线程同步渲染图件与工作簿（约 1 s/数据集），期间事件循环停转。写盘前先把提示画出来、
    光标换成等待，读者看到的是「正在导出」，而不是一个不响应的窗口。发布是原子的，中途无法
    安全取消，所以提示框不带取消键。
    """
    busy = QProgressDialog(f"正在导出 {dataset_count} 个数据集…", None, 0, 0, parent)
    busy.setObjectName("exportBusyDialog")
    busy.setWindowTitle("导出结果")
    busy.setWindowModality(Qt.WindowModality.WindowModal)
    busy.setMinimumDuration(0)
    QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
    try:
        busy.show()
        # 同步写盘会冻住事件循环，先让提示框真的画出来。
        QApplication.processEvents()
        yield
    finally:
        QApplication.restoreOverrideCursor()
        busy.close()
        busy.deleteLater()


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
        formats: Sequence[api.ExportFormat] = api.DEFAULT_FORMATS,
    ) -> api.ExportManifest:
        if self._is_running():
            raise RuntimeError("cannot export a project while an operation is running")
        manifest = api.export_result(
            self._document.project,
            Path(directory),
            formats=formats,
        )
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
        option = OrtOptionDialog(parent, project=self._document.project)
        if option.exec() != QDialog.DialogCode.Accepted:
            return None
        formats = option.selected_formats
        name = QFileDialog.getExistingDirectory(parent, "导出拟合结果")
        if not name:
            return None
        destination = Path(name)
        try:
            with _busy_export(parent, len(self._document.project.datasets)):
                manifest = self.export_results(destination, formats=formats)
        except EXPECTED_EXPORT_ERRORS as error:
            QMessageBox.critical(
                parent,
                "导出失败",
                f"目标目录：{destination}\n{type(error).__name__}: {error}\n请检查目标目录的写入权限和可用空间后重试。",
            )
            return None
        ExportSummaryDialog(self._summary_text, parent, run_directory=manifest.run_directory).exec()
        return manifest
