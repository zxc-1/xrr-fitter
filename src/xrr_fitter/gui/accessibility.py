"""Accessible metadata, keyboard order, and deterministic focus recovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QDialogButtonBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class AccessibilitySpec:
    object_name: str
    accessible_name: str
    tooltip: str
    description: str = ""


ACCESSIBILITY_SPECS = (
    # Dock titles carry their own accessible names from window_layout; these
    # specs describe the panels the docks contain.
    AccessibilitySpec("dataPanel", "数据与掩膜", "导入数据并管理拟合掩膜", "数据集、数据源和拟合范围"),
    AccessibilitySpec("structurePanel", "样品结构", "编辑活动数据集的样品结构", "样品层、周期块和氧化层建议"),
    AccessibilitySpec("structureEditor", "结构编辑", "编辑样品层和周期结构", "添加、删除和排序结构组件"),
    AccessibilitySpec("plotPanel", "反射率、SLD 与拟合诊断", "查看曲线和诊断图", "原始曲线、残差、SLD 和候选解诊断"),
    AccessibilitySpec("parametersPanel", "参数与共享", "编辑拟合参数和共享规则", "参数边界、专家设置和共享关系"),
    AccessibilitySpec("parameterTable", "拟合参数与边界表", "逐行查看和编辑参数初值、边界、拟合状态与共享关系"),
    AccessibilitySpec("fitPanel", "拟合控制", "启动、取消和监视拟合", "批次模式、拟合进度和操作状态"),
    AccessibilitySpec("resultsPanel", "拟合结果", "查看候选解和不确定度", "候选解、证据和专家 MCMC"),
    AccessibilitySpec("newProjectButton", "新建项目", "创建新的空项目"),
    AccessibilitySpec("openProjectButton", "打开项目", "打开已有 XRR 项目"),
    AccessibilitySpec("saveProjectButton", "保存项目", "保存当前 XRR 项目"),
    AccessibilitySpec("saveAsProjectButton", "项目另存为", "将当前项目保存到新位置并重定位相对数据源"),
    AccessibilitySpec("reloadSourceButton", "重新加载活动数据源", "从当前路径重新读取活动数据集并核对哈希"),
    AccessibilitySpec("relinkSourceButton", "重新链接活动数据源", "为活动数据集选择新源文件并核对哈希"),
    AccessibilitySpec("importFilesButton", "导入文件", "选择一个或多个 XRR 数据文件并确认导入设置"),
    AccessibilitySpec("importFolderButton", "导入文件夹", "选择 XRR 数据文件夹并确认批量导入设置"),
    AccessibilitySpec("datasetTree", "数据集列表", "使用键盘选择活动数据集"),
    AccessibilitySpec("initializeStructureButton", "初始化结构", "建立 Air、空层栈和 Si 基底"),
    AccessibilitySpec("startFitButton", "开始一键拟合", "运行当前项目的拟合工作流"),
    AccessibilitySpec("cancelFitButton", "取消拟合", "请求取消当前拟合"),
    AccessibilitySpec("forceStopFitButton", "强制停止拟合", "强制终止当前拟合进程"),
    AccessibilitySpec("mcmcButton", "运行专家 MCMC", "对当前候选解运行显式专家 MCMC"),
    AccessibilitySpec(
        "openUncertaintyDialogButton",
        "打开不确定度分析",
        "对当前候选解运行专家 MCMC 采样",
        "在独立窗口中配置并运行 MCMC 采样",
    ),
    AccessibilitySpec("cancelMcmcButton", "取消 MCMC", "请求取消当前 MCMC"),
    AccessibilitySpec("forceStopMcmcButton", "强制停止 MCMC", "强制终止当前 MCMC 进程"),
    AccessibilitySpec("mcmcWalkers", "MCMC walkers 数", "设置 MCMC walkers 数量"),
    AccessibilitySpec("mcmcBurnIn", "MCMC burn-in 步数", "设置 MCMC burn-in 步数"),
    AccessibilitySpec("mcmcProduction", "MCMC production 步数", "设置 MCMC production 步数"),
    AccessibilitySpec("mcmcThin", "MCMC thinning 间隔", "设置 MCMC thinning 间隔"),
    AccessibilitySpec("addLayerButton", "添加普通层", "在当前结构末尾添加普通层"),
    AccessibilitySpec("addPeriodicBlockButton", "添加周期块", "在当前结构末尾添加周期块"),
    AccessibilitySpec("removeComponentButton", "删除结构组件", "删除当前选中的结构组件"),
    AccessibilitySpec("moveComponentUpButton", "上移结构组件", "将当前结构组件上移"),
    AccessibilitySpec("moveComponentDownButton", "下移结构组件", "将当前结构组件下移"),
    AccessibilitySpec("oxideSuggestionButton", "接受氧化层建议", "应用当前自然氧化层建议"),
    AccessibilitySpec("oxideSuggestionRefuseButton", "忽略氧化层建议", "记录并隐藏当前氧化层建议"),
    AccessibilitySpec("twoThetaColumnEditor", "2θ 数据列", "设置 2θ 数据的列号"),
    AccessibilitySpec("intensityColumnEditor", "强度数据列", "设置强度数据的列号"),
    AccessibilitySpec("intensitySigmaEnabled", "启用强度不确定度列", "包含强度不确定度数据列"),
    AccessibilitySpec("intensitySigmaColumnEditor", "强度不确定度数据列", "设置强度不确定度的列号"),
    AccessibilitySpec("resolutionEnabled", "启用分辨率列", "包含逐点分辨率数据列"),
    AccessibilitySpec("resolutionColumnEditor", "分辨率数据列", "设置分辨率数据的列号"),
    AccessibilitySpec("resolutionKindEditor", "分辨率数据类型", "选择分辨率列的单位和定义"),
    AccessibilitySpec("recursiveFolderImportCheck", "递归导入子目录", "同时扫描所选文件夹的子目录"),
    AccessibilitySpec("monochromaticWavelengthEditor", "单色波长", "设置单色光源波长，单位 Å"),
    AccessibilitySpec("mixedWavelength1Editor", "混合 Kα 第一波长", "设置 Kα1 波长，单位 Å"),
    AccessibilitySpec("mixedWavelength2Editor", "混合 Kα 第二波长", "设置 Kα2 波长，单位 Å"),
    AccessibilitySpec("mixedIntensityRatioEditor", "混合 Kα 强度比", "设置 I2/I1 强度比"),
    AccessibilitySpec("instrumentIdEditor", "仪器标识", "设置可选的仪器标识"),
    AccessibilitySpec("footprintModeEditor", "足迹修正模式", "选择几何、拟合或无足迹修正"),
    AccessibilitySpec("sampleLengthEditor", "样品长度", "设置样品长度，单位 mm"),
    AccessibilitySpec("beamWidthEditor", "光束宽度", "设置光束宽度，单位 mm"),
    AccessibilitySpec("backgroundModelEditor", "背景模型", "选择拟合背景模型"),
    AccessibilitySpec("resolutionDomainEditor", "分辨率域", "选择 q 或 θ 分辨率域"),
    AccessibilitySpec("columnMappingButton", "高级列映射", "为特殊多列源文件配置各数据列的含义"),
    AccessibilitySpec("emptyStateImportButton", "导入数据文件", "选择反射率数据文件并确认导入设置"),
    AccessibilitySpec("datasetDetails", "活动数据集详情", "显示活动数据集的源文件、光路、仪器与校验摘要"),
    AccessibilitySpec("fitReadinessStatus", "拟合就绪状态", "显示当前一键拟合的就绪状态"),
    AccessibilitySpec("activeDatasetStatus", "活动数据集", "显示当前活动数据集名称"),
    AccessibilitySpec("mainToolbar", "主工具栏", "项目命令、拟合控制与导出入口"),
)


FOCUS_ORDER = (
    "newProjectButton",
    "openProjectButton",
    "saveProjectButton",
    "saveAsProjectButton",
    "reloadSourceButton",
    "relinkSourceButton",
    "importFilesButton",
    "importFolderButton",
    "datasetTree",
    "initializeStructureButton",
    "structureTree",
    # The plot toolbar's own row, in the order it is read on screen. Listing
    # only the first button used to hand focus straight from it to the export
    # command, so a keyboard user could reach "查看" but never the range, mask,
    # navigation or zoom controls sitting beside it.
    "plotModeView",
    "plotModeRange",
    "plotModeMask",
    "plotNavPan",
    "plotNavZoom",
    "plotNavHome",
    "plotZoomToRange",
    "plotResetZoom",
    "diagnosticTabs",
    "expertModeToggle",
    "parameterTable",
    "batchModeSelector",
    "startFitButton",
    "cancelFitButton",
    "forceStopFitButton",
    "candidateList",
    "uncertaintyEvidence",
    # The MCMC inputs moved into an on-demand dialog, so the main window's tab
    # chain stops at the entry point; the dialog owns its own internal order.
    "openUncertaintyDialogButton",
    "clearResultsButton",
    "exportResultsButton",
)


def _named_widgets(root: QWidget, object_name: str) -> tuple[QWidget, ...]:
    values = list(root.findChildren(QWidget, object_name))
    if root.objectName() == object_name:
        values.insert(0, root)
    return tuple(values)


def _named_widget(root: QWidget, object_name: str) -> QWidget | None:
    values = _named_widgets(root, object_name)
    return None if not values else values[0]


def _apply_spec(root: QWidget, spec: AccessibilitySpec) -> QWidget | None:
    widget = _named_widget(root, spec.object_name)
    if widget is None:
        return None
    widget.setAccessibleName(spec.accessible_name)
    widget.setToolTip(spec.tooltip)
    if spec.description:
        widget.setAccessibleDescription(spec.description)
    return widget


def _standard_button_name(root: QWidget, standard: QDialogButtonBox.StandardButton) -> str:
    if standard == QDialogButtonBox.StandardButton.Cancel:
        return "取消"
    if standard == QDialogButtonBox.StandardButton.Close:
        return "关闭"
    if root.objectName() == "importDialog":
        return "确认导入"
    if root.objectName() == "columnMappingDialog":
        return "确认列映射"
    return "确认"


def _configure_dialog_buttons(root: QWidget) -> None:
    standards = (
        QDialogButtonBox.StandardButton.Ok,
        QDialogButtonBox.StandardButton.Cancel,
        QDialogButtonBox.StandardButton.Close,
    )
    for box in root.findChildren(QDialogButtonBox):
        for standard in standards:
            button = box.button(standard)
            if button is not None:
                name = _standard_button_name(root, standard)
                button.setAccessibleName(name)
                button.setToolTip(name)


def _configure_beam_buttons(root: QWidget) -> None:
    names = {"单色": "单色光路", "混合 Kα": "混合 Kα 光路"}
    for button in root.findChildren(QRadioButton):
        name = names.get(button.text())
        if name is not None:
            button.setAccessibleName(name)
            button.setToolTip(f"选择{name}")


def _configure_parameter_tables(root: QWidget) -> None:
    for table in _named_widgets(root, "parameterTable"):
        if not isinstance(table, QTableWidget):
            continue
        blocker = QSignalBlocker(table)
        table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for row in range(table.rowCount()):
            for column in range(1, min(table.columnCount(), 5)):
                item = table.item(row, column)
                if item is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        del blocker


def configure_accessibility(root: QWidget) -> tuple[QWidget, ...]:
    """Apply stable names without depending on concrete widget classes."""
    configured = tuple(widget for spec in ACCESSIBILITY_SPECS if (widget := _apply_spec(root, spec)) is not None)
    _configure_dialog_buttons(root)
    _configure_beam_buttons(root)
    _configure_parameter_tables(root)
    return configured


def accessible_error_text(path: str | Path, error: BaseException) -> str:
    return f"{Path(path)}\n{type(error).__name__}: {error}"


def _copy_text(editor: QPlainTextEdit) -> None:
    QApplication.clipboard().setText(editor.toPlainText())


def create_copy_button(
    editor: QPlainTextEdit,
    parent: QWidget | None = None,
) -> QPushButton:
    button = QPushButton("复制", parent)
    button.setObjectName("copyEvidenceButton")
    button.setAccessibleName("复制拟合证据")
    button.setToolTip("将当前候选解的证据文本复制到剪贴板")
    button.clicked.connect(partial(_copy_text, editor))
    return button


def configure_focus_navigation(root: QWidget) -> tuple[QWidget, ...]:
    QApplication.styleHints().setTabFocusBehavior(Qt.TabFocusBehavior.TabFocusAllControls)
    ordered = tuple(widget for name in FOCUS_ORDER if (widget := _named_widget(root, name)) is not None)
    for first, second in zip(ordered[:-1], ordered[1:], strict=True):
        QWidget.setTabOrder(first, second)
    return ordered


def focus_named(root: QWidget, object_name: str) -> QWidget:
    candidates = _named_widgets(root, object_name)
    if not candidates:
        raise LookupError(f"focus target is missing: {object_name}")
    target = next(
        (widget for widget in reversed(candidates) if widget.isEnabled() and widget.isVisibleTo(root)),
        candidates[-1],
    )
    target.setFocus(Qt.FocusReason.OtherFocusReason)
    return target


def run_with_error_focus(
    root: QWidget,
    object_name: str,
    operation: Callable[[], object],
) -> object:
    try:
        return operation()
    except Exception:
        focus_named(root, object_name)
        raise


@dataclass(frozen=True, slots=True)
class _ScrollSnapshot:
    widget: QAbstractScrollArea
    horizontal: int
    vertical: int


def _scroll_snapshots(root: QWidget) -> tuple[_ScrollSnapshot, ...]:
    return tuple(
        _ScrollSnapshot(
            widget,
            widget.horizontalScrollBar().value(),
            widget.verticalScrollBar().value(),
        )
        for widget in root.findChildren(QAbstractScrollArea)
    )


def _restore_scroll(snapshots: tuple[_ScrollSnapshot, ...]) -> None:
    for snapshot in snapshots:
        snapshot.widget.horizontalScrollBar().setValue(snapshot.horizontal)
        snapshot.widget.verticalScrollBar().setValue(snapshot.vertical)


def preserve_focus(root: QWidget, operation: Callable[[], object]) -> object:
    focused = QApplication.focusWidget()
    object_name = "" if focused is None else focused.objectName()
    scroll = _scroll_snapshots(root)
    try:
        return operation()
    finally:
        if object_name:
            focus_named(root, object_name)
        _restore_scroll(scroll)


def focus_layer_error(dialog: QWidget, error: ValueError) -> QWidget:
    message = str(error)
    targets = (
        (("unknown element", "material formula"), "layerFormulaInput"),
        (("bulk density",), "layerDensityInput"),
        (("thickness_a",), "layerThicknessInput"),
        (("roughness_a",), "layerRoughnessInput"),
    )
    for fragments, name in targets:
        if any(fragment in message for fragment in fragments):
            return focus_named(dialog, name)
    return focus_named(dialog, "layerNameInput")


def focus_periodic_error(dialog: QWidget, row: int | None) -> QWidget:
    table = _named_widget(dialog, "periodicLayerTable")
    if not isinstance(table, QTableWidget) or row is None:
        return focus_named(dialog, "periodicNameInput")
    editor = table.cellWidget(row, 1)
    if editor is not None:
        editor.setFocus(Qt.FocusReason.OtherFocusReason)
        return editor
    table.setCurrentCell(row, 1)
    table.setFocus(Qt.FocusReason.OtherFocusReason)
    return table


def _active_mapping_editors(dialog: QWidget) -> tuple[QWidget, ...]:
    names = ["twoThetaColumnEditor", "intensityColumnEditor"]
    optional = (
        ("intensitySigmaEnabled", "intensitySigmaColumnEditor"),
        ("resolutionEnabled", "resolutionColumnEditor"),
    )
    for check_name, editor_name in optional:
        check = _named_widget(dialog, check_name)
        if check is not None and bool(check.isChecked()):
            names.append(editor_name)
    return tuple(focus for name in names if (focus := _named_widget(dialog, name)) is not None)


def _focus_duplicate_mapping(dialog: QWidget) -> QWidget:
    editors = _active_mapping_editors(dialog)
    seen: set[int] = set()
    for editor in editors:
        value = int(editor.value())
        if value in seen:
            editor.setFocus(Qt.FocusReason.OtherFocusReason)
            return editor
        seen.add(value)
    if not editors:
        raise LookupError("column mapping has no active editors")
    editors[0].setFocus(Qt.FocusReason.OtherFocusReason)
    return editors[0]


def _mapping_accept_finished(dialog: QWidget) -> None:
    error = _named_widget(dialog, "columnMappingError")
    if error is not None and error.isVisible():
        _focus_duplicate_mapping(dialog)


def configure_column_mapping_focus(dialog: QWidget) -> None:
    if bool(dialog.property("xrrColumnMappingFocusConfigured")):
        return
    boxes = dialog.findChildren(QDialogButtonBox)
    if len(boxes) != 1:
        raise LookupError("column mapping requires one button box")
    boxes[0].accepted.connect(partial(_mapping_accept_finished, dialog))
    dialog.setProperty("xrrColumnMappingFocusConfigured", True)
