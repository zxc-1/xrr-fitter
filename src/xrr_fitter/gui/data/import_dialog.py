"""Explicit data-import, beam, instrument, and column-mapping dialogs."""

from __future__ import annotations

import logging
from math import asin, degrees
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.accessibility import localize_standard_buttons
from xrr_fitter.gui.noise import NOISE_MODE_LABELS, NOISE_MODE_REQUIREMENTS
from xrr_fitter.gui.sizing import CompactDoubleSpinBox, ContentSizedTable

LOG = logging.getLogger(__name__)

VALIDATION_TEXT = "请选择光路类型：单色 / 混合 Kα"
DIALOG_HEADING = "📥 导入数据 · 批量预览"
# 批量预览表：每个待导入文件一行，角度约定、列映射、行数与校验状态在确认前就摆在明面上。
# 状态一律"字形 + 文字"双编码（可访问性要求），坏文件保留成行并标红，而不是静默吞掉。
PREVIEW_HEADERS = ("文件", "角度", "列映射", "行数", "状态")
# 这一格说的是「这一次按哪个轴读的」，所以它跟着「角度约定」那两档走，而不是钉死在 2θ。
# 选错了轴，行数与「✓ 就绪」照样成立，导进来的角度却整段差一倍——写出来才看得见。
# 读不出来的文件连角度都无从谈起，写破折号。
PREVIEW_ANGLE_TWO_THETA = "2θ"
PREVIEW_ANGLE_THETA = "θ"
PREVIEW_UNKNOWN = "—"
PREVIEW_READY = "✓ 就绪"
# 解析成功但有效点不足 30（或 qz 不单调）——文件是好的，只是拟合用不了。既不能报「就绪」
# 骗人，也不该跟无数据列的坏文件同色，所以第三档信息色单独一行。
PREVIEW_THIN = "ℹ 点数不足"
PREVIEW_FAILED = "✕ 无数据列"
PREVIEW_HELP = "已自动检测分隔符、列映射与角度约定；可逐文件覆盖。无有效数据列的文件将被跳过，不影响其余导入。"
PREVIEW_MAPPING_TIP = "列号从 1 开始（文件里的第几列），与「列映射」对话框中的列编辑器一致。"
# 用户看得见的列号一律从 1 数起：文件里数得出来的是第一列、第二列，而
# ``DataColumnMapping`` 存的是 0 基下标。差的这个 1 在显示层加、读回时减，界面上就
# 不会出现「第 0 列」这种打开文件也数不出来的东西。
COLUMN_DISPLAY_OFFSET = 1
ANGLE_CAPTION = "角度约定（全局默认）"
ANGLE_TWO_THETA_TEXT = "2θ（衍射角）"
ANGLE_THETA_TEXT = "θ（掠射角）"
ANGLE_TWO_THETA_TIP = "源文件第一列就是散射角，照原样读入。"
ANGLE_THETA_TIP = "源文件第一列是入射角：导入时 ×2 归一到散射角，之后与 2θ 数据走同一条计算路径。"
# 「自动检测」只读源文件表头里写的轴名。按数值范围猜会在半数真实文件上翻车（2θ 扫到 4°
# 收尾的薄膜曲线很常见），而猜错就是整段角度差一倍——所以没有依据时一律不动那两个单选，
# 只把理由写在状态行里。回填时也必须出示依据：轴被改了却看不见凭据，用户没法核对。
DETECT_CAPTION = "自动检测"
DETECT_TIP = "读源文件表头里写的轴名。表头没写、或这一批写得不一致时不改选择，只说明原因。"
DETECT_UNCHANGED = "选择保持不动。"
DETECT_DECLARED = "按表头判定为 {label}：{header}"
DETECT_UNDECLARED = f"{{reason}}，{DETECT_UNCHANGED}"
DETECT_CONFLICT = f"这一批文件的表头写了不同的轴，{DETECT_UNCHANGED}"
DETECT_NOTHING = f"没有可读的文件，{DETECT_UNCHANGED}"
DELIMITER_CAPTION = "列分隔符"
DELIMITER_TEXT = "自动检测（空白/逗号）"
DELIMITER_TIP = "读取器逐行把逗号当空白处理再切分，空白与逗号混排也读得出来，因此没有别的分隔符可选。"
ADVANCED_TITLE = "高级"
HEADER_SKIP_CAPTION = "跳过表头行"
HEADER_SKIP_TEXT = "自动"
HEADER_SKIP_TIP = "读取器从第一处连续两行都是数字的位置开始取数，表头有几行都不用数。"
LOG_INTENSITY_CAPTION = "强度列取对数"
LOG_INTENSITY_TEXT = "是（若为线性计数）"
LOG_INTENSITY_TIP = "读取器只收线性强度，取对数发生在拟合的目标函数里（残差按量级算），导入这一步没有这个开关。"
RESOLUTION_KINDS = (
    ("σ(q) / Å⁻¹", "sigma_q_a_inv"),
    ("FWHM(q) / Å⁻¹", "fwhm_q_a_inv"),
    ("σ(2θ) / °", "sigma_two_theta_deg"),
    ("FWHM(2θ) / °", "fwhm_two_theta_deg"),
)


def preview_mapping_text(mapping: api.DataColumnMapping | None) -> str:
    """Name the columns this scan read as 「角度列 → 强度列」, counting from 1.

    行数与状态只说解析成功了，不说解析成了什么：一个把强度读成 2θ 的文件同样报
    「✓ 就绪」，而错在哪只有把这次实际用的列写出来才看得见。箭头的两头就是这一次
    真正读的那两列，列号与「列映射」对话框里的编辑器同一套口径。
    """
    active = mapping or api.DataColumnMapping()
    text = f"{active.two_theta + COLUMN_DISPLAY_OFFSET} → {active.intensity + COLUMN_DISPLAY_OFFSET}"
    if active.intensity_sigma is not None:
        text = f"{text} · σ:{active.intensity_sigma + COLUMN_DISPLAY_OFFSET}"
    if active.resolution is not None:
        text = f"{text} · 分辨率:{active.resolution + COLUMN_DISPLAY_OFFSET}"
    return text


def _column_spin(name: str, index: int) -> QSpinBox:
    """A column picker showing 1-based column numbers over a 0-based API index."""
    editor = QSpinBox()
    editor.setObjectName(name)
    editor.setRange(COLUMN_DISPLAY_OFFSET, 1024)
    editor.setValue(index + COLUMN_DISPLAY_OFFSET)
    return editor


def _column_index(editor: QSpinBox) -> int:
    """The API's 0-based index behind a 1-based column picker."""
    return editor.value() - COLUMN_DISPLAY_OFFSET


def _double_spin(
    name: str,
    value: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1_000_000.0,
    suffix: str = "",
) -> QDoubleSpinBox:
    editor = CompactDoubleSpinBox()
    editor.setObjectName(name)
    editor.setDecimals(8)
    editor.setRange(minimum, maximum)
    # 单位跟着数字走，而不是挤进左边的标签。「样品长度 / mm」要 86px，「样品长度」只要
    # 53px，而后缀是画在输入框里的——省下来的 33px 两栏各两处，正是把对话框从 777px
    # 收到设计稿那个宽度的那一笔。读屏那边本来就从 accessibility.py 的描述里拿单位。
    editor.setSuffix(suffix)
    editor.setValue(value)
    return editor


def _combo(name: str, choices: tuple[tuple[str, str], ...]) -> QComboBox:
    editor = QComboBox()
    editor.setObjectName(name)
    for text, value in choices:
        editor.addItem(text, value)
    return editor


def _wrap2(box: QWidget | None = None) -> QGridLayout:
    """帧⑥ 的 ``.wrap2``：一行两个字段，而不是一行一个。

    设计稿的 ``.groupbox`` 里装的正是这个网格（「跳过表头行 / 强度列取对数」并排）。一行
    一个字段时，光路 5 行加仪器 6 行是 240px 的刚性下限，对话框整体要到 703px——比
    1440×900 的可用高度还高，窗管会把页脚连着「导入」按钮一起裁掉。两两并排把这两组各自
    收成三行。

    字段的标签贴在左边，而不是设计稿 ``.field`` 那样压在上面：竖排每个字段要占两行文字，
    省下来的高度又还回去了，而这个软件其余的表单一律是标签在左。

    ``box`` 省略时给出一个待嵌入的无主网格——设计稿里「角度约定 / 列分隔符」那一组
    ``.wrap2`` 不在任何 ``.groupbox`` 里，直接摞在表格下面。
    """
    grid = QGridLayout(box) if box is not None else QGridLayout()
    grid.setHorizontalSpacing(theme.SPACE_MD)
    grid.setVerticalSpacing(theme.SPACE_XS)
    # 两个字段列吃掉多余的宽度，两个标签列贴着文字走，这样两栏才对得齐。
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(3, 1)
    return grid


def _wrap2_field(grid: QGridLayout, index: int, caption: str, field: QWidget, *, row_offset: int = 0) -> None:
    """把第 ``index`` 个字段摆进 ``.wrap2`` 网格：偶数在左、奇数在右。"""
    row, column = divmod(index, 2)
    label = QLabel(caption, grid.parentWidget())
    # QFormLayout 会自己认领 buddy，换成网格之后得手接上：点标签跳到输入框，读屏也靠它。
    label.setBuddy(field)
    grid.addWidget(label, row + row_offset, column * 2)
    grid.addWidget(field, row + row_offset, column * 2 + 1)


def _undecided_convention_text(
    evidence: tuple[api.AngleConventionEvidence, ...],
    declared: set[str],
) -> str:
    """为什么没敢动那两个单选：批内互相矛盾，还是压根没有声明。"""
    if len(declared) > 1:
        return DETECT_CONFLICT
    reason = next((item.warning for item in evidence if item.warning), None)
    return DETECT_NOTHING if reason is None else DETECT_UNDECLARED.format(reason=reason)


class ColumnMappingDialog(QDialog):
    """Collect an explicit validated mapping for optional source columns."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("columnMappingDialog")
        self.setWindowTitle("列映射")
        self.setAccessibleName("列映射")
        self.setModal(True)
        self.two_theta = _column_spin("twoThetaColumnEditor", 0)
        self.intensity = _column_spin("intensityColumnEditor", 1)
        self.sigma_enabled = QCheckBox("强度不确定度列")
        self.sigma_enabled.setObjectName("intensitySigmaEnabled")
        self.intensity_sigma = _column_spin("intensitySigmaColumnEditor", 2)
        self.resolution_enabled = QCheckBox("分辨率列")
        self.resolution_enabled.setObjectName("resolutionEnabled")
        self.resolution = _column_spin("resolutionColumnEditor", 3)
        self.resolution_kind = _combo("resolutionKindEditor", RESOLUTION_KINDS)
        self.error_label = QLabel()
        self.error_label.setObjectName("columnMappingError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        localize_standard_buttons(self.buttons, confirm="应用映射")
        self._connect_controls()
        self._arrange()

    def _connect_controls(self) -> None:
        self.sigma_enabled.toggled.connect(self.intensity_sigma.setEnabled)
        self.resolution_enabled.toggled.connect(self.resolution.setEnabled)
        self.resolution_enabled.toggled.connect(self.resolution_kind.setEnabled)
        self.intensity_sigma.setEnabled(False)
        self.resolution.setEnabled(False)
        self.resolution_kind.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("2θ 列", self.two_theta)
        form.addRow("强度列", self.intensity)
        form.addRow(self.sigma_enabled, self.intensity_sigma)
        form.addRow(self.resolution_enabled, self.resolution)
        form.addRow("分辨率类型", self.resolution_kind)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def mapping(self) -> api.DataColumnMapping:
        """Return the API value represented by the visible controls."""
        sigma = _column_index(self.intensity_sigma) if self.sigma_enabled.isChecked() else None
        resolution = _column_index(self.resolution) if self.resolution_enabled.isChecked() else None
        kind = self.resolution_kind.currentData() if resolution is not None else None
        return api.DataColumnMapping(
            _column_index(self.two_theta),
            _column_index(self.intensity),
            sigma,
            resolution,
            kind,
        )

    def accept(self) -> None:
        """Keep invalid input open and show the API validation error unchanged."""
        try:
            self.mapping()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            if not self.isVisible():
                self.show()
            return
        self.error_label.hide()
        super().accept()


class ImportDialog(QDialog):
    """Require explicit beam selection before an import can be confirmed."""

    def __init__(
        self,
        paths: tuple[Path, ...] | list[Path],
        *,
        folder_mode: bool = False,
        noise_model: str = "robust_log",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paths = tuple(Path(path) for path in paths)
        self._column_mapping: api.DataColumnMapping | None = None
        self._noise_model = noise_model
        self._failed_count = 0
        self.setObjectName("importDialog")
        self.setWindowTitle("导入 XRR 数据")
        self.setAccessibleName("导入 XRR 数据")
        self.setModal(True)
        self._build_source_controls(folder_mode)
        self._build_beam_controls()
        self._build_instrument_controls()
        self._build_convention_controls()
        self._build_advanced_controls()
        self._build_buttons()
        # 首次预览放在按钮构建之后：刷新要把批次计数写到确认按钮上，顺序反了就没有按钮可写。
        self._refresh_preview()
        self._arrange()

    def _build_source_controls(self, folder_mode: bool) -> None:
        self.heading = QLabel(DIALOG_HEADING)
        self.heading.setObjectName("importDialogHeading")
        # 两张导入/导出对话框在设计稿里是并排的两张卡，各自靠头一行说明自己是哪一张；
        # 只有页脚按钮时，读者要反推这是导入还是导出。
        self.heading.setProperty("sectionHeader", True)
        self.preview_help = QLabel(PREVIEW_HELP)
        self.preview_help.setObjectName("importPreviewHelp")
        self.preview_help.setProperty("mutedText", True)
        self.preview_help.setWordWrap(True)
        self.preview_table = ContentSizedTable(0, len(PREVIEW_HEADERS))
        self.preview_table.setObjectName("importPreviewTable")
        self.preview_table.setAccessibleName("待导入文件预览")
        self.preview_table.setHorizontalHeaderLabels(list(PREVIEW_HEADERS))
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.preview_table.horizontalHeader()
        # 文件名吃掉多余的宽度，其余四列贴着自己的内容——角度与状态都是短字形串，
        # 平分列宽会把文件名挤成省略号。
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(PREVIEW_HEADERS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.noise_label = QLabel(f"噪声模式：{NOISE_MODE_LABELS[self._noise_model]}")
        self.noise_label.setObjectName("importNoiseModel")
        self.noise_requirements = QLabel(NOISE_MODE_REQUIREMENTS[self._noise_model])
        self.noise_requirements.setObjectName("importNoiseRequirements")
        self.noise_requirements.setWordWrap(True)
        self.recursive_check = QCheckBox("递归导入子目录")
        self.recursive_check.setObjectName("recursiveFolderImportCheck")
        self.recursive_check.setVisible(bool(folder_mode))

    def _build_beam_controls(self) -> None:
        self.beam_group = QButtonGroup(self)
        self.mono_button = QRadioButton("单色")
        self.mixed_button = QRadioButton("混合 Kα")
        self.beam_group.addButton(self.mono_button)
        self.beam_group.addButton(self.mixed_button)
        self.mono_wavelength = _double_spin("monochromaticWavelengthEditor", 1.5406, suffix=" Å")
        self.wavelength_1 = _double_spin("mixedWavelength1Editor", 1.54056, suffix=" Å")
        self.wavelength_2 = _double_spin("mixedWavelength2Editor", 1.54439, suffix=" Å")
        self.intensity_ratio = _double_spin("mixedIntensityRatioEditor", 0.5)
        self.beam_group.buttonToggled.connect(self._refresh_validation)

    def _build_instrument_controls(self) -> None:
        self.instrument_id = QLineEdit()
        self.instrument_id.setObjectName("instrumentIdEditor")
        self.footprint = _combo(
            "footprintModeEditor",
            (("几何换算", "geometry"), ("拟合", "fit"), ("无", "none")),
        )
        self.sample_length = _double_spin("sampleLengthEditor", 10.0, suffix=" mm")
        self.beam_width = _double_spin("beamWidthEditor", 0.1, suffix=" mm")
        self.background = _combo(
            "backgroundModelEditor",
            (("常数", "constant"), ("线性", "linear"), ("幂律", "powerlaw")),
        )
        self.resolution_domain = _combo(
            "resolutionDomainEditor",
            (("q", "q"), ("θ", "theta")),
        )
        self.footprint.setCurrentIndex(self.footprint.findData("fit"))
        self.footprint.currentIndexChanged.connect(self._refresh_geometry_controls)
        self._refresh_geometry_controls()

    def _build_convention_controls(self) -> None:
        """帧⑥ 的「角度约定」与「列分隔符」：读取器认得的那一种，摆出来给人看。

        角度约定两档都是真的：``2θ（衍射角）`` 照原样读，``θ（掠射角）`` 在导入时把角度列
        ×2 归一到散射角——模型内部的轴始终是 ``two_theta_deg``，两条路径之后走同一套计算。
        分隔符那一栏才真的只有一档：每行先把逗号换成空白再切分，空白与逗号一律认得，所以
        它灰着并在提示里说清为什么不必选，比摆一个按不下去也没别的可选的下拉更接近实情。
        """
        self.angle_group = QButtonGroup(self)
        self.two_theta_convention = QRadioButton(ANGLE_TWO_THETA_TEXT)
        self.two_theta_convention.setObjectName("twoThetaConventionOption")
        self.two_theta_convention.setToolTip(ANGLE_TWO_THETA_TIP)
        self.two_theta_convention.setChecked(True)
        self.theta_convention = QRadioButton(ANGLE_THETA_TEXT)
        self.theta_convention.setObjectName("thetaConventionOption")
        self.theta_convention.setToolTip(ANGLE_THETA_TIP)
        self.angle_group.addButton(self.two_theta_convention)
        self.angle_group.addButton(self.theta_convention)
        # 预览是按当前约定解析出来的，换了轴就得整表重画：行数与「✓ 就绪」在两档下都成立，
        # 变的是角度那一格和它背后的数。两档互斥，所以只接 θ 这一路——它在来回切换时各触发
        # 一次，接两路会让一次切换重画两遍。
        self.theta_convention.toggled.connect(self._refresh_preview)
        self.angle_row = QWidget()
        self.angle_row.setObjectName("angleConventionRow")
        choices = QHBoxLayout(self.angle_row)
        choices.setContentsMargins(0, 0, 0, 0)
        choices.setSpacing(theme.SPACE_SM)
        choices.addWidget(self.two_theta_convention)
        choices.addWidget(self.theta_convention)
        self.detect_convention = QPushButton(DETECT_CAPTION)
        self.detect_convention.setObjectName("detectAngleConventionButton")
        self.detect_convention.setToolTip(DETECT_TIP)
        self.detect_convention.clicked.connect(self._detect_convention)
        choices.addWidget(self.detect_convention)
        # 判定依据（那一行表头）与「为什么没敢改」都写在这里，检测按下之前是空的。
        self.convention_status = QLabel()
        self.convention_status.setObjectName("angleConventionStatus")
        self.convention_status.setWordWrap(True)
        self.delimiter_select = _combo("columnDelimiterEditor", ((DELIMITER_TEXT, "auto"),))
        self.delimiter_select.setEnabled(False)
        self.delimiter_select.setToolTip(DELIMITER_TIP)

    def _build_advanced_controls(self) -> None:
        """帧⑥「高级」里的「跳过表头行」与「强度列取对数」。"""
        self.header_skip_field = QLineEdit(HEADER_SKIP_TEXT)
        self.header_skip_field.setObjectName("headerSkipEditor")
        self.header_skip_field.setReadOnly(True)
        self.header_skip_field.setToolTip(HEADER_SKIP_TIP)
        self.log_intensity_check = QCheckBox(LOG_INTENSITY_TEXT)
        self.log_intensity_check.setObjectName("logIntensityCheck")
        self.log_intensity_check.setEnabled(False)
        self.log_intensity_check.setToolTip(LOG_INTENSITY_TIP)

    def _preview_beam(self) -> api.BeamSpec:
        """The beam to parse previews with, falling back before a choice is made.

        The preview is drawn before the reader has committed to a light path, so
        an unselected or half-edited beam must still yield a readable curve
        rather than an empty pane blaming the file.
        """
        if self.beam_kind() is None:
            return api.BeamSpec("monochromatic", wavelength_a=1.5406)
        try:
            return self.beam_spec()
        except (ValueError, TypeError):
            return api.BeamSpec("monochromatic", wavelength_a=1.5406)

    def _detect_convention(self) -> None:
        """把源文件表头里写着的轴回填到那两个单选上，没有依据就只说明原因。

        静默回填才是这里真正的危险：把没有依据的猜测写进单选，用户会以为那是文件说的。
        所以三种「说不清」——一份都没声明、某一份自己的表头写了两个轴、批内两份写了不同
        的轴——一律不动选择，只在状态行里写清是哪一种。回填成功时也要出示那一行表头：轴
        被改了却看不见凭据，用户没法核对，而选错轴的代价是整段角度差一倍。
        """
        evidence = tuple(api.detect_angle_convention(path) for path in self._paths)
        declared = {item.convention for item in evidence if item.declared}
        if len(declared) != 1:
            self.convention_status.setText(_undecided_convention_text(evidence, declared))
            return
        convention = declared.pop()
        label = ANGLE_THETA_TEXT if convention == "theta" else ANGLE_TWO_THETA_TEXT
        # 互斥按钮组里只勾该勾的那一个，另一个会自动松开；两个都设会让预览重画两遍。
        button = self.theta_convention if convention == "theta" else self.two_theta_convention
        button.setChecked(True)
        header = next(item.header_line for item in evidence if item.declared)
        self.convention_status.setText(DETECT_DECLARED.format(label=label, header=header))

    def _refresh_preview(self) -> None:
        """Re-scan the batch and re-label the confirm button."""
        self._refresh_preview_table()
        self._apply_import_button_text()

    def _scan_path(self, path: Path, beam: api.BeamSpec) -> tuple[int, str, str, str]:
        """Row count, angle convention, status and detail for one candidate file.

        A source that cannot be parsed is reported rather than raised: the batch
        preview has to survive an unreadable path to be able to show it, which is
        the whole point of listing failures instead of dropping them.

        解析成功还分两档。``fit_ready`` 为假的文件（有效点不足 30，或 qz 不单调）读得出
        曲线却拟不了，报「就绪」是在骗人；它的告警本来就在 ``warnings`` 里，直接拿来当
        这一格的提示。
        """
        convention = self.angle_convention()
        try:
            data = api.import_data(
                path,
                beam,
                column_mapping=self._column_mapping,
                angle_convention=convention,
                noise_model=self._noise_model,
            )
        except Exception as exc:
            LOG.debug("preview scan failed for %s: %s", path, exc)
            return 0, PREVIEW_UNKNOWN, PREVIEW_FAILED, f"{type(exc).__name__}: {exc}"
        rows = len(data.two_theta_deg)
        # 这一格报的是「刚才按哪个轴读的」，所以它跟着选择走：θ 那一档读进来的数已经 ×2
        # 归一过了，仍写 ``2θ`` 会把唯一能看出轴选错的地方也遮掉。
        label = PREVIEW_ANGLE_THETA if convention == "theta" else PREVIEW_ANGLE_TWO_THETA
        if data.fit_ready:
            return rows, label, PREVIEW_READY, str(path)
        detail = "；".join(data.warnings) or f"有效点不足，无法参与拟合：{path}"
        return rows, label, PREVIEW_THIN, detail

    def _refresh_preview_table(self) -> None:
        """Re-scan every selected file so the table matches the current beam."""
        tokens = theme.palette_tokens(self.palette())
        colours = {
            PREVIEW_READY: tokens.ok,
            PREVIEW_THIN: tokens.info,
            PREVIEW_FAILED: tokens.error,
        }
        beam = self._preview_beam()
        self.preview_table.setRowCount(len(self._paths))
        self._failed_count = 0
        for row, path in enumerate(self._paths):
            rows, angle, status, detail = self._scan_path(path, beam)
            self._failed_count += int(status == PREVIEW_FAILED)
            name = QTableWidgetItem(path.name)
            name.setToolTip(str(path))
            kind = QTableWidgetItem(angle)
            mapping_text = PREVIEW_UNKNOWN if status == PREVIEW_FAILED else preview_mapping_text(self._column_mapping)
            mapping = QTableWidgetItem(mapping_text)
            mapping.setToolTip(PREVIEW_MAPPING_TIP)
            count = QTableWidgetItem(str(rows))
            count.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            marker = QTableWidgetItem(status)
            marker.setToolTip(detail)
            marker.setForeground(QColor(colours[status]))
            for column, item in enumerate((name, kind, mapping, count, marker)):
                self.preview_table.setItem(row, column, item)

    def _apply_import_button_text(self) -> None:
        """Name the batch on the confirm button: what imports, and what is skipped.

        A bare 导入 over a mixed batch hides the one piece of arithmetic the
        reader needs before committing — that a chosen file carries no data and
        will be dropped — so both counts are stated where the click happens.
        """
        importable = len(self._paths) - self._failed_count
        label = f"导入 {importable} 个"
        if self._failed_count:
            label = f"{label}（跳过 {self._failed_count} 个失败）"
        self._import_button.setText(label)

    def _build_buttons(self) -> None:
        self.mapping_button = QPushButton("高级列映射…")
        self.mapping_button.setObjectName("columnMappingButton")
        self.mapping_button.clicked.connect(self._edit_column_mapping)
        self.validation_label = QLabel(VALIDATION_TEXT)
        self.validation_label.setObjectName("importValidation")
        self.validation_label.setWordWrap(True)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._import_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self._import_button.setText("导入")
        self._import_button.setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        # 只本地化取消：确认按钮的文案由 ``_apply_import_button_text`` 按批次计数接管，
        # 交给共享入口会把「导入 4 个（跳过 1 个失败）」覆写成一个笼统的确认。
        localize_standard_buttons(self.button_box)

    def _arrange(self) -> None:
        beam_box = QGroupBox("光路")
        beam_grid = _wrap2(beam_box)
        choice_row = QHBoxLayout()
        choice_row.addWidget(self.mono_button)
        choice_row.addWidget(self.mixed_button)
        # 光路类型管着下面四个波长字段读哪一组，所以它横跨整行，不参与配对。
        beam_grid.addLayout(choice_row, 0, 0, 1, 4)
        _wrap2_field(beam_grid, 0, "单色波长", self.mono_wavelength, row_offset=1)
        _wrap2_field(beam_grid, 1, "Kα₁", self.wavelength_1, row_offset=1)
        _wrap2_field(beam_grid, 2, "Kα₂", self.wavelength_2, row_offset=1)
        _wrap2_field(beam_grid, 3, "I₂/I₁", self.intensity_ratio, row_offset=1)
        instrument_box = QGroupBox("仪器")
        instrument_grid = _wrap2(instrument_box)
        for index, (caption, field) in enumerate(
            (
                ("标识", self.instrument_id),
                ("足迹模式", self.footprint),
                ("样品长度", self.sample_length),
                ("光束宽度", self.beam_width),
                ("背景模型", self.background),
                ("分辨率域", self.resolution_domain),
            )
        ):
            _wrap2_field(instrument_grid, index, caption, field)
        layout = QVBoxLayout(self)
        layout.addWidget(self.heading)
        layout.addWidget(self.preview_help)
        layout.addWidget(self.preview_table)
        layout.addWidget(self.noise_label)
        layout.addWidget(self.noise_requirements)
        layout.addWidget(self.recursive_check)
        convention_grid = _wrap2()
        _wrap2_field(convention_grid, 0, ANGLE_CAPTION, self.angle_row)
        # 角度约定与列分隔符各占一行（0 与 2 都落在左半边），而不是并排在同一行：角度那格是
        # 一对单选钮，「2θ（衍射角）」与「θ（掠射角）」加上 132px 的标签，一行摆开要 699px，
        # 那已经超过整个对话框允许的最小宽度。竖着排把这一格的下限降到一半出头。
        _wrap2_field(convention_grid, 2, DELIMITER_CAPTION, self.delimiter_select)
        layout.addLayout(convention_grid)
        # 判定依据是一整行表头原文，独占整宽才不至于把那一栏挤成两个字一行。
        layout.addWidget(self.convention_status)
        layout.addWidget(self._advanced_section(beam_box, instrument_box))
        layout.addWidget(self.validation_label)
        layout.addWidget(self.button_box)

    def _advanced_section(self, beam_box: QGroupBox, instrument_box: QGroupBox) -> QGroupBox:
        """帧⑥a 的 ``高级`` 组：设计稿的两个字段，加上本版真要人选的光路与仪器。

        光路是导入的必填项（未选中时「导入」按钮就是灰的），设计稿这张卡上没有它——那是
        因为设计稿默认数据从仪器文件里自带波长。摘掉它导入根本走不通，所以它跟仪器一起
        收进「高级」：默认折叠不了，但至少不再占着卡面最显眼的两栏。
        """
        self.advanced_box = QGroupBox(ADVANCED_TITLE)
        self.advanced_box.setObjectName("importAdvancedGroup")
        advanced = QVBoxLayout(self.advanced_box)
        advanced.setSpacing(theme.SPACE_SM)
        advanced_grid = _wrap2()
        _wrap2_field(advanced_grid, 0, HEADER_SKIP_CAPTION, self.header_skip_field)
        _wrap2_field(advanced_grid, 1, LOG_INTENSITY_CAPTION, self.log_intensity_check)
        advanced.addLayout(advanced_grid)
        # 帧⑥ 的 ``.wrap2`` 把成对的字段并排放，正是这一点让对话框收得住：这两组表单
        # 竖着摞是 408px 的刚性下限，未按下的「导入」按钮会被 1440×900 的窗管裁掉。
        settings_row = QHBoxLayout()
        settings_row.addWidget(beam_box)
        settings_row.addWidget(instrument_box)
        advanced.addLayout(settings_row)
        advanced.addWidget(self.mapping_button)
        return self.advanced_box

    def _refresh_geometry_controls(self, _index: int = -1) -> None:
        enabled = self.footprint.currentData() == "geometry"
        self.sample_length.setEnabled(enabled)
        self.beam_width.setEnabled(enabled)

    def _refresh_validation(self, _button: object, _checked: bool) -> None:
        selected = self.beam_kind() is not None
        self._import_button.setEnabled(selected)
        self.validation_label.setText("" if selected else VALIDATION_TEXT)
        self.validation_label.setVisible(not selected)
        self._refresh_preview()

    def _edit_column_mapping(self) -> None:
        dialog = ColumnMappingDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._column_mapping = dialog.mapping()
            self._refresh_preview()

    def beam_kind(self) -> str | None:
        if self.mono_button.isChecked():
            return "monochromatic"
        if self.mixed_button.isChecked():
            return "mixed_kalpha"
        return None

    def select_beam_kind(self, kind: str) -> None:
        buttons = {
            "monochromatic": self.mono_button,
            "mixed_kalpha": self.mixed_button,
        }
        try:
            buttons[kind].setChecked(True)
        except KeyError as error:
            raise ValueError(f"unsupported beam kind: {kind}") from error

    def import_button(self) -> QPushButton:
        return self._import_button

    def validation_text(self) -> str:
        return self.validation_label.text()

    def beam_spec(self) -> api.BeamSpec:
        kind = self.beam_kind()
        if kind is None:
            raise ValueError("beam kind must be selected")
        if kind == "monochromatic":
            return api.BeamSpec(kind, wavelength_a=self.mono_wavelength.value())
        return api.BeamSpec(
            kind,
            wavelength_1_a=self.wavelength_1.value(),
            wavelength_2_a=self.wavelength_2.value(),
            intensity_ratio_21=self.intensity_ratio.value(),
        )

    def instrument_spec(self) -> api.InstrumentSpec:
        mode = str(self.footprint.currentData())
        instrument_id = self.instrument_id.text().strip() or None
        if mode != "geometry":
            return api.InstrumentSpec(
                instrument_id=instrument_id,
                footprint_mode=mode,
                background_kind=str(self.background.currentData()),
                resolution_domain=str(self.resolution_domain.currentData()),
            )
        length = self.sample_length.value()
        width = self.beam_width.value()
        if length <= 0.0 or width <= 0.0 or width > length:
            raise ValueError("geometry requires 0 < beam_width_mm <= sample_length_mm")
        return api.InstrumentSpec(
            instrument_id=instrument_id,
            footprint_mode=mode,
            footprint_spill_angle_deg=degrees(asin(width / length)),
            sample_length_mm=length,
            beam_width_mm=width,
            background_kind=str(self.background.currentData()),
            resolution_domain=str(self.resolution_domain.currentData()),
        )

    def angle_convention(self) -> str:
        return "theta" if self.theta_convention.isChecked() else "two_theta"

    def column_mapping(self) -> api.DataColumnMapping | None:
        return self._column_mapping

    def set_column_mapping(
        self,
        two_theta: int = 0,
        intensity: int = 1,
        intensity_sigma: int | None = None,
        resolution: int | None = None,
        resolution_kind: str | None = None,
    ) -> None:
        self._column_mapping = api.DataColumnMapping(
            two_theta,
            intensity,
            intensity_sigma,
            resolution,
            resolution_kind,
        )
        # 预览是按当前映射解析出来的，所以映射一变，行数、状态和「列映射」那一格必须
        # 一起重画；否则表上留着的是上一套列号解析出来的结果。
        self._refresh_preview()

    def cancel_column_mapping(self) -> None:
        self._column_mapping = None
        self._refresh_preview()

    def recursive_folder_import(self) -> bool:
        return self.recursive_check.isChecked()
