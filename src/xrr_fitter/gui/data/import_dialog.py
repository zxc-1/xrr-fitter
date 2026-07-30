"""Explicit data-import, beam, instrument, and column-mapping dialogs."""

from __future__ import annotations

from math import asin, degrees
from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api


VALIDATION_TEXT = "请选择光路类型：单色 / 混合 Kα"
RESOLUTION_KINDS = (
    ("σ(q) / Å⁻¹", "sigma_q_a_inv"),
    ("FWHM(q) / Å⁻¹", "fwhm_q_a_inv"),
    ("σ(2θ) / °", "sigma_two_theta_deg"),
    ("FWHM(2θ) / °", "fwhm_two_theta_deg"),
)


def _spin(name: str, value: int) -> QSpinBox:
    editor = QSpinBox()
    editor.setObjectName(name)
    editor.setRange(0, 1024)
    editor.setValue(value)
    return editor


def _double_spin(
    name: str,
    value: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1_000_000.0,
) -> QDoubleSpinBox:
    editor = QDoubleSpinBox()
    editor.setObjectName(name)
    editor.setDecimals(8)
    editor.setRange(minimum, maximum)
    editor.setValue(value)
    return editor


def _combo(name: str, choices: tuple[tuple[str, str], ...]) -> QComboBox:
    editor = QComboBox()
    editor.setObjectName(name)
    for text, value in choices:
        editor.addItem(text, value)
    return editor


class ColumnMappingDialog(QDialog):
    """Collect an explicit validated mapping for optional source columns."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("columnMappingDialog")
        self.setWindowTitle("列映射")
        self.setAccessibleName("列映射")
        self.setModal(True)
        self.two_theta = _spin("twoThetaColumnEditor", 0)
        self.intensity = _spin("intensityColumnEditor", 1)
        self.sigma_enabled = QCheckBox("强度不确定度列")
        self.sigma_enabled.setObjectName("intensitySigmaEnabled")
        self.intensity_sigma = _spin("intensitySigmaColumnEditor", 2)
        self.resolution_enabled = QCheckBox("分辨率列")
        self.resolution_enabled.setObjectName("resolutionEnabled")
        self.resolution = _spin("resolutionColumnEditor", 3)
        self.resolution_kind = _combo("resolutionKindEditor", RESOLUTION_KINDS)
        self.error_label = QLabel()
        self.error_label.setObjectName("columnMappingError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
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
        sigma = self.intensity_sigma.value() if self.sigma_enabled.isChecked() else None
        resolution = self.resolution.value() if self.resolution_enabled.isChecked() else None
        kind = self.resolution_kind.currentData() if resolution is not None else None
        return api.DataColumnMapping(
            self.two_theta.value(),
            self.intensity.value(),
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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paths = tuple(Path(path) for path in paths)
        self._column_mapping: api.DataColumnMapping | None = None
        self.setObjectName("importDialog")
        self.setWindowTitle("导入 XRR 数据")
        self.setAccessibleName("导入 XRR 数据")
        self.setModal(True)
        self._build_source_controls(folder_mode)
        self._build_beam_controls()
        self._build_instrument_controls()
        self._build_buttons()
        self._arrange()

    def _build_source_controls(self, folder_mode: bool) -> None:
        names = "\n".join(str(path) for path in self._paths)
        self.source_label = QLabel(names)
        self.source_label.setObjectName("importSourcePaths")
        self.source_label.setWordWrap(True)
        self.recursive_check = QCheckBox("递归导入子目录")
        self.recursive_check.setObjectName("recursiveFolderImportCheck")
        self.recursive_check.setVisible(bool(folder_mode))

    def _build_beam_controls(self) -> None:
        self.beam_group = QButtonGroup(self)
        self.mono_button = QRadioButton("单色")
        self.mixed_button = QRadioButton("混合 Kα")
        self.beam_group.addButton(self.mono_button)
        self.beam_group.addButton(self.mixed_button)
        self.mono_wavelength = _double_spin("monochromaticWavelengthEditor", 1.5406)
        self.wavelength_1 = _double_spin("mixedWavelength1Editor", 1.54056)
        self.wavelength_2 = _double_spin("mixedWavelength2Editor", 1.54439)
        self.intensity_ratio = _double_spin("mixedIntensityRatioEditor", 0.5)
        self.beam_group.buttonToggled.connect(self._refresh_validation)

    def _build_instrument_controls(self) -> None:
        self.instrument_id = QLineEdit()
        self.instrument_id.setObjectName("instrumentIdEditor")
        self.footprint = _combo(
            "footprintModeEditor",
            (("几何换算", "geometry"), ("拟合", "fit"), ("无", "none")),
        )
        self.sample_length = _double_spin("sampleLengthEditor", 10.0)
        self.beam_width = _double_spin("beamWidthEditor", 0.1)
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

    def _build_buttons(self) -> None:
        self.mapping_button = QPushButton("高级列映射…")
        self.mapping_button.setObjectName("columnMappingButton")
        self.mapping_button.clicked.connect(self._edit_column_mapping)
        self.validation_label = QLabel(VALIDATION_TEXT)
        self.validation_label.setObjectName("importValidation")
        self.validation_label.setWordWrap(True)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._import_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self._import_button.setText("导入")
        self._import_button.setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _arrange(self) -> None:
        beam_box = QGroupBox("光路")
        beam_form = QFormLayout(beam_box)
        choice_row = QHBoxLayout()
        choice_row.addWidget(self.mono_button)
        choice_row.addWidget(self.mixed_button)
        beam_form.addRow(choice_row)
        beam_form.addRow("单色波长 / Å", self.mono_wavelength)
        beam_form.addRow("Kα₁ / Å", self.wavelength_1)
        beam_form.addRow("Kα₂ / Å", self.wavelength_2)
        beam_form.addRow("I₂/I₁", self.intensity_ratio)
        instrument_box = QGroupBox("仪器")
        instrument_form = QFormLayout(instrument_box)
        instrument_form.addRow("标识", self.instrument_id)
        instrument_form.addRow("足迹模式", self.footprint)
        instrument_form.addRow("样品长度 / mm", self.sample_length)
        instrument_form.addRow("光束宽度 / mm", self.beam_width)
        instrument_form.addRow("背景模型", self.background)
        instrument_form.addRow("分辨率域", self.resolution_domain)
        layout = QVBoxLayout(self)
        layout.addWidget(self.source_label)
        layout.addWidget(self.recursive_check)
        layout.addWidget(beam_box)
        layout.addWidget(instrument_box)
        layout.addWidget(self.mapping_button)
        layout.addWidget(self.validation_label)
        layout.addWidget(self.button_box)

    def _refresh_geometry_controls(self, _index: int = -1) -> None:
        enabled = self.footprint.currentData() == "geometry"
        self.sample_length.setEnabled(enabled)
        self.beam_width.setEnabled(enabled)

    def _refresh_validation(self, _button: object, _checked: bool) -> None:
        selected = self.beam_kind() is not None
        self._import_button.setEnabled(selected)
        self.validation_label.setVisible(not selected)

    def _edit_column_mapping(self) -> None:
        dialog = ColumnMappingDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._column_mapping = dialog.mapping()

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

    def cancel_column_mapping(self) -> None:
        self._column_mapping = None

    def recursive_folder_import(self) -> bool:
        return self.recursive_check.isChecked()
