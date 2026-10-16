"""Modal forms that build one complete immutable layer, gradient layer or backing.

The periodic block has its own form (``periodic_dialog``); the input parts all of
them share live in ``fields``.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.structure.fields import (
    NominalDensity,
    density_editor,
    dialog_buttons,
    number_editor,
    show_rejection,
    table_cell,
)

# 梯度层的 SLD 按 10⁻⁶ Å⁻² 输入，和 SLD 剖面图的纵轴同一个单位。
SLD_DISPLAY_SCALE = 1e6


class LayerDialog(QDialog):
    """Collect one formula-backed layer using nm for visible lengths."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        commit_layer: Callable[[api.LayerSpec], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("layerDialog")
        self.setWindowTitle("添加普通层")
        self.setAccessibleName("添加普通层")
        self.setModal(True)
        self._commit_layer = commit_layer
        self._layer: api.LayerSpec | None = None
        self.name_editor = QLineEdit()
        self.name_editor.setObjectName("layerNameInput")
        self.formula_editor = QLineEdit()
        self.formula_editor.setObjectName("layerFormulaInput")
        self.density_editor = density_editor("layerDensityInput")
        self._density = NominalDensity(self.formula_editor, self.density_editor)
        self.thickness_editor = number_editor("layerThicknessInput", 0.2, 1.0)
        self.roughness_editor = number_editor("layerRoughnessInput", 0.0)
        self.transition_toggle = QCheckBox()
        self.transition_toggle.setObjectName("layerTransitionToggle")
        self.transition_toggle.setAccessibleName("启用界面过渡")
        self.transition_toggle.setToolTip("用可归一化的过渡函数族替代 Névot-Croce 粗糙度")
        self.microslab_editor = number_editor("layerMicroslabInput", 0.001, 0.1)
        self.branch_table = QTableWidget(2, 3)
        self.branch_table.setObjectName("layerTransitionTable")
        self.branch_table.setHorizontalHeaderLabels(("过渡类型", "权重", "宽度 (nm)"))
        self.error_label = QLabel()
        self.error_label.setObjectName("layerDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = dialog_buttons("layerDialogButtons")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        self.transition_toggle.toggled.connect(self._sync_transition_inputs)
        self._arrange()
        self._sync_transition_inputs(False)

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("名称", self.name_editor)
        form.addRow("化学式", self.formula_editor)
        form.addRow("密度 (g/cm³)", self.density_editor)
        form.addRow("厚度 (nm)", self.thickness_editor)
        form.addRow("粗糙度 (nm)", self.roughness_editor)
        form.addRow("界面过渡", self.transition_toggle)
        form.addRow("切片上限 (nm)", self.microslab_editor)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.branch_table)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _sync_transition_inputs(self, enabled: bool) -> None:
        """Make the roughness/transition exclusion visible before commit time.

        A transition already sets the interface width, so leaving the roughness
        editable would let the user fill a field the model then rejects with an
        English message pointing at ``roughness_a``.
        """
        self.roughness_editor.setEnabled(not enabled)
        if enabled:
            self.roughness_editor.setValue(0.0)
        self.microslab_editor.setEnabled(enabled)
        self.branch_table.setEnabled(enabled)

    def _branch_at(self, row: int) -> api.TransitionBranch:
        kind, weight, width = (table_cell(self.branch_table, row, column, "transition branch") for column in range(3))
        return api.TransitionBranch(kind, float(weight), float(width) * 10.0)

    def _transition(self) -> api.InterfaceTransition | None:
        if not self.transition_toggle.isChecked():
            return None
        branches = tuple(self._branch_at(row) for row in range(self.branch_table.rowCount()))
        return api.InterfaceTransition(branches, self.microslab_editor.value() * 10.0)

    def _accept_fields(self) -> None:
        try:
            formula = self.formula_editor.text().strip()
            material = api.MaterialSpec(formula, formula, self._density.value())
            candidate = api.LayerSpec(
                self.name_editor.text().strip(),
                material,
                self.thickness_editor.value() * 10.0,
                roughness_a=self.roughness_editor.value() * 10.0,
                transition=self._transition(),
            )
            if self._commit_layer is not None:
                self._commit_layer(candidate)
        except (TypeError, ValueError) as error:
            show_rejection(self, self.error_label, error, self.formula_editor)
            return
        self._layer = candidate
        self.error_label.hide()
        self.accept()

    def layer(self) -> api.LayerSpec:
        if self._layer is None:
            raise LookupError("layer has not been confirmed")
        return self._layer


class BackingDialog(QDialog):
    """Edit one formula-backed substrate and its connecting roughness."""

    def __init__(
        self,
        backing: api.MaterialSpec,
        roughness_a: float,
        parent: QWidget | None = None,
        *,
        commit_backing: Callable[[api.MaterialSpec, float], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("backingDialog")
        self.setWindowTitle("编辑基底")
        self.setAccessibleName("编辑基底")
        self.setModal(True)
        self._commit_backing = commit_backing
        self._backing: api.MaterialSpec | None = None
        self._roughness_a: float | None = None
        self.formula_editor = QLineEdit()
        self.formula_editor.setObjectName("backingFormulaInput")
        self.formula_editor.setText("" if backing.formula is None else backing.formula)
        self.density_editor = density_editor("backingDensityInput")
        self._density = NominalDensity(self.formula_editor, self.density_editor, backing.bulk_density_g_cm3)
        self.roughness_editor = number_editor(
            "backingRoughnessInput",
            0.0,
            roughness_a / 10.0,
        )
        self.error_label = QLabel()
        self.error_label.setObjectName("backingDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = dialog_buttons("backingDialogButtons")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        self._arrange()

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("化学式", self.formula_editor)
        form.addRow("密度 (g/cm³)", self.density_editor)
        form.addRow("连接界面粗糙度 (nm)", self.roughness_editor)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _accept_fields(self) -> None:
        try:
            formula = self.formula_editor.text().strip()
            backing = api.MaterialSpec(formula, formula, self._density.value())
            roughness_a = self.roughness_editor.value() * 10.0
            if self._commit_backing is not None:
                self._commit_backing(backing, roughness_a)
        except (TypeError, ValueError) as error:
            show_rejection(self, self.error_label, error, self.formula_editor)
            return
        self._backing = backing
        self._roughness_a = roughness_a
        self.error_label.hide()
        self.accept()

    def backing(self) -> tuple[api.MaterialSpec, float]:
        if self._backing is None or self._roughness_a is None:
            raise LookupError("backing has not been confirmed")
        return self._backing, self._roughness_a


class GradientDialog(QDialog):
    """Collect one gradient layer: SLD goes linearly from the upper to the lower value.

    编辑已有梯度层时预填它的值；没碰过的框逐位照抄原值——10⁻⁶ 与 nm 的显示换算来回一趟会
    改掉末几位，确定一次就成了一次「结构变了」。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        commit_gradient: Callable[[api.GradientLayerSpec], object] | None = None,
        gradient: api.GradientLayerSpec | None = None,
    ) -> None:
        super().__init__(parent)
        title = "添加梯度层" if gradient is None else "编辑梯度层"
        self.setObjectName("gradientLayerDialog")
        self.setWindowTitle(title)
        self.setAccessibleName(title)
        self.setModal(True)
        self._commit_gradient = commit_gradient
        self._gradient: api.GradientLayerSpec | None = None
        self.name_editor = QLineEdit()
        self.name_editor.setObjectName("gradientNameInput")
        self.upper_editor = _sld_number("gradientUpperSldInput", -1_000_000.0)
        self.upper_absorption_editor = _sld_number("gradientUpperAbsorptionInput", 0.0)
        self.lower_editor = _sld_number("gradientLowerSldInput", -1_000_000.0)
        self.lower_absorption_editor = _sld_number("gradientLowerAbsorptionInput", 0.0)
        self.thickness_editor = number_editor("gradientThicknessInput", 0.2, 1.0)
        self.roughness_editor = number_editor("gradientRoughnessInput", 0.0)
        self.microslab_editor = number_editor("gradientMicroslabInput", 0.001, 0.1)
        self.error_label = QLabel()
        self.error_label.setObjectName("gradientDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = dialog_buttons("gradientDialogButtons")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        # 每只框预填时显示的值与它背后的原值；确定时框里还是那个数，就交回原值。
        self._exact: dict[QDoubleSpinBox, tuple[float, float]] = {}
        if gradient is not None:
            self._prefill(gradient)
        self._arrange()

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("名称", self.name_editor)
        form.addRow("上侧 SLD 实部 (10⁻⁶ Å⁻²)", self.upper_editor)
        form.addRow("上侧 SLD 吸收 (10⁻⁶ Å⁻²)", self.upper_absorption_editor)
        form.addRow("下侧 SLD 实部 (10⁻⁶ Å⁻²)", self.lower_editor)
        form.addRow("下侧 SLD 吸收 (10⁻⁶ Å⁻²)", self.lower_absorption_editor)
        form.addRow("厚度 (nm)", self.thickness_editor)
        form.addRow("上界面粗糙度 (nm)", self.roughness_editor)
        form.addRow("切片上限 (nm)", self.microslab_editor)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _prefill(self, gradient: api.GradientLayerSpec) -> None:
        self.name_editor.setText(gradient.name)
        for editor, exact, scale in (
            (self.upper_editor, gradient.upper_sld_a2.real, SLD_DISPLAY_SCALE),
            (self.upper_absorption_editor, gradient.upper_sld_a2.imag, SLD_DISPLAY_SCALE),
            (self.lower_editor, gradient.lower_sld_a2.real, SLD_DISPLAY_SCALE),
            (self.lower_absorption_editor, gradient.lower_sld_a2.imag, SLD_DISPLAY_SCALE),
            (self.thickness_editor, gradient.thickness_a, 0.1),
            (self.roughness_editor, gradient.roughness_a, 0.1),
            (self.microslab_editor, gradient.microslab_max_a, 0.1),
        ):
            editor.setValue(exact * scale)
            self._exact[editor] = (editor.value(), exact)

    def _value(self, editor: QDoubleSpinBox, scale: float) -> float:
        shown, exact = self._exact.get(editor, (None, 0.0))
        return exact if editor.value() == shown else editor.value() / scale

    def _sld(self, real: QDoubleSpinBox, imaginary: QDoubleSpinBox) -> complex:
        return complex(self._value(real, SLD_DISPLAY_SCALE), self._value(imaginary, SLD_DISPLAY_SCALE))

    def _accept_fields(self) -> None:
        try:
            candidate = api.GradientLayerSpec(
                self.name_editor.text().strip(),
                self._sld(self.upper_editor, self.upper_absorption_editor),
                self._sld(self.lower_editor, self.lower_absorption_editor),
                self._value(self.thickness_editor, 0.1),
                roughness_a=self._value(self.roughness_editor, 0.1),
                microslab_max_a=self._value(self.microslab_editor, 0.1),
            )
            if self._commit_gradient is not None:
                self._commit_gradient(candidate)
        except (TypeError, ValueError) as error:
            show_rejection(self, self.error_label, error)
            return
        self._gradient = candidate
        self.error_label.hide()
        self.accept()

    def gradient(self) -> api.GradientLayerSpec:
        if self._gradient is None:
            raise LookupError("gradient layer has not been confirmed")
        return self._gradient


def _sld_number(name: str, minimum: float) -> QDoubleSpinBox:
    editor = number_editor(name, minimum)
    editor.setDecimals(6)
    return editor
