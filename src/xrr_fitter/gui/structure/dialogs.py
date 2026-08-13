"""Modal forms that build complete immutable structure components."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api


def _number(name: str, minimum: float, value: float = 0.0) -> QDoubleSpinBox:
    editor = QDoubleSpinBox()
    editor.setObjectName(name)
    editor.setDecimals(8)
    editor.setRange(minimum, 1_000_000.0)
    editor.setValue(value)
    return editor


def _buttons(name: str) -> QDialogButtonBox:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.setObjectName(name)
    return buttons


def _table_cell(table: QTableWidget, row: int, column: int, label: str) -> str:
    item: QTableWidgetItem | None = table.item(row, column)
    value = "" if item is None else item.text().strip()
    if not value:
        raise ValueError(f"{label} row {row + 1} is incomplete")
    return value


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
        self.density_editor = _number("layerDensityInput", 0.000001, 1.0)
        self.thickness_editor = _number("layerThicknessInput", 0.2, 1.0)
        self.roughness_editor = _number("layerRoughnessInput", 0.0)
        self.transition_toggle = QCheckBox()
        self.transition_toggle.setObjectName("layerTransitionToggle")
        self.transition_toggle.setAccessibleName("启用界面过渡")
        self.transition_toggle.setToolTip("用可归一化的过渡函数族替代 Névot-Croce 粗糙度")
        self.microslab_editor = _number("layerMicroslabInput", 0.001, 0.1)
        self.branch_table = QTableWidget(2, 3)
        self.branch_table.setObjectName("layerTransitionTable")
        self.branch_table.setHorizontalHeaderLabels(("过渡类型", "权重", "宽度 (nm)"))
        self.error_label = QLabel()
        self.error_label.setObjectName("layerDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = _buttons("layerDialogButtons")
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
        kind, weight, width = (_table_cell(self.branch_table, row, column, "transition branch") for column in range(3))
        return api.TransitionBranch(kind, float(weight), float(width) * 10.0)

    def _transition(self) -> api.InterfaceTransition | None:
        if not self.transition_toggle.isChecked():
            return None
        branches = tuple(self._branch_at(row) for row in range(self.branch_table.rowCount()))
        return api.InterfaceTransition(branches, self.microslab_editor.value() * 10.0)

    def _accept_fields(self) -> None:
        try:
            formula = self.formula_editor.text().strip()
            material = api.MaterialSpec(formula, formula, self.density_editor.value())
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
            self._show_error(error)
            return
        self._layer = candidate
        self.error_label.hide()
        self.accept()

    def _show_error(self, error: BaseException) -> None:
        self.error_label.setText(str(error))
        self.error_label.show()
        if not self.isVisible():
            self.show()

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
        self.density_editor = _number("backingDensityInput", 0.000001, 1.0)
        if backing.bulk_density_g_cm3 is not None:
            self.density_editor.setValue(backing.bulk_density_g_cm3)
        self.roughness_editor = _number(
            "backingRoughnessInput",
            0.0,
            roughness_a / 10.0,
        )
        self.error_label = QLabel()
        self.error_label.setObjectName("backingDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = _buttons("backingDialogButtons")
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
            backing = api.MaterialSpec(formula, formula, self.density_editor.value())
            roughness_a = self.roughness_editor.value() * 10.0
            if self._commit_backing is not None:
                self._commit_backing(backing, roughness_a)
        except (TypeError, ValueError) as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            if not self.isVisible():
                self.show()
            return
        self._backing = backing
        self._roughness_a = roughness_a
        self.error_label.hide()
        self.accept()

    def backing(self) -> tuple[api.MaterialSpec, float]:
        if self._backing is None or self._roughness_a is None:
            raise LookupError("backing has not been confirmed")
        return self._backing, self._roughness_a


class PeriodicDialog(QDialog):
    """Collect one ordered periodic cell using an explicit layer table."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        commit_block: Callable[[api.PeriodicBlock], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("periodicBlockDialog")
        self.setWindowTitle("添加周期块")
        self.setAccessibleName("添加周期块")
        self.setModal(True)
        self._commit_block = commit_block
        self._block: api.PeriodicBlock | None = None
        self.name_editor = QLineEdit()
        self.name_editor.setObjectName("periodicNameInput")
        self.repeats_editor = QSpinBox()
        self.repeats_editor.setObjectName("periodicRepeatsInput")
        self.repeats_editor.setRange(1, 1_000_000)
        self.repeats_editor.setValue(2)
        self.table = QTableWidget(2, 5)
        self.table.setObjectName("periodicLayerTable")
        self.table.setHorizontalHeaderLabels(("名称", "化学式", "密度", "厚度 (nm)", "粗糙度 (nm)"))
        self.error_label = QLabel()
        self.error_label.setObjectName("periodicDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = _buttons("periodicDialogButtons")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        self._arrange()

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("名称", self.name_editor)
        form.addRow("重复次数", self.repeats_editor)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.table)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _accept_fields(self) -> None:
        try:
            layers = tuple(self._layer_at(row) for row in range(self.table.rowCount()))
            candidate = api.PeriodicBlock(
                self.name_editor.text().strip(),
                layers,
                self.repeats_editor.value(),
            )
            if self._commit_block is not None:
                self._commit_block(candidate)
        except (TypeError, ValueError) as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            if not self.isVisible():
                self.show()
            return
        self._block = candidate
        self.error_label.hide()
        self.accept()

    def _layer_at(self, row: int) -> api.LayerSpec:
        fields = tuple(self._cell(row, column) for column in range(5))
        name, formula, density, thickness, roughness = fields
        material = api.MaterialSpec(formula, formula, float(density))
        return api.LayerSpec(
            name,
            material,
            float(thickness) * 10.0,
            roughness_a=float(roughness) * 10.0,
        )

    def _cell(self, row: int, column: int) -> str:
        return _table_cell(self.table, row, column, "periodic layer")

    def block(self) -> api.PeriodicBlock:
        if self._block is None:
            raise LookupError("periodic block has not been confirmed")
        return self._block
