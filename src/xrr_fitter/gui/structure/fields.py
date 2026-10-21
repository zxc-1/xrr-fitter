"""The input parts every structure dialog is built from: numbers, density, buttons, the rejection line."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
)

import xrr_fitter.api as api
from xrr_fitter.gui.accessibility import localize_standard_buttons
from xrr_fitter.gui.structure.errors import is_formula_error, structure_error_text

# 密度框空着时显示的字。``QDoubleSpinBox`` 只能把「没有数」画成顶在下限上的特殊文本。
DENSITY_BLANK_TEXT = "请填写"


def number_editor(name: str, minimum: float, value: float = 0.0) -> QDoubleSpinBox:
    editor = QDoubleSpinBox()
    editor.setObjectName(name)
    editor.setDecimals(8)
    editor.setRange(minimum, 1_000_000.0)
    editor.setValue(value)
    return editor


def density_editor(name: str) -> QDoubleSpinBox:
    """密度框：下限顶着「请填写」，表示这一格还没有人给过数。"""
    editor = number_editor(name, 0.0)
    editor.setSpecialValueText(DENSITY_BLANK_TEXT)
    return editor


class NominalDensity:
    """让密度框跟着化学式走，直到读者自己改过它。

    表里有的化学式带入名义密度；表外的把框清空，要读者自己填——以前这里先填 1.0，一个没人写
    过的数，提交出去就成了这层的密度。
    """

    def __init__(self, formula: QLineEdit, density: QDoubleSpinBox, initial: float | None = None) -> None:
        self._formula = formula
        self._density = density
        if initial is not None:
            density.setValue(initial)
        self._typed = False
        density.valueChanged.connect(self._mark_typed)
        formula.textChanged.connect(self._follow)

    def _mark_typed(self, *_args: object) -> None:
        self._typed = True

    def _follow(self, text: str) -> None:
        if self._typed:
            return
        nominal = api.INITIAL_DENSITIES_G_CM3.get(text.strip())
        blocker = QSignalBlocker(self._density)
        self._density.setValue(self._density.minimum() if nominal is None else nominal)
        del blocker

    def value(self) -> float:
        if self._density.value() > self._density.minimum():
            return self._density.value()
        formula = self._formula.text().strip()
        unlisted = "" if formula in api.INITIAL_DENSITIES_G_CM3 else "（它不在初始密度表里；元素符号区分大小写）"
        raise ValueError(f"请填写 {formula or '这一层'} 的密度 (g/cm³){unlisted}。")


def dialog_buttons(name: str) -> QDialogButtonBox:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.setObjectName(name)
    localize_standard_buttons(buttons, confirm="确定")
    return buttons


def show_rejection(dialog: QDialog, label: QLabel, error: BaseException, formula: QLineEdit | None = None) -> None:
    """把拒绝写成中文留在对话框里；化学式错了就整段选中它。

    对话框里没有「上一个有效值」可退：留住读者打的字，整段选中，重打一遍就是替换。
    """
    label.setText(structure_error_text(error))
    label.show()
    if not dialog.isVisible():
        dialog.show()
    if formula is not None and is_formula_error(error):
        formula.setFocus()
        formula.selectAll()


def table_cell(table: QTableWidget, row: int, column: int, label: str) -> str:
    item: QTableWidgetItem | None = table.item(row, column)
    value = "" if item is None else item.text().strip()
    if not value:
        raise ValueError(f"{label} row {row + 1} is incomplete")
    return value
