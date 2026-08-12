"""Modal form that builds one parameter prior through the public API only."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api

# Per kind, the visible scalar-parameter labels in stored order. The labels are
# a display concern owned here; the arity and admissibility stay with the API's
# PriorSpec, which this dialog defers to at commit time. uniform carries none.
PRIOR_FIELDS: dict[str, tuple[str, ...]] = {
    "uniform": (),
    "normal": ("μ", "σ"),
    "lognormal": ("μ (log)", "σ (log)"),
    "soft_range": ("下限", "上限", "σ"),
}

# The widest kind fixes how many reusable spin boxes the form owns; narrower
# kinds hide the surplus rather than rebuilding the layout on every switch.
MAX_PRIOR_FIELDS = max(map(len, PRIOR_FIELDS.values()))


def _param_spin(name: str) -> QDoubleSpinBox:
    editor = QDoubleSpinBox()
    editor.setObjectName(name)
    editor.setDecimals(8)
    # The lower bound is deliberately negative and permissive: bound and scale
    # admissibility belong to PriorSpec, so a non-positive sigma must reach it
    # to raise rather than being clamped away into a silently valid entry.
    editor.setRange(-1_000_000.0, 1_000_000.0)
    return editor


class PriorDialog(QDialog):
    """Collect a prior kind and its scalars, returning a validated PriorSpec."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        commit_spec: Callable[[api.PriorSpec], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("priorDialog")
        self.setWindowTitle("编辑先验")
        self.setAccessibleName("编辑参数先验")
        self.setModal(True)
        self._commit_spec = commit_spec
        self._spec: api.PriorSpec | None = None
        self.kind_select = QComboBox()
        self.kind_select.setObjectName("priorKindSelect")
        self.kind_select.addItems(tuple(PRIOR_FIELDS))
        self._labels = tuple(QLabel() for _ in range(MAX_PRIOR_FIELDS))
        self._params = tuple(_param_spin(f"priorParam{index}") for index in range(MAX_PRIOR_FIELDS))
        self.error_label = QLabel()
        self.error_label.setObjectName("priorDialogError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.setObjectName("priorDialogButtons")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        self.kind_select.currentTextChanged.connect(self._sync_fields)
        self._arrange()
        self._sync_fields(self.kind_select.currentText())

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("类型", self.kind_select)
        for label, editor in zip(self._labels, self._params, strict=True):
            form.addRow(label, editor)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _sync_fields(self, kind: str) -> None:
        labels = PRIOR_FIELDS[kind]
        for index, (label, editor) in enumerate(zip(self._labels, self._params, strict=True)):
            active = index < len(labels)
            label.setText(labels[index] if active else "")
            label.setVisible(active)
            editor.setVisible(active)

    def _accept_fields(self) -> None:
        kind = self.kind_select.currentText()
        parameters = tuple(editor.value() for editor in self._params[: len(PRIOR_FIELDS[kind])])
        try:
            spec = api.PriorSpec(kind, parameters)
            if self._commit_spec is not None:
                self._commit_spec(spec)
        except (TypeError, ValueError) as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            if not self.isVisible():
                self.show()
            return
        self._spec = spec
        self.error_label.hide()
        self.accept()

    def spec(self) -> api.PriorSpec | None:
        return self._spec
