"""Modal form that builds one parameter prior through the public API only."""

from __future__ import annotations

from collections.abc import Callable
from math import log

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
from xrr_fitter.gui.parameters.table import _uses_nm

# Per kind, the scalar labels in stored order. Unit annotations are added from
# the current declaration, while arity and admissibility remain API concerns.
PRIOR_FIELDS: dict[str, tuple[str, ...]] = {
    "uniform": (),
    "normal": ("μ", "σ"),
    "lognormal": ("μ (log)", "σ (log)"),
    "soft_range": ("下限", "上限", "σ"),
}

# The widest kind fixes how many reusable spin boxes the form owns; narrower
# kinds hide the surplus rather than rebuilding the layout on every switch.
MAX_PRIOR_FIELDS = max(map(len, PRIOR_FIELDS.values()))
ANGSTROM_PER_NM = 10.0


def _param_spin(name: str) -> QDoubleSpinBox:
    editor = QDoubleSpinBox()
    editor.setObjectName(name)
    editor.setDecimals(8)
    # The lower bound is deliberately negative and permissive: bound and scale
    # admissibility belong to PriorSpec, so a non-positive sigma must reach it
    # to raise rather than being clamped away into a silently valid entry.
    editor.setRange(-1_000_000.0, 1_000_000.0)
    return editor


def _prior_uses_nm(definition: api.ParameterDefinition) -> bool:
    """Use nm only when the prior itself lives in a physical length space."""
    return _uses_nm(definition) and definition.transform != "roughness_fraction"


def _quantity_unit(definition: api.ParameterDefinition) -> str:
    if definition.transform == "roughness_fraction":
        return "分数"
    if _prior_uses_nm(definition):
        return "nm"
    return definition.unit


def _field_labels(
    definition: api.ParameterDefinition,
    kind: str,
) -> tuple[str, ...]:
    labels = PRIOR_FIELDS[kind]
    unit = _quantity_unit(definition)
    if not unit:
        return labels
    if kind == "lognormal":
        return (f"μ (log {unit})", "σ (log)")
    return tuple(f"{label} ({unit})" for label in labels)


def _to_persisted_parameters(
    definition: api.ParameterDefinition,
    kind: str,
    values: tuple[float, ...],
) -> tuple[float, ...]:
    if not _prior_uses_nm(definition):
        return values
    if kind == "lognormal":
        return (values[0] + log(ANGSTROM_PER_NM), values[1])
    return tuple(value * ANGSTROM_PER_NM for value in values)


def _to_display_parameters(
    definition: api.ParameterDefinition,
    prior: api.PriorSpec,
) -> tuple[float, ...]:
    values = prior.parameters
    if not _prior_uses_nm(definition):
        return values
    if prior.kind == "lognormal":
        return (values[0] - log(ANGSTROM_PER_NM), values[1])
    return tuple(value / ANGSTROM_PER_NM for value in values)


class PriorDialog(QDialog):
    """Collect a prior kind and its scalars, returning a validated PriorSpec."""

    def __init__(
        self,
        definition: api.ParameterDefinition,
        parent: QWidget | None = None,
        *,
        existing_prior: api.PriorSpec | None = None,
        commit_spec: Callable[[api.PriorSpec], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("priorDialog")
        self.setWindowTitle("编辑先验")
        self.setAccessibleName("编辑参数先验")
        self.setModal(True)
        self._definition = definition
        self._commit_spec = commit_spec
        self._spec: api.PriorSpec | None = None
        self.kind_select = QComboBox()
        self.kind_select.setObjectName("priorKindSelect")
        self.kind_select.addItems(tuple(PRIOR_FIELDS))
        self._labels = tuple(QLabel() for _ in range(MAX_PRIOR_FIELDS))
        for index, label in enumerate(self._labels):
            label.setObjectName(f"priorParamLabel{index}")
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
        if existing_prior is not None:
            self._load_existing(existing_prior)

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
        labels = _field_labels(self._definition, kind)
        for index, (label, editor) in enumerate(zip(self._labels, self._params, strict=True)):
            active = index < len(labels)
            label.setText(labels[index] if active else "")
            label.setVisible(active)
            editor.setVisible(active)

    def _load_existing(self, prior: api.PriorSpec) -> None:
        self.kind_select.setCurrentText(prior.kind)
        values = _to_display_parameters(self._definition, prior)
        for editor, value in zip(self._params, values, strict=False):
            editor.setValue(value)

    def _accept_fields(self) -> None:
        kind = self.kind_select.currentText()
        displayed = tuple(editor.value() for editor in self._params[: len(PRIOR_FIELDS[kind])])
        try:
            parameters = _to_persisted_parameters(self._definition, kind, displayed)
            spec = api.PriorSpec(kind, parameters)
            api.validate_parameter_priors(
                (self._definition,),
                (api.ParameterPrior(self._definition.name, spec),),
            )
            if self._commit_spec is not None:
                self._commit_spec(spec)
        except (OverflowError, TypeError, ValueError) as error:
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
