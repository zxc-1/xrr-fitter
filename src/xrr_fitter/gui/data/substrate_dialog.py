"""Group-scoped substrate selection for ambiguous automatic structures."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class SubstrateDialog(QDialog):
    """Collect one explicit substrate token shared by a structure group."""

    def __init__(
        self,
        layers_backing_to_surface: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("substrateDialog")
        self.setWindowTitle("选择实际基底")
        self.setAccessibleName("选择实际基底")
        self.setModal(True)
        self.substrate_editor = QLineEdit()
        self.substrate_editor.setObjectName("substrateMaterialEditor")
        self.substrate_editor.setPlaceholderText("材料，例如 Al2O3")
        self.stack_label = QLabel(" + ".join(layers_backing_to_surface))
        self.stack_label.setObjectName("substrateLayerStack")
        self.error_label = QLabel()
        self.error_label.setObjectName("substrateValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form = QFormLayout()
        form.addRow("共享层序", self.stack_label)
        form.addRow("实际基底", self.substrate_editor)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def substrate_token(self) -> str:
        token = self.substrate_editor.text().strip()
        if not token:
            raise ValueError("substrate material must not be empty")
        return token

    def accept(self) -> None:
        try:
            self.substrate_token()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        self.error_label.hide()
        super().accept()
