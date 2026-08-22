"""Shared dialog template for consistent styling across all modal dialogs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from xrr_fitter.gui import theme


class StyledDialog(QDialog):
    """Base dialog with structured layout: header, body card, button box.

    Subclasses populate the body by adding widgets to ``self.body_layout``.
    The button box is accessible via ``self.button_box``.
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        parent: QWidget | None = None,
        *,
        buttons: QDialogButtonBox.StandardButton = (
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        ),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        # Root layout
        root = QVBoxLayout(self)
        root.setContentsMargins(theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG)
        root.setSpacing(theme.SPACE_MD)

        # Header
        header = QLabel(title)
        header.setProperty("sectionHeader", True)
        header.setObjectName("dialogHeader")
        root.addWidget(header)

        if description:
            desc = QLabel(description)
            desc.setProperty("mutedText", True)
            desc.setWordWrap(True)
            desc.setObjectName("dialogDescription")
            root.addWidget(desc)

        # Body card
        self._card = QFrame()
        self._card.setProperty("sectionCard", True)
        self.body_layout = QVBoxLayout(self._card)
        self.body_layout.setContentsMargins(theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_MD)
        self.body_layout.setSpacing(theme.SPACE_SM)
        root.addWidget(self._card)

        # Button box
        self.button_box = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box, 0, Qt.AlignmentFlag.AlignRight)
