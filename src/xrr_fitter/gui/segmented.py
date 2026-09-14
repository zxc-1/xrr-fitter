"""The design's segmented control: one exclusive choice drawn as a lit half.

A pair of checkable menu entries can be checked together, checked neither, and
says nothing at all while the menu is shut.  A segment answers "which one is it"
without being opened, which is why the mockup draws every either/or choice this
way: one pill, one border, one filled half.

The control owns no state of its own.  It reports the half the user pressed and
lights the half it is told to, so whatever already owns the choice -- a QAction,
a project field -- stays the single source of truth.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget


class SegmentedControl(QWidget):
    """Choose one of several named halves, drawn as the design's ``.seg`` pill."""

    chosen = Signal(str)

    def __init__(
        self,
        specs: tuple[tuple[str, str, str, str], ...],
        *,
        name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        # Qt paints the background of a plain QWidget only when it is told to
        # style itself; without this the pill's border and its filled half are
        # both absent and the halves read as two loose buttons.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("segmented", True)
        self._buttons: dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No spacing: the halves share the pill's single border, so a gap would
        # split one control into two.
        layout.setSpacing(0)
        for index, (value, object_name, text, tooltip) in enumerate(specs):
            button = QToolButton(self)
            button.setObjectName(object_name)
            button.setText(text)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setAccessibleName(text)
            button.setToolTip(tooltip)
            button.setProperty("segmentValue", value)
            self._group.addButton(button, index)
            self._buttons[value] = button
            layout.addWidget(button)
            button.clicked.connect(self._segment_clicked)

    def buttons(self) -> dict[str, QToolButton]:
        return dict(self._buttons)

    def value(self) -> str:
        """The lit half, or the empty string before anything has been lit."""
        for value, button in self._buttons.items():
            if button.isChecked():
                return value
        return ""

    def set_value(self, value: str) -> None:
        """Light the named half without reporting it, so the owner can drive it.

        Silent on purpose: this is the path taken when the state changed
        elsewhere, and re-announcing it there would send the owner a change it
        just made.
        """
        self._buttons[value].setChecked(True)

    def _segment_clicked(self) -> None:
        self.chosen.emit(str(self.sender().property("segmentValue")))
