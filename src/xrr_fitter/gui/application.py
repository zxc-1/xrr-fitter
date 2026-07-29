"""Construction rules for the process-wide Qt application."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Reuse the QApplication or create one from an owned argument copy."""
    existing = QCoreApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    if existing is not None:
        raise RuntimeError("existing QCoreApplication is not a QApplication")
    return QApplication(list(argv or ()))
