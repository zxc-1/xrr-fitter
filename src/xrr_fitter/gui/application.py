"""Construction rules for the process-wide Qt application."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from xrr_fitter.gui.theme import apply_theme


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Reuse the QApplication or create one from an owned argument copy."""
    existing = QCoreApplication.instance()
    if isinstance(existing, QApplication):
        apply_theme(existing)
        return existing
    if existing is not None:
        raise RuntimeError("existing QCoreApplication is not a QApplication")
    application = QApplication(list(argv or ()))
    apply_theme(application)
    return application
