"""Construction rules for the process-wide Qt application."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QSplashScreen

from xrr_fitter.gui.theme import apply_theme


def _build_app_icon() -> QIcon:
    """Generate a multi-resolution app icon: XRR curve on accent background."""
    icon = QIcon()
    for size in (64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Rounded rect background with accent gradient
        radius = size * 0.18
        gradient = QLinearGradient(0, 0, 0, size)
        gradient.setColorAt(0.0, QColor("#3A7BF2"))
        gradient.setColorAt(1.0, QColor("#2558B8"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(gradient)
        p.drawRoundedRect(QRectF(0, 0, size, size), radius, radius)

        # Draw an XRR-like oscillating decay curve in white
        pen = QPen(QColor(255, 255, 255, 230), size * 0.03)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        path = QPainterPath()
        margin = size * 0.15
        w = size - 2 * margin
        h = size - 2 * margin
        n_points = 80
        for i in range(n_points):
            t = i / (n_points - 1)
            x = margin + t * w
            envelope = math.exp(-3.0 * t)
            oscillation = math.sin(t * 6.0 * math.pi)
            y = margin + h * 0.15 + h * 0.7 * (1.0 - envelope * (0.3 + 0.7 * (0.5 + 0.5 * oscillation)))
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        p.drawPath(path)
        p.end()
        icon.addPixmap(pixmap)
    return icon


def show_splash(application: QApplication) -> QSplashScreen:
    """Display a lightweight splash screen matching the current theme."""
    size = 360
    pixmap = QPixmap(size, size)
    palette = application.palette()
    is_dark = palette.window().color().lightness() < 128
    bg = QColor("#1E1F22") if is_dark else QColor("#FFFFFF")
    fg = QColor("#E8E8EA") if is_dark else QColor("#202020")
    pixmap.fill(bg)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Draw icon in center
    icon = _build_app_icon()
    icon_size = 80
    icon_rect = QRectF((size - icon_size) / 2, size * 0.25, icon_size, icon_size)
    icon.paint(p, icon_rect.toAlignedRect())

    # Title
    title_font = p.font()
    title_font.setPointSize(18)
    title_font.setBold(True)
    p.setFont(title_font)
    p.setPen(fg)
    p.drawText(QRectF(0, size * 0.55, size, 30), Qt.AlignmentFlag.AlignHCenter, "XRR Fitter")

    # Loading text
    sub_font = p.font()
    sub_font.setPointSize(11)
    sub_font.setBold(False)
    p.setFont(sub_font)
    muted = QColor(fg)
    muted.setAlpha(160)
    p.setPen(muted)
    p.drawText(QRectF(0, size * 0.68, size, 24), Qt.AlignmentFlag.AlignHCenter, "正在加载…")

    p.end()
    splash = QSplashScreen(pixmap)
    splash.show()
    application.processEvents()
    return splash


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
    application.setWindowIcon(_build_app_icon())
    return application
