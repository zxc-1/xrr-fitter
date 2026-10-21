"""Construction rules for the process-wide Qt application."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QRectF, Qt, QTranslator
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

from xrr_fitter.gui.theme import FONT_PX_MD, apply_theme, light_palette

LOG = logging.getLogger(__name__)

# 启动画面标题的字号。原先写 18pt，在 96 dpi 落 24px、在 Mac 上只剩 18px；按像素给两边一样大。
SPLASH_TITLE_FONT_PX = 24
QT_TRANSLATION_LOCALE = QLocale(QLocale.Language.Chinese, QLocale.Country.China)


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


class _StagedSplash(QSplashScreen):
    """启动画面：入口逐段调 ``showMessage`` 报告进度，文字统一排在底部居中。"""

    def __init__(self, pixmap: QPixmap, color: QColor) -> None:
        super().__init__(pixmap)
        self._color = color
        font = self.font()
        font.setPixelSize(FONT_PX_MD)
        self.setFont(font)

    def showMessage(self, message: str, *_args: object) -> None:  # noqa: N802 - Qt override
        # 图标和标题画在上半部；消息固定在底边居中，每段替换上一段。
        alignment = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
        super().showMessage(message, alignment, self._color)


def show_splash(application: QApplication) -> QSplashScreen:
    """Display a lightweight splash screen matching the current theme."""
    size = 360
    pixmap = QPixmap(size, size)
    palette = application.palette()
    # 取应用调色板而非写死的色值，启动画面与随后出现的主窗口同一个底色。
    bg = palette.window().color()
    fg = palette.windowText().color()
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
    title_font.setPixelSize(SPLASH_TITLE_FONT_PX)
    title_font.setBold(True)
    p.setFont(title_font)
    p.setPen(fg)
    p.drawText(QRectF(0, size * 0.55, size, 30), Qt.AlignmentFlag.AlignHCenter, "XRR Fitter")

    p.end()
    muted = QColor(fg)
    muted.setAlpha(160)
    splash = _StagedSplash(pixmap, muted)
    splash.showMessage("正在加载…")
    splash.show()
    application.processEvents()
    return splash


def _name_settings_scope(application: QApplication) -> None:
    """给 ``QSettings()`` 一个落脚处：明暗偏好（F52）存在这个组织 / 应用名下。

    只在还没有名字时才给——复用别人建好的应用（测试、嵌入）时不改它的身份。
    """
    if not application.organizationName():
        application.setOrganizationName("XRR Fitter")
    if not application.applicationName():
        application.setApplicationName("XRR Fitter")


def install_qt_translations(application: QApplication) -> QTranslator | None:
    """Give Qt's own strings the application's language.

    Standard buttons (是/否/确定), the editing menu of every text field and the macOS
    application menu are written by Qt, not by this package; without its catalogue
    they read in English inside a Chinese window. The catalogue ships with PySide6.
    The interface is Chinese whatever the desktop's locale, so the locale is fixed.
    """
    directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    translator = QTranslator(application)
    if not translator.load(QT_TRANSLATION_LOCALE, "qtbase", "_", directory):
        # 装不上只是 Qt 那几处字退回英文，界面照常可用；但要留下痕迹，好查打包是否漏了 .qm。
        LOG.warning("Qt translation qtbase_%s not found in %s", QT_TRANSLATION_LOCALE.name(), directory)
        return None
    application.installTranslator(translator)
    return translator


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Reuse the QApplication or create one from an owned argument copy."""
    existing = QCoreApplication.instance()
    if isinstance(existing, QApplication):
        _name_settings_scope(existing)
        apply_theme(existing)
        return existing
    if existing is not None:
        raise RuntimeError("existing QCoreApplication is not a QApplication")
    application = QApplication(list(argv or ()))
    _name_settings_scope(application)
    install_qt_translations(application)
    # 壳、面板、机架三层洗色得先装进调色板，样式表才有得可读：``build_stylesheet`` 是从
    # Window / Base / AlternateBase 三个角色里取这三层的，而平台默认的亮色板把它们给成同一
    # 个白，机架那一层于是一个像素也画不出来。复用已有应用时不动它的板——那多半是深色外观，
    # 或者是测试自己摆好的。
    application.setPalette(light_palette())
    apply_theme(application)
    application.setWindowIcon(_build_app_icon())
    return application
