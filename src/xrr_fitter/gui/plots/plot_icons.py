"""Theme-aware vector glyphs for the plot interaction controls.

The plot toolbar drives eight commands -- three interaction modes, three
navigation actions and two one-shot zooms -- and each reached the user as a bare
text button, so a graphical tool advertised itself as a wall of words.  A glyph
belongs to each command here, painted rather than loaded: the bundled Matplotlib
toolbar PNGs are a fixed black that vanishes on a dark canvas, and QStyle carries
no standard pixmap for "select a fit range" or "toggle a point mask".  Painting
from the active palette's text colour instead keeps every glyph legible in both
the light and the dark theme.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QApplication

# Canonical design grid.  Every painter draws as though the icon were 16 px and
# the transform in plot_icon scales it to the requested size, so one set of
# coordinates serves every device pixel ratio.
GRID = 16.0


def _text_color() -> QColor:
    """The active palette's text colour, so glyphs flip with the theme.

    The bundled Matplotlib icons are a fixed black; reading the palette instead
    is what keeps a glyph visible once the application adopts a dark palette.
    """
    application = QApplication.instance()
    if application is None:
        return QColor(0, 0, 0)
    return application.palette().windowText().color()


def _pen(color: QColor, width: float = 1.6) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _arrow_head(painter: QPainter, tip: QPointF, dx: float, dy: float, color: QColor) -> None:
    """Fill a small triangle at ``tip`` pointing along the unit vector (dx, dy)."""
    back = QPointF(tip.x() - dx * 3.2, tip.y() - dy * 3.2)
    left = QPointF(back.x() - dy * 2.0, back.y() + dx * 2.0)
    right = QPointF(back.x() + dy * 2.0, back.y() - dx * 2.0)
    painter.setBrush(color)
    painter.drawPolygon(QPolygonF([tip, left, right]))
    painter.setBrush(Qt.BrushStyle.NoBrush)


def _paint_view(p: QPainter, c: QColor) -> None:
    # An eye: the "查看" mode is for reading the plot rather than acting on it.
    path = QPainterPath()
    path.moveTo(2.5, 8.0)
    path.quadTo(8.0, 3.0, 13.5, 8.0)
    path.quadTo(8.0, 13.0, 2.5, 8.0)
    p.drawPath(path)
    p.setBrush(c)
    p.drawEllipse(QPointF(8.0, 8.0), 1.9, 1.9)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _paint_range(p: QPainter, c: QColor) -> None:
    # Two uprights with a translucent band between: a selected angle window.
    fill = QColor(c)
    fill.setAlpha(60)
    p.fillRect(QRectF(4.5, 3.0, 7.0, 10.0), fill)
    p.drawLine(QPointF(4.5, 2.5), QPointF(4.5, 13.5))
    p.drawLine(QPointF(11.5, 2.5), QPointF(11.5, 13.5))


def _paint_mask(p: QPainter, c: QColor) -> None:
    # A single point struck through: toggling one point in or out of the fit.
    p.setBrush(c)
    p.drawEllipse(QPointF(8.0, 8.0), 2.6, 2.6)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(2.8, 13.2), QPointF(13.2, 2.8))


def _paint_pan(p: QPainter, c: QColor) -> None:
    # A four-way cross: drag the plot around under the cursor.
    p.drawLine(QPointF(8.0, 3.0), QPointF(8.0, 13.0))
    p.drawLine(QPointF(3.0, 8.0), QPointF(13.0, 8.0))
    _arrow_head(p, QPointF(8.0, 2.2), 0.0, -1.0, c)
    _arrow_head(p, QPointF(8.0, 13.8), 0.0, 1.0, c)
    _arrow_head(p, QPointF(2.2, 8.0), -1.0, 0.0, c)
    _arrow_head(p, QPointF(13.8, 8.0), 1.0, 0.0, c)


def _paint_zoom(p: QPainter, c: QColor) -> None:
    # A magnifier over a "+": drag a rectangle to magnify it.
    p.drawEllipse(QPointF(6.5, 6.5), 4.0, 4.0)
    p.drawLine(QPointF(9.4, 9.4), QPointF(14.0, 14.0))
    p.drawLine(QPointF(6.5, 4.6), QPointF(6.5, 8.4))
    p.drawLine(QPointF(4.6, 6.5), QPointF(8.4, 6.5))


def _paint_home(p: QPainter, c: QColor) -> None:
    # A house: return the view to the limits the plot was drawn with.
    roof = QPainterPath()
    roof.moveTo(2.5, 8.2)
    roof.lineTo(8.0, 2.8)
    roof.lineTo(13.5, 8.2)
    p.drawPath(roof)
    p.drawRect(QRectF(4.5, 8.2, 7.0, 5.3))
    p.drawRect(QRectF(7.0, 10.5, 2.0, 3.0))


def _paint_zoom_to_range(p: QPainter, c: QColor) -> None:
    # Brackets closing inward on a span: pull the view onto the fit range.
    p.drawLine(QPointF(2.5, 3.5), QPointF(2.5, 12.5))
    p.drawLine(QPointF(13.5, 3.5), QPointF(13.5, 12.5))
    p.drawLine(QPointF(4.6, 8.0), QPointF(7.2, 8.0))
    p.drawLine(QPointF(11.4, 8.0), QPointF(8.8, 8.0))
    _arrow_head(p, QPointF(4.3, 8.0), -1.0, 0.0, c)
    _arrow_head(p, QPointF(11.7, 8.0), 1.0, 0.0, c)


def _paint_reset_zoom(p: QPainter, c: QColor) -> None:
    # Four corners opening outward: restore the complete view.
    corners = (
        ((5.2, 2.6), (2.6, 2.6), (2.6, 5.2), (-1.0, -1.0)),
        ((10.8, 2.6), (13.4, 2.6), (13.4, 5.2), (1.0, -1.0)),
        ((5.2, 13.4), (2.6, 13.4), (2.6, 10.8), (-1.0, 1.0)),
        ((10.8, 13.4), (13.4, 13.4), (13.4, 10.8), (1.0, 1.0)),
    )
    for (ax, ay), (bx, by), (cx, cy), (dx, dy) in corners:
        p.drawLine(QPointF(ax, ay), QPointF(bx, by))
        p.drawLine(QPointF(bx, by), QPointF(cx, cy))
        norm = (dx * dx + dy * dy) ** 0.5
        _arrow_head(p, QPointF(bx, by), dx / norm, dy / norm, c)


PAINTERS: dict[str, Callable[[QPainter, QColor], None]] = {
    "view": _paint_view,
    "range": _paint_range,
    "mask": _paint_mask,
    "pan": _paint_pan,
    "zoom": _paint_zoom,
    "home": _paint_home,
    "zoom_to_range": _paint_zoom_to_range,
    "reset_zoom": _paint_reset_zoom,
}


def plot_icon(name: str, *, size: int = 16) -> QIcon:
    """The glyph for one plot control, painted in the active theme's text colour.

    Painting on demand -- rather than caching -- means the next toolbar built
    after a palette change picks up the new colour, matching how the figures
    themselves re-read the palette on every redraw.
    """
    painter_fn = PAINTERS.get(name)
    if painter_fn is None:
        raise KeyError(f"no plot glyph for control: {name!r}")
    color = _text_color()
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.scale(size / GRID, size / GRID)
        painter.setPen(_pen(color))
        painter_fn(painter, color)
    finally:
        painter.end()
    return QIcon(pixmap)
