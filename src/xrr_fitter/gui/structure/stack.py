"""Proportional section drawing of a sample stack, selectable by band.

The tree states the same numbers exactly, but a column of "40" and "4000" reads
as two similar rows; drawn to scale the second is the whole sample and the first
is a line.  This renders the section with QPainter rather than matplotlib because
the diagram has to stay inside the structure package to track the tree's current
row, and `tests/architecture/test_dependency_rules.py` admits numpy and
matplotlib only under `gui.plots`.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme

# Air and the substrate are semi-infinite, so they have no thickness to be
# proportional to and instead get a constant strip that reads as a boundary.
MEDIUM_BAND_H = 18

# 3D pseudo-perspective parameters: the side/top faces that give depth.
DEPTH_X = 14  # horizontal extent of the side face
DEPTH_Y = 5  # vertical extent of the top face

# A native oxide beside a micron film earns a fraction of a pixel.  Rounding it
# away would erase the layer the diagram exists to show, so every component keeps
# at least a band tall enough to click and to carry a label.
MIN_BAND_H = 14

# Enough rows to seat the media plus a few components before the panel scrolls.
MIN_VIEW_H = 160


@dataclass(frozen=True, slots=True)
class Band:
    """One drawn strip: a component when `index` is set, else a bounding medium."""

    index: int | None
    label: str
    detail: str
    top: int
    height: int
    fill: str


def _component_thickness(component: object) -> float:
    """Total thickness a component occupies, expanded over any repeats."""
    if isinstance(component, api.PeriodicBlock):
        cell = sum(float(layer.thickness_a) for layer in component.layers)
        return cell * int(component.repeats)
    return float(component.thickness_a)


def _component_detail(component: object) -> str:
    """Name the quantity that explains the band's size."""
    if isinstance(component, api.PeriodicBlock):
        cell = sum(float(layer.thickness_a) for layer in component.layers)
        return f"{component.repeats} × {cell / 10:g} nm"
    return f"{_component_thickness(component) / 10:g} nm"


def _medium_detail(material: api.MaterialSpec) -> str:
    if material.formula is not None:
        return material.formula.strip()
    return material.name


def _even_split(budget: int, count: int) -> list[int]:
    """An even split for when proportion cannot apply: too little room for every
    band to clear the floor, or no thickness to weight by.  It stops being
    proportional but keeps every band clickable."""
    base, extra = divmod(max(budget, 0), count)
    return [base + (1 if position < extra else 0) for position in range(count)]


def _settled_shares(thicknesses: list[float], budget: int, count: int) -> tuple[list[float], set[int]]:
    """Proportional shares, with any band that rounds below MIN_BAND_H pinned to it.

    A pin shrinks what the rest have to share and can starve a further band, so
    the pass repeats until it settles; the last unpinned band always clears the
    floor because `budget` seats every band at it, which keeps `order` non-empty.
    """
    pinned: set[int] = set()
    while True:
        free = budget - MIN_BAND_H * len(pinned)
        weight = sum(thicknesses[position] for position in range(count) if position not in pinned)
        shares = [
            float(MIN_BAND_H) if position in pinned else thicknesses[position] / weight * free
            for position in range(count)
        ]
        starved = {position for position, share in enumerate(shares) if share < MIN_BAND_H}
        if not starved:
            return shares, pinned
        pinned |= starved


def _rounded_heights(shares: list[float], pinned: set[int], count: int, budget: int) -> list[int]:
    """Floor the shares to whole pixels and hand the truncated remainder to the
    largest fractional parts, so the section meets the substrate without a seam."""
    heights = [int(share) for share in shares]
    order = sorted(
        (position for position in range(count) if position not in pinned),
        key=lambda position: shares[position] - int(shares[position]),
        reverse=True,
    )
    for offset in range(budget - sum(heights)):
        heights[order[offset % len(order)]] += 1
    return heights


def _proportional_heights(thicknesses: list[float], budget: int) -> list[int]:
    """Split `budget` px across thicknesses in proportion, floored at MIN_BAND_H.

    The honest split is the plain proportional one, so the floor applies only to
    the bands it has to rescue: those round to nothing, get pinned at MIN_BAND_H,
    and the pixels they borrow are charged to the bands that still have room.
    Spending the floor on every band instead would dilute the ratio the diagram
    exists to show, leaving a 20 nm and a 60 nm layer at 57 px and 143 px.
    """
    count = len(thicknesses)
    if count == 0:
        return []
    if budget < MIN_BAND_H * count or sum(thicknesses) <= 0:
        return _even_split(budget, count)
    shares, pinned = _settled_shares(thicknesses, budget, count)
    return _rounded_heights(shares, pinned, count, budget)


def stack_bands(structure: api.StructureSpec | None, height: int) -> tuple[Band, ...]:
    """Lay the structure out as bands from fronting down to backing."""
    if structure is None:
        return ()
    usable = max(int(height), MIN_VIEW_H)
    components = list(structure.components)
    budget = usable - 2 * MEDIUM_BAND_H
    heights = _proportional_heights([_component_thickness(value) for value in components], budget)
    bands: list[Band] = [
        Band(None, "Air", _medium_detail(structure.fronting), 0, MEDIUM_BAND_H, theme.DATA_NEUTRAL),
    ]
    cursor = MEDIUM_BAND_H
    for index, (component, band_height) in enumerate(zip(components, heights, strict=True)):
        bands.append(
            Band(
                index,
                component.name,
                _component_detail(component),
                cursor,
                band_height,
                theme.DATA_SEQUENCE[index % len(theme.DATA_SEQUENCE)],
            )
        )
        cursor += band_height
    bands.append(Band(None, "基底", _medium_detail(structure.backing), cursor, usable - cursor, theme.DATA_NEUTRAL))
    return tuple(bands)


class StackView(QWidget):
    """Draw the stack to scale and report which component a click lands on."""

    component_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("structureStack")
        self.setAccessibleName("样品结构示意图")
        self.setMinimumHeight(MIN_VIEW_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self._structure: api.StructureSpec | None = None
        self._selected: int | None = None
        self._hovered: int | None = None
        self._bands: tuple[Band, ...] = ()

    def load(self, structure: api.StructureSpec) -> None:
        self._structure = structure
        self._relayout()

    def clear(self) -> None:
        self._structure = None
        self._selected = None
        self._relayout()

    def bands(self) -> tuple[Band, ...]:
        return self._bands

    def selected_index(self) -> int | None:
        return self._selected

    def set_selected_index(self, index: int | None) -> None:
        """Track a selection made elsewhere without re-emitting the signal."""
        if index == self._selected:
            return
        self._selected = index
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._relayout()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().mousePressEvent(event)
        band = self._band_at(int(event.position().y()))
        if band is None or band.index is None:
            return
        self._selected = band.index
        self.update()
        self.component_selected.emit(band.index)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().mouseMoveEvent(event)
        band = self._band_at(int(event.position().y()))
        hovered = None if band is None else band.index
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().leaveEvent(event)
        if self._hovered is not None:
            self._hovered = None
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        tokens = theme.palette_tokens(self.palette())
        font = QFont(self.font())
        font.setPointSize(theme.FONT_PT_SM)
        painter.setFont(font)
        w = self.width()
        margin = 2
        front_left = float(margin)
        front_right = float(w - margin - DEPTH_X)
        front_width = front_right - front_left
        edge_pen = QPen(QColor(0, 0, 0, 90), 1.0)
        for idx, band in enumerate(self._bands):
            fill = QColor(band.fill)
            is_selected = band.index == self._selected
            is_hovered = band.index == self._hovered and not is_selected
            fill.setAlpha(240 if is_selected else 210)
            ft = float(band.top + DEPTH_Y)
            fh = float(band.height)
            front_rect = QRectF(front_left, ft, front_width, fh)
            self._paint_band_3d(painter, fill, front_left, front_right, ft, fh, band.index is not None, idx == 0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRect(front_rect)
            self._paint_band_overlay(painter, tokens, front_rect, is_hovered, is_selected)
            self._paint_band_edges(painter, edge_pen, front_left, front_right, ft, fh, band.index is not None)
            self._draw_caption(painter, band, QRect(int(front_left), int(ft), int(front_width), int(fh)))
        painter.end()

    def _paint_band_3d(
        self,
        painter: QPainter,
        fill: QColor,
        front_left: float,
        front_right: float,
        ft: float,
        fh: float,
        is_component: bool,
        is_first: bool,
    ) -> None:
        """Draw the 3D side face and optional top cap."""
        if is_component and fh >= MIN_BAND_H:
            side = fill.darker(170)
            side_poly = QPolygonF(
                [
                    QPointF(front_right, ft),
                    QPointF(front_right + DEPTH_X, ft - DEPTH_Y),
                    QPointF(front_right + DEPTH_X, ft + fh - DEPTH_Y),
                    QPointF(front_right, ft + fh),
                ]
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(side)
            painter.drawPolygon(side_poly)
        if is_first:
            top_face = fill.lighter(145)
            top_poly = QPolygonF(
                [
                    QPointF(front_left, ft),
                    QPointF(front_left + DEPTH_X, ft - DEPTH_Y),
                    QPointF(front_right + DEPTH_X, ft - DEPTH_Y),
                    QPointF(front_right, ft),
                ]
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(top_face)
            painter.drawPolygon(top_poly)

    @staticmethod
    def _paint_band_overlay(
        painter: QPainter, tokens: object, front_rect: QRectF, is_hovered: bool, is_selected: bool
    ) -> None:
        """Draw hover highlight or selection accent border."""
        if is_hovered:
            overlay = QColor(255, 255, 255, 40) if tokens is theme.DARK_TOKENS else QColor(0, 0, 0, 22)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(overlay)
            painter.drawRect(front_rect)
        if is_selected:
            accent = QColor(tokens.accent)
            accent.setAlpha(200)
            painter.setPen(QPen(accent, 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(front_rect.adjusted(1, 1, -1, -1))

    @staticmethod
    def _paint_band_edges(
        painter: QPainter,
        edge_pen: QPen,
        front_left: float,
        front_right: float,
        ft: float,
        fh: float,
        is_component: bool,
    ) -> None:
        """Draw inter-layer separation lines."""
        painter.setPen(edge_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(front_left, ft + fh), QPointF(front_right, ft + fh))
        if is_component and fh >= MIN_BAND_H:
            painter.drawLine(QPointF(front_right, ft + fh), QPointF(front_right + DEPTH_X, ft + fh - DEPTH_Y))

    def _draw_caption(self, painter: QPainter, band: Band, rect: QRect) -> None:
        """Label a band only when its own height leaves room to read one."""
        if rect.height() < MIN_BAND_H:
            return
        # The fills span most of the luminance range, so one fixed caption
        # colour cannot stay readable on all of them.
        painter.setPen(QColor(theme.band_label_colour(band.fill)))
        text = band.label if band.detail == "" else f"{band.label} · {band.detail}"
        painter.drawText(
            rect.adjusted(theme.SPACE_XS, 0, -theme.SPACE_XS, 0),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            text,
        )

    def _band_at(self, y: int) -> Band | None:
        for band in self._bands:
            if band.top <= y < band.top + band.height:
                return band
        return None

    def _relayout(self) -> None:
        self._bands = stack_bands(self._structure, self.height())
        if self._selected is not None and self._selected >= len(self._bands) - 2:
            self._selected = None
        self.update()
