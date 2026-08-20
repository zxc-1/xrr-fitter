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

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme

# Air and the substrate are semi-infinite, so they have no thickness to be
# proportional to and instead get a constant strip that reads as a boundary.
MEDIUM_BAND_H = 18

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
        self._structure: api.StructureSpec | None = None
        self._selected: int | None = None
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

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        tokens = theme.palette_tokens(self.palette())
        font = QFont(self.font())
        font.setPointSize(theme.FONT_PT_SM)
        painter.setFont(font)
        width = self.width()
        for band in self._bands:
            rect = QRect(0, band.top, width, band.height)
            fill = QColor(band.fill)
            fill.setAlpha(255 if band.index == self._selected else 190)
            painter.fillRect(rect, fill)
            border = QColor(tokens.accent if band.index == self._selected else band.fill)
            painter.setPen(QPen(border, 2 if band.index == self._selected else 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            self._draw_caption(painter, band, rect)
        painter.end()

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
