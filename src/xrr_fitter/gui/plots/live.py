"""pyqtgraph-backed live reflectivity view for real-time preview and fluid gestures.

matplotlib stays the renderer for the static diagnostics (correlation matrix,
credible-band profile, residual and parameter heatmaps, batch trends) and for the
byte-identity export in ``io/export_plots.py``; both are decoupled from what draws on
screen, so this module can own the interactive reflectivity family without touching
them.  The win here is a search publishing a new model many times per second into a
single mutating curve, plus a draggable region and a click that reads back a q, in
place of the matplotlib navigation toolbar.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from xrr_fitter.gui import theme


class LiveReflectivityPlot(pg.PlotWidget):
    """One reflectivity pane with an owned preview curve and native gestures.

    The observed and model artists are created once and updated in place; the
    preview artist is created lazily on its first publish so a pane that never
    hosts a search carries no spare line.  Selection and masking are opt-in so a
    read-only pane does not intercept the drags a reader uses to pan and zoom.
    """

    fit_range_selected = Signal(float, float)
    point_mask_requested = Signal(float)
    cursor_moved = Signal(float, float)
    cursor_left = Signal()

    def __init__(self, parent: object | None = None) -> None:
        super().__init__(parent)
        self.plot_item = self.getPlotItem()
        self._released = False
        self._masking = False
        self.preview_item: pg.PlotDataItem | None = None
        self.range_item: pg.LinearRegionItem | None = None
        self.observed_item = self.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolPen=None,
            symbolBrush=pg.mkColor(theme.DATA_OBSERVED),
            name="归一化数据",
        )
        self.model_item = self.plot(
            [],
            [],
            pen=pg.mkPen(theme.DATA_CANDIDATE, width=1.4, style=Qt.PenStyle.DashLine),
            name="当前候选模型",
        )
        # The raw view struck-out points and the residual zero baseline are always
        # present but empty; a pane that never renders those projections simply
        # leaves them blank rather than paying a lazy-construction branch per draw.
        self.excluded_item = self.plot(
            [],
            [],
            pen=None,
            symbol="x",
            symbolSize=7,
            symbolPen=pg.mkPen(theme.DATA_NEUTRAL),
            symbolBrush=None,
            name="排除点",
        )
        self.reference_item = self.plot(
            [],
            [],
            pen=pg.mkPen(theme.DATA_NEUTRAL, width=1.0, style=Qt.PenStyle.DotLine),
            name="零参考线",
        )
        self.fit_range_item: pg.LinearRegionItem | None = None
        self._title = ""
        self._xlabel = ""
        self._ylabel = ""
        self._caption_text: str | None = None
        self._placeholder_text: str | None = None
        self._nav_mode = "pan"
        # A reference grid lets a reader read values off the curve, matching the
        # grid the matplotlib draw_* functions drew.
        self.plot_item.showGrid(x=True, y=True, alpha=0.25)
        # The quality caption and the empty-state placeholder ride the scene as
        # annotations that never widen autoRange (ignoreBounds); a range-change
        # handler pins them to the view corners so they track pan and zoom the way
        # the matplotlib axes-fraction text did.  Their colour comes from the
        # palette applied below, not a fixed grey: a light note on a light desktop
        # is the failure this pane's matplotlib predecessor was fixed for.
        self._caption_item = pg.TextItem(anchor=(1, 1))
        self._caption_item.setZValue(20)
        self._caption_item.hide()
        self.addItem(self._caption_item, ignoreBounds=True)
        self._placeholder_item = pg.TextItem(anchor=(0.5, 0.5))
        self._placeholder_item.setZValue(20)
        self._placeholder_item.hide()
        self.addItem(self._placeholder_item, ignoreBounds=True)
        self._palette = theme.LIGHT_PLOT_PALETTE
        self.apply_palette(theme.current_plot_palette())
        self.plot_item.vb.sigRangeChanged.connect(self._reposition_annotations)
        self.scene().sigMouseMoved.connect(self._on_scene_moved)
        self.scene().sigMouseClicked.connect(self._on_scene_clicked)

    def _apply_labels(self, title: str, xlabel: str, ylabel: str) -> None:
        self._title, self._xlabel, self._ylabel = title, xlabel, ylabel
        self.plot_item.setTitle(title)
        self.plot_item.setLabel("bottom", xlabel)
        self.plot_item.setLabel("left", ylabel)

    def axis_labels(self) -> tuple[str, str, str]:
        """Return the drawn (title, x-label, y-label) for parity assertions."""
        return (self._title, self._xlabel, self._ylabel)

    def apply_palette(self, palette: theme.PlotPalette) -> None:
        """Paint the pane's structural colours from a resolved theme palette.

        A pg pane is drawn outside the Qt stylesheet and is not reached by
        ``diagnostics.apply_figure_palette``, so — like the matplotlib figures —
        it must be handed the palette explicitly and re-read it on every draw;
        otherwise it keeps pyqtgraph's default black canvas when the desktop
        switches appearance.  The Okabe-Ito data-series hues are left fixed on
        purpose: a curve that changed colour with the desktop would invalidate
        the reference a reader built ("the blue curve is my data").
        """
        self._palette = palette
        self.setBackground(QColor.fromRgbF(*palette.background))
        foreground = QColor.fromRgbF(*palette.foreground)
        muted = QColor.fromRgbF(*palette.muted)
        for side in ("left", "bottom"):
            axis = self.plot_item.getAxis(side)
            # Tick-value text and the axis label take the foreground; the axis
            # line and ticks take the quieter muted, matching the matplotlib
            # spines/tick_params split in apply_figure_palette.
            axis.setTextPen(foreground)
            axis.setPen(muted)
        # The title label caches its colour in opts, so store it there and only
        # re-render when a title is already present; the first _apply_labels
        # applies the stored colour otherwise, keeping the label hidden until a
        # draw sets its text.
        self.plot_item.titleLabel.setAttr("color", foreground.name())
        if self._title:
            self.plot_item.setTitle(self._title)
        self._caption_item.setColor(muted)
        self._placeholder_item.setColor(muted)

    def background_color(self) -> tuple[float, float, float, float]:
        """Read back the painted canvas background as an RGBA 0..1 tuple.

        QColor stores channels at reduced precision, so this round-trips to only
        ~1e-6 of the palette tuple; callers compare with a tolerance, not ==.
        """
        return self.backgroundBrush().color().getRgbF()

    def set_observed(self, angles: object, values: object) -> None:
        self.observed_item.setData(np.asarray(angles, dtype=float), np.asarray(values, dtype=float))

    def set_model(self, angles: object | None, values: object | None) -> None:
        if angles is None or values is None:
            self.model_item.setData([], [])
            return
        self.model_item.setData(np.asarray(angles, dtype=float), np.asarray(values, dtype=float))

    def show_log_reflectivity(
        self,
        angles: object,
        observed: object,
        model: object | None,
        *,
        r_floor: float,
    ) -> None:
        """Project the reflectivity family onto the log axis with draw_log's floor.

        ``r_floor`` is a presentation clamp, not a fit input: a measured or a
        modelled point below it reads at the floor rather than diverging down the
        log axis, so this pane and the matplotlib ``draw_log`` render the same data
        identically.  A missing candidate leaves the model curve empty rather than
        drawing a flat line at the floor.
        """
        self.apply_palette(theme.current_plot_palette())
        floor = float(r_floor)
        x = np.asarray(angles, dtype=float)
        self.set_observed(x, np.maximum(np.asarray(observed, dtype=float), floor))
        if model is None:
            self.set_model(None, None)
        else:
            self.set_model(x, np.maximum(np.asarray(model, dtype=float), floor))
        self._apply_labels("对数反射率", "2θ (deg)", f"归一化 R（显示下限 {floor:g}）")
        self._hide_placeholder()
        self.set_log_mode(True)

    def show_raw_reflectivity(
        self,
        angles: object,
        raw: object,
        mask: object,
        model: object | None,
    ) -> None:
        """Render stored angle / raw intensity with fit and excluded points split.

        The raw family reads on a linear axis: raw intensity spans well under a
        decade, so the log projection the ``show_log_reflectivity`` pane applies
        would only crush it against the baseline.  Included and excluded points
        live on separate artists so the mask a reader clicked reads back visually.
        """
        self.apply_palette(theme.current_plot_palette())
        x = np.asarray(angles, dtype=float)
        y = np.asarray(raw, dtype=float)
        keep = np.asarray(mask, dtype=bool)
        finite = np.isfinite(x) & np.isfinite(y)
        included = finite & keep
        excluded = finite & ~keep
        self.observed_item.setData(x[included], y[included])
        self.excluded_item.setData(x[excluded], y[excluded])
        if model is None:
            self.set_model(None, None)
        else:
            self.set_model(x, np.asarray(model, dtype=float))
        self._apply_labels("原始数据与模型", "2θ (deg)", "原始强度")
        self._hide_placeholder()
        self.set_log_mode(False)

    def show_qz4(
        self,
        data_qz: object,
        data_values: object,
        model_qz: object | None,
        model_values: object | None,
        *,
        ylabel: str = "qz⁴R",
    ) -> None:
        """Render the qz⁴R diagnostic transform on a linear axis.

        The qz⁴ weighting is computed upstream (it needs overflow-aware scaling),
        so this pane only plots the ready pairs and shows the ``ylabel`` that
        projection chose, which records the scaling reference when it had to
        normalize.  A missing candidate leaves the model curve empty rather than
        drawing a bare data series as a fit.
        """
        self.apply_palette(theme.current_plot_palette())
        self.observed_item.setData(np.asarray(data_qz, dtype=float), np.asarray(data_values, dtype=float))
        self.excluded_item.setData([], [])
        if model_qz is None or model_values is None:
            self.set_model(None, None)
        else:
            self.set_model(np.asarray(model_qz, dtype=float), np.asarray(model_values, dtype=float))
        self._apply_labels("qz⁴R 诊断变换（非拟合数据）", "qz (Å⁻¹)", ylabel)
        self._hide_placeholder()
        self.set_log_mode(False)

    def show_residual(self, qz: object, weighted: object) -> None:
        """Render the weighted residual as a marked line above a zero baseline."""
        self.apply_palette(theme.current_plot_palette())
        x = np.asarray(qz, dtype=float)
        y = np.asarray(weighted, dtype=float)
        # The residual curve carries a connecting line, unlike the marker-only
        # observed series of the reflectivity panes, so it opts its pen back in.
        self.observed_item.setPen(pg.mkPen(theme.DATA_OBSERVED, width=1.2))
        self.observed_item.setData(x, y)
        self.reference_item.setData(x, np.zeros_like(x))
        self.set_model(None, None)
        self.excluded_item.setData([], [])
        self._apply_labels("加权残差", "qz (Å⁻¹)", "加权残差")
        self._hide_placeholder()
        self.set_log_mode(False)

    def clear_series(self) -> None:
        """Empty every managed curve for the panel's no-data state."""
        self.observed_item.setData([], [])
        self.model_item.setData([], [])
        self.excluded_item.setData([], [])
        self.reference_item.setData([], [])

    def set_quality_caption(self, text: str | None) -> None:
        """Caption the fit quality in the corner, or clear it when text is None.

        This is the pg twin of draw_*'s bottom-right ``J=… · 平均残差 …`` note; the
        panel supplies the string so the widget stays ignorant of the objective.
        """
        self._caption_text = text
        if text:
            self._caption_item.setText(text)
            self._caption_item.show()
            self._reposition_annotations()
        else:
            self._caption_item.hide()

    def quality_caption_text(self) -> str | None:
        return self._caption_text

    def show_placeholder(self, text: str) -> None:
        """Empty the pane and centre a placeholder, matching draw_empty.

        A pane with no data or no candidate reads as an explicit "暂无…" note
        rather than a blank canvas, and the stale quality caption goes with it.
        """
        self.apply_palette(theme.current_plot_palette())
        self.clear_series()
        self.set_quality_caption(None)
        self._placeholder_text = text
        self._placeholder_item.setText(text)
        self._placeholder_item.show()
        self._reposition_annotations()

    def placeholder_text(self) -> str | None:
        return self._placeholder_text

    def _hide_placeholder(self) -> None:
        self._placeholder_text = None
        self._placeholder_item.hide()

    def _reposition_annotations(self, *_args: object) -> None:
        (x0, x1), (y0, y1) = self.plot_item.vb.viewRange()
        self._caption_item.setPos(x1, y0)
        self._placeholder_item.setPos((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def show_fit_range(self, low: float, high: float) -> None:
        """Shade the committed fit range as a read-only band.

        This is the presentation twin of the matplotlib ``axvspan`` mirror, not
        the drag selector: the band is not user-movable and does not arm the
        interactive handles, so a read-only pane can show the range it fitted.
        """
        lo, hi = sorted((float(low), float(high)))
        if self.fit_range_item is None:
            fill = pg.mkColor(theme.DATA_RANGE)
            fill.setAlpha(41)
            region = pg.LinearRegionItem(
                values=(lo, hi),
                brush=pg.mkBrush(fill),
                pen=pg.mkPen(theme.DATA_RANGE),
                movable=False,
            )
            region.setZValue(-10)
            self.addItem(region)
            self.fit_range_item = region
        else:
            self.fit_range_item.setRegion((lo, hi))

    def clear_fit_range(self) -> None:
        item = self.fit_range_item
        self.fit_range_item = None
        if item is None or self._released:
            return
        self.removeItem(item)

    def set_view_xrange(self, low: float, high: float) -> None:
        self.plot_item.vb.setXRange(float(low), float(high), padding=0)

    def autoscale_view(self) -> None:
        self.plot_item.vb.autoRange()

    def set_navigation_mode(self, mode: str) -> None:
        """Select pan (drag translates) or zoom (drag rubber-bands a rectangle).

        These are the pg twins of the matplotlib pan and zoom toolbar buttons, so
        the same toolbar can drive whichever backend renders the visible tab.
        """
        vb = self.plot_item.vb
        if mode == "zoom":
            self._nav_mode = "zoom"
            vb.setMouseMode(pg.ViewBox.RectMode)
        else:
            self._nav_mode = "pan"
            vb.setMouseMode(pg.ViewBox.PanMode)

    def navigation_mode(self) -> str:
        return self._nav_mode

    def go_home(self) -> None:
        """Restore the latest draw's data bounds, the pg twin of the home button."""
        self.plot_item.vb.autoRange()

    def _emit_cursor(self, x_view: float, y_view: float) -> None:
        if self._released:
            return
        # A log-mode viewbox carries y in log10 space; report the physical value so
        # the readout matches what the reader reads off the axis.
        y = 10.0**y_view if self.plot_item.ctrl.logYCheck.isChecked() else y_view
        self.cursor_moved.emit(float(x_view), float(y))

    def _emit_cursor_left(self) -> None:
        if self._released:
            return
        self.cursor_left.emit()

    def _on_scene_moved(self, scene_pos: object) -> None:
        if self._released:
            return
        if self.plot_item.sceneBoundingRect().contains(scene_pos):
            point = self.plot_item.vb.mapSceneToView(scene_pos)
            self._emit_cursor(point.x(), point.y())
        else:
            self._emit_cursor_left()

    def set_preview(self, angles: object, values: object) -> bool:
        """Publish the searching model into one owned curve; False once released."""
        if self._released:
            return False
        x = np.asarray(angles, dtype=float)
        y = np.asarray(values, dtype=float)
        if self.preview_item is None:
            self.preview_item = self.plot(x, y, pen=pg.mkPen(theme.DATA_PREVIEW, width=1.4), name="搜索中模型")
        else:
            self.preview_item.setData(x, y)
        return True

    def clear_preview(self) -> None:
        item = self.preview_item
        self.preview_item = None
        if item is None or self._released:
            return
        self.removeItem(item)

    def set_log_mode(self, enabled: bool) -> None:
        self.plot_item.setLogMode(y=bool(enabled))

    def enable_range_selection(self, enabled: bool) -> None:
        if enabled:
            if self.range_item is None:
                brush = pg.mkColor(theme.DATA_RANGE)
                brush.setAlpha(50)
                region = pg.LinearRegionItem(brush=pg.mkBrush(brush), pen=pg.mkPen(theme.DATA_RANGE))
                region.sigRegionChangeFinished.connect(self._emit_range)
                self.addItem(region)
                self.range_item = region
            return
        if self.range_item is not None:
            self.removeItem(self.range_item)
            self.range_item = None

    def enable_masking(self, enabled: bool) -> None:
        self._masking = bool(enabled)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for signal, slot in (
            (self.scene().sigMouseClicked, self._on_scene_clicked),
            (self.scene().sigMouseMoved, self._on_scene_moved),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                # Idempotent teardown: the scene C++ object may already be gone, or
                # the slot may have been disconnected by an earlier release.
                pass

    def _emit_range(self, *_args: object) -> None:
        region = self.range_item
        if region is None:
            return
        low, high = sorted(float(value) for value in region.getRegion())
        self.fit_range_selected.emit(low, high)

    def _mask_from_view_x(self, x_view: float) -> None:
        if self._released or not self._masking:
            return
        self.point_mask_requested.emit(float(x_view))

    def _on_scene_clicked(self, event: object) -> None:
        if self._released or not self._masking or event.button() != Qt.MouseButton.LeftButton:
            return
        scene_pos = event.scenePos()
        if not self.plot_item.sceneBoundingRect().contains(scene_pos):
            return
        self._mask_from_view_x(self.plot_item.vb.mapSceneToView(scene_pos).x())
