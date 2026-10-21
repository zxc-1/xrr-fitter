"""Atomic interaction-mode controls for diagnostic plots.

One controller owns every Matplotlib callback and Qt event filter installed for
a plot panel.  It caches the parent watched for teardown events so destruction
never needs to dereference an already deleted panel wrapper, and its release
path disconnects callback state before queued Qt events can observe it.  The
bar the reader clicks lives in :mod:`xrr_fitter.gui.plots.toolbar`; this module
owns what each mode and navigation request does to the plots.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
from matplotlib.backend_bases import NavigationToolbar2
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QWidget

from xrr_fitter.gui.plots.diagnostics import ANALYSIS_KEYS, REFLECTIVITY_KEYS
from xrr_fitter.gui.plots.live import LiveReflectivityPlot
from xrr_fitter.gui.plots.sld_drag import SldDrag, SldHandleDragMixin
from xrr_fitter.gui.plots.toolbar import PlotInteractionToolbar

PAN_MODE = "pan/zoom"

# One wheel notch scales each axis span by this factor.  Matplotlib reports a
# scroll-up as a positive step, which reads as "look closer", so scrolling up
# divides the span (zooms in) and scrolling down multiplies it (zooms out).
WHEEL_ZOOM_STEP = 1.2


class PlotNavigator(NavigationToolbar2):
    """Drive one canvas's pan, box zoom and view history without Qt chrome.

    Matplotlib's ready-made Qt toolbar is a widget carrying ten buttons of its
    own, and it binds to a single canvas.  This panel has eight canvases and
    already owns its button row, so only the navigation *behaviour* is wanted:
    each canvas gets its own navigator, and the panel's own three buttons drive
    whichever one belongs to the view currently on screen.
    """

    def __init__(self, canvas: object, message_sink: object | None = None) -> None:
        super().__init__(canvas)
        self._message_sink = message_sink
        self._connected = True

    def push_baseline(self) -> None:
        """Record the freshly drawn limits as the view ``home`` returns to.

        The history stack starts empty, and an empty stack makes ``home`` a
        no-op, so a user who had panned away would find the button dead.  It
        cannot be reconstructed by autoscaling either: the correlation matrix is
        drawn with ``imshow``, whose inverted y-axis autoscale would silently
        flip top for bottom.  So the limits are captured as drawn.
        """
        if self._connected:
            self._nav_stack.clear()
            self.push_current()

    def disconnect_events(self) -> None:
        """Drop every callback and the canvas's back-reference to this navigator.

        The base constructor assigns itself to ``canvas.toolbar``, so leaving it
        in place would keep a released panel's canvases and navigators pointing
        at each other after teardown.
        """
        for identifier in (self._id_press, self._id_release, self._id_drag):
            self.canvas.mpl_disconnect(identifier)
        self._nav_stack.clear()
        self.canvas.toolbar = None
        self._connected = False

    def draw_rubberband(self, event: object, x0: float, y0: float, x1: float, y1: float) -> None:
        """Show the zoom rectangle being dragged.

        The base implementation is a no-op meant for backends without one, which
        would leave a box zoom with no feedback until the mouse is released.
        """
        height = self.canvas.get_width_height(physical=True)[1]
        self.canvas.drawRectangle([int(value) for value in (x0, height - y0, x1 - x0, y0 - y1)])

    def remove_rubberband(self) -> None:
        self.canvas.drawRectangle(None)

    def set_message(self, message: str) -> None:
        """Route the cursor read-out to the panel's coordinate label.

        Matplotlib feeds this on every ``motion_notify_event`` with the data
        coordinates under the pointer, and clears it with an empty string when
        the pointer leaves the axes.  This panel carries no Matplotlib status
        bar, so the message is forwarded to a Qt label instead of discarded.
        """
        sink = self._message_sink
        if sink is not None:
            sink(str(message))

    def set_history_buttons(self) -> None:
        """No back/forward buttons are exposed, so there is none to enable."""

    def save_figure(self, *args: object) -> None:
        """Exporting is owned by the project's export workflow, not this panel."""
        raise NotImplementedError("plot export is owned by the export workflow")


def ordered_finite_range(first: float, second: float) -> tuple[float, float]:
    lower = float(first)
    upper = float(second)
    if not isfinite(lower) or not isfinite(upper):
        raise ValueError("fit range values must be finite")
    return (lower, upper) if lower <= upper else (upper, lower)


def prepared_point_index(index: int, point_count: int) -> int:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < point_count:
        raise IndexError("point index is outside the prepared data")
    return index


class PlotInteractionController(SldHandleDragMixin, QObject):
    """Own Matplotlib callbacks, tab visibility, and scoped keyboard input."""

    def __init__(self, panel: object, toolbar: PlotInteractionToolbar) -> None:
        super().__init__(panel)
        self._panel = panel
        self._toolbar = toolbar
        self._editing_enabled = True
        self._data_available = True
        self._reflectivity_tabs = panel.reflectivity_tabs
        self._analysis_tabs = panel.analysis_tabs
        # Backward-compat alias used by tests that reference self._tabs
        self._tabs = self._reflectivity_tabs
        self._views = panel._views
        self._watched_parent = None
        self._requested_reflectivity_index = 0
        self._requested_analysis_index = 0
        # Which group the user last interacted with ("reflectivity" or "analysis")
        self._active_group: str = "reflectivity"
        self._projecting_tabs = False
        # The reflectivity panes render through pyqtgraph and carry their own
        # gestures: a draggable region for range selection, a masking click,
        # native wheel zoom and pan/box-zoom modes all live on the widget, so the
        # controller only wires their signals back to the panel.  The static
        # matplotlib views keep the NavigationToolbar2 driver, one per canvas: a
        # navigator binds to the canvas it drives, and the companion SLD pane is
        # on screen alongside whichever tab is selected, so no single canvas could
        # stand in for the panel.
        self._navigators = {
            key: PlotNavigator(view.canvas, self._show_cursor_message)
            for key, view in self._views.items()
            if not isinstance(view, LiveReflectivityPlot)
        }
        # Wheel zoom on the matplotlib views is a direct-manipulation gesture that
        # stays live in every mode, so it is connected once per canvas here rather
        # than armed by a mode.  Connecting once also keeps the per-canvas callback
        # counts stable across redraws, which the teardown contract asserts.  The
        # pyqtgraph panes zoom on the wheel natively and need no such connection.
        self._scroll_ids = {
            key: view.canvas.mpl_connect("scroll_event", self._wheel_zoom)
            for key, view in self._views.items()
            if not isinstance(view, LiveReflectivityPlot)
        }
        # The companion SLD pane carries two kinds of grabbable handle: a vertical
        # interface line that sets a layer's thickness by eye, and a horizontal
        # level line spanning a layer that raises or lowers its SLD.  The three
        # gesture callbacks connect once here for the same reason wheel zoom does:
        # a constant per-canvas callback count is what the teardown contract
        # checks.  ``_sld_drag`` holds the grab between press and release (see
        # ``SldDrag``) and is ``None`` when no drag is in flight; ``_sld_hover`` is
        # the handle the pointer rests on, with the line width and marker size it
        # had before it was emphasised.
        self._sld_drag: SldDrag | None = None
        self._sld_hover: tuple[object, float, float] | None = None
        self._sld_drag_ids: tuple[int, ...] = ()
        sld_view = self._views.get("sld")
        if sld_view is not None and not isinstance(sld_view, LiveReflectivityPlot):
            sld_canvas = sld_view.canvas
            self._sld_drag_ids = (
                sld_canvas.mpl_connect("button_press_event", self._sld_press),
                sld_canvas.mpl_connect("motion_notify_event", self._sld_motion),
                sld_canvas.mpl_connect("button_release_event", self._sld_release),
            )
        for signal, slot in self._signal_slots():
            signal.connect(slot)
        panel.installEventFilter(self)
        for child in panel.findChildren(QWidget):
            child.installEventFilter(self)
        # 设计稿开局亮着 ``✥``，而一张刚画好的 pyqtgraph 面板本来就在平移态：左键拖就是
        # 平移，没有「什么都不做」这一档。换 tab 时这一句已经在跑；开局不跑，条上四枚全灰，
        # 读者据此以为得先点一下才能拖。
        self._sync_navigation_after_tab_change()
        self.watch_parent()

    def _live_panes(self) -> tuple[LiveReflectivityPlot, ...]:
        return tuple(view for view in self._views.values() if isinstance(view, LiveReflectivityPlot))

    def _signal_slots(self) -> tuple[tuple[object, object], ...]:
        """控制器挂上的每一条 Qt 信号。接线与 ``release`` 拆线读的是同一张表，两边才不会漏掉一条。"""
        pairs: list[tuple[object, object]] = []
        for pane in self._live_panes():
            pairs += [
                (pane.fit_range_selected, self._pg_range_selected),
                (pane.point_mask_requested, self._pg_mask_requested),
                (pane.cursor_moved, self._pg_cursor_moved),
                (pane.cursor_left, self._pg_cursor_left),
            ]
        toolbar = self._toolbar
        pairs += [
            (toolbar.mode_changed, self._mode_changed),
            (toolbar.zoom_to_range_requested, self._zoom_to_range),
            (toolbar.reset_zoom_requested, self._reset_zoom),
            (toolbar.navigation_requested, self._navigation_requested),
            (self._reflectivity_tabs.currentChanged, self._reflectivity_tab_changed),
            (self._analysis_tabs.currentChanged, self._analysis_tab_changed),
        ]
        return tuple(pairs)

    def _current_live_pane(self) -> LiveReflectivityPlot | None:
        """Return the pyqtgraph pane on screen, or None when a static view shows."""
        if self._panel is None:
            return None
        view = self._views.get(self.current_view_key())
        return view if isinstance(view, LiveReflectivityPlot) else None

    def current_view_key(self) -> str:
        if self._active_group == "analysis":
            idx = self._analysis_tabs.currentIndex()
            if 0 <= idx < len(ANALYSIS_KEYS):
                return ANALYSIS_KEYS[idx]
        idx = self._reflectivity_tabs.currentIndex()
        if 0 <= idx < len(REFLECTIVITY_KEYS):
            return REFLECTIVITY_KEYS[idx]
        return REFLECTIVITY_KEYS[0]

    def select_view(self, key: str) -> None:
        # 换页时 tab 的回调也会报一次 ``view_changed``，而页没换时不会——同一个动作报几次就取决于读者原先
        # 停在哪一页，主窗口据此把工作区写两遍。所以换页期间让回调噤声，结尾这里只报一次。
        self._projecting_tabs = True
        try:
            self._select_tab(key)
        finally:
            self._projecting_tabs = False
        self._sync_navigation_after_tab_change()
        self._emit_view_changed()

    def _select_tab(self, key: str) -> None:
        if key in REFLECTIVITY_KEYS:
            index = REFLECTIVITY_KEYS.index(key)
            if not self._reflectivity_tabs.isTabVisible(index):
                raise ValueError(f"diagnostic view is hidden: {key}")
            self._active_group = "reflectivity"
            self._reflectivity_tabs.setCurrentIndex(index)
            self._requested_reflectivity_index = index
        elif key in ANALYSIS_KEYS:
            index = ANALYSIS_KEYS.index(key)
            if not self._analysis_tabs.isTabVisible(index):
                raise ValueError(f"diagnostic view is hidden: {key}")
            self._active_group = "analysis"
            self._analysis_tabs.setCurrentIndex(index)
            self._requested_analysis_index = index
        else:
            raise KeyError(f"unknown diagnostic view: {key}")

    def set_expert_mode(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("expert mode must be bool")
        self._apply_tabs(enabled, self._requested_reflectivity_index, self._requested_analysis_index)

    def apply_workspace(self, expert_mode: bool, tab_index: int, analysis_tab_index: int = 0) -> None:
        # A saved index is clamped rather than rejected: the strip has gained and
        # lost tabs across releases, so a project saved under an older layout can
        # name an index the bar no longer has.  Refusing to open such a project
        # would make a layout change break the reader's own files; falling back to
        # the leading view costs them one click.
        if not 0 <= tab_index < self._reflectivity_tabs.count():
            tab_index = 0
        if not 0 <= analysis_tab_index < self._analysis_tabs.count():
            analysis_tab_index = 0
        self._requested_reflectivity_index = tab_index
        self._requested_analysis_index = analysis_tab_index
        self._apply_tabs(expert_mode, tab_index, analysis_tab_index)

    def select_fit_range(self, first: float, second: float) -> bool:
        if not self._click_modes_live() or self._toolbar.mode() != "range":
            return False
        lower, upper = ordered_finite_range(first, second)
        self._panel.fit_range_requested.emit(lower, upper)
        return True

    def request_point_mask(self, index: int) -> bool:
        if not self._click_modes_live() or self._toolbar.mode() != "mask":
            return False
        data = self._panel._active_data()
        value = prepared_point_index(index, data.two_theta_deg.size)
        self._panel.point_mask_requested.emit(value)
        return True

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing_enabled = enabled
        self._sync_click_modes()
        if not enabled:
            self._sld_cancel_drag()

    def set_data_available(self, available: bool) -> None:
        """当前数据集读不出来时，范围与掩膜无从谈起：两档灰掉，模式退回查看。"""
        self._data_available = bool(available)
        self._sync_click_modes()

    def _click_modes_live(self) -> bool:
        return self._editing_enabled and self._data_available

    def _sync_click_modes(self) -> None:
        live = self._click_modes_live()
        for mode in ("range", "mask"):
            self._toolbar.mode_actions()[mode].setEnabled(live)
        if not live:
            self._toolbar.set_mode("view")

    def cancel(self) -> None:
        # Esc 先撤最里层那个手势：剖面上正拖着把手时，只把把手放回原处，模式与范围都不动。
        if self._sld_cancel_drag():
            return
        self._panel._drop_pending_range()
        self._toolbar.set_mode("view")

    def callback_counts(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        # Only the matplotlib views expose a callback registry; the pyqtgraph
        # panes carry their gestures on the widget and are torn down through
        # ``release`` on the widget itself, so they never enter this contract.
        return tuple(
            (key, self._canvas_callback_counts(view.canvas))
            for key, view in self._views.items()
            if not isinstance(view, LiveReflectivityPlot)
        )

    def _canvas_callback_counts(self, canvas: object) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((name, len(callbacks)) for name, callbacks in canvas.callbacks.callbacks.items()))

    def watch_parent(self) -> None:
        panel = self._panel
        if panel is None:
            return
        parent = panel.parent()
        if parent is not None and parent is not self._watched_parent:
            parent.installEventFilter(self)
            self._watched_parent = parent

    def release(self, *_args: object) -> None:
        if self._panel is None:
            return
        # The pyqtgraph panes, the toolbar and both tab bars emit into this
        # controller; drop every one of those first so a widget that outlives
        # this call cannot reach a half-released controller -- a tab switched
        # after release used to go looking for ``tab_keys`` on a panel that was
        # already gone.  The pane's own ``release`` (driven by the panel) then
        # silences its scene callbacks.
        for signal, slot in self._signal_slots():
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                # Idempotent teardown: a connection may already be gone if the
                # emitter's C++ object was destroyed first.
                pass
        for key, identifier in self._scroll_ids.items():
            view = self._views.get(key)
            if view is not None:
                view.canvas.mpl_disconnect(identifier)
        self._scroll_ids = {}
        sld_view = self._views.get("sld")
        for identifier in self._sld_drag_ids:
            if sld_view is not None:
                sld_view.canvas.mpl_disconnect(identifier)
        self._sld_drag_ids = ()
        self._sld_drag = None
        self._sld_hover = None
        for navigator in self._navigators.values():
            navigator.disconnect_events()
        self._navigators = {}
        self._panel = None
        self._watched_parent = None

    def navigators(self) -> dict[str, PlotNavigator]:
        return dict(self._navigators)

    def refresh_navigation_baselines(self) -> None:
        """Make ``home`` mean "the limits this redraw produced".

        A redraw replaces the data, so limits recorded against the previous
        candidate would restore a window belonging to a plot that no longer
        exists.
        """
        for navigator in self._navigators.values():
            navigator.push_baseline()

    def navigation_mode(self) -> str:
        pane = self._current_live_pane()
        if pane is not None:
            return pane.navigation_mode()
        navigator = self._navigators.get(self.current_view_key())
        return "" if navigator is None else str(navigator.mode)

    def _navigation_requested(self, action: str) -> None:
        """Drive pan/box-zoom/home for the view the user is looking at.

        A pyqtgraph pane is always in pan or box-zoom, so home just restores the
        data bounds and the button row is put back in step with the pane's own
        mode.  A matplotlib view keeps the latching NavigationToolbar2 behaviour,
        where claiming the widget lock is what makes the range and mask handlers
        stand down, so ``_leave_click_mode`` drops the mode row alongside it.
        """
        pane = self._current_live_pane()
        if pane is not None:
            if action == "home":
                pane.go_home()
            else:
                self._leave_click_mode()
                pane.set_navigation_mode(action)
            self._toolbar.show_navigation_mode(pane.navigation_mode())
            return
        navigator = self._navigators.get(self.current_view_key())
        if navigator is None:
            return
        if action == "home":
            navigator.home()
        else:
            self._leave_click_mode()
            getattr(navigator, action)()
        self._sync_navigation_buttons(navigator)

    def _leave_click_mode(self) -> None:
        """Drop out of range or mask when the user starts navigating instead."""
        if self._toolbar.mode() != "view":
            self._toolbar.set_mode("view")

    def _sync_navigation_buttons(self, navigator: PlotNavigator) -> None:
        mode = str(navigator.mode)
        self._toolbar.show_navigation_mode("pan" if mode == PAN_MODE else "zoom" if mode else "")

    def _mode_changed(self, mode: str) -> None:
        if not self._click_modes_live() and mode != "view":
            self._toolbar.set_mode("view")
            return
        # Range and mask are angle-domain gestures, so they arm on the two
        # reflectivity panes that draw the measured points (raw and log); qz⁴R
        # lives in qz space, and the residual -- on the same angle axis -- is a
        # read-out of the fit, not the data a mask or window is chosen on.  The
        # pyqtgraph region and mask click coexist with pan, so arming them does
        # not need to claim a lock.
        for key in ("raw", "log"):
            pane = self._views.get(key)
            if isinstance(pane, LiveReflectivityPlot):
                pane.enable_range_selection(mode == "range")
                pane.enable_masking(mode == "mask")
        if mode != "view":
            # On a matplotlib view, range and mask work by clicking the curve, so
            # an active pan or box zoom would swallow the very press they need.
            # Whichever the user picks second wins; the earlier one is switched
            # off rather than silently ignored.
            self._stop_navigation()

    def _stop_navigation(self) -> None:
        for navigator in self._navigators.values():
            if navigator.mode:
                getattr(navigator, "pan" if str(navigator.mode) == PAN_MODE else "zoom")()
        self._toolbar.show_navigation_mode("")

    def _zoom_to_range(self) -> None:
        if self._panel is not None:
            self._panel.zoom_to_range()

    def _reset_zoom(self) -> None:
        if self._panel is not None:
            self._panel.reset_zoom()

    def _wheel_zoom(self, event: object) -> None:
        """Zoom the axes under the pointer around the cursor, in any mode.

        The wheel is the one gesture every user already knows for "look
        closer", so it answers directly without first arming a mode.  The point
        under the pointer keeps its place on screen because each axis is scaled
        about the cursor's data coordinate, which makes the zoom feel anchored
        rather than recentred.
        """
        axes = getattr(event, "inaxes", None)
        step = float(getattr(event, "step", 0.0) or 0.0)
        if axes is None or not axes.get_navigate() or step == 0.0:
            return
        factor = 1.0 / WHEEL_ZOOM_STEP if step > 0.0 else WHEEL_ZOOM_STEP
        self._zoom_axis(axes.get_xlim(), axes.set_xlim, axes.get_xscale(), event.xdata, factor)
        self._zoom_axis(axes.get_ylim(), axes.set_ylim, axes.get_yscale(), event.ydata, factor)
        axes.figure.canvas.draw_idle()

    @staticmethod
    def _zoom_axis(limits: object, setter: object, scale: str, center: object, factor: float) -> None:
        """Scale one axis about ``center`` by ``factor``, honouring a log scale.

        Both endpoints move toward the centre by the same factor, so their order
        is preserved: an inverted ``imshow`` axis keeps reading top-down.  On a
        log axis the interpolation happens in decade space, which keeps both
        limits strictly positive and never crosses zero.
        """
        low, high = float(limits[0]), float(limits[1])
        if center is None or not isfinite(float(center)):
            center = 0.5 * (low + high)
        center = float(center)
        if scale == "log":
            if low <= 0.0 or high <= 0.0 or center <= 0.0:
                return
            log_low, log_high, log_center = np.log10(low), np.log10(high), np.log10(center)
            setter(
                10.0 ** (log_center + (log_low - log_center) * factor),
                10.0 ** (log_center + (log_high - log_center) * factor),
            )
        else:
            setter(center + (low - center) * factor, center + (high - center) * factor)

    def _show_cursor_message(self, message: str) -> None:
        """Forward a navigator's cursor read-out to the panel's coordinate label."""
        panel = self._panel
        if panel is not None:
            panel.set_cursor_readout(message)

    def _pg_range_selected(self, low: float, high: float) -> None:
        """Commit a range the reader dragged on a pyqtgraph pane.

        The widget only emits while its region is armed (range mode), so this
        mirrors the old span selector: shade the window immediately and notify
        the app that a fit range was requested.
        """
        panel = self._panel
        if panel is None or not self._click_modes_live():
            return
        lower, upper = ordered_finite_range(low, high)
        panel.show_range(lower, upper)
        panel.fit_range_requested.emit(lower, upper)

    def _pg_mask_requested(self, index: int) -> None:
        """Forward the prepared point a masking click landed on.

        The pane resolves the hit itself, by on-screen distance to the drawn
        point, and emits only while masking is armed; the panel then applies the
        mode and range guards before the window toggles it.
        """
        panel = self._panel
        if panel is not None:
            panel.request_point_mask(index)

    def _pg_cursor_moved(self, x_view: float, y_view: float) -> None:
        """Read out the pointer over a pyqtgraph pane in that pane's own units.

        发信号的那张卡才是指针所在的卡：残差卡钉在反射率页下面，选中的 tab 可能是 qz⁴·R 页，也可能是
        一张静态分析页——按当前 tab 取轴名，读数就会写错单位，或者干脆不写。
        """
        panel = self._panel
        pane = self.sender()
        if panel is None or not isinstance(pane, LiveReflectivityPlot):
            return
        xlabel, ylabel = pane.axis_labels()
        panel.set_cursor_readout(f"{xlabel} {x_view:.4g} · {ylabel} {y_view:.4g}")

    def _pg_cursor_left(self) -> None:
        panel = self._panel
        if panel is not None:
            panel.set_cursor_readout("")

    def _reflectivity_tab_changed(self, index: int) -> None:
        if self._projecting_tabs or index < 0:
            return
        self._requested_reflectivity_index = index
        self._active_group = "reflectivity"
        self._sync_navigation_after_tab_change()
        self._emit_view_changed()

    def _analysis_tab_changed(self, index: int) -> None:
        if self._projecting_tabs or index < 0:
            return
        self._requested_analysis_index = index
        self._active_group = "analysis"
        self._sync_navigation_after_tab_change()
        self._emit_view_changed()

    def _sync_navigation_after_tab_change(self) -> None:
        pane = self._current_live_pane()
        if pane is not None:
            self._toolbar.show_navigation_mode(pane.navigation_mode())
        else:
            navigator = self._navigators.get(self.current_view_key())
            if navigator is not None:
                self._sync_navigation_buttons(navigator)

    def _emit_view_changed(self) -> None:
        """Emit a global index (0-8) for chrome sync compatibility."""
        key = self.current_view_key()
        all_keys = self._panel.tab_keys()
        global_index = all_keys.index(key) if key in all_keys else 0
        self._panel.view_changed.emit(global_index)

    def _apply_tabs(self, expert_mode: bool, ref_index: int, ana_index: int = 0) -> None:
        """Project expert mode onto the companion pane, not the tab bar.

        The SLD profile left the tab bar for a permanent pane, so expert mode
        now shows or hides that pane. Every tab stays selectable in both modes,
        which means a persisted selection can always be honoured.

        交给面板记下来再由它统一落可见性，而不是在这里直接 ``setVisible``：那块面板上还有一道
        按流程步的闸（``STEP_PLOT_PANES``），两处各自写同一个属性的话，谁后跑谁说话。
        """
        self._projecting_tabs = True
        try:
            self._panel.set_expert_pane_scope(expert_mode)
            self._reflectivity_tabs.setCurrentIndex(ref_index)
            self._analysis_tabs.setCurrentIndex(ana_index)
        finally:
            self._projecting_tabs = False

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        panel = getattr(self, "_panel", None)
        if panel is None:
            return False
        if self._is_scoped_escape(watched, event, panel):
            self.cancel()
            event.accept()
            return True
        if watched is self._watched_parent and event.type() in (
            QEvent.Type.Close,
            QEvent.Type.DeferredDelete,
        ):
            panel.release_resources()
        return super().eventFilter(watched, event)

    def _is_scoped_escape(
        self,
        watched: object,
        event: QEvent,
        panel: object,
    ) -> bool:
        return bool(
            event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
            and (watched is panel or (isinstance(watched, QWidget) and panel.isAncestorOf(watched)))
        )
