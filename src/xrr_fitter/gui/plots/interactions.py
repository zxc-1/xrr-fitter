"""Atomic interaction-mode controls for diagnostic plots.

One controller owns every Matplotlib callback and Qt event filter installed for
a plot panel.  It caches the parent watched for teardown events so destruction
never needs to dereference an already deleted panel wrapper, and its release
path disconnects callback state before queued Qt events can observe it.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

from xrr_fitter.gui import theme

MODE_SPECS = (
    ("view", "plotModeView", "查看", "查看和缩放诊断图"),
    ("range", "plotModeRange", "范围", "选择拟合角度范围"),
    ("mask", "plotModeMask", "掩膜", "切换单个预处理点的掩膜"),
)

# Deliberately not prefixed "plotMode": these are not members of the exclusive
# mode group, and the panel tests assert that prefix selects exactly the three
# interaction modes.
NAVIGATION_SPECS = (
    ("pan", "plotNavPan", "平移", "按住左键拖动图像，按住右键拖动缩放"),
    ("zoom", "plotNavZoom", "框选放大", "拖出一个矩形，放大到该区域"),
    ("home", "plotNavHome", "复位", "恢复当前图刚绘制时的坐标范围"),
)

PAN_MODE = "pan/zoom"


class PlotNavigator(NavigationToolbar2):
    """Drive one canvas's pan, box zoom and view history without Qt chrome.

    Matplotlib's ready-made Qt toolbar is a widget carrying ten buttons of its
    own, and it binds to a single canvas.  This panel has eight canvases and
    already owns its button row, so only the navigation *behaviour* is wanted:
    each canvas gets its own navigator, and the panel's own three buttons drive
    whichever one belongs to the view currently on screen.
    """

    def __init__(self, canvas: object) -> None:
        super().__init__(canvas)
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
        """Discard the cursor read-out; this panel has no status bar for it."""

    def set_history_buttons(self) -> None:
        """No back/forward buttons are exposed, so there is none to enable."""

    def save_figure(self, *args: object) -> None:
        """Exporting is owned by the project's export workflow, not this panel."""
        raise NotImplementedError("plot export is owned by the export workflow")


class PlotInteractionToolbar(QWidget):
    """Own one exclusive, programmatically validated plot mode."""

    mode_changed = Signal(str)
    zoom_to_range_requested = Signal()
    reset_zoom_requested = Signal()
    navigation_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotInteractionToolbar")
        self.setAccessibleName("绘图交互模式")
        self._buttons: dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        for index, (mode, name, text, description) in enumerate(MODE_SPECS):
            button = QToolButton(self)
            button.setObjectName(name)
            button.setText(text)
            button.setCheckable(True)
            button.setAccessibleName(text)
            button.setToolTip(description)
            self._group.addButton(button, index)
            self._buttons[mode] = button
            layout.addWidget(button)
            button.setProperty("plotMode", mode)
            button.clicked.connect(self._button_clicked)
        # Three related groups read as three groups because of the gaps between
        # them, and the row ends with the stretch rather than carrying it in the
        # middle: a spring between the modes and the zoom actions used to open a
        # void as wide as the panel, leaving the buttons stranded at both edges.
        layout.addSpacing(theme.SPACE_MD)
        self._install_navigation_buttons(layout)
        layout.addSpacing(theme.SPACE_MD)
        self._install_zoom_buttons(layout)
        layout.addStretch(1)
        self._buttons["view"].setChecked(True)
        self._mode = "view"

    def _install_navigation_buttons(self, layout: QHBoxLayout) -> None:
        """Add the pan, box-zoom and reset controls onto the same row.

        Pan and box zoom latch, so they are checkable, but they are not part of
        the exclusive mode group: they answer "how do I move around this plot",
        not "what does clicking it do", and the panel makes the two families
        mutually exclusive through the canvas widget lock instead.
        """
        self._navigation: dict[str, QToolButton] = {}
        for action, name, text, description in NAVIGATION_SPECS:
            button = QToolButton(self)
            button.setObjectName(name)
            button.setText(text)
            button.setCheckable(action != "home")
            button.setAccessibleName(text)
            button.setToolTip(description)
            button.setProperty("plotNavigation", action)
            button.clicked.connect(self._navigation_clicked)
            self._navigation[action] = button
            layout.addWidget(button)

    def _navigation_clicked(self) -> None:
        self.navigation_requested.emit(str(self.sender().property("plotNavigation")))

    def navigation_buttons(self) -> dict[str, QToolButton]:
        return dict(self._navigation)

    def show_navigation_mode(self, mode: str) -> None:
        """Mirror the canvas's navigation state onto the latching buttons.

        The state lives on the navigator, not on the buttons: pan also ends when
        a range drag takes the widget lock, or when the user switches to a view
        whose navigator is idle.  Reflecting it back keeps a button from staying
        pressed over a canvas that is no longer panning.
        """
        for action, button in self._navigation.items():
            if button.isCheckable():
                button.setChecked(action == mode)

    def _install_zoom_buttons(self, layout: QHBoxLayout) -> None:
        """Add non-exclusive zoom actions distinct from the mode button group.

        Zooming to the fit range is a one-shot action, not a persistent mode,
        so these stay outside the exclusive group; pressing one must not clear
        the active view/range/mask mode the user is working in.
        """
        self._zoom_to_range = QToolButton(self)
        self._zoom_to_range.setObjectName("plotZoomToRange")
        self._zoom_to_range.setText("缩放拟合区")
        self._zoom_to_range.setAccessibleName("缩放到拟合范围")
        self._zoom_to_range.setToolTip("将反射率视图缩放到当前拟合角度范围")
        self._zoom_to_range.clicked.connect(lambda: self.zoom_to_range_requested.emit())
        self._reset_zoom = QToolButton(self)
        self._reset_zoom.setObjectName("plotResetZoom")
        self._reset_zoom.setText("全览")
        self._reset_zoom.setAccessibleName("恢复完整视图")
        self._reset_zoom.setToolTip("恢复到完整角度范围")
        self._reset_zoom.clicked.connect(lambda: self.reset_zoom_requested.emit())
        layout.addWidget(self._zoom_to_range)
        layout.addWidget(self._reset_zoom)

    def buttons(self) -> dict[str, QToolButton]:
        return dict(self._buttons)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in self._buttons:
            raise ValueError(f"unsupported plot interaction mode: {mode}")
        if mode == self._mode:
            return
        self._buttons[mode].setChecked(True)
        self._mode = mode
        self.mode_changed.emit(mode)

    def _button_clicked(self, checked: bool) -> None:
        button = self.sender()
        mode = str(button.property("plotMode"))
        if not checked:
            self._buttons[self._mode].setChecked(True)
            return
        self._mode = mode
        self.mode_changed.emit(mode)


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


class PlotInteractionController(QObject):
    """Own Matplotlib callbacks, tab visibility, and scoped keyboard input."""

    def __init__(self, panel: object, toolbar: PlotInteractionToolbar) -> None:
        super().__init__(panel)
        self._panel = panel
        self._toolbar = toolbar
        self._tabs = panel.tabs
        self._views = panel._views
        self._watched_parent = None
        self._requested_tab_index = 0
        self._projecting_tabs = False
        self._range_selector = SpanSelector(
            panel.view("raw").axes,
            self._span_selected,
            "horizontal",
            useblit=False,
            button=1,
        )
        self._range_selector.set_active(False)
        self._mask_callback_id = panel.view("raw").canvas.mpl_connect(
            "button_press_event",
            self._point_clicked,
        )
        # One navigator per canvas: a navigator binds to the canvas it drives,
        # and the companion SLD pane is on screen alongside whichever tab is
        # selected, so no single canvas could stand in for the panel.
        self._navigators = {key: PlotNavigator(view.canvas) for key, view in self._views.items()}
        toolbar.mode_changed.connect(self._mode_changed)
        toolbar.zoom_to_range_requested.connect(self._zoom_to_range)
        toolbar.reset_zoom_requested.connect(self._reset_zoom)
        toolbar.navigation_requested.connect(self._navigation_requested)
        self._tabs.currentChanged.connect(self._tab_changed)
        panel.installEventFilter(self)
        for child in panel.findChildren(QWidget):
            child.installEventFilter(self)
        self.watch_parent()

    def current_view_key(self) -> str:
        return self._panel.tab_keys()[self._tabs.currentIndex()]

    def select_view(self, key: str) -> None:
        keys = self._panel.tab_keys()
        if key not in keys:
            raise KeyError(f"unknown diagnostic view: {key}")
        index = keys.index(key)
        if not self._tabs.isTabVisible(index):
            raise ValueError(f"diagnostic view is hidden: {key}")
        self._tabs.setCurrentIndex(index)

    def set_expert_mode(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("expert mode must be bool")
        self._apply_tabs(enabled, self._requested_tab_index)

    def apply_workspace(self, expert_mode: bool, tab_index: int) -> None:
        if not 0 <= tab_index < self._tabs.count():
            raise IndexError("plot tab index is out of range")
        self._requested_tab_index = tab_index
        self._apply_tabs(expert_mode, tab_index)

    def select_fit_range(self, first: float, second: float) -> bool:
        if self._toolbar.mode() != "range":
            return False
        lower, upper = ordered_finite_range(first, second)
        self._panel.fit_range_requested.emit(lower, upper)
        return True

    def request_point_mask(self, index: int) -> bool:
        if self._toolbar.mode() != "mask":
            return False
        data = self._panel._active_data()
        value = prepared_point_index(index, data.two_theta_deg.size)
        self._panel.point_mask_requested.emit(value)
        return True

    def cancel(self) -> None:
        self._panel._clear_visible_range()
        self._toolbar.set_mode("view")

    def callback_counts(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return tuple((key, self._canvas_callback_counts(view.canvas)) for key, view in self._views.items())

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
        selector = self._range_selector
        self._range_selector = None
        selector.disconnect_events()
        selector.set_active(False)
        raw = self._views.get("raw")
        if raw is not None and self._mask_callback_id is not None:
            raw.canvas.mpl_disconnect(self._mask_callback_id)
        self._mask_callback_id = None
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
        navigator = self._navigators.get(self.current_view_key())
        return "" if navigator is None else str(navigator.mode)

    def _navigation_requested(self, action: str) -> None:
        """Drive the navigator behind the view the user is looking at."""
        navigator = self._navigators.get(self.current_view_key())
        if navigator is None:
            return
        if action == "home":
            navigator.home()
        else:
            # Latching pan or box zoom claims the canvas widget lock, which is
            # what makes the range and mask handlers stand down: both refuse to
            # act while another owner holds it.  The button is put back in step
            # with that so the mode row does not keep advertising a mode whose
            # clicks are now being swallowed.
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
        if self._range_selector is not None:
            self._range_selector.set_active(mode == "range")
        if mode != "view":
            # Range and mask work by clicking the curve, so an active pan or box
            # zoom would swallow the very press they need.  Whichever the user
            # picks second wins; the earlier one is switched off rather than
            # silently ignored.
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

    def _span_selected(self, first: float, second: float) -> None:
        panel = self._panel
        selector = self._range_selector
        canvas = panel.view("raw").canvas
        if selector is None or not canvas.widgetlock.available(selector):
            return
        lower, upper = ordered_finite_range(first, second)
        panel.show_range(lower, upper)
        panel.fit_range_requested.emit(lower, upper)

    def _point_clicked(self, event: object) -> None:
        panel = self._panel
        canvas = panel.view("raw").canvas
        if not self._point_event_is_available(event, panel, canvas):
            return
        data = panel._active_data()
        indices = np.asarray(panel.displayed_prepared_indices(), dtype=int)
        if indices.size:
            angles = np.asarray(data.two_theta_deg, dtype=float)
            nearest = int(indices[np.argmin(np.abs(angles[indices] - float(event.xdata)))])
            panel.request_point_mask(nearest)

    def _point_event_is_available(
        self,
        event: object,
        panel: object,
        canvas: object,
    ) -> bool:
        return bool(
            self._toolbar.mode() == "mask"
            and event.inaxes is panel.view("raw").axes
            and event.button == 1
            and event.xdata is not None
            and canvas.widgetlock.available(self)
        )

    def _tab_changed(self, index: int) -> None:
        if self._projecting_tabs or index < 0:
            return
        self._requested_tab_index = index
        # Each canvas navigates independently, so the buttons have to follow the
        # view that is now on screen.  Otherwise pan left latched on the previous
        # tab would keep the button pressed over a canvas that is not panning.
        navigator = self._navigators.get(self.current_view_key())
        if navigator is not None:
            self._sync_navigation_buttons(navigator)
        self._panel.view_changed.emit(index)

    def _apply_tabs(self, expert_mode: bool, requested_index: int) -> None:
        """Project expert mode onto the companion pane, not the tab bar.

        The SLD profile left the tab bar for a permanent pane, so expert mode
        now shows or hides that pane. Every tab stays selectable in both modes,
        which means a persisted selection can always be honoured.
        """
        self._projecting_tabs = True
        try:
            self._panel.sld_pane.setVisible(expert_mode)
            self._tabs.setCurrentIndex(requested_index)
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
