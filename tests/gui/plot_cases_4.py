"""Plot contract cases, partition 4; collected via test_plots.py."""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403


@pytest.mark.parametrize("mode", ("mask", "range"))
def test_plot_navigation_gesture_cancels_click_mode_and_blocks_selection(qtbot, mode) -> None:
    """Starting a navigation gesture drops the armed click mode.

    The pyqtgraph panes have no widget lock: pan and box-zoom live on the view
    box itself.  Clicking a navigation button therefore disarms range/mask
    (via ``_leave_click_mode``) so the same drag cannot both pan and select; the
    disarmed gesture then commits nothing.
    """
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    ranges: list[tuple[float, float]] = []
    points: list[int] = []
    panel.fit_range_requested.connect(lambda low, high: ranges.append((low, high)))
    panel.point_mask_requested.connect(points.append)
    panel.select_view("raw")
    panel.set_interaction_mode(mode)

    panel.navigation_buttons()["pan"].click()
    assert panel.interaction_mode() == "view"

    if mode == "range":
        _drag_range(panel, 0.5, 2.5)
    else:
        panel.request_point_mask(2)

    assert ranges == []
    assert points == []


def test_plot_renders_excluded_points_as_scene_markers(qtbot) -> None:
    data = prepared_data(size=4, fit_mask=np.array([False, True, False, True]))
    panel = _panel(qtbot, data=data)
    raw = panel.view("raw")

    assert _marker(raw, "排除点") == "x"
    assert not _marker_filled(raw, "排除点")
    np.testing.assert_array_equal(_line_x(raw, "排除点"), data.two_theta_deg[[0, 2]])


def test_standard_mode_hides_sld_and_expert_mode_restores_it(qtbot) -> None:
    """Expert mode now gates the companion pane instead of an SLD tab."""
    panel = _panel(qtbot, data=prepared_data(size=4))

    panel.set_expert_mode(False)
    assert panel.sld_pane.isVisibleTo(panel) is False

    panel.set_expert_mode(True)
    assert panel.sld_pane.isVisibleTo(panel) is True


def test_number_key_shortcuts_address_tabs_by_visible_position(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_expert_mode(True)

    assert panel.select_visible_view(4) is True
    assert panel.current_view_key() == "candidates"


def test_number_key_shortcut_positions_are_mode_independent(qtbot) -> None:
    """Every tab is selectable in both modes, so ordinals never shift."""
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_expert_mode(False)

    assert panel.select_visible_view(4) is True
    assert panel.current_view_key() == "candidates"


def test_number_key_shortcuts_reject_out_of_range_ordinal(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    before = panel.current_view_key()

    assert panel.select_visible_view(20) is False
    assert panel.current_view_key() == before


def test_view_shortcuts_register_one_key_per_tab(qtbot) -> None:
    from PySide6.QtGui import QKeySequence

    panel = _panel(qtbot, data=prepared_data(size=4))
    keys = [shortcut.key() for shortcut in panel.view_shortcuts]

    assert keys[0] == QKeySequence("Alt+1")
    assert keys[-1] == QKeySequence(f"Alt+{len(panel.tab_keys())}")


def _drag_on_view(panel, key: str, dx: int = 60, dy: int = 0) -> None:
    """Drag across the middle of a view, as a pan or a box zoom would.

    The gesture is built in display coordinates rather than data coordinates:
    the axes middle is inside every view regardless of whether it carries a log
    reflectivity scale or an inverted correlation image.
    """
    from matplotlib.backend_bases import MouseButton

    view = panel.view(key)
    view.canvas.draw()
    x0, y0 = view.axes.transAxes.transform((0.5, 0.5))
    places = ((x0, y0), (x0 + dx, y0 + dy), (x0 + dx, y0 + dy))
    names = ("button_press_event", "motion_notify_event", "button_release_event")
    for name, (x, y) in zip(names, places, strict=True):
        # A motion event also has to say which buttons are *still* down: pan
        # treats a drag whose button set no longer matches the press as a gesture
        # that lost focus mid-way, and cancels itself.
        held = {MouseButton.LEFT} if name == "motion_notify_event" else None
        event = MouseEvent(name, view.canvas, x, y, button=MouseButton.LEFT, buttons=held)
        view.canvas.callbacks.process(name, event)


def test_plot_toolbar_exposes_accessible_navigation_controls(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    buttons = panel.navigation_buttons()

    assert set(buttons) == {"pan", "zoom", "home"}
    # Pan and box zoom latch until switched off; resetting the view is one shot.
    assert {name: button.isCheckable() for name, button in buttons.items()} == {
        "pan": True,
        "zoom": True,
        "home": False,
    }
    assert all(button.accessibleName() and button.toolTip() for button in buttons.values())


def test_plot_navigation_pan_then_home_restores_the_drawn_limits(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.select_view("raw")
    raw = panel.view("raw")
    raw.autoscale_view()
    drawn = _view_xrange(raw)
    buttons = panel.navigation_buttons()

    buttons["pan"].click()
    assert panel.navigation_mode() == "pan"
    # A pan shifts the view box; the pyqtgraph pane exposes that move directly.
    raw.set_view_xrange(drawn[0] + 1.0, drawn[1] + 1.0)
    assert _view_xrange(raw) != drawn

    buttons["home"].click()
    assert _view_xrange(raw) == pytest.approx(drawn)


def test_plot_navigation_home_keeps_the_correlation_image_orientation(qtbot) -> None:
    """The correlation matrix reads top-down, and resetting must not flip it.

    ``imshow`` inverts the y axis, and autoscaling an inverted axis silently
    restores it the right way up, so the reset has to replay the recorded limits
    instead of autoscaling.
    """
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    panel.select_view("uncertainty")
    drawn = panel.view("uncertainty").axes.get_ylim()
    assert drawn[0] > drawn[1]

    panel.navigation_buttons()["pan"].click()
    _drag_on_view(panel, "uncertainty", dx=30, dy=30)
    panel.navigation_buttons()["home"].click()

    assert panel.view("uncertainty").axes.get_ylim() == drawn


def test_plot_navigation_and_click_modes_switch_each_other_off(qtbot) -> None:
    """Both families share the one toolbar, so the later choice unlatches the first."""
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.select_view("raw")
    buttons = panel.navigation_buttons()

    buttons["pan"].click()
    assert buttons["pan"].isChecked() is True
    assert panel.navigation_mode() == "pan"

    # Arming a click mode drops the latched navigation button.
    panel.set_interaction_mode("range")
    assert buttons["pan"].isChecked() is False

    # Clicking a navigation button in turn leaves range/mask for plain view.
    buttons["zoom"].click()
    assert panel.interaction_mode() == "view"
    assert buttons["zoom"].isChecked() is True


def test_plot_navigation_buttons_follow_the_visible_view(qtbot) -> None:
    """Each pane navigates on its own, so a latched button cannot outlive its tab."""
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.select_view("raw")
    buttons = panel.navigation_buttons()
    buttons["zoom"].click()
    assert panel.navigation_mode() == "zoom"
    assert buttons["zoom"].isChecked() is True

    panel.select_view("log")
    # The log pane holds its own mode, still the pane default, so zoom is not latched.
    assert panel.navigation_mode() == "pan"
    assert buttons["zoom"].isChecked() is False

    panel.select_view("raw")
    # The raw pane kept its box-zoom mode, so returning to it re-latches the button.
    assert panel.navigation_mode() == "zoom"
    assert buttons["zoom"].isChecked() is True


def test_plot_navigation_home_targets_the_latest_redraw(qtbot) -> None:
    """Home restores the current data's bounds, not a replaced dataset's window."""
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_dataset("wide", prepared_data(size=4, two_theta_deg=np.linspace(5.0, 9.0, 4)))
    panel.select_view("raw")
    raw = panel.view("raw")
    raw.autoscale_view()
    drawn = _view_xrange(raw)
    assert drawn[1] > 5.0
    buttons = panel.navigation_buttons()

    buttons["pan"].click()
    raw.set_view_xrange(drawn[0] - 3.0, drawn[0] - 1.0)
    assert _view_xrange(raw) != drawn

    buttons["home"].click()
    assert _view_xrange(raw) == pytest.approx(drawn)


def test_plot_panel_release_disconnects_every_navigator(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    mpl_keys = _mpl_view_keys(panel)
    # Only the static matplotlib canvases carry a navigator; the pyqtgraph panes
    # drive their own gestures and are never entered into the navigator map.
    assert set(panel.navigators()) == set(mpl_keys)
    canvases = [panel.view(key).canvas for key in mpl_keys]
    assert all(canvas.toolbar is not None for canvas in canvases)

    panel.release_resources()

    assert panel.navigators() == {}
    assert all(canvas.toolbar is None for canvas in canvases)


# Every plot control the user clicks -- the three modes, the three navigation
# actions and the two one-shot zooms -- so a bare text button reads as "什么玩意"
# rather than as a recognisable tool.
PLOT_CONTROL_NAMES = (
    "plotModeView",
    "plotModeRange",
    "plotModeMask",
    "plotNavPan",
    "plotNavZoom",
    "plotNavHome",
    "plotZoomToRange",
    "plotResetZoom",
)


def _control_glyph(button) -> bytes:
    """The rendered 16px glyph, so two controls compare by what the eye sees."""
    return bytes(button.icon().pixmap(16, 16).toImage().constBits())


def test_plot_controls_each_wear_an_icon(qtbot) -> None:
    """The literal complaint: a graphical tool shows tools, not a wall of text."""
    panel = _panel(qtbot, data=prepared_data(size=4))

    bare = [name for name in PLOT_CONTROL_NAMES if panel.findChild(QToolButton, name).icon().isNull()]

    assert bare == []


def test_plot_controls_wear_distinct_icons(qtbot) -> None:
    """A shared glyph would make two tools look like one, defeating the point."""
    panel = _panel(qtbot, data=prepared_data(size=4))

    seen: dict[bytes, str] = {}
    for name in PLOT_CONTROL_NAMES:
        glyph = _control_glyph(panel.findChild(QToolButton, name))
        assert glyph not in seen, f"{name} 与 {seen[glyph]} 用了同一个图标"
        seen[glyph] = name


def test_plot_controls_show_the_glyph_alone(qtbot) -> None:
    """The bar floats over the plot, so it shows icons and no words.

    Labels are what made the bar as wide as the panel; over the data they would
    curtain the curve the tool acts on.  The name is not lost, it moves to the
    tooltip and the accessible name, which is where peer charting tools keep it.
    """
    panel = _panel(qtbot, data=prepared_data(size=4))

    for name in PLOT_CONTROL_NAMES:
        button = panel.findChild(QToolButton, name)
        assert button.text() == ""
        assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
        assert button.toolTip() != ""
        assert button.accessibleName() != ""


def _scroll_on_view(panel, key: str, *, step: float, xfrac: float = 0.5, yfrac: float = 0.5):
    """Fire a wheel-scroll over a fraction of the view, as the mouse wheel would.

    The point is built in axes-fraction space so the same helper reaches every
    view whatever its scale, mirroring ``_drag_on_view``.  ``step`` follows the
    Matplotlib convention: positive scrolls up (zoom in), negative scrolls down.
    """
    view = panel.view(key)
    view.canvas.draw()
    x, y = view.axes.transAxes.transform((xfrac, yfrac))
    event = MouseEvent("scroll_event", view.canvas, x, y, step=step)
    view.canvas.callbacks.process("scroll_event", event)


def test_plot_wheel_scroll_up_zooms_in_around_the_cursor(qtbot) -> None:
    """The wheel is the one gesture every user already knows for "look closer".

    Direct manipulation means the plot answers the wheel without first arming a
    mode: scrolling up narrows both axes, and the data point under the pointer
    keeps its place on screen so the zoom feels anchored rather than recentred.
    The candidate-comparison view keeps this matplotlib gesture; its qz axis is
    linear, so the cursor-anchor arithmetic reads directly off the limits.
    """
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    panel.select_view("candidates")
    axes = panel.view("candidates").axes
    low, high = axes.get_xlim()
    cursor_x = low + 0.3 * (high - low)
    y_low, y_high = axes.get_ylim()

    _scroll_on_view(panel, "candidates", step=1.0, xfrac=0.3)

    new_low, new_high = axes.get_xlim()
    assert (new_high - new_low) < (high - low)
    assert (axes.get_ylim()[1] - axes.get_ylim()[0]) < (y_high - y_low)
    # The point under the pointer keeps its fractional position: the zoom is
    # anchored to the cursor, not to the axes centre.
    assert (cursor_x - new_low) / (new_high - new_low) == pytest.approx(0.3, abs=1e-6)


def test_plot_wheel_scroll_down_zooms_out(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    panel.select_view("candidates")
    axes = panel.view("candidates").axes
    width = axes.get_xlim()[1] - axes.get_xlim()[0]

    _scroll_on_view(panel, "candidates", step=-1.0)

    assert (axes.get_xlim()[1] - axes.get_xlim()[0]) > width


def test_plot_wheel_zoom_respects_a_log_axis(qtbot) -> None:
    """Zooming a decade axis must stay in log space, never crossing zero."""
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    panel.select_view("candidates")
    axes = panel.view("candidates").axes
    low, high = axes.get_ylim()
    assert low > 0.0
    ratio = high / low

    _scroll_on_view(panel, "candidates", step=1.0)

    new_low, new_high = axes.get_ylim()
    assert new_low > 0.0
    assert (new_high / new_low) < ratio


def test_plot_cursor_readout_reports_coordinates_under_the_pointer(qtbot) -> None:
    """A graphical plot should say where the pointer is, like every peer tool."""
    from PySide6.QtWidgets import QLabel

    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.select_view("raw")
    readout = panel.findChild(QLabel, "plotCursorReadout")
    assert readout is not None
    idle = readout.text()

    # The pyqtgraph pane reports the pointer through its own scene signals; the
    # controller turns those into the coordinate read-out.
    raw = panel.view("raw")
    raw.cursor_moved.emit(0.7, 123.0)
    hovering = readout.text()
    assert any(character.isdigit() for character in hovering)

    # Leaving the pane clears the coordinate and restores the standing hint, so a
    # stale reading never lingers as if the pointer were still on the curve.
    raw.cursor_left.emit()
    assert readout.text() == idle
    assert hovering != idle
