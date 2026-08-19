"""Plot contract cases, partition 4; collected via test_plots.py."""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403


@pytest.mark.parametrize("mode", ("mask", "range"))
def test_plot_panel_widgetlock_owner_blocks_plot_selection(qtbot, mode) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    ranges: list[tuple[float, float]] = []
    points: list[int] = []
    panel.fit_range_requested.connect(lambda low, high: ranges.append((low, high)))
    panel.point_mask_requested.connect(points.append)
    panel.set_interaction_mode(mode)
    canvas = panel.view("raw").canvas
    owner = object()
    canvas.widgetlock(owner)

    if mode == "range":
        _drag_range(panel, 0.5, 2.5)
    else:
        canvas.callbacks.process(
            "button_press_event",
            _mouse_event("button_press_event", panel, float(data.two_theta_deg[2])),
        )

    canvas.widgetlock.release(owner)
    assert ranges == []
    assert points == []


def test_plot_renders_excluded_points_as_scene_markers(qtbot) -> None:
    data = prepared_data(size=4, fit_mask=np.array([False, True, False, True]))
    panel = _panel(qtbot, data=data)
    excluded = next(line for line in panel.view("raw").axes.lines if line.get_label() == "排除点")

    assert excluded.get_marker() == "x"
    assert excluded.get_linestyle() == "None"
    np.testing.assert_array_equal(excluded.get_xdata(), data.two_theta_deg[[0, 2]])


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
    buttons = panel.navigation_buttons()
    drawn = panel.view("raw").axes.get_xlim()

    buttons["pan"].click()
    assert panel.navigation_mode() == "pan/zoom"
    _drag_on_view(panel, "raw")
    assert panel.view("raw").axes.get_xlim() != drawn

    buttons["home"].click()
    assert panel.view("raw").axes.get_xlim() == drawn


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
    """Both families need the same mouse press, so the later choice wins."""
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.select_view("raw")
    buttons = panel.navigation_buttons()
    canvas = panel.view("raw").canvas

    buttons["pan"].click()
    assert canvas.widgetlock.available(None) is False

    panel.set_interaction_mode("range")
    assert buttons["pan"].isChecked() is False
    assert canvas.widgetlock.available(None) is True

    buttons["zoom"].click()
    assert panel.interaction_mode() == "view"
    assert buttons["zoom"].isChecked() is True


def test_plot_navigation_buttons_follow_the_visible_view(qtbot) -> None:
    """Each canvas navigates on its own, so a latched button cannot outlive its tab."""
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.select_view("raw")
    buttons = panel.navigation_buttons()
    buttons["pan"].click()

    panel.select_view("log")
    assert panel.navigation_mode() == ""
    assert buttons["pan"].isChecked() is False

    panel.select_view("raw")
    assert panel.navigation_mode() == "pan/zoom"
    assert buttons["pan"].isChecked() is True


def test_plot_navigation_home_targets_the_latest_redraw(qtbot) -> None:
    """Limits recorded against a replaced dataset would restore a vanished window."""
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_dataset("wide", prepared_data(size=4, two_theta_deg=np.linspace(5.0, 9.0, 4)))
    panel.select_view("raw")
    buttons = panel.navigation_buttons()
    drawn = panel.view("raw").axes.get_xlim()
    assert drawn[1] > 5.0

    buttons["pan"].click()
    _drag_on_view(panel, "raw")
    buttons["home"].click()

    assert panel.view("raw").axes.get_xlim() == drawn


def test_plot_panel_release_disconnects_every_navigator(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    canvases = [panel.view(key).canvas for key in panel.view_keys()]
    assert all(canvas.toolbar is not None for canvas in canvases)

    panel.release_resources()

    assert panel.navigators() == {}
    assert all(canvas.toolbar is None for canvas in canvases)
