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
    excluded = next(
        line for line in panel.view("raw").axes.lines if line.get_label() == "排除点"
    )

    assert excluded.get_marker() == "x"
    assert excluded.get_linestyle() == "None"
    np.testing.assert_array_equal(excluded.get_xdata(), data.two_theta_deg[[0, 2]])

def test_standard_mode_hides_sld_and_expert_mode_restores_it(qtbot) -> None:
    panel = _panel(qtbot)
    sld_index = panel.view_keys().index("sld")

    panel.set_expert_mode(False)
    assert panel.tabs.isTabVisible(sld_index) is False

    panel.set_expert_mode(True)
    assert panel.tabs.isTabVisible(sld_index) is True

def test_number_key_shortcuts_follow_visible_position_in_expert_mode(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_expert_mode(True)

    assert panel.select_visible_view(4) is True
    assert panel.current_view_key() == "sld"

def test_number_key_shortcuts_skip_hidden_tabs_in_standard_mode(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_expert_mode(False)

    assert panel.select_visible_view(4) is True
    assert panel.current_view_key() == "candidates"

def test_number_key_shortcuts_reject_out_of_range_ordinal(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    before = panel.current_view_key()

    assert panel.select_visible_view(20) is False
    assert panel.current_view_key() == before

def test_view_shortcuts_register_alt_1_through_8(qtbot) -> None:
    from PySide6.QtGui import QKeySequence

    panel = _panel(qtbot, data=prepared_data(size=4))
    keys = [shortcut.key() for shortcut in panel.view_shortcuts]

    assert keys[0] == QKeySequence("Alt+1")
    assert keys[7] == QKeySequence("Alt+8")
