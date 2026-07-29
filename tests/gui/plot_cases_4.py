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
