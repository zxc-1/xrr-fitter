"""The pyqtgraph-backed live reflectivity view honours the existing draw contract.

This widget is the interactive half of the plot migration: matplotlib still renders
the static diagnostics and the byte-identity export, while the reflectivity family
gains a mutating preview curve and native drag-to-select range / click-to-mask in
place of the matplotlib toolbar. These tests pin that contract without a visible
window, so they never depend on the desktop focus the offscreen platform withholds.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.live import LiveReflectivityPlot


def _collect(signal) -> list:
    received: list = []
    signal.connect(lambda *args: received.append(args if len(args) != 1 else args[0]))
    return received


def test_observed_points_carry_the_entered_arrays(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2, 0.3])
    values = np.array([1.0, 0.5, 0.25])
    plot.set_observed(angles, values)
    x, y = plot.observed_item.getData()
    assert np.allclose(x, angles)
    assert np.allclose(y, values)


def test_preview_mutates_one_owned_curve_across_updates(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2, 0.3])
    assert plot.set_preview(angles, np.array([0.9, 0.4, 0.1])) is True
    first = plot.preview_item
    assert first is not None
    assert plot.set_preview(angles, np.array([0.8, 0.3, 0.05])) is True
    # A live search updates many times per second, so the second publish must
    # reuse the same artist rather than stack a new one behind it.
    assert plot.preview_item is first
    _, y = plot.preview_item.getData()
    assert np.allclose(y, np.array([0.8, 0.3, 0.05]))


def test_clearing_preview_drops_the_curve_and_allows_recreation(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2])
    plot.set_preview(angles, np.array([0.9, 0.4]))
    plot.clear_preview()
    assert plot.preview_item is None
    assert plot.set_preview(angles, np.array([0.5, 0.2])) is True
    assert plot.preview_item is not None


def test_released_widget_refuses_further_preview(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.release()
    assert plot.set_preview(np.array([0.1]), np.array([0.5])) is False


def test_log_mode_toggles_the_left_axis(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_log_mode(True)
    assert plot.plot_item.ctrl.logYCheck.isChecked() is True
    plot.set_log_mode(False)
    assert plot.plot_item.ctrl.logYCheck.isChecked() is False


def test_range_selection_emits_sorted_bounds(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.enable_range_selection(True)
    received = _collect(plot.fit_range_selected)
    region = plot.range_item
    assert region is not None
    # A user can drag either handle first, so the widget must publish the
    # interval low-to-high whatever order the region reports internally.
    region.setRegion((0.22, 0.08))
    region.sigRegionChangeFinished.emit(region)
    assert received
    low, high = received[-1]
    assert low < high
    assert np.isclose(low, 0.08)
    assert np.isclose(high, 0.22)


def test_range_item_is_hidden_until_selection_is_enabled(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    assert plot.range_item is None
    plot.enable_range_selection(True)
    assert plot.range_item is not None
    plot.enable_range_selection(False)
    assert plot.range_item is None


def test_masking_emits_the_clicked_position_only_when_enabled(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    received = _collect(plot.point_mask_requested)
    plot._mask_from_view_x(0.137)
    assert received == []
    plot.enable_masking(True)
    plot._mask_from_view_x(0.137)
    assert received and np.isclose(received[-1], 0.137)


def test_observed_and_model_use_the_shared_palette_colours(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_observed(np.array([0.1, 0.2]), np.array([1.0, 0.5]))
    plot.set_model(np.array([0.1, 0.2]), np.array([0.9, 0.45]))
    # Matching the matplotlib views keeps a candidate curve the same colour
    # whichever backend drew the tab the user is looking at.
    observed_colour = plot.observed_item.opts["pen"] or plot.observed_item.opts["symbolBrush"]
    assert observed_colour is not None
    assert pg.mkColor(theme.DATA_CANDIDATE).name() == plot.model_item.opts["pen"].color().name()


def test_log_projection_applies_the_display_floor_and_log_axis(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2, 0.3])
    observed = np.array([1.0, 1e-7, 0.25])
    model = np.array([0.9, 1e-9, 0.2])
    plot.show_log_reflectivity(angles, observed, model, r_floor=1e-6)
    # The floor is the same presentation clamp draw_log applies, so a value below
    # it reads at the floor rather than dropping off the axis.  pyqtgraph reports
    # a log-mode curve in log10 display space, so the floored point reads
    # log10(1e-6) = -6 rather than log10(1e-7) = -7.
    _, obs_y = plot.observed_item.getData()
    _, model_y = plot.model_item.getData()
    assert np.allclose(obs_y, np.log10(np.maximum(observed, 1e-6)))
    assert np.allclose(model_y, np.log10(np.maximum(model, 1e-6)))
    assert plot.plot_item.ctrl.logYCheck.isChecked() is True


def test_log_projection_without_candidate_clears_the_model_curve(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2])
    plot.show_log_reflectivity(angles, np.array([1.0, 0.5]), None, r_floor=1e-6)
    model_x, model_y = plot.model_item.getData()
    assert model_x is None or len(model_x) == 0
    assert model_y is None or len(model_y) == 0


def test_raw_reflectivity_splits_included_and_excluded_on_a_linear_axis(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2, 0.3, 0.4])
    raw = np.array([100.0, 80.0, 60.0, 40.0])
    mask = np.array([True, False, True, False])
    model = np.array([99.0, 79.0, 59.0, 39.0])
    plot.show_raw_reflectivity(angles, raw, mask, model)
    # The raw view keeps the fit points and the struck-out excluded points on two
    # separate artists so a reader sees at a glance which points the search used.
    inc_x, inc_y = plot.observed_item.getData()
    exc_x, exc_y = plot.excluded_item.getData()
    assert np.allclose(inc_x, angles[mask]) and np.allclose(inc_y, raw[mask])
    assert np.allclose(exc_x, angles[~mask]) and np.allclose(exc_y, raw[~mask])
    _, model_y = plot.model_item.getData()
    assert np.allclose(model_y, model)
    # Raw intensity spans one order at most, so it reads on a linear axis.
    assert plot.plot_item.ctrl.logYCheck.isChecked() is False


def test_raw_reflectivity_without_candidate_leaves_model_empty(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2, 0.3])
    raw = np.array([100.0, 80.0, 60.0])
    mask = np.array([True, True, False])
    plot.show_raw_reflectivity(angles, raw, mask, None)
    model_x, model_y = plot.model_item.getData()
    assert model_x is None or len(model_x) == 0
    assert model_y is None or len(model_y) == 0


def test_qz4_plots_data_and_model_on_a_linear_axis(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    data_qz = np.array([0.10, 0.20, 0.30])
    data_values = np.array([1e-4, 2e-4, 3e-4])
    model_qz = np.array([0.10, 0.20, 0.30])
    model_values = np.array([1.1e-4, 1.9e-4, 3.1e-4])
    plot.show_qz4(data_qz, data_values, model_qz, model_values)
    obs_x, obs_y = plot.observed_item.getData()
    mod_x, mod_y = plot.model_item.getData()
    assert np.allclose(obs_x, data_qz) and np.allclose(obs_y, data_values)
    assert np.allclose(mod_x, model_qz) and np.allclose(mod_y, model_values)
    assert plot.plot_item.ctrl.logYCheck.isChecked() is False


def test_qz4_without_model_leaves_model_curve_empty(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_qz4(np.array([0.1, 0.2]), np.array([1e-4, 2e-4]), None, None)
    model_x, model_y = plot.model_item.getData()
    assert model_x is None or len(model_x) == 0
    assert model_y is None or len(model_y) == 0


def test_residual_draws_the_curve_and_a_zero_reference(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    qz = np.array([0.1, 0.2, 0.3])
    weighted = np.array([0.5, -0.2, 0.1])
    plot.show_residual(qz, weighted)
    obs_x, obs_y = plot.observed_item.getData()
    ref_x, ref_y = plot.reference_item.getData()
    assert np.allclose(obs_x, qz) and np.allclose(obs_y, weighted)
    # The dotted baseline anchors the eye at zero misfit across the whole q range.
    assert np.allclose(ref_x, qz) and np.allclose(ref_y, np.zeros_like(qz))
    assert plot.plot_item.ctrl.logYCheck.isChecked() is False


def test_clear_series_empties_every_managed_curve(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.array([0.1, 0.2, 0.3])
    plot.show_raw_reflectivity(
        angles, np.array([9.0, 8.0, 7.0]), np.array([True, False, True]), np.array([9.0, 8.0, 7.0])
    )
    plot.clear_series()
    for item in (plot.observed_item, plot.model_item, plot.excluded_item, plot.reference_item):
        x, _ = item.getData()
        assert x is None or len(x) == 0


def test_fit_range_highlight_is_shaded_and_distinct_from_the_selector(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_fit_range(0.28, 0.12)
    assert plot.fit_range_item is not None
    # The committed-range band is not the drag selector; a read-only pane shows
    # the band without arming the interactive handles.
    assert plot.range_item is None
    low, high = sorted(plot.fit_range_item.getRegion())
    assert np.isclose(low, 0.12) and np.isclose(high, 0.28)
    assert plot.fit_range_item.movable is False
    plot.clear_fit_range()
    assert plot.fit_range_item is None


def test_view_xrange_and_autoscale_drive_the_viewbox(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.5, 0.25]), None, r_floor=1e-6)
    plot.set_view_xrange(0.15, 0.25)
    low, high = plot.plot_item.vb.viewRange()[0]
    assert np.isclose(low, 0.15, atol=1e-6) and np.isclose(high, 0.25, atol=1e-6)
    plot.autoscale_view()
    low2, high2 = plot.plot_item.vb.viewRange()[0]
    assert low2 <= 0.1 + 1e-9 and high2 >= 0.3 - 1e-9


def test_log_pane_annotates_title_axes_and_floor_label(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.1, 0.2]), np.array([1.0, 0.5]), None, r_floor=1e-6)
    title, xlabel, ylabel = plot.axis_labels()
    # The pg pane must state the same axis identity draw_log wrote, so a reader
    # sees the display-floor caveat rather than a bare "R" whatever backend drew.
    assert title == "对数反射率"
    assert xlabel == "2θ (deg)"
    assert "显示下限" in ylabel and "display floor" not in ylabel


def test_raw_pane_annotates_title_and_axes(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_raw_reflectivity(np.array([0.1, 0.2]), np.array([9.0, 8.0]), np.array([True, True]), None)
    assert plot.axis_labels() == ("原始数据与模型", "2θ (deg)", "原始强度")


def test_qz4_pane_uses_the_supplied_dynamic_ylabel(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    dynamic = "归一化 (qz/1 Å⁻¹)⁴R"
    plot.show_qz4(np.array([0.1, 0.2]), np.array([1e-4, 2e-4]), None, None, ylabel=dynamic)
    title, xlabel, ylabel = plot.axis_labels()
    # qz⁴ scaling is decided upstream and can relabel the axis when it overflows,
    # so the pane renders whatever ylabel the projection handed it.
    assert title == "qz⁴R 诊断变换（非拟合数据）"
    assert xlabel == "qz (Å⁻¹)"
    assert ylabel == dynamic and "归一化" in ylabel


def test_residual_pane_annotates_title_and_axes(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.1, 0.2]), np.array([0.3, -0.3]))
    assert plot.axis_labels() == ("加权残差", "qz (Å⁻¹)", "加权残差")


def test_panes_carry_a_reference_grid_from_construction(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    # draw_* enabled a grid so values read against gridlines; the pg pane keeps it
    # on both axes from the start rather than per draw.
    assert plot.plot_item.ctrl.xGridCheck.isChecked() is True
    assert plot.plot_item.ctrl.yGridCheck.isChecked() is True


def test_quality_caption_reads_back_and_clears(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    assert plot.quality_caption_text() is None
    plot.set_quality_caption("J=0.25 · 平均残差 0.1 decade")
    assert "J=0.25" in plot.quality_caption_text()
    plot.set_quality_caption(None)
    assert plot.quality_caption_text() is None


def test_placeholder_shows_and_drawing_data_hides_it(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_placeholder("暂无当前候选")
    assert plot.placeholder_text() == "暂无当前候选"
    # Drawing a real series is the pane leaving its placeholder state, matching how
    # the matplotlib qz⁴/residual views drop draw_empty once a candidate arrives.
    plot.show_residual(np.array([0.1, 0.2]), np.array([0.1, -0.1]))
    assert plot.placeholder_text() is None


def test_placeholder_clears_series_and_quality_caption(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.1, 0.2]), np.array([0.1, -0.1]))
    plot.set_quality_caption("J=0.25")
    plot.show_placeholder("暂无当前候选")
    obs_x, _ = plot.observed_item.getData()
    assert obs_x is None or len(obs_x) == 0
    assert plot.quality_caption_text() is None


def test_cursor_report_emits_coordinates_then_leave(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    moved = _collect(plot.cursor_moved)
    left = _collect(plot.cursor_left)
    # The readout replaces the matplotlib toolbar coordinate display; a move under
    # the pointer publishes the view coordinate, leaving the area publishes idle.
    plot._emit_cursor(0.2, 0.5)
    assert moved and np.allclose(moved[-1], (0.2, 0.5))
    plot._emit_cursor_left()
    assert len(left) == 1


def test_cursor_report_delinearizes_y_on_a_log_axis(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_log_mode(True)
    moved = _collect(plot.cursor_moved)
    # pg holds a log-mode viewbox in log10 space; the readout must report the
    # physical reflectivity a reader expects, not the exponent.
    plot._emit_cursor(0.2, -3.0)
    x, y = moved[-1]
    assert np.isclose(x, 0.2) and np.isclose(y, 1e-3)


def test_released_widget_stops_cursor_reports(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.release()
    moved = _collect(plot.cursor_moved)
    left = _collect(plot.cursor_left)
    plot._emit_cursor(0.2, 0.5)
    plot._emit_cursor_left()
    assert moved == [] and left == []


def test_navigation_mode_toggles_the_viewbox_mouse_mode(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    # pan drags translate, zoom drags rubber-band a rectangle: the pg equivalents
    # of the matplotlib pan/zoom toolbar buttons the reader used to click.
    assert plot.navigation_mode() == "pan"
    plot.set_navigation_mode("zoom")
    assert plot.navigation_mode() == "zoom"
    assert plot.plot_item.vb.state["mouseMode"] == pg.ViewBox.RectMode
    plot.set_navigation_mode("pan")
    assert plot.navigation_mode() == "pan"
    assert plot.plot_item.vb.state["mouseMode"] == pg.ViewBox.PanMode


def test_go_home_restores_the_drawn_data_bounds_after_pan(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.5, 0.25]), None, r_floor=1e-6)
    plot.set_view_xrange(0.15, 0.18)
    # Home returns to the latest draw's data bounds, so a reader who panned away
    # gets the whole curve back like the matplotlib home button restored.
    plot.go_home()
    low, high = plot.plot_item.vb.viewRange()[0]
    assert low <= 0.1 + 1e-9 and high >= 0.3 - 1e-9
