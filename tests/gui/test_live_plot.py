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
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.live import LiveReflectivityPlot


def _collect(signal) -> list:
    received: list = []
    signal.connect(lambda *args: received.append(args if len(args) != 1 else args[0]))
    return received


def _axis_texts(plot: LiveReflectivityPlot, side: str) -> list[str]:
    """坐标轴真正会画出来的刻度文字。

    pyqtgraph 的刻度文字要过两道筛：``maxTextLevel`` 决定几级刻度带标签，拥挤判定
    再决定次级标签是否让位。断言 ``tickStrings`` 只看得到候选字符串，看不到这两道筛
    之后剩下什么，所以这里读 ``generateDrawSpecs`` 的第三项——绘制时用的正是它。
    """
    axis = plot.plot_item.getAxis(side)
    image = QImage(4, 4, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        _axis_spec, _tick_specs, text_specs = axis.generateDrawSpecs(painter)
    finally:
        painter.end()
    return [spec[2] for spec in text_specs]


def _laid_out(qtbot, plot: LiveReflectivityPlot, height: int) -> LiveReflectivityPlot:
    """给已经画好的面板一个真实尺寸，并把布局与自动量程跑完。

    刻度文字是照坐标轴当时的几何与量程算的：没跑过布局的面板量出来的是一张空表，
    没跑过事件循环的面板量到的还是空面板那条 0–1 的默认量程。两种都不是设计稿要
    检查的东西，所以画完数据再进来，等布局和量程都落定。
    """
    qtbot.addWidget(plot)
    plot.resize(840, height)
    plot.show()
    qtbot.waitExposed(plot)
    qtbot.wait(1)
    plot.plot_item.layout.activate()
    return plot


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


def test_range_drag_claims_the_viewbox_and_restores_navigation(qtbot) -> None:
    """Dragging a range handle must not translate the plot underneath it."""
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.resize(800, 400)
    plot.show()
    qtbot.wait(1)
    plot.plot_item.vb.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0)
    plot.set_navigation_mode("pan")
    before = plot.plot_item.vb.viewRange()
    original_mouse_buttons = plot.plot_item.vb.acceptedMouseButtons()

    plot.enable_range_selection(True)
    region = plot.range_item
    assert region is not None
    region.setRegion((0.2, 0.4))
    assert plot.plot_item.vb.state["mouseEnabled"] == [False, False]
    assert plot.plot_item.vb.acceptedMouseButtons() == Qt.MouseButton.NoButton

    received = _collect(plot.fit_range_selected)
    view = plot.scene().views()[0]

    def viewport_position(x: float) -> object:
        scene = plot.plot_item.vb.mapViewToScene(QPointF(x, 0.5))
        return view.mapFromScene(scene)

    start = viewport_position(0.2)
    finish = viewport_position(0.3)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), finish, delay=20)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=finish)
    qtbot.wait(1)

    low, high = sorted(region.getRegion())
    assert np.isclose(low, 0.3, atol=0.02)
    assert np.isclose(high, 0.4, atol=0.02)
    assert plot.plot_item.vb.viewRange() == before
    assert received and np.isclose(received[-1][0], low) and np.isclose(received[-1][1], high)

    plot.enable_range_selection(False)
    assert plot.plot_item.vb.state["mouseEnabled"] == [True, True]
    assert plot.plot_item.vb.acceptedMouseButtons() == original_mouse_buttons


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
    # The baseline anchors the eye at zero misfit across the whole q range.
    assert np.allclose(ref_x, qz) and np.allclose(ref_y, np.zeros_like(qz))
    assert plot.plot_item.ctrl.logYCheck.isChecked() is False


def test_the_residual_pane_shades_the_one_sigma_band_its_card_keys(qtbot) -> None:
    """设计稿残差图里那条横贯的浅带就是 ±1σ，加上 ±2σ 两条虚线参考。

    残差已经除过 σ，所以「大不大」有绝对刻度：±1 之内是噪声量级，越过 ±2 就该解释。
    没有这条带，读者只能拿纵轴刻度反推，而纵轴随这一轮残差自动缩放——同一张图形状
    一样、量级差十倍时看起来是一模一样的。带是常数 ±1，所以它同时也把纵轴的自动缩放
    读成了「这次偏离了几个 σ」。

    ±2σ 用虚线而不是第二条实线：它是提示线不是基准线，实线只留给零。
    """
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)

    plot.show_residual(np.array([0.1, 0.2, 0.3]), np.array([0.5, -0.2, 0.1]))

    band = plot.sigma_band_item
    assert band is not None, "残差图没有 ±1σ 带，而卡片图例已经在给它标名字"
    low, high = sorted(band.getRegion())
    assert np.isclose(low, -1.0) and np.isclose(high, 1.0)
    # 带是读数参考，不是可拖的选区：拖动它等于让读者改掉 σ 的定义。
    assert band.movable is False
    assert band.orientation == "horizontal"
    assert tuple(sorted(round(line.value(), 6) for line in plot.sigma_guide_items)) == (-2.0, 2.0)
    for guide in plot.sigma_guide_items:
        assert guide.pen.style() == Qt.PenStyle.DashLine
        assert guide.movable is False
    # 零线是这张图唯一的基准线，实线；点线在设计稿里没有出现过。
    assert plot.reference_item.opts["pen"].style() == Qt.PenStyle.SolidLine


def test_the_sigma_band_is_installed_once_and_survives_a_redraw(qtbot) -> None:
    """一秒里可能重画很多次残差，带与参考线不能每次都新长一份。"""
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)

    plot.show_residual(np.array([0.1, 0.2]), np.array([0.5, -0.2]))
    band, guides = plot.sigma_band_item, plot.sigma_guide_items
    plot.show_residual(np.array([0.1, 0.2]), np.array([0.1, -0.1]))

    assert plot.sigma_band_item is band
    assert plot.sigma_guide_items == guides


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


def test_log_pane_annotates_axes_the_way_the_design_names_them(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.1, 0.2]), np.array([1.0, 0.5]), None, r_floor=1e-6)
    # 设计稿把两条轴写成整句中文：横轴是入射角，不是裸的 2θ；纵轴点明这是归一化之后
    # 的对数刻度。显示下限不再写进轴标题——被夹到下限的点由 ▽ 空心标记就地标出，
    # 轴标题只回答「这条轴是什么量」。
    assert plot.axis_labels() == ("入射角 2θ (°)", "反射率 R（归一化 · 对数）")
    ylabel = plot.axis_labels()[1]
    assert "display floor" not in ylabel and "显示下限" not in ylabel


def test_raw_pane_annotates_its_axes(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_raw_reflectivity(np.array([0.1, 0.2]), np.array([9.0, 8.0]), np.array([True, True]), None)
    # 同一个物理量在设计稿里只有一种写法，原始视图跟着对数视图叫入射角。
    assert plot.axis_labels() == ("入射角 2θ (°)", "原始强度")


def test_qz4_pane_uses_the_supplied_dynamic_ylabel(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    dynamic = "归一化 (qz/1 Å⁻¹)⁴R"
    plot.show_qz4(np.array([0.1, 0.2]), np.array([1e-4, 2e-4]), None, None, ylabel=dynamic)
    xlabel, ylabel = plot.axis_labels()
    # qz⁴ scaling is decided upstream and can relabel the axis when it overflows,
    # so the pane renders whatever ylabel the projection handed it.
    assert xlabel == "散射矢量 qz (Å⁻¹)"
    assert ylabel == dynamic and "归一化" in ylabel


def test_residual_pane_annotates_its_axes(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.1, 0.2]), np.array([0.3, -0.3]))
    # 残差以 σ 为单位，设计稿把单位写进轴标题：没有它，纵轴上的 2 是两倍误差棒还是
    # 两个反射率单位就只能靠猜。
    assert plot.axis_labels() == ("散射矢量 qz (Å⁻¹)", "加权残差 (σ)")


def test_log_pane_labels_whole_decades_as_powers_of_ten(qtbot) -> None:
    plot = LiveReflectivityPlot()
    plot.show_log_reflectivity(np.linspace(0.15, 6.0, 200), np.logspace(0.0, -6.4, 200), None, r_floor=1e-7)
    texts = _axis_texts(_laid_out(qtbot, plot, 300), "left")
    # pyqtgraph 自己的对数刻度先按 "%0.1g" 写成十进制，只在字符串里出现 "e" 时才改写
    # 成幂次，于是同一条轴上混着 1 / 0.01 / 0.0001 / 10⁻⁶ 四种写法。设计稿是齐整的
    # 10ⁿ：一列同宽的记号才能一眼数出差了几个数量级。
    assert texts, "面板没有量出刻度，这个断言没有在检查任何东西"
    assert all(text.startswith("10") for text in texts), texts

    # 设计稿每个十倍频程都有网格线，但隔一格才写数字。pyqtgraph 默认给次级刻度也配
    # 标签（``maxTextLevel`` 是 2），那会把奇数频程一并写上，纵轴挤成一列连续数字。
    assert "10⁻¹" not in texts and "10⁻³" not in texts and "10⁻⁵" not in texts, texts


def test_log_pane_still_labels_decades_when_the_curve_spans_barely_two(qtbot) -> None:
    """量程只跨一两个频程时，纵轴照样写 10ⁿ，不退回一列十进制小数。

    上一条给的是六个多频程的曲线，pyqtgraph 第一级刻度恰好落在整数指数上，所以它一直绿。屏上
    不是这个量程：帧① 的曲线从 1 掉到 0.03，不到两个频程，此时 ``AxisItem.tickValues`` 只给一
    级、间距 0.05 个频程，值全是非整数指数（-1.45、-1.40…）。``maxTextLevel`` 是 0，只有第一级
    写字，于是整条轴写成 ``0.04 0.05 … 0.9 1``——密集小数，读者数不出差了几个数量级。
    """
    plot = LiveReflectivityPlot()
    plot.show_log_reflectivity(np.linspace(0.15, 6.0, 64), np.logspace(0.0, -1.5, 64), None, r_floor=1e-7)
    texts = _axis_texts(_laid_out(qtbot, plot, 300), "left")

    assert texts, "面板没有量出刻度，这个断言没有在检查任何东西"
    assert all(text.startswith("10") for text in texts), texts
    # 两端那两个整数频程都要写出来：窄量程里可写的记号本来就少，再隔一格就一个都不剩。
    assert "10⁰" in texts and "10⁻¹" in texts, texts


def test_residual_pane_leaves_its_value_axis_bare_and_left_aligned(qtbot) -> None:
    residual = LiveReflectivityPlot()
    residual.show_residual(np.linspace(0.01, 0.44, 120), np.linspace(-2.4, 2.4, 120))
    _laid_out(qtbot, residual, 140)
    reflectivity = LiveReflectivityPlot()
    reflectivity.show_log_reflectivity(np.linspace(0.15, 6.0, 120), np.logspace(0.0, -6.0, 120), None, r_floor=1e-7)
    _laid_out(qtbot, reflectivity, 300)

    # 设计稿的残差面板左侧只有旋转的「加权残差 (σ)」，一个刻度数字都没有：判读残差看的
    # 是 ±1σ 带、±2σ 虚线和零线的相对位置，纵轴上的数字不参与这个判断。
    assert _axis_texts(residual, "left") == [], _axis_texts(residual, "left")

    # 两块面板在画布里上下叠放，绘图区左边界必须落在同一条线上。pyqtgraph 按刻度文字
    # 宽度自动伸缩轴宽，一块写着 10⁻⁶、一块什么都不写，自动宽度会差出近 30px，叠起来
    # 就是两条错开的左边界。
    gutter = residual.plot_item.getAxis("left").minimumWidth()
    assert gutter == reflectivity.plot_item.getAxis("left").minimumWidth()
    assert gutter == theme.PLOT_LEFT_GUTTER_PX


def test_no_draw_titles_the_pane_from_inside(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles, values = np.array([0.1, 0.2]), np.array([1.0, 0.5])
    plot.show_log_reflectivity(angles, values, None, r_floor=1e-6)
    plot.show_raw_reflectivity(angles, np.array([9.0, 8.0]), np.array([True, True]), None)
    plot.show_qz4(angles, np.array([1e-4, 2e-4]), None, None, ylabel="qz⁴R")
    plot.show_residual(angles, np.array([0.3, -0.3]))
    # The pane is framed by a titled plot card, and pyqtgraph's own title sits
    # inside the axes -- indented by the y-axis width, so it lines up with
    # neither the card header above it nor the plot beneath.  Naming the picture
    # is the card's job; the pane keeps the 30px band that title reserved.
    label = plot.plot_item.titleLabel
    assert (label.text, label.isVisible(), label.maximumHeight()) == ("", False, 0.0)


def test_panes_carry_a_reference_grid_from_construction(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    # draw_* enabled a grid so values read against gridlines; the pg pane keeps it
    # on both axes from the start rather than per draw.
    assert plot.plot_item.ctrl.xGridCheck.isChecked() is True
    assert plot.plot_item.ctrl.yGridCheck.isChecked() is True


def test_placeholder_shows_and_drawing_data_hides_it(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_placeholder("暂无当前候选")
    assert plot.placeholder_text() == "暂无当前候选"
    # Drawing a real series is the pane leaving its placeholder state, matching how
    # the matplotlib qz⁴/residual views drop draw_empty once a candidate arrives.
    plot.show_residual(np.array([0.1, 0.2]), np.array([0.1, -0.1]))
    assert plot.placeholder_text() is None


def test_placeholder_clears_series_and_annotations(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.1, 0.2]), np.array([0.1, -0.1]))
    plot.show_fit_range(0.1, 0.2)
    plot.show_placeholder("暂无当前候选")
    obs_x, _ = plot.observed_item.getData()
    assert obs_x is None or len(obs_x) == 0
    # 空面板上留着上一份拟合窗口的带子和说明，等于给一张没有数据的图配了个区间。
    assert plot.fit_range_caption() is None


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


def test_axes_keep_the_labelled_unit_instead_of_an_si_prefix(qtbot) -> None:
    """qz ticks read in the label's own unit, not a rescaled "(x0.001)" decade.

    pyqtgraph offers to divide the ticks by a power of ten and print the factor
    next to the axis label.  On a qz axis spanning 0..0.4 Å⁻¹ that turns the
    ticks into 0..400 captioned "(x0.001)", so the numbers a reader lifts off
    the axis are a thousand times the quantity the label names.
    """
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.02, 0.2, 0.4]), np.array([0.4, -0.2, 0.1]))
    for side in ("bottom", "left"):
        axis = plot.plot_item.getAxis(side)
        assert axis.autoSIPrefix is False
        assert axis.labelUnitPrefix == ""


def _pen_is_off(pen: object) -> bool:
    """「这条线不画」在 pg 里有两种落地形态，断言得都认。

    构造时传 ``pen=None`` 会把 ``None`` 原样存进 ``opts``；之后再调 ``setPen(None)``
    存进去的却是一支 ``style()`` 为 ``NoPen`` 的 QPen。只认一种，散点面板和残差面板
    互相切换之后就会假红。
    """
    return pen is None or pen.style() == Qt.PenStyle.NoPen


def _range_label(plot: LiveReflectivityPlot) -> pg.TextItem:
    return plot.fit_range_label_item


def test_fit_range_band_carries_its_window_caption_inside_the_plot(qtbot) -> None:
    """设计稿把「拟合窗口 0.5–5.4°」写在带子顶端，而不是靠图例或卡片副标题交代。

    带子只有 16% 不透明度，读者要在同一处知道这段横轴到底覆盖了什么区间；把区间数字
    挪到画外，带子就退化成一块没有刻度含义的色块。
    """
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.5, 3.0, 5.4]), np.array([1.0, 0.1, 0.01]), None, r_floor=1e-6)
    plot.show_fit_range(0.5, 5.4)

    assert plot.fit_range_caption() == "拟合窗口 0.5–5.4°"
    assert _range_label(plot).toPlainText() == "拟合窗口 0.5–5.4°"
    assert _range_label(plot).isVisible() is True
    # 设计稿 font-size:10.5px；Qt 这边按 px*0.75 折成点值。
    assert np.isclose(_range_label(plot).textItem.font().pointSizeF(), 7.875)


def test_fit_range_caption_takes_its_unit_from_the_pane_axis(qtbot) -> None:
    """同一个数字在角度面板和 qz 面板不是一回事，单位必须跟着当前横轴走。"""
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.12, 0.2, 0.28]), np.array([0.4, -0.2, 0.1]))
    plot.show_fit_range(0.28, 0.12)

    # 散射矢量轴的单位是 Å⁻¹，不像度号那样贴着数字写。
    assert plot.fit_range_caption() == "拟合窗口 0.12–0.28 Å⁻¹"


def test_fit_range_caption_follows_a_relabelled_axis(qtbot) -> None:
    """面板换了画法，带子还在，说明里的单位不能留着上一张图的。

    ``_draw_range`` 会在切换视图之后重新铺带子，但重画本身也可能先发生：一个只改
    坐标轴、不动带子的顺序里，写死在 ``show_fit_range`` 里的单位就会留在原地。
    """
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.5, 3.0, 5.4]), np.array([1.0, 0.1, 0.01]), None, r_floor=1e-6)
    plot.show_fit_range(0.5, 5.4)
    plot.show_residual(np.array([0.5, 3.0, 5.4]), np.array([0.4, -0.2, 0.1]))

    assert plot.fit_range_caption() == "拟合窗口 0.5–5.4 Å⁻¹"


def test_fit_range_caption_omits_a_unit_before_any_axis_is_named(qtbot) -> None:
    """还没画过任何一路数据时横轴没有名字，说明就只报数字，不硬凑一个单位。"""
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_fit_range(0.28, 0.12)

    assert plot.fit_range_caption() == "拟合窗口 0.12–0.28"


def test_fit_range_caption_centres_on_the_visible_part_of_the_band(qtbot) -> None:
    """标签跟着「看得见的那段带子」居中，放大到带子一角时不会被推到画外。"""
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.5, 3.0, 5.4]), np.array([1.0, 0.1, 0.01]), None, r_floor=1e-6)
    plot.show_fit_range(0.5, 5.4)

    plot.set_view_xrange(0.0, 6.0)
    (_x0, _x1), (_y0, y1) = plot.plot_item.vb.viewRange()
    _px_x, px_y = plot.plot_item.vb.viewPixelSize()
    position = _range_label(plot).pos()
    assert np.isclose(position.x(), 2.95)
    # 设计稿基线 y=31、画区顶边 y=18，扣掉 10.5px 的 ascent 约剩 4px。
    assert np.isclose(position.y(), y1 - theme.SPACE_XS * px_y, atol=abs(px_y))

    plot.set_view_xrange(3.0, 6.0)
    assert np.isclose(_range_label(plot).pos().x(), 4.2)


def test_clearing_the_fit_range_takes_its_caption_with_it(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_log_reflectivity(np.array([0.5, 3.0, 5.4]), np.array([1.0, 0.1, 0.01]), None, r_floor=1e-6)
    plot.show_fit_range(0.5, 5.4)
    plot.clear_fit_range()

    assert plot.fit_range_caption() is None
    assert _range_label(plot).isVisible() is False


def test_fit_range_caption_reads_against_the_canvas_it_is_drawn_on(qtbot) -> None:
    """标签压在半透明带子上，实际底色是画布，颜色得由画布决定。"""
    application = QApplication.instance()
    assert application is not None
    previous_palette = application.palette()
    try:
        # show_log_reflectivity re-resolves the application palette, so make the
        # first half independent of the desktop appearance running the test.
        application.setPalette(theme.light_palette())
        plot = LiveReflectivityPlot()
        qtbot.addWidget(plot)
        plot.show_log_reflectivity(
            np.array([0.5, 3.0, 5.4]),
            np.array([1.0, 0.1, 0.01]),
            None,
            r_floor=1e-6,
        )
        plot.show_fit_range(0.5, 5.4)
        light = _range_label(plot).color.name().upper()

        plot.apply_palette(theme.DARK_PLOT_PALETTE)
        dark = _range_label(plot).color.name().upper()

        assert light == theme.RANGE_LABEL_ON_LIGHT.upper()
        assert dark == theme.RANGE_LABEL_ON_DARK.upper()
    finally:
        application.setPalette(previous_palette)


def test_residual_pane_names_its_sigma_band_in_the_plot(qtbot) -> None:
    """设计稿在 ±1σ 带子左上角写了「±1σ」，否则那条淡带没有任何说明。"""
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.1, 0.2, 0.3]), np.array([0.4, -0.2, 0.1]))
    plot.set_view_xrange(0.1, 0.3)

    label = plot.sigma_label_item
    assert label.toPlainText() == "±1σ"
    assert label.isVisible() is True
    # 设计稿 font-size:9.5px。
    assert np.isclose(label.textItem.font().pointSizeF(), 7.125)
    (x0, _x1), _y = plot.plot_item.vb.viewRange()
    px_x, _px_y = plot.plot_item.vb.viewPixelSize()
    # 设计稿 x=62 对画区左边 58，差 4px；基线压在带子顶边 y=1.0 之上。
    assert np.isclose(label.pos().x(), x0 + theme.SPACE_XS * px_x, atol=abs(px_x))
    assert np.isclose(label.pos().y(), 1.0)


def test_sigma_label_leaves_with_the_band_on_a_reflectivity_pane(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.show_residual(np.array([0.1, 0.2, 0.3]), np.array([0.4, -0.2, 0.1]))
    plot.show_log_reflectivity(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.5, 0.25]), None, r_floor=1e-6)

    assert plot.sigma_label_item.isVisible() is False


def _draw(plot: LiveReflectivityPlot, key: str) -> None:
    angles = np.array([0.1, 0.2, 0.3])
    values = np.array([1.0, 0.5, 0.25])
    if key == "log":
        plot.show_log_reflectivity(angles, values, None, r_floor=1e-6)
    elif key == "raw":
        plot.show_raw_reflectivity(angles, values, np.array([True, True, True]), None)
    elif key == "qz4":
        plot.show_qz4(angles, values, None, None)
    else:
        plot.show_residual(angles, np.array([0.4, -0.2, 0.1]))


@pytest.mark.parametrize(("key", "diameter"), [("log", 5.2), ("raw", 5.2), ("qz4", 5.2), ("residual", 4.4)])
def test_observed_markers_match_the_designed_diameters(qtbot, key: str, diameter: float) -> None:
    """设计稿的测量点半径 r=2.6（反射率三张）和 r=2.2（残差），不是默认的 6。

    残差面板点上还压着一条连线，点子跟着缩一号，密集区才不会糊成一条带；反射率面板
    只有点，反而要大一点点。写死一个尺寸就必然有一张对不上。
    """
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    _draw(plot, key)

    assert np.isclose(plot.observed_item.opts["symbolSize"], diameter)


def test_floor_marker_matches_the_designed_triangle_size(qtbot) -> None:
    """截断点那个 ▽ 在设计稿里是 6px 宽的小三角，不是比测量点还大的 8。"""
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    _draw(plot, "log")

    assert np.isclose(plot.clipped_item.opts["symbolSize"], 6.0)


def test_residual_line_is_thin_and_translucent_but_leaves_no_pen_behind(qtbot) -> None:
    """残差折线设计稿写的是 stroke-width 1.3 / opacity 0.55，且只属于残差面板。

    同一个 ``observed_item`` 在四张图之间复用：残差面板给它接上连线，切回反射率面板
    必须再摘掉，否则散点图上会多出一条把测量点串起来的假曲线。
    """
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    _draw(plot, "residual")

    pen = plot.observed_item.opts["pen"]
    assert np.isclose(pen.widthF(), 1.3)
    assert pen.color().alpha() == 140

    _draw(plot, "log")
    assert _pen_is_off(plot.observed_item.opts["pen"])
