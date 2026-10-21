"""坐标轴说的必须是同一件事，读得出数，名字是读者认得的那个。

审计实测：残差卡写着「加权残差 vs 散射矢量 qz」，横轴是 qz，压在横轴是 2θ 的反射率卡正下方，同一个
数据点在两张卡上不在同一条竖线上，缩放其中一张另一张也不跟；残差纵轴一个刻度数字都没有，残差一大，
自动缩放把 ±1σ 带压成一条线，读者既看不出偏了几个 σ，也没有带子可比；热图和后验标题写的是
``component.0.thickness_a`` 这种机器路径。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.gui.plot_support import _panel, _result, _uncertainty
from tests.support.model_cases import final_fit_result, fit_candidate, prepared_data

import xrr_fitter.api as api
from xrr_fitter.gui.plots.live import LiveReflectivityPlot
from xrr_fitter.model.parameters import ParameterValue

NAMES = ("component.0.thickness_a", "component.0.roughness_a")
DISPLAYS = ("ox 厚度", "ox 粗糙度")


def test_the_residual_pane_shares_the_reflectivity_angle_axis_and_reads_its_values(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    residual, log = panel.view("residual"), panel.view("log")

    assert residual.axis_labels()[0] == log.axis_labels()[0] == "散射角 2θ (°)"
    x, _y = residual.observed_item.getData()
    assert np.allclose(x, data.two_theta_deg)
    # 残差一大，带子和虚线都出了框，纵轴上的数字是剩下唯一的读数。
    assert residual.plot_item.getAxis("left").style["showValues"] is True


def _gesture_zoom(pane: LiveReflectivityPlot, low: float, high: float) -> None:
    """滚轮、拖动、框选落到 ViewBox 上都是这两步：改量程，再报一声「这是读者改的」。"""
    view_box = pane.plot_item.vb
    view_box.setXRange(low, high, padding=0)
    view_box.sigRangeChangedManually.emit([True, False])


def test_zooming_the_reflectivity_pane_carries_the_residual_with_it(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    low, high = (float(value) for value in data.two_theta_deg[1:3])

    _gesture_zoom(panel.view("log"), low, high)

    for key in ("residual", "raw"):
        x0, x1 = panel.view(key).plot_item.vb.viewRange()[0]
        assert np.isclose(x0, low) and np.isclose(x1, high), key


def test_zoom_to_the_fit_range_moves_the_residual_too(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    low, high = (float(value) for value in data.two_theta_deg[1:3])
    panel.show_range(low, high)

    assert panel.zoom_to_range()

    x0, x1 = panel.view("residual").plot_item.vb.viewRange()[0]
    assert np.isclose(x0, low) and np.isclose(x1, high)


def test_a_large_residual_is_clamped_to_an_edge_marker_so_the_band_stays_readable(qtbot) -> None:
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    angles = np.linspace(0.2, 2.0, 6)
    plot.show_residual(angles, np.array([0.4, -0.3, 40.0, 0.2, -12.0, 0.1]))
    plot.plot_item.vb.autoRange()

    y0, y1 = plot.plot_item.vb.viewRange()[1]
    assert 2.0 / (y1 - y0) > 0.15
    edge_x, edge_y = plot.residual_edge_item.getData()
    assert np.allclose(edge_x, (angles[2], angles[4]))
    assert np.allclose(np.abs(edge_y), 5.0)


def _definition(name: str, display_name: str) -> api.ParameterDefinition:
    return api.ParameterDefinition(
        name=name,
        display_name=display_name,
        unit="Å",
        category="layer",
        initial=10.0,
        lower=0.0,
        upper=20.0,
        transform="linear",
        locked=False,
    )


def _two_parameter_result() -> api.FitResult:
    def candidate(candidate_id: str, objective: float, thickness: float) -> object:
        parameters = (
            ParameterValue(NAMES[0], thickness, 0.0, 20.0),
            ParameterValue(NAMES[1], 3.0, 0.0, 20.0),
        )
        return replace(fit_candidate(candidate_id, objective), parameters=parameters)

    result = final_fit_result(candidate("candidate-a", 0.2, 12.0), candidate("candidate-b", 0.3, 14.0))
    return replace(result, parameter_definitions=tuple(map(_definition, NAMES, DISPLAYS)))


def test_the_parameter_heatmap_names_its_columns_the_way_the_matrix_does(qtbot) -> None:
    from xrr_fitter.gui.plots.diagnostics import build_tabs
    from xrr_fitter.gui.plots.heatmaps import draw_parameter_heatmap

    reflectivity, analysis, views = build_tabs()
    qtbot.addWidget(reflectivity)
    qtbot.addWidget(analysis)
    draw_parameter_heatmap(views["parameter_map"], _two_parameter_result(), "candidate-a")

    labels = [label.get_text() for label in views["parameter_map"].axes.get_xticklabels()]
    assert labels == ["d·ox", "σ·ox"]


def test_the_posterior_marginals_are_titled_with_the_quantity_not_the_path(qtbot) -> None:
    from xrr_fitter.gui.plots.posterior import McmcHistogramPlot

    chain = api.McmcReport(
        config=api.McmcConfig(walkers=4, burn_in=0, production_steps=4),
        child_seed=7,
        parameter_names=NAMES,
        samples_physical=np.array([[10.0, 3.0], [11.0, 3.2], [12.0, 2.9], [13.0, 3.1]]),
        log_probability=np.zeros(4),
        acceptance_fraction=np.full(4, 0.4),
        split_rhat=np.ones(2),
        effective_sample_size=np.full(2, 4.0),
        boundary_hits=(),
        candidate_id="candidate-a",
    )
    result = replace(_two_parameter_result(), uncertainty=replace(_uncertainty(), mcmc=chain))
    plot = McmcHistogramPlot()
    qtbot.addWidget(plot)

    plot.show_report(result, "candidate-a")

    assert [axes.get_title() for axes in plot.figure.axes] == ["厚度 d·ox", "粗糙度 σ·ox"]
