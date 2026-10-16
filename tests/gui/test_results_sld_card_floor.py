"""结果步加上拟合 SLD 剖面卡、卡被压到地板高度时，图例只能占宽度，不能占剖面的高度（F16）。

结果步的画布列里还排着反射率与残差两张卡，SLD 卡加进来只分到 ``MIN_SLD_PLOT_H`` 那条地板。
拟合剖面的轴内图例有四行（虚部、标称、外层不确定带、其他候选），比地板高度下的轴还高；
constrained_layout 为了给它让边距会把轴压成一条线，读者看到的是挤扁的剖面、被裁掉一半的图例。
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget
from tests.gui.plot_cases_5 import _sld_project

import xrr_fitter.api as api


def _floored_results_sld_view(controls, tmp_path):
    project, _result = _sld_project(tmp_path, "curve")
    saved = tmp_path / "saved-sld.xrrproj.json"
    api.save_project(project, saved)
    window = controls.window(api.load_project(saved))
    controls.named_click(window, "pipelineStep_结果")
    panel = window.plot_panel
    controls.context_menu(panel.toolbar, panel.toolbar.rect().center(), "plotToolAction:sld_profile")
    pane = panel.findChild(QWidget, "sldPane")
    splitter = panel.plot_splitter
    sizes = splitter.sizes()
    sizes[splitter.indexOf(pane)] = 0
    splitter.setSizes(sizes)
    controls.qtbot.waitUntil(lambda: pane.height() == pane.minimumHeight())
    return panel.view("sld")


def test_the_fitted_profile_key_costs_width_not_plot_height(controls, tmp_path) -> None:
    view = _floored_results_sld_view(controls, tmp_path)
    legend = view.axes.get_legend()
    assert len(legend.get_texts()) > 1

    # 轴的窗口范围是跟着轴位置走的活对象，量完得冻住，否则两次读到的是同一个最新值。
    view.canvas.draw()
    renderer = view.canvas.get_renderer()
    with_key = view.axes.get_window_extent(renderer).frozen()
    key = legend.get_window_extent(renderer).frozen()
    legend.set_visible(False)
    view.canvas.draw()
    without_key = view.axes.get_window_extent(renderer).frozen()
    legend.set_visible(True)

    assert abs(with_key.height - without_key.height) <= 1.0
    assert key.x0 >= with_key.x1
    assert 0.0 <= key.y0 and key.y1 <= view.figure.bbox.height and key.x1 <= view.figure.bbox.width
