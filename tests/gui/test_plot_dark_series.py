"""深色外观下图上每一笔都读得出来，数据系列只用 Okabe-Ito 那一套颜色。

审计在深色外观下实测：工具栏字形在创建那一刻按当时的调色板烧死，切到深色后是深墨压在深底上；
matplotlib 图例沿用默认的白底黑字；热图色标与相关矩阵的读数要等下一次投影才换；没传颜色的
曲线落进 matplotlib 默认的 tab10 轮转，17 条参数剖面只有 10 种颜色，第 1 条和第 11 条同色。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgba
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QIcon, QImage
from PySide6.QtWidgets import QApplication
from tests.gui.plot_support import _panel, _result, _uncertainty
from tests.support.model_cases import prepared_data

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.model.inference import CovarianceEvidence

# matplotlib 默认轮转 tab10 的十个色。它和 Okabe-Ito 没有一个色重合，所以图上出现其中任何一个，
# 就是有一条曲线没拿到显式颜色、落进了默认轮转。
TAB10 = frozenset(
    (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
)


@pytest.fixture
def dark(qapp):
    """从亮色起步，返回一个把整个应用切到深色的开关；用完换回来。"""
    previous_palette = qapp.palette()
    previous_stylesheet = qapp.styleSheet()
    previous_light = theme.LIGHT_PALETTE
    theme.LIGHT_PALETTE = None
    qapp.setPalette(theme.light_palette())
    theme.apply_theme(qapp)
    try:
        yield lambda: theme.set_dark_appearance(qapp, True)
    finally:
        theme.set_dark_appearance(qapp, False)
        theme.LIGHT_PALETTE = previous_light
        qapp.setPalette(previous_palette)
        qapp.setStyleSheet(previous_stylesheet)


def _ink(icon: QIcon, mode: QIcon.Mode = QIcon.Mode.Normal) -> QColor:
    """字形落墨最重的那一格的颜色：抗锯齿边缘是半透明的，最实的那一格才是笔的颜色。"""
    image = icon.pixmap(QSize(16, 16), mode).toImage().convertToFormat(QImage.Format.Format_ARGB32)
    pixels = (image.pixelColor(x, y) for x in range(image.width()) for y in range(image.height()))
    return max(pixels, key=lambda colour: colour.alpha())


def _line_colours(figure) -> set[str]:
    return {to_hex(line.get_color()) for axes in figure.axes for line in axes.lines}


def test_toolbar_glyphs_repaint_in_the_text_colour_after_switching_to_dark(qtbot, dark) -> None:
    from xrr_fitter.gui.plots.plot_icons import plot_icon
    from xrr_fitter.gui.plots.toolbar import PlotInteractionToolbar

    toolbar = PlotInteractionToolbar()
    qtbot.addWidget(toolbar)
    icons = [action.icon() for action in toolbar.tool_actions() if not action.isSeparator()]
    icons += [button.icon() for button in toolbar.navigation_buttons().values()]
    icons.append(plot_icon("start_fit"))

    dark()

    text = QApplication.palette().windowText().color()
    for icon in icons:
        assert abs(_ink(icon).lightness() - text.lightness()) <= 16


def test_a_disabled_glyph_still_reads_dimmer_than_an_enabled_one_on_dark(qtbot, dark) -> None:
    from xrr_fitter.gui.plots.plot_icons import plot_icon

    icon = plot_icon("sld_profile")
    dark()

    window = QApplication.palette().window().color().lightness()
    normal = abs(_ink(icon).lightness() - window)
    disabled = abs(_ink(icon, QIcon.Mode.Disabled).lightness() - window)
    assert 0 < disabled < normal


def test_a_matplotlib_legend_wears_the_canvas_colours_after_switching_to_dark(qtbot, dark) -> None:
    panel = _panel(qtbot)
    panel.set_batch_trends(("样品一", "样品二"), (100.0, 120.0), (50.0, 55.0))

    dark()

    palette = theme.DARK_PLOT_PALETTE
    legend = panel.view("trend").axes.get_legend()
    assert legend is not None
    assert legend.get_frame().get_facecolor()[:3] == pytest.approx(palette.background[:3])
    for text in legend.get_texts():
        assert text.get_color()[:3] == pytest.approx(palette.foreground[:3])


def test_a_legend_already_on_a_standalone_canvas_follows_the_switch(qtbot, dark) -> None:
    """不归绘图面板管的画布（不确定度对话框那几页）没人替它重画，图例得在主题钩子里自己换色。"""
    from xrr_fitter.gui.plots.diagnostics import build_tabs, draw_batch_trends

    reflectivity, analysis, views = build_tabs()
    qtbot.addWidget(reflectivity)
    qtbot.addWidget(analysis)
    draw_batch_trends(views["trend"], (("样品一", "样品二"), (100.0, 120.0), (50.0, 55.0)))

    dark()

    legend = views["trend"].axes.get_legend()
    assert legend.get_frame().get_facecolor()[:3] == pytest.approx(theme.DARK_PLOT_PALETTE.background[:3])


def _weak_pair_result(data):
    names = ("component.0.thickness_a", "instrument.scale")
    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    report = replace(
        _uncertainty(),
        correlation_matrix=matrix,
        covariance_evidence=CovarianceEvidence(names, matrix, "gaussian_known_sigma", 2),
    )
    return replace(_result(data), uncertainty=report)


def test_heatmap_and_matrix_colours_follow_the_switch_without_a_new_projection(qtbot, dark) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_weak_pair_result(data))

    dark()

    palette = theme.DARK_PLOT_PALETTE
    for key in ("residual_map", "uncertainty"):
        image = panel.view(key).axes.images[0]
        assert image.cmap(0.5)[:3] == pytest.approx(palette.diverging_mid[:3], abs=0.01)
    caption = next(text for text in panel.view("residual_map").axes.texts if text.get_text().startswith("固定范围"))
    assert caption.get_color()[:3] == pytest.approx(palette.foreground[:3])
    weak = next(text for text in panel.view("uncertainty").axes.texts if text.get_text() == "+0.20")
    assert weak.get_color()[:3] == pytest.approx(palette.foreground[:3])


def test_placeholder_messages_are_not_left_in_the_default_black(qtbot, dark) -> None:
    """没传颜色的说明字落成 matplotlib 默认的黑，在深色画布上就看不见了。"""
    from xrr_fitter.gui.plots.diagnostics import draw_candidate_comparison

    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    dark()
    draw_candidate_comparison(panel.view("candidates"), _result(data), None)
    empty = _panel(qtbot)
    draw_candidate_comparison(empty.view("candidates"), None, None)

    black = to_rgba("black")
    figures = (panel.view("candidates").figure, empty.view("candidates").figure, panel.view("uncertainty").figure)
    texts = [text for figure in figures for axes in figure.axes for text in axes.texts if text.get_text()]
    assert {text.get_text() for text in texts} >= {"尚未选择候选解", "暂无候选解", "剖面与区间证据未执行/不可用"}
    for text in texts:
        assert to_rgba(text.get_color()) != black, text.get_text()


def _profiles(count: int) -> tuple[api.ParameterProfile, ...]:
    offsets = np.linspace(-1.0, 1.0, 5)
    return tuple(
        api.ParameterProfile(
            f"component.{index}.thickness_a",
            1.0 + offsets,
            1.0 + (1.0 + 0.1 * index) * offsets**2,
            True,
            True,
            objective_threshold=2.0,
        )
        for index in range(count)
    )


def _evidence_panel(qtbot, profile_count: int = 3):
    data = prepared_data(size=4)
    result = replace(_result(data), uncertainty=_uncertainty(profiles=_profiles(profile_count)))
    panel = _panel(qtbot, data=data, result=result)
    panel.set_batch_trends(("样品一", "样品二", "样品三"), (100.0, 120.0, 140.0), (50.0, 55.0, 60.0))
    return panel


def test_no_series_falls_back_to_the_default_tab10_cycle(qtbot) -> None:
    """系列色只取 Okabe-Ito：tab10 的红绿对色觉缺陷读者是同一种颜色，深浅两种画布上的对比也没人校过。"""
    panel = _evidence_panel(qtbot)

    for key in ("candidates", "sld", "uncertainty", "trend"):
        assert not _line_colours(panel.view(key).figure) & TAB10, key


def test_seventeen_profile_curves_never_share_a_look(qtbot) -> None:
    """17 个自由参数是真实样品的规模：颜色轮完一圈之后要换线型与记号，任何两条曲线都分得开。"""
    from xrr_fitter.gui.plots.sld import PROFILE_TITLE

    panel = _evidence_panel(qtbot, profile_count=17)

    figure = panel.view("uncertainty").figure
    axes = next(axes for axes in figure.axes if axes.get_title() == PROFILE_TITLE)
    curves = [line for line in axes.lines if not str(line.get_label()).startswith("区间闭合阈值")]
    looks = {(to_hex(line.get_color()), line.get_linestyle(), line.get_marker()) for line in curves}
    assert len(curves) == 17
    assert len(looks) == 17


def test_the_inspected_sld_profile_keeps_the_candidate_hue_beside_overlays(qtbot) -> None:
    """卡片色标说「当前 SLD 剖面」是候选橙；对比候选先画时，默认轮转会把蓝色发给它。"""
    from xrr_fitter.gui.plots.sld import CANDIDATE_REAL_LABEL

    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))

    lines = {str(line.get_label()): to_hex(line.get_color()) for line in panel.view("sld").axes.lines}
    assert lines[CANDIDATE_REAL_LABEL] == theme.DATA_CANDIDATE.lower()
    assert lines["SLD 虚部"] == theme.DATA_CANDIDATE.lower()
