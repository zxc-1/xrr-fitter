"""设计稿帧③ 那张 SLD 剖面的介质分区、界面手柄与画框。

设计稿把深度轴分成四段竖色带：空气、SiO₂、a-Si、c-Si 各一条，半无限的那两段取中性色，
只有真正的层才着自己那一号色——帧③ 的引文写的就是这一句。带子顶上各写一行「这段是
什么」，界面处一根蓝虚线从曲线落到轴底、末端顶一个白心蓝环的圆点，那是这张图上唯一能改
结构的东西。轴只在左边和下边有框，网格只有横向五条。

此前这张图画的是另一套东西：三根琥珀色 axvline 从轴顶贯到轴底，每层一条 1.6px 琥珀横杠，
每个界面一个 10px 琥珀实心圆点，加起来比设计稿多一倍装饰；分区和分区标注一个没有，纵轴从
0.0 起每 2.5 一格、四面都有边框、纵横两向都铺网格。
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_rgba

import xrr_fitter.api as api

pytest.importorskip("pytestqt")

# 设计稿帧③ 层列表里的那四行（HTML 656-663）：厚度 3.42 / 48.70 nm，粗糙 0.51 / 0.44 /
# 0.30 nm，密度 2.19 / 2.28 / 2.33。分区标注要写出的正是这套读数。
STRUCTURE = api.StructureSpec(
    api.MaterialSpec("Air", None, None, 0.0j),
    (
        api.LayerSpec("SiO₂ · 表面氧化层", api.MaterialSpec("SiO2", "SiO2", 2.19), 34.2, roughness_a=5.1),
        api.LayerSpec("a-Si · 非晶硅薄膜", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),
    ),
    api.MaterialSpec("c-Si · 晶体硅基底", "Si", 2.329),
    backing_roughness_a=3.0,
)

OXIDE_NM = 3.42
TOTAL_NM = 52.12


def _draw(qtbot):
    """画出设计稿帧③ 那张图：只有结构、还没有候选。"""
    from xrr_fitter.gui.plots.panel import PlotPanel
    from xrr_fitter.gui.plots.sld import draw_sld

    panel = PlotPanel()
    qtbot.addWidget(panel)
    view = panel.view("sld")
    draw_sld(view, None, structure=STRUCTURE, wavelength_a=1.5406)
    return view


def _line(view, label: str):
    return next(line for line in view.axes.lines if str(line.get_label()) == label)


def _stroke(line) -> dict[str, object]:
    """一根线画成什么样：颜色、虚线、粗细，以及线头那个点的三笔。

    读成一张表再整张比，而不是一笔一个 ``assert``：手柄的样子是九项一起成立才对得上设计稿，
    逐笔断言时第一笔不合就停，后面八笔的实际值一个都看不到。

    ``_unscaled_dash_pattern`` 是私有的，可 matplotlib 没给虚线间隔留公开读法，而设计稿写的
    正是 ``dasharray="3 3"``——这一项不读就没法区分「虚线」和「设计稿那种虚线」。
    """
    return {
        "color": to_rgba(line.get_color()),
        "linestyle": line.get_linestyle(),
        "dashes": tuple(line._unscaled_dash_pattern[1]),
        "linewidth": line.get_linewidth(),
        "markevery": list(line.get_markevery()),
        "marker_face": to_rgba(line.get_markerfacecolor()),
        "marker_edge": to_rgba(line.get_markeredgecolor()),
        "marker_edge_width": line.get_markeredgewidth(),
        "alpha": line.get_alpha(),
    }


def _spans(view) -> tuple[object, ...]:
    """轴上那几条竖色带，按左边界从浅到深排好。

    设计稿画的是四个 ``<rect>``，matplotlib 里对应 ``axvspan``——它返回的是 ``Rectangle``，
    横向按数据坐标、纵向按轴比例铺满，正是设计稿那种 ``y=16 height=180`` 的整条竖带。
    """
    from matplotlib.patches import Rectangle

    spans = [patch for patch in view.axes.patches if isinstance(patch, Rectangle)]
    return tuple(sorted(spans, key=lambda patch: float(patch.get_x())))


def _extent(span) -> tuple[float, float]:
    left = float(span.get_x())
    return left, left + float(span.get_width())


def _captions(view) -> tuple[object, ...]:
    """带子顶上那几行分区标注，按写下的横坐标从浅到深排好。"""
    return tuple(sorted(view.axes.texts, key=lambda text: float(text.get_position()[0])))


def test_each_medium_along_the_depth_axis_gets_its_own_band(qtbot) -> None:
    """设计稿在图区里画了四条竖色带（HTML 675 的四个 rect），一段介质一条。

    没有它们，读者只能从曲线的台阶反推「哪一段是氧化层」；而这张图的用处恰恰是一眼认出
    分层。两条层带按层的边界精确落位，两条半无限带铺到画框边缘——设计稿的 rect 就是从左
    边框铺到 0、从 52.12 铺到右边框。
    """
    view = _draw(qtbot)

    spans = _spans(view)
    left, right = view.axes.get_xlim()

    assert len(spans) == 4
    assert _extent(spans[1]) == pytest.approx((0.0, OXIDE_NM))
    assert _extent(spans[2]) == pytest.approx((OXIDE_NM, TOTAL_NM))
    assert _extent(spans[0])[0] <= left
    assert _extent(spans[0])[1] == pytest.approx(0.0)
    assert _extent(spans[3])[0] == pytest.approx(TOTAL_NM)
    assert _extent(spans[3])[1] >= right


def test_only_the_real_layers_take_a_colour(qtbot) -> None:
    """帧③ 的引文：「空气与基底是半无限介质，取中性色，只有真正的层才着色。」

    层带取的必须是层列表那一行的色块同一号色（``component_fill``），两处同源读者才能把
    带子和行对上；设计稿的 rect 写的正是 ``--d-observed`` 和 ``--d-candidate``，也就是
    ``DATA_SEQUENCE`` 的头两号。
    """
    from xrr_fitter.gui import theme
    from xrr_fitter.gui.structure.stack import component_fill

    spans = _spans(_draw(qtbot))

    assert spans[0].get_facecolor() == pytest.approx(to_rgba(theme.DATA_NEUTRAL, 0.05))
    assert spans[3].get_facecolor() == pytest.approx(to_rgba(theme.DATA_NEUTRAL, 0.05))
    assert spans[1].get_facecolor() == pytest.approx(to_rgba(component_fill(0), 0.10))
    assert spans[2].get_facecolor() == pytest.approx(to_rgba(component_fill(1), 0.10))


def test_every_band_says_which_medium_it_is(qtbot) -> None:
    """设计稿每条带子顶上都写了一行（HTML 675 里四个 ``y=29`` 的 text）。

    色带只说明「这里换了介质」，写出名字和厚度才让人不必回头数层列表。名字取的是层自己
    的名字（``naming.expert_name`` 的前半段，与树里那一行同源），厚度以 nm 记一位小数——
    设计稿写的正是「SiO₂ 3.4nm」「a-Si 48.7nm」，半无限的两段只写「空气」「c-Si 基底」。
    """
    from xrr_fitter.gui import theme

    captions = _captions(_draw(qtbot))

    assert [text.get_text() for text in captions] == ["空气", "SiO₂ 3.4nm", "a-Si 48.7nm", "c-Si 基底"]
    for text in captions:
        assert text.get_horizontalalignment() == "center"
        assert text.get_verticalalignment() == "top"
        assert text.get_fontweight() == "semibold"
        assert text.get_fontsize() == pytest.approx(theme.FONT_PT_SM)
        # 设计稿把这行字压在图区顶边下方一点（y=29 落在 16…196 这条画框里）。
        assert 0.95 <= float(text.get_position()[1]) <= 1.0


def test_each_caption_sits_over_the_middle_of_what_is_visible(qtbot) -> None:
    """标注要落在「看得见的那一段」的正中，而不是整段介质的正中。

    空气和基底是半无限的，中点无从谈起；设计稿把这两行字放在画框内那一截的中间——空气那
    行在 x=74.5px 即 −3.0nm，基底那行在 404.8px 即 57.05nm。所以位置必须在定好视野之后算。
    """
    view = _draw(qtbot)
    left, right = view.axes.get_xlim()
    positions = [float(text.get_position()[0]) for text in _captions(view)]

    assert positions[0] == pytest.approx(0.5 * (left + 0.0))
    assert positions[1] == pytest.approx(0.5 * OXIDE_NM)
    assert positions[2] == pytest.approx(0.5 * (OXIDE_NM + TOTAL_NM))
    assert positions[3] == pytest.approx(0.5 * (TOTAL_NM + right))


def test_the_depth_axis_frames_the_stack_the_way_the_design_does(qtbot) -> None:
    """设计稿的视野是 −6.0…62.0nm，不是「采样网格再放 5%」。

    标称网格只铺到 −2.55…53.62nm，照它取景基底那条带只剩一丝，空气那条带几乎看不见；设计
    稿留的是按总厚度算的边距，两侧各占画框 8.82% 和 14.53%。用比例而不是常数，换一套层厚
    时构图仍然成立；再与网格取 min/max，曲线永远不会被裁掉。
    """
    view = _draw(qtbot)
    left, right = view.axes.get_xlim()
    span = right - left

    assert (left, right) == pytest.approx((-0.115 * TOTAL_NM, 1.19 * TOTAL_NM))
    assert (0.0 - left) / span == pytest.approx(0.0882, abs=0.001)
    assert (right - TOTAL_NM) / span == pytest.approx(0.1453, abs=0.002)


def test_the_sld_axis_stops_just_above_the_profile_and_ticks_in_fives(qtbot) -> None:
    """设计稿的纵轴刻度是 0 / 5 / 10 / 15 / 20，顶到 22.0（即峰值的 1.10 倍）。

    此前这张图从 0.0 起每 2.5 一格，画出十来条横线，把一条只有三级台阶的曲线埋在网格里。
    五条横线是设计稿的读数密度，也正好让每级台阶各自贴着一条线。
    """
    from xrr_fitter.gui.plots.sld import CANDIDATE_REAL_LABEL

    view = _draw(qtbot)
    curve = _line(view, CANDIDATE_REAL_LABEL)
    peak = float(np.max(curve.get_ydata()))
    bottom, top = view.axes.get_ylim()

    assert (bottom, top) == pytest.approx((0.0, 1.10 * peak))
    in_view = [float(tick) for tick in view.axes.get_yticks() if bottom <= tick <= top]
    assert in_view == pytest.approx([0.0, 5.0, 10.0, 15.0, 20.0])


def test_an_interface_handle_is_a_dashed_accent_drop_topped_by_a_ringed_dot(qtbot) -> None:
    """设计稿的界面手柄只有两笔：一根从曲线落到轴底的蓝虚线，线头一个白心蓝环的圆点。

    蓝是这套设计里「可以动」的意思（图例那行写着「可拖拽界面」），所以手柄必须跟着主题的
    accent 走；此前用的琥珀色是「范围」的意思，而且 axvline 从轴顶贯到轴底，把「界面在曲线
    这个高度」这条信息也抹掉了。圆点单独给白心蓝环，不能靠整条线的 alpha——那会把点也调淡。
    """
    from xrr_fitter.gui import theme

    view = _draw(qtbot)
    accent = theme.current_accent()
    handle = _line(view, "_interface_1")

    assert tuple(handle.get_xdata()) == pytest.approx((TOTAL_NM, TOTAL_NM))
    # 线从轴底（0）落到曲线那个高度，而不是从轴顶贯到轴底——那会把「界面在这个 SLD 上」抹掉。
    assert float(handle.get_ydata()[0]) == pytest.approx(0.0)
    assert float(handle.get_ydata()[1]) > 0.0
    # 圆点单独给白心蓝环，``alpha`` 留空：整条线调淡会把点也调淡。
    assert _stroke(handle) == {
        "color": pytest.approx(to_rgba(accent, 0.6)),
        "linestyle": "--",
        "dashes": (3, 3),
        "linewidth": pytest.approx(1.0),
        "markevery": [1],
        "marker_face": pytest.approx(to_rgba("#FFFFFF")),
        "marker_edge": pytest.approx(to_rgba(accent)),
        "marker_edge_width": pytest.approx(2.0),
        "alpha": None,
    }


def test_the_surface_is_marked_without_becoming_a_handle(qtbot) -> None:
    """设计稿在 x=91px（深度 0）也画了虚线和圆点，一共三组手柄样式的装饰。

    但表面不是可拖的界面——层的边界是 3.42 和 52.12 两处。这一笔取的标签必须落在
    ``sld_drag`` 认的 ``_interface_{i}`` 之外，否则按下它会被当成某个界面而改错层厚。
    """
    from xrr_fitter.gui.plots.sld import draggable_interfaces

    view = _draw(qtbot)
    surface = _line(view, "_surface")
    grabbable = {f"_interface_{index}" for index, _ in draggable_interfaces(STRUCTURE)}

    assert tuple(surface.get_xdata()) == pytest.approx((0.0, 0.0))
    assert "_surface" not in grabbable
    assert grabbable == {"_interface_0", "_interface_1"}
    assert all(float(_line(view, label).get_xdata()[0]) > 0.0 for label in grabbable)


def test_a_roughness_whisker_does_not_grow_a_second_dot(qtbot) -> None:
    """设计稿一个界面只有一个圆点。

    此前每个界面画两个：界面处一个 10px 的琥珀实心点，粗糙度末端再一个。设计稿这套读数下
    σ 只有 0.44 / 0.30nm，折成 2.4 / 1.65px，末端那个点整个压在界面那个点底下，看上去就是
    重影。所以这一笔留下线段本身（真正粗糙的界面才看得出宽度，也仍然能拖），但不带点。
    """
    from xrr_fitter.gui import theme

    view = _draw(qtbot)
    whisker = _line(view, "_roughness_0")

    assert tuple(whisker.get_xdata()) == pytest.approx((OXIDE_NM, OXIDE_NM + 0.44))
    assert str(whisker.get_marker()) == "None"
    assert to_rgba(whisker.get_color()) == pytest.approx(to_rgba(theme.current_accent(), 0.6))
    assert whisker.get_linewidth() == pytest.approx(1.0)


def test_a_level_bar_is_the_curve_segment_it_edits(qtbot) -> None:
    """设计稿里没有第二条横杠——层的高度就是曲线自己那一段平台。

    此前这条杠是 1.6px、alpha 0.7 的琥珀线，压在 2.2px 的曲线上，看着像曲线破了个口子。
    改成与曲线同色同粗、不带 alpha，它就是曲线本身，同时仍然是 ``_level_{i}`` 那个可拖的把手。
    """
    from xrr_fitter.gui.plots.sld import CANDIDATE_REAL_LABEL

    view = _draw(qtbot)
    curve = _line(view, CANDIDATE_REAL_LABEL)
    level = _line(view, "_level_1")

    assert to_rgba(level.get_color()) == pytest.approx(to_rgba(curve.get_color()))
    assert level.get_linewidth() == pytest.approx(curve.get_linewidth())
    assert level.get_linestyle() == curve.get_linestyle()
    assert level.get_alpha() is None
    assert str(level.get_marker()) == "None"


def test_only_the_horizontal_gridlines_are_left(qtbot) -> None:
    """设计稿只画五条横线（HTML 675 的 5 个 y 相同的 line），一条竖线也没有。

    竖向的分界已经由色带和界面虚线交代过了，再铺一层竖网格就是三套线在同一处重复。次刻度
    也一并去掉——``apply_axes_grid`` 会装上 ``AutoMinorLocator``，留着就多出一批没画完的短线。
    """
    view = _draw(qtbot)
    axes = view.axes
    horizontal = axes.get_ygridlines()

    assert horizontal
    assert all(line.get_visible() for line in horizontal)
    assert not any(line.get_visible() for line in axes.get_xgridlines())
    assert len(axes.xaxis.get_minorticklocs()) == 0
    assert len(axes.yaxis.get_minorticklocs()) == 0


def test_the_frame_is_open_on_the_top_and_the_right(qtbot) -> None:
    """设计稿只在左边和下边描了框（x=58 与 y=196 两条 ``rgba(27,31,39,.45)`` 线）。

    ``apply_figure_palette`` 会把四条边都描上，围出一个盒子；设计稿要的是开放式坐标轴，右上
    两边留白，视线不被框住。
    """
    spines = _draw(qtbot).axes.spines

    assert spines["left"].get_visible()
    assert spines["bottom"].get_visible()
    assert not spines["top"].get_visible()
    assert not spines["right"].get_visible()


def test_the_handle_colour_follows_the_live_appearance(qtbot) -> None:
    """手柄的蓝要跟着当前主题走，和界面上其他「可操作」的蓝是同一支。

    ``plot_palette`` 里没有 accent 这一项，绘图模块此前拿不到它。深色主题的 accent 是
    ``#5A8DEE`` 而不是 ``#2F6BD8``，写死任何一个都会让另一套主题下的手柄跑色。
    """
    from PySide6.QtWidgets import QApplication

    from xrr_fitter.gui import theme

    _ = qtbot
    tokens = theme.palette_tokens(QApplication.instance().palette())

    assert theme.current_accent() == tokens.accent
