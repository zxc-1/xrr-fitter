"""设计稿帧③ 那张 SLD 深度剖面的两条轴和一份图例。

轴此前写作「深度 (nm)」与「SLD (Å⁻²)」。前者没说深度从哪头算起——同一张图既可以从表面
往基底走，也可以反过来，而剖面正是靠这个方向才读得出「先是氧化层，然后才是薄膜」。后者
的单位对不上画出来的数：X 射线的 SLD 落在 10⁻⁶ Å⁻² 这个量级，标成 Å⁻² 就得让刻度写成
1.9e-05，读者得自己数六位小数才知道自己在看什么。

图例此前只有曲线，没有那几个手柄。手柄是这张图上唯一能改结构的东西，不进图例就只是几根
橙色虚线。
"""

from __future__ import annotations

import numpy as np
import pytest

import xrr_fitter.api as api

pytest.importorskip("pytestqt")

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)

STRUCTURE = api.StructureSpec(
    AIR,
    (
        api.LayerSpec("SiO2", SIO2, 34.2, roughness_a=5.1),
        api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),
    ),
    SI,
    backing_roughness_a=3.0,
)


def _view(qtbot):
    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    qtbot.addWidget(panel)
    return panel.view("sld")


def _candidate():
    from dataclasses import replace

    from tests.support.model_cases import fit_candidate

    return replace(
        fit_candidate("candidate-a", 0.2),
        sld_depth_a=np.array([0.0, 200.0, 520.0]),
        sld_profile_a2=np.array([0.0 + 0.0j, 2e-5 + 1e-7j, 4e-6 + 0.0j]),
    )


def _labels(view) -> tuple[str, ...]:
    legend = view.axes.get_legend()
    return () if legend is None else tuple(text.get_text() for text in legend.get_texts())


def test_the_depth_axis_says_which_way_is_down(qtbot) -> None:
    """设计稿写「深度 z（nm，从表面向基底）」。

    只写「深度 (nm)」的话，剖面从左到右是往里还是往外全靠猜；而这张图的整个读法——先氧
    化层、后薄膜、最后基底——就建立在这个方向上。
    """
    from xrr_fitter.gui.plots.sld import draw_sld

    view = _view(qtbot)

    draw_sld(view, _candidate())

    assert view.axes.get_xlabel() == "深度 z（nm，从表面向基底）"


def test_the_sld_axis_states_the_unit_the_numbers_are_actually_in(qtbot) -> None:
    """纵轴标成 10⁻⁶ Å⁻²，画出来的数也换算到这个单位。

    只改标注不换算等于在科学图上写错单位；只换算不改标注则让刻度平白大出六个数量级。
    两件事必须一起做。
    """
    from xrr_fitter.gui.plots.sld import draw_sld

    view = _view(qtbot)
    candidate = _candidate()

    draw_sld(view, candidate)

    assert view.axes.get_ylabel() == "散射长度密度 SLD（10⁻⁶ Å⁻²）"
    real = next(line for line in view.axes.lines if line.get_label() == "当前 SLD 剖面")
    np.testing.assert_allclose(real.get_ydata(), candidate.sld_profile_a2.real * 1e6)


def test_the_curve_carries_the_name_the_card_key_uses(qtbot) -> None:
    """曲线自己叫「当前 SLD 剖面」，不是「SLD 实部」。

    「实部」是把复折射率的记法直接搬到了图上；读者要认的是「这条就是我现在这套结构的
    剖面」，虚部另有一条说自己是虚部。这个名字同时是卡片色标那一行的字（两处同源，见
    ``sld_state.SLD_LEGEND_ENTRIES``），改一处两处一起改。
    """
    from xrr_fitter.gui.plots.sld import draw_sld

    view = _view(qtbot)

    draw_sld(view, _candidate())

    assert any(line.get_label() == "当前 SLD 剖面" for line in view.axes.lines)


def test_with_no_candidate_the_editable_profile_is_the_current_profile(qtbot) -> None:
    """还没拟合时，读者在拖的那条标称剖面就是「当前 SLD 剖面」。

    设计稿帧③ 的图里只有一条曲线，候选橙实线，手柄骑在它上面。把它画成灰点线再叫「结构
    标称」，等于在一张没有第二条曲线的图上说「这是拿来对比的那条」。

    有候选时它才真是第二条曲线（拿手上的声明比对拟合结果），那时才叫回「结构标称 实部」。
    """
    from xrr_fitter.gui.plots.sld import draw_sld

    view = _view(qtbot)

    draw_sld(view, None, structure=STRUCTURE, wavelength_a=1.5406)

    labels = tuple(str(line.get_label()) for line in view.axes.lines)
    assert "当前 SLD 剖面" in labels
    assert "结构标称 实部" not in labels
    # 手柄仍按原名字画：``SldDragMixin._sld_handle`` 精确匹配它，改名就拖不动。
    assert "_interface_0" in labels


def test_the_in_axes_key_does_not_repeat_the_cards_own_key(qtbot) -> None:
    """卡片抬头下面那排色标说过的记号，轴里不再画一遍。

    设计稿帧③ 的图区里没有图例框——四条 pyqtgraph 面板同样只有卡片色标、不画内嵌图例。
    这里画一份就是同一张卡里两份图例，还正好压在剖面上。
    """
    from xrr_fitter.gui.plots.sld import CARD_KEY_LABELS, draw_sld

    view = _view(qtbot)

    draw_sld(view, None, structure=STRUCTURE, wavelength_a=1.5406)

    assert not set(_labels(view)) & set(CARD_KEY_LABELS)


def test_a_pane_with_nothing_left_to_key_draws_no_key_at_all(qtbot) -> None:
    """色标说完了就没有图例框——那正是设计稿帧③ 的样子。

    留一个空框比多一行还糟：一个没有字的白盒子盖在曲线上。
    """
    from xrr_fitter.gui.plots.sld import draw_sld

    view = _view(qtbot)

    draw_sld(view, None, structure=STRUCTURE, wavelength_a=1.5406)

    assert view.axes.get_legend() is None


def test_a_second_opinion_still_gets_keyed(qtbot) -> None:
    """叠在拟合结果上的结构标称是色标说不到的第二条曲线，轴里就得有一行。

    不画的话，图上多出一条灰点线却没有任何地方说它是什么。
    """
    from xrr_fitter.gui.plots.sld import draw_sld

    view = _view(qtbot)

    draw_sld(view, _candidate(), structure=STRUCTURE, wavelength_a=1.5406)

    assert "结构标称 实部" in _labels(view)


def test_the_uncertainty_band_is_named_as_a_band(qtbot) -> None:
    """设计稿写「16–84% 不确定带」，不是光秃秃的「16–84%」。

    单写一个区间，读者得先猜它说的是哪个量的百分位；这张图上同时还有分位数着色的对比
    候选，两者都能被读成「16–84%」。这个名字同样和卡片色标那一行同源。
    """
    from xrr_fitter.gui.plots.sld import BAND_PAIRS

    assert BAND_PAIRS[0][2] == "16–84% 不确定带"
    assert BAND_PAIRS[1][2] == "2.5–97.5% 不确定带"


def test_the_band_is_drawn_in_the_same_unit_as_the_curve(qtbot) -> None:
    """带和曲线同乘一个换算：只换算曲线，带就会缩成轴底的一条线。"""
    from xrr_fitter.gui.plots.sld import draw_sld
    from xrr_fitter.model.sld_bands import SldUncertaintyBands

    view = _view(qtbot)
    bands = SldUncertaintyBands(
        depth_a=np.array([0.0, 200.0, 520.0]),
        quantiles=(0.16, 0.84),
        real=np.array([[0.0, 1.8e-5, 3.6e-6], [0.0, 2.2e-5, 4.4e-6]]),
        imaginary=np.array([[0.0, 0.9e-7, 0.0], [0.0, 1.1e-7, 0.0]]),
        align_label="厚度 d",
        sample_count=64,
        total_samples=64,
        failure_rate=0.0,
    )

    draw_sld(view, _candidate(), bands=bands)

    top = max(
        float(np.max(path.vertices[:, 1])) for collection in view.axes.collections for path in collection.get_paths()
    )
    assert top == pytest.approx(22.0, rel=1e-6)
